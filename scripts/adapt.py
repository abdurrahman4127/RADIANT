import argparse
import os

import torch
from torch.utils.data import DataLoader

from core import config, data, radiomics
from training.evaluate import validate_on_loader
from modeling.losses import SelfFeedbackScheduler
from modeling.models import (
    DomainDiscriminator,
    RadRegressor,
    UNet3D,
    decoder_parameters,
    encoder_parameters,
    global_pool,
)
from training.train import adapt_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--rad-feats-path", default=config.rad_feats_save)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config.set_seed()
    os.makedirs(config.ckpt_dir, exist_ok=True)

    hgg_dir = os.path.join(args.root, "HGG")
    lgg_dir = os.path.join(args.root, "LGG")
    hgg_train, hgg_val, lgg_train, lgg_val = data.build_splits(hgg_dir, lgg_dir)

    hgg_loader = DataLoader(
        data.Brats3DDataset(hgg_train, labeled=True, augment=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    lgg_loader = DataLoader(
        data.Brats3DDataset(lgg_train, labeled=False, augment=False),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    hgg_val_loader = DataLoader(
        data.Brats3DDataset(hgg_val, labeled=True, augment=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    lgg_val_loader = DataLoader(
        data.Brats3DDataset(lgg_val, labeled=True, augment=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    rad_feats, rad_keys, rad_cases = radiomics.load_hgg_radiomics(args.rad_feats_path)
    rad_mean, rad_std = radiomics.build_normalization_stats(rad_feats)
    rad_lookup = radiomics.build_case_lookup(rad_feats, rad_cases, rad_mean, rad_std)
    rad_out_dim = rad_feats.shape[1]

    model = UNet3D(in_ch=4, base_ch=16, out_ch=3).to(config.device)
    with torch.no_grad():
        dummy = torch.randn(1, 4, *config.patch_size).to(config.device)
        _, bott = model(dummy)
        feat_dim = global_pool(bott).shape[1]

    rad_reg = RadRegressor(in_dim=feat_dim, out_dim=rad_out_dim).to(config.device)
    discriminator = DomainDiscriminator(in_dim=feat_dim).to(config.device)

    ckpt = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(ckpt["model"])
    rad_reg.load_state_dict(ckpt["rad"])

    encoder_params = encoder_parameters(model)
    decoder_params = decoder_parameters(model)

    adapt_optimizer = torch.optim.Adam(
        [
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": decoder_params + list(rad_reg.parameters()), "lr": config.decoder_lr},
        ],
        weight_decay=config.weight_decay,
    )

    adapt_scheduler = torch.optim.lr_scheduler.StepLR(
        adapt_optimizer,
        step_size=config.adapt_step_size,
        gamma=config.adapt_gamma,
    )

    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.disc_lr,
        betas=config.disc_betas,
    )

    self_feedback = SelfFeedbackScheduler(
        beta_max=config.self_feedback_beta_max,
        e_start=config.self_feedback_e_start,
        e_end=config.self_feedback_e_end,
    )

    baseline_lgg = validate_on_loader(
        model, 
        lgg_val_loader, 
        config.device, 
        "lgg baseline"
    )
    
    print("lgg baseline before adaptation:", baseline_lgg)

    for epoch in range(1, config.adapt_epochs + 1):
        grl_alpha = config.grl_alpha_start + (
            epoch / config.adapt_epochs
        ) * (config.grl_alpha_end - config.grl_alpha_start)

        stats = adapt_epoch(
            epoch,
            model,
            rad_reg,
            discriminator,
            hgg_loader,
            lgg_loader,
            adapt_optimizer,
            disc_optimizer,
            self_feedback,
            rad_lookup,
            rad_keys,
            rad_mean,
            rad_std,
            rad_out_dim,
            config.device,
            grl_alpha,
        )
        adapt_scheduler.step()

        print(
            f"epoch {epoch} seg {stats['seg']:.4f} rad_h {stats['rad_h']:.4f} "
            f"rad_cons {stats['rad_cons']:.4f} adv {stats['adv']:.4f} "
            f"disc {stats['disc']:.4f} coral {stats['coral']:.4f}"
        )

        if epoch % 5 == 0 or epoch == config.adapt_epochs:
            torch.save(
                {
                    "model": model.state_dict(),
                    "rad": rad_reg.state_dict(),
                    "disc": discriminator.state_dict(),
                },
                os.path.join(config.ckpt_dir, f"adapt_epoch_{epoch}.pth"),
            )

    final_lgg = validate_on_loader(model, lgg_val_loader, config.device, "lgg final")
    final_hgg = validate_on_loader(model, hgg_val_loader, config.device, "hgg final")

    print("final lgg dice:", final_lgg)
    print("final hgg dice:", final_hgg)


if __name__ == "__main__":
    main()
