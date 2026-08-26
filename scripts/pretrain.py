import argparse
import os

import torch
from torch.utils.data import DataLoader

from core import config, data, radiomics
from training.evaluate import validate_on_loader
from modeling.models import RadRegressor, UNet3D, global_pool
from training.train import pretrain_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--rad-feats-path", default=config.rad_feats_save)
    args = parser.parse_args()

    config.set_seed()
    os.makedirs(config.ckpt_dir, exist_ok=True)

    hgg_dir = os.path.join(args.root, "HGG")
    lgg_dir = os.path.join(args.root, "LGG")
    hgg_train, hgg_val, _, _ = data.build_splits(hgg_dir, lgg_dir)

    train_loader = DataLoader(
        data.Brats3DDataset(hgg_train, labeled=True, augment=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        data.Brats3DDataset(hgg_val, labeled=True, augment=False),
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

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(rad_reg.parameters()),
        lr=config.sup_lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.sup_step_size,
        gamma=config.sup_gamma,
    )

    for epoch in range(1, config.sup_epochs + 1):
        stats = pretrain_epoch(
            epoch,
            model,
            rad_reg,
            train_loader,
            optimizer,
            rad_lookup,
            rad_out_dim,
            config.device,
        )

        val_dice = validate_on_loader(
            model, 
            val_loader, 
            config.device, 
            "hgg val"
        )

        scheduler.step()

        print(
            f"epoch {epoch} loss {stats['loss']:.4f} "
            f"seg {stats['seg']:.4f} rad {stats['rad']:.4f} "
            f"train dice {stats['dice']} val dice {val_dice}"
        )

        if epoch % 5 == 0 or epoch == config.sup_epochs:
            torch.save(
                {
                    "model": model.state_dict(),
                    "rad": rad_reg.state_dict(),
                    "optim": optimizer.state_dict(),
                },
                os.path.join(config.ckpt_dir, f"sup_epoch_{epoch}.pth"),
            )


if __name__ == "__main__":
    main()
