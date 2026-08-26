import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core import config, radiomics
from modeling.losses import avg_dice_metric, coral_loss, seg_loss
from modeling.models import global_pool


def pretrain_epoch(
    epoch,
    model,
    rad_reg,
    loader,
    optimizer,
    rad_lookup,
    rad_out_dim,
    device,
):
    """
    runs one epoch of supervised training on the source domain,
    jointly optimizing segmentation accuracy and radiomics
    regression from ground truth annotations
    """
    model.train()
    rad_reg.train()

    running_loss = 0.0
    seg_loss_acc = 0.0
    rad_loss_acc = 0.0
    dice_acc = np.zeros(3)
    steps = 0

    for vols, regions, cases in tqdm(loader, desc=f"sup epoch {epoch}"):
        vols = vols.to(device)
        regions = regions.to(device)

        preds, bott = model(vols)
        seg_l = seg_loss(preds, regions)

        pooled = global_pool(bott)
        batch_rad = np.stack(
            [
                rad_lookup.get(c, np.zeros(rad_out_dim, dtype=np.float32))
                for c in cases
            ],
            axis=0,
        )
        batch_rad = torch.from_numpy(batch_rad).to(device)

        pred_rad = rad_reg(pooled)
        rad_l = F.mse_loss(pred_rad, batch_rad)

        rad_weight = (
            config.rad_weight_early
            if epoch <= config.rad_weight_switch_epoch
            else config.rad_weight_late
        )
        loss = seg_l + rad_weight * rad_l

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(rad_reg.parameters()),
            config.clip_grad,
        )
        optimizer.step()

        running_loss += loss.item()
        seg_loss_acc += seg_l.item()
        rad_loss_acc += rad_l.item()
        dice_acc += avg_dice_metric(preds, regions)
        steps += 1

    return {
        "loss": running_loss / max(1, steps),
        "seg": seg_loss_acc / max(1, steps),
        "rad": rad_loss_acc / max(1, steps),
        "dice": (dice_acc / max(1, steps)).tolist(),
    }


def discriminator_step(
    discriminator,
    disc_optimizer,
    pooled_src,
    pooled_tgt,
    device,
    alpha=1.0,
):
    """
    updates the domain discriminator alone using detached source and
    target features, keeping the encoder untouched by this step
    """
    discriminator.train()

    src_label = torch.ones(pooled_src.size(0), 1, device=device)
    tgt_label = torch.zeros(pooled_tgt.size(0), 1, device=device)

    disc_out_src = discriminator(pooled_src.detach(), grl_alpha=alpha)
    disc_out_tgt = discriminator(pooled_tgt.detach(), grl_alpha=alpha)

    loss_src = F.binary_cross_entropy(disc_out_src, src_label)
    loss_tgt = F.binary_cross_entropy(disc_out_tgt, tgt_label)
    loss = 0.5 * (loss_src + loss_tgt)

    disc_optimizer.zero_grad()
    loss.backward()
    disc_optimizer.step()

    return loss.item()


def adapt_epoch(
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
    device,
    grl_alpha,
):
    """
    runs one epoch of unsupervised adaptation from the source domain
    to the target domain, combining source segmentation, source
    radiomics regression, target radiomics self consistency,
    adversarial domain alignment, and correlation alignment
    """
    model.train()
    rad_reg.train()
    discriminator.train()

    totals = {
        "seg": 0.0,
        "rad_h": 0.0,
        "rad_cons": 0.0,
        "adv": 0.0,
        "disc": 0.0,
        "coral": 0.0,
        "loss": 0.0,
    }
    steps = 0

    current_lambda_rad = (
        0.0 if epoch <= config.rad_cons_warmup_epochs 
        else config.lambda_rad_cons
    )

    hgg_iter = iter(hgg_loader)

    for lvols, _, lcases in tqdm(lgg_loader, desc=f"adapt epoch {epoch}"):
        try:
            hvols, hregions, hcases = next(hgg_iter)
        except StopIteration:
            hgg_iter = iter(hgg_loader)
            hvols, hregions, hcases = next(hgg_iter)

        hvols = hvols.to(device)
        hregions = hregions.to(device)
        lvols = lvols.to(device)

        hp, hb = model(hvols)
        seg_l = seg_loss(hp, hregions)
        pooled_h = global_pool(hb)

        batch_rad_h = np.stack(
            [
                rad_lookup.get(c, np.zeros(rad_out_dim, dtype=np.float32))
                for c in hcases
            ],
            axis=0,
        )

        batch_rad_h = torch.from_numpy(batch_rad_h).to(device)
        pred_rad_h = rad_reg(pooled_h)
        rad_h_loss = F.mse_loss(pred_rad_h, batch_rad_h)

        lp, lb = model(lvols)
        pooled_l = global_pool(lb)

        lpred_np = (lp.detach().cpu().numpy()[:, 0] > 0.5).astype(np.uint8)

        rad_consistency = torch.tensor(0.0, device=device)
        beta_self = 0.0

        if current_lambda_rad > 0.0:
            batch_rad_l = []
            for b in range(lpred_np.shape[0]):
                mask = lpred_np[b]
                img_np = lvols[b].detach().cpu().numpy()[0]

                if mask.sum() == 0:
                    vec_norm = np.zeros(rad_out_dim, dtype=np.float32)
                else:
                    od = radiomics.extract_radiomics(img_np, mask)
                    vec = radiomics.flatten_od(od, rad_keys)
                    vec_norm = radiomics.normalize_vector(vec, rad_mean, rad_std)

                batch_rad_l.append(vec_norm)

            batch_rad_l = torch.from_numpy(
                np.stack(batch_rad_l, axis=0)
            ).to(device)
            pred_rad_l = rad_reg(pooled_l)
            rad_consistency = F.mse_loss(pred_rad_l, batch_rad_l)

            beta_self = self_feedback.step(epoch, lcases, list(lpred_np))
            rad_consistency = rad_consistency * beta_self
        else:
            self_feedback.step(epoch, lcases, list(lpred_np))

        disc_out_tgt_for_gen = discriminator(
            pooled_l, 
            grl_alpha=grl_alpha
        )
        
        adv_loss = F.binary_cross_entropy(
            disc_out_tgt_for_gen, torch.ones_like(disc_out_tgt_for_gen)
        )

        coral_l = coral_loss(pooled_h, pooled_l)

        loss_model = (
            seg_l
            + config.rad_h_weight * rad_h_loss
            + rad_consistency
            + config.lambda_adv * adv_loss
            + config.lambda_coral * coral_l
        )

        adapt_optimizer.zero_grad()
        loss_model.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(rad_reg.parameters()),
            config.clip_grad,
        )
        adapt_optimizer.step()

        disc_loss = discriminator_step(
            discriminator,
            disc_optimizer,
            pooled_h,
            pooled_l,
            device,
            alpha=grl_alpha,
        )

        totals["seg"] += seg_l.item()
        totals["rad_h"] += rad_h_loss.item()
        totals["rad_cons"] += rad_consistency.item()
        totals["adv"] += adv_loss.item()
        totals["disc"] += disc_loss
        totals["coral"] += coral_l.item()
        totals["loss"] += loss_model.item()
        steps += 1

    for k in totals:
        totals[k] = totals[k] / max(1, steps)

    return totals
