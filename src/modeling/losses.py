from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

dice_region_weights = (1.0, 2.5, 3.5)


def dice_per_channel(pred, target, eps=1e-6):
    """
    computes the dice score independently for each output channel,
    treating an empty prediction and an empty target as a perfect
    match rather than a near zero score
    """
    p = pred.reshape(pred.size(0), pred.size(1), -1)
    t = target.reshape(target.size(0), target.size(1), -1)

    inter = (p * t).sum(-1)
    union = p.sum(-1) + t.sum(-1)
    dice = (2 * inter + eps) / (union + eps)

    dice = torch.where(union < 1.0, torch.ones_like(dice), dice)
    return dice


def avg_dice_metric(pred, target):
    """
    averages the per channel dice score across the batch and
    returns it as a plain numpy array
    """
    return dice_per_channel(pred, target).mean(dim=0).detach().cpu().numpy()


def dice_loss(pred, target):
    """
    weighted dice loss across whole tumor, tumor core, and
    enhancing tumor regions
    """
    dices = dice_per_channel(pred, target)
    weights = torch.tensor(dice_region_weights, device=pred.device)
    weighted_dice = (dices * weights).sum() / weights.sum()
    return 1.0 - weighted_dice


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    focal loss variant of binary cross entropy that down-weights
    easy examples to counter class imbalance
    """
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target > 0.5, pred, 1 - pred)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def seg_loss(pred, target, focal_weight=0.5):
    """
    combined dice and focal loss used for both source and target
    segmentation supervision
    """
    return dice_loss(pred, target) + focal_weight * focal_loss(pred, target)


def coral_loss(source, target):
    """
    correlation alignment loss between pooled source and target
    bottleneck features, matching second order statistics without
    any gradient reversal
    """
    d = source.size(1)
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)

    source_c = (
        (source - source_mean).t().mm(source - source_mean)
        / (source.size(0) - 1 + 1e-6)
    )
    target_c = (
        (target - target_mean).t().mm(target - target_mean)
        / (target.size(0) - 1 + 1e-6)
    )

    return ((source_c - target_c) ** 2).sum() / (4.0 * (d ** 2))


def compute_iou(
    mask_a,
    mask_b,
    eps=1e-6,
):
    """
    computes intersection over union between two hard binary masks
    of identical shape, treating two empty masks as a perfect match.
    """
    intersection = float(
        np.logical_and(mask_a, mask_b).sum()
    )

    union = float(
        np.logical_or(mask_a, mask_b).sum()
    )

    if union == 0:
        return 1.0

    return intersection / (union + eps)


class SelfFeedbackScheduler:
    """
    tracks each target case's previous prediction across epochs and
    produces the dynamic self consistency weight from the warmup
    factor and the intersection over union stability score
    """

    def __init__(
        self,
        beta_max,
        e_start,
        e_end,
    ):
        self.beta_max = beta_max
        self.e_start = e_start
        self.e_end = e_end
        self.prev_preds = {}

    def warmup_factor(self, epoch):
        """
        returns a value between zero and one that ramps linearly
        from e_start to e_end and stays at zero before e_start
        """
        if epoch <= self.e_start:
            return 0.0

        frac = (epoch - self.e_start) / max(1, (self.e_end - self.e_start))
        return float(min(1.0, max(0.0, frac)))


    def step(
        self,
        epoch,
        case_names,
        hard_masks,
    ):
        """
        computes the average self consistency weight across a batch
        of cases and updates the stored previous prediction for
        each case.
        """
        beta_e = self.warmup_factor(epoch)

        if beta_e == 0.0:
            for name, mask in zip(case_names, hard_masks):
                self.prev_preds[name] = mask
            return 0.0

        ious = []
        for name, mask in zip(case_names, hard_masks):
            if name in self.prev_preds:
                ious.append(compute_iou(mask, self.prev_preds[name]))
            else:
                ious.append(1.0)
            self.prev_preds[name] = mask

        q = float(np.mean(ious)) if len(ious) > 0 else 1.0
        return beta_e * self.beta_max * q
