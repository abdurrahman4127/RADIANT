import numpy as np
import torch

from tqdm import tqdm
from modeling.losses import avg_dice_metric


def validate_on_loader(
    model,
    loader,
    device,
    name="val",
):
    """
    runs the model in evaluation mode over a loader and returns the
    average per-region Dice score across all batches.
    """
    model.eval()

    dice_acc = np.zeros(3)
    steps = 0

    with torch.no_grad():
        for vols, regions, _ in tqdm(
            loader,
            desc=f"{name} eval",
        ):
            vols = vols.to(device)
            regions = regions.to(device)

            preds, _ = model(vols)

            dice_acc += avg_dice_metric(
                preds,
                regions,
            )

            steps += 1

    return (
        dice_acc / max(1, steps)
    ).tolist()