import argparse
import os

import torch
from torch.utils.data import DataLoader

from core import config, data
from training.evaluate import validate_on_loader
from modeling.models import UNet3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    hgg_dir = os.path.join(args.root, "HGG")
    lgg_dir = os.path.join(args.root, "LGG")
    hgg_train, hgg_val, lgg_train, lgg_val = data.build_splits(hgg_dir, lgg_dir)

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

    model = UNet3D(in_ch=4, base_ch=16, out_ch=3).to(config.device)
    ckpt = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(ckpt["model"])

    hgg_dice = validate_on_loader(model, hgg_val_loader, config.device, "hgg")
    lgg_dice = validate_on_loader(model, lgg_val_loader, config.device, "lgg")

    print("hgg dice:", hgg_dice)
    print("lgg dice:", lgg_dice)


if __name__ == "__main__":
    main()
