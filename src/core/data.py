import glob
import os
import random
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from core import config


# returns a sorted list of case directories found under a domain folder
def list_case_dirs(domain_dir):
    return sorted(glob.glob(os.path.join(domain_dir, "BraTS*")))

# loads a nifti volume from disk and returns it as a float32 array
def read_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)


"""
centers a crop of the input array within out_shape, zero-padding
any dimension where the input is smaller than the target
"""
def center_crop_or_pad(arr, out_shape):
    z0, y0, x0 = arr.shape
    oz, oy, ox = out_shape
    res = np.zeros(out_shape, dtype=arr.dtype)

    sz, sy, sx = min(z0, oz), min(y0, oy), min(x0, ox)
    zs, ys, xs = (z0 - sz) // 2, (y0 - sy) // 2, (x0 - sx) // 2
    rs, ry, rx = (oz - sz) // 2, (oy - sy) // 2, (ox - sx) // 2

    res[rs:rs + sz, ry:ry + sy, rx:rx + sx] = arr[zs:zs + sz, ys:ys + sy, xs:xs + sx]
    return res


# Standardizes a volume to zero mean and unit variance
def normalize_image(img):
    m = img.mean()
    s = img.std()

    return (img - m) / (s + 1e-8)


def load_case(
    case_dir,
    patch_size=config.patch_size,
):
    """
    Reads all four modalities and the ground truth mask for one case,
    center-crops each to patch_size, normalizes each modality
    independently, and stacks them into a single 4-channel volume.
    """
    base = os.path.basename(case_dir)

    flair = read_nifti(
        os.path.join(case_dir, f"{base}_flair.nii")
    )
    t1 = read_nifti(
        os.path.join(case_dir, f"{base}_t1.nii")
    )
    t1ce = read_nifti(
        os.path.join(case_dir, f"{base}_t1ce.nii")
    )
    t2 = read_nifti(
        os.path.join(case_dir, f"{base}_t2.nii")
    )
    seg = read_nifti(
        os.path.join(case_dir, f"{base}_seg.nii")
    )

    flair_c = center_crop_or_pad(
        flair,
        patch_size,
    )

    t1_c = center_crop_or_pad(
        t1,
        patch_size,
    )

    t1ce_c = center_crop_or_pad(
        t1ce,
        patch_size,
    )

    t2_c = center_crop_or_pad(
        t2,
        patch_size,
    )

    seg_c = center_crop_or_pad(
        seg,
        patch_size,
    ).astype(np.int16)

    vol = np.stack(
        [
            normalize_image(flair_c),
            normalize_image(t1_c),
            normalize_image(t1ce_c),
            normalize_image(t2_c),
        ],
        axis=0,
    ).astype(np.float32)

    return vol, seg_c


def get_brain_bbox(image):
    """
    finds the bounding box of nonzero voxels across the first channel
    of a 4d volume, or across a 3d volume directly, and returns none
    when no foreground voxels are present
    """
    mask = image[0] > 0 if image.ndim == 4 else image > 0

    if mask.sum() == 0:
        return None

    coords = np.argwhere(mask)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    return (
        slice(z_min, z_max + 1),
        slice(y_min, y_max + 1),
        slice(x_min, x_max + 1),
    )


def crop_around_center(vol, center, patch_size):
    """
    extracts a fixed-size patch centered at a given 3d coordinate,
    zero-padding any side that extends past the volume boundary
    """
    c, z, y, x = vol.shape
    pz, py, px = patch_size
    cz, cy, cx = center

    z_start = max(0, cz - pz // 2)
    y_start = max(0, cy - py // 2)
    x_start = max(0, cx - px // 2)

    z_end = min(z, z_start + pz)
    y_end = min(y, y_start + py)
    x_end = min(x, x_start + px)

    crop = vol[:, z_start:z_end, y_start:y_end, x_start:x_end]

    pad_z = pz - (z_end - z_start)
    pad_y = py - (y_end - y_start)
    pad_x = px - (x_end - x_start)

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        crop = np.pad(
            crop,
            ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
        )

    return crop


def smart_load_case(
    case_dir,
    patch_size=config.patch_size,
    labeled=True,
    force_tumor_crop=False,
):
    """
    Loads a case with content-aware cropping, first isolating the brain
    region to skip large black borders, then optionally centering the
    crop on tumor voxels when force_tumor_crop is set.
    """
    base = os.path.basename(case_dir)

    flair = read_nifti(
        os.path.join(case_dir, f"{base}_flair.nii")
    )
    
    t1 = read_nifti(
        os.path.join(case_dir, f"{base}_t1.nii")
    )
    
    t1ce = read_nifti(
        os.path.join(case_dir, f"{base}_t1ce.nii")
    )
    
    t2 = read_nifti(
        os.path.join(case_dir, f"{base}_t2.nii")
    )
    
    seg = read_nifti(
        os.path.join(case_dir, f"{base}_seg.nii")
    ).astype(np.int16)

    vol_raw = np.stack(
        [flair, t1, t1ce, t2],
        axis=0,
    ).astype(np.float32)

    bbox = get_brain_bbox(vol_raw)

    if bbox is None:
        return (
            np.zeros((4,) + patch_size, dtype=np.float32),
            np.zeros(patch_size, dtype=np.int16),
        )

    vol_brain = vol_raw[:, bbox[0], bbox[1], bbox[2]]
    seg_brain = seg[bbox[0], bbox[1], bbox[2]]

    if labeled and force_tumor_crop:
        tumor_coords = np.argwhere(seg_brain > 0)

        center = (
            random.choice(tumor_coords)
            if len(tumor_coords) > 0
            else np.array(seg_brain.shape) // 2
        )
    else:
        center = np.array(seg_brain.shape) // 2

    seg_brain_exp = seg_brain[None, ...]

    vol_crop = crop_around_center(
        vol_brain,
        center,
        patch_size,
    )

    seg_crop = crop_around_center(
        seg_brain_exp,
        center,
        patch_size,
    )[0]

    vol_norm = [
        normalize_image(vol_crop[ch])
        for ch in range(4)
    ]

    return (
        np.stack(vol_norm, axis=0).astype(np.float32),
        seg_crop,
    )


def seg_to_regions(seg):
    """
    converts a multi-class segmentation map into three binary channels
    for whole tumor, tumor core, and enhancing tumor
    """
    wt = (seg > 0).astype(np.float32)
    tc = ((seg == 1) | (seg == 4)).astype(np.float32)
    et = (seg == 4).astype(np.float32)
    return np.stack([wt, tc, et], axis=0)


def has_complete_tumor(
    case_dir,
    min_wt=10,
    min_tc=5,
    min_et=5,
):
    """
    Checks whether a case has a minimum voxel count in each tumor
    region, used to filter out cases with tiny or missing tumors.
    """
    try:
        _, seg = load_case(case_dir)

        wt = (seg > 0).sum()
        tc = ((seg == 1) | (seg == 4)).sum()
        et = (seg == 4).sum()

        return (
            wt >= min_wt
            and tc >= min_tc
            and et >= min_et
        )

    except Exception:
        return False


class Brats3DDataset(Dataset):
    """
    wraps a list of case directories and returns loaded volumes on
    demand, applying foreground-biased cropping and light augmentation
    when enabled
    """

    def __init__(self, case_dirs: 
                 List[str], 
                 labeled=True, 
                 augment=False):
        self.cases = case_dirs
        self.labeled = labeled
        self.augment = augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        cdir = self.cases[idx]

        force_tumor = (
            self.labeled and self.augment and random.random() < 0.8
        )

        vol, seg = smart_load_case(
            cdir,
            patch_size=config.patch_size,
            labeled=self.labeled,
            force_tumor_crop=force_tumor,
        )

        if self.augment and random.random() < 0.5:
            vol = np.flip(vol, axis=2).copy()
            seg = np.flip(seg, axis=1).copy()

        vol_t = torch.from_numpy(vol).float()

        if self.labeled:
            regs = seg_to_regions(seg)
            return vol_t, torch.from_numpy(regs).float(), cdir

        dummy = np.zeros((3,) + config.patch_size, dtype=np.float32)
        return vol_t, torch.from_numpy(dummy).float(), cdir


def build_splits(hgg_dir, lgg_dir, train_frac=0.7):
    """
    builds hgg and lgg train/val splits, then filters the lgg splits
    down to cases with a meaningful tumor in all three regions
    """
    hgg_cases = list_case_dirs(hgg_dir)
    lgg_cases = list_case_dirs(lgg_dir)

    random.shuffle(hgg_cases)
    split_idx = int(train_frac * len(hgg_cases))
    hgg_train = hgg_cases[:split_idx]
    hgg_val = hgg_cases[split_idx:]
    if len(hgg_val) == 0:
        hgg_val = hgg_train[: max(1, len(hgg_train) // 10)]

    random.shuffle(lgg_cases)
    lgg_split_idx = int(train_frac * len(lgg_cases))
    lgg_train = lgg_cases[:lgg_split_idx]
    lgg_val = lgg_cases[lgg_split_idx:]
    if len(lgg_val) == 0:
        lgg_val = lgg_train[: max(1, len(lgg_train) // 10)]

    lgg_train = [c for c in lgg_train if has_complete_tumor(c)]
    lgg_val = [c for c in lgg_val if has_complete_tumor(c)]

    return hgg_train, hgg_val, lgg_train, lgg_val
