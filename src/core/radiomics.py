from collections import OrderedDict
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from skimage import measure
from skimage.morphology import ball, binary_closing
from tqdm import tqdm

try:
    from radiomics import featureextractor
    pyradiomics_available = True
except Exception:
    pyradiomics_available = False

radiomics_settings = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
}

radiomics_enabled_classes = [
    "firstorder",
    "shape",
    "glcm",
    "glrlm",
]


def get_extractor():
    """
    builds a pyradiomics extractor restricted to the four enabled
    feature classes, disabling every other class the library would
    otherwise enable by default
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings.update(radiomics_settings)
    extractor.disableAllFeatures()
    for cls in radiomics_enabled_classes:
        extractor.enableFeatureClassByName(cls)
    return extractor


def compute_simple_geometric(mask):
    """
    computes volume, surface area, compactness, and eccentricity as
    fallback shape descriptors when extraction through pyradiomics
    fails or is unavailable
    """
    od = OrderedDict()
    m = (mask > 0).astype(np.uint8)
    vol_vox = int(m.sum())
    od["volume"] = float(vol_vox)

    if vol_vox == 0:
        od["surface_area"] = 0.0
        od["compactness"] = 0.0
        od["eccentricity"] = 0.0
        return od

    closed = binary_closing(m, ball(1))
    try:
        verts, faces, _, _ = measure.marching_cubes(closed.astype(np.float32), 0)
        tris = verts[faces]
        a = np.linalg.norm(tris[:, 0] - tris[:, 1], axis=1)
        b = np.linalg.norm(tris[:, 1] - tris[:, 2], axis=1)
        c = np.linalg.norm(tris[:, 2] - tris[:, 0], axis=1)
        s = 0.5 * (a + b + c)
        area = np.sum(np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0)))
        od["surface_area"] = float(area)
    except Exception:
        od["surface_area"] = 0.0

    od["compactness"] = (
        float((vol_vox ** 2) / (od["surface_area"] + 1e-8))
        if od["surface_area"] > 0
        else 0.0
    )

    try:
        labels = measure.label(m)
        props = measure.regionprops(labels)
        ecc = max([getattr(p, "eccentricity", 0.0) for p in props], default=0.0)
        od["eccentricity"] = float(ecc)
    except Exception:
        od["eccentricity"] = 0.0

    return od


def extract_radiomics(image, mask):
    """
    extracts a 72-dimensional radiomics vector from an image and a
    binary mask, restricted to first order, shape, glcm, and glrlm
    features, falling back to geometric proxies on failure or on an
    empty mask
    """
    if not pyradiomics_available or mask.sum() == 0:
        return compute_simple_geometric(mask)

    try:
        img_nii = nib.Nifti1Image(image.astype(np.float32), np.eye(4))
        mask_nii = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
        extractor = get_extractor()
        result = extractor.execute(img_nii, mask_nii)

        od = OrderedDict()
        for k, v in result.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and "original" in k:
                od[k] = float(v)

        return od if len(od) > 0 else compute_simple_geometric(mask)
    except Exception:
        return compute_simple_geometric(mask)


def flatten_od(od, keys_ref):
    """
    converts a feature dictionary into a fixed-length vector following
    a fixed key order, filling any missing key with zero.
    """
    return np.asarray(
        [
            float(od.get(k, 0.0))
            for k in keys_ref
        ],
        dtype=np.float32,
    )


def precompute_hgg_radiomics(
    case_dirs,
    load_case_fn,
    save_path,
):
    """
    extracts radiomics features for every case in case_dirs and
    caches the resulting feature matrix, key list, and case list to
    a compressed npz file.
    """
    feats = []
    keys_ref = None

    for c in tqdm(case_dirs, desc="precomputing hgg radiomics"):
        vol, seg = load_case_fn(c)
        img = vol[0]
        mask = (seg > 0).astype(np.uint8)

        od = extract_radiomics(img, mask)

        vec = flatten_od(
            od,
            keys_ref
            if keys_ref is not None
            else list(od.keys()),
        )

        if keys_ref is None:
            keys_ref = list(od.keys())

        feats.append(vec)

    feats = np.stack(feats, axis=0).astype(np.float32)

    np.savez_compressed(
        save_path,
        feats=feats,
        keys=keys_ref,
        cases=case_dirs,
    )
    return feats, keys_ref, case_dirs


def load_hgg_radiomics(path):
    """
    loads a cached radiomics matrix and confirms it has the expected
    seventy two dimensional feature vector before returning it
    """
    data = np.load(path, allow_pickle=True)
    feats = data["feats"].astype(np.float32)
    keys = data["keys"].tolist()
    cases = data["cases"].tolist()

    if feats.shape[1] != 72:
        raise ValueError(
            f"loaded radiomics cache has dimensionality {feats.shape[1]}, "
            f"expected 72"
        )

    return feats, keys, cases


def build_normalization_stats(feats):
    """
    computes per-feature mean and standard deviation across the
    source domain cache, clamping near-zero variance features to
    avoid division blowups
    """
    mean = np.mean(feats, axis=0)
    std = np.std(feats, axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def build_case_lookup(feats, cases, mean, std):
    """
    builds a dictionary mapping each case path to its normalized
    radiomics vector for fast lookup during training
    """
    return {cases[i]: (feats[i] - mean) / std for i in range(len(cases))}


def normalize_vector(vec, mean, std):
    """
    normalizes a single radiomics vector using precomputed source
    domain statistics
    """
    return (vec - mean) / std
