"""Framework-free helpers (NumPy / OpenCV only) shared by training, evaluation and tests.

Keeping these free of PyTorch makes them cheap to unit-test and reuse.
"""
import cv2
import numpy as np


# ---------------------------------------------------------------- data / splitting
def to_uint8(img):
    """Scale a 8/16-bit slice into uint8 without wrap-around (16-bit TIFFs are common)."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img // 257).astype(np.uint8)  # 65535 -> 255
    raise ValueError(f"Unsupported slice dtype: {img.dtype}")


def val_region_start(height, val_frac=0.2):
    """First row of the held-out validation band (bottom `val_frac` of the fragment)."""
    return int(round(height * (1.0 - val_frac)))


def tile_origins(y0, y1, x0, x1, tile_size, stride, cover=True):
    """Top-left corners of tiles that fit fully inside [y0:y1, x0:x1].

    cover=True shifts the last row/col of tiles to touch the border so the whole region is covered
    (used for inference and overlapping training tiles). cover=False drops any remainder, which
    keeps non-overlapping validation tiles from counting a pixel twice.
    """
    def axis(a0, a1):
        n = a1 - a0
        if n < tile_size:
            return []
        pos = list(range(a0, a1 - tile_size + 1, stride))
        if cover and pos[-1] != a1 - tile_size:
            pos.append(a1 - tile_size)
        return pos
    return [(y, x) for y in axis(y0, y1) for x in axis(x0, x1)]


def spatial_split_tiles(mask, tile_size=256, stride=128, val_frac=0.2, min_mask_frac=0.05):
    """Split tiles into train / val by a *spatial* holdout with zero pixel overlap.

    Validation = bottom `val_frac` band of the fragment (validation tiles do not overlap it with
    training tiles because training tiles must end above the band). Tiles with less than
    `min_mask_frac` valid (papyrus) pixels are skipped.
    """
    h, w = mask.shape
    split_y = val_region_start(h, val_frac)

    def keep(t):
        y, x = t
        return (mask[y:y + tile_size, x:x + tile_size] > 0).mean() >= min_mask_frac

    train = [t for t in tile_origins(0, split_y, 0, w, tile_size, stride, cover=True) if keep(t)]
    val = [t for t in tile_origins(split_y, h, 0, w, tile_size, tile_size, cover=False) if keep(t)]  # non-overlapping
    if not train or not val:
        raise ValueError(
            f"Split produced {len(train)} train / {len(val)} val tiles for a {h}x{w} mask "
            f"(val band = {h - split_y} rows, tile = {tile_size}). Increase val_frac or use a smaller tile."
        )
    return {"train": train, "val": val, "split_y": split_y}


# ---------------------------------------------------------------- metrics
def confusion_counts(pred_bin, target_bin, valid=None):
    pred_bin = pred_bin.astype(bool)
    target_bin = target_bin.astype(bool)
    if valid is not None:
        valid = valid.astype(bool)
        pred_bin, target_bin = pred_bin[valid], target_bin[valid]
    tp = int(np.logical_and(pred_bin, target_bin).sum())
    fp = int(np.logical_and(pred_bin, ~target_bin).sum())
    fn = int(np.logical_and(~pred_bin, target_bin).sum())
    return tp, fp, fn


def dice_from_counts(tp, fp, fn, eps=1e-8):
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def fbeta_from_counts(tp, fp, fn, beta=0.5, eps=1e-8):
    """F-beta; beta=0.5 is the metric used by the Vesuvius ink-detection competition."""
    b2 = beta * beta
    return ((1 + b2) * tp + eps) / ((1 + b2) * tp + b2 * fn + fp + eps)


def precision_recall(tp, fp, fn, eps=1e-8):
    return tp / (tp + fp + eps), tp / (tp + fn + eps)


def metrics_at_threshold(probs, labels, valid, threshold):
    tp, fp, fn = confusion_counts(probs > threshold, labels > 0.5, valid)
    p, r = precision_recall(tp, fp, fn)
    return {"threshold": float(threshold), "dice": dice_from_counts(tp, fp, fn),
            "f0.5": fbeta_from_counts(tp, fp, fn, 0.5), "precision": p, "recall": r}


def sweep_thresholds(probs, labels, valid=None, thresholds=None, metric="f0.5"):
    """Evaluate a grid of thresholds on real predictions and return (best, all_rows)."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)
    rows = [metrics_at_threshold(probs, labels, valid, t) for t in thresholds]
    return max(rows, key=lambda r: r[metric]), rows


# ---------------------------------------------------------------- post-processing
def post_process(prob, threshold=0.5, kernel_size=3):
    """Threshold + morphological opening (remove speckle) and closing (fill pin-holes)."""
    binary = (prob > threshold).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    out = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
