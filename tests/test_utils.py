"""Framework-free tests (run without PyTorch)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import evaluate, predict_region
from inksense_utils import (dice_from_counts, fbeta_from_counts, post_process, spatial_split_tiles,
                            sweep_thresholds, tile_origins, to_uint8)


def test_16bit_slices_do_not_wrap():
    img = np.array([0, 257, 65535], dtype=np.uint16)
    assert to_uint8(img).tolist() == [0, 1, 255]


def test_spatial_split_has_no_pixel_overlap():
    mask = np.full((2000, 900), 255, np.uint8)
    sp = spatial_split_tiles(mask, tile_size=256, stride=128, val_frac=0.2)
    assert sp["train"] and sp["val"]
    assert max(y + 256 for y, _ in sp["train"]) <= sp["split_y"]
    assert min(y for y, _ in sp["val"]) >= sp["split_y"]


def test_split_fails_loudly_when_band_too_small():
    with pytest.raises(ValueError):
        spatial_split_tiles(np.full((1000, 600), 255, np.uint8), 256, 128, 0.2)


def test_tiles_cover_whole_region():
    seen = np.zeros((700, 530), bool)
    for y, x in tile_origins(0, 700, 0, 530, 256, 128):
        seen[y:y + 256, x:x + 256] = True
    assert seen.all()


def test_metric_formulas():
    assert dice_from_counts(50, 10, 20) == pytest.approx(100 / 130, abs=1e-6)
    # F0.5 = 1.25*TP / (1.25*TP + 0.25*FN + FP)
    assert fbeta_from_counts(50, 10, 20, 0.5) == pytest.approx(62.5 / (62.5 + 5 + 10), abs=1e-6)
    assert dice_from_counts(0, 0, 0) == pytest.approx(1.0)


def test_threshold_sweep_uses_real_predictions_and_finds_good_cut():
    rng = np.random.default_rng(0)
    labels = (rng.random((128, 128)) > 0.8)
    probs = np.where(labels, 0.85, 0.15) + rng.normal(0, 0.05, labels.shape)
    best, rows = sweep_thresholds(probs, labels)
    assert 0.3 <= best["threshold"] <= 0.7
    assert best["f0.5"] > 0.95 and len(rows) > 5


def test_morphology_removes_speckle_and_keeps_blob():
    prob = np.zeros((64, 64), np.float32)
    prob[20:40, 20:40] = 0.9
    prob[5, 5] = 0.9  # isolated speckle
    out = post_process(prob, 0.5, 3)
    assert out[5, 5] == 0 and out[30, 30] == 1


def test_end_to_end_evaluation_with_fake_model():
    """Sliding-window inference + sweep + post-processing with a stand-in predictor."""
    h, w, z = 600, 520, 4
    rng = np.random.default_rng(1)
    labels = np.zeros((h, w), np.uint8)
    labels[300:500, 100:400] = 255
    mask = np.full((h, w), 255, np.uint8)
    volume = np.stack([labels // 2 + rng.integers(0, 20, (h, w), dtype=np.uint8) for _ in range(z)])

    def fake_model(tile):  # "detects" ink from mean brightness
        return (tile.mean(0) > 0.25).astype(np.float32) * 0.9 + 0.05

    res = evaluate(fake_model, volume, labels, mask, (240, h, 0, w), tile_size=256, stride=128)
    assert res["after_morphology"]["f0.5"] > 0.9
    assert res["probs"].shape == (h - 240, w)
    assert predict_region(fake_model, volume, (240, h, 0, w), 256, 128).shape == (h - 240, w)


def test_validation_tiles_never_overlap_each_other():
    mask = np.full((2000, 900), 255, np.uint8)
    val = spatial_split_tiles(mask, 256, 128, 0.2)["val"]
    covered = np.zeros((2000, 900), np.uint8)
    for y, x in val:
        covered[y:y + 256, x:x + 256] += 1
    assert covered.max() == 1
