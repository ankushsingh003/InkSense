"""PyTorch tests (skipped automatically if torch is not installed)."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import VesuviusModel, dice_loss
from train import VesuviusDataset


def test_model_output_shape():
    model = VesuviusModel().eval()
    out = model(torch.randn(2, 1, 32, 128, 128))
    assert out.shape == (2, 1, 128, 128)


def test_attention_runs_on_downsampled_tokens_only():
    """Guard against regressing to per-pixel attention: tokens must be (H/8)*(W/8)."""
    model = VesuviusModel()
    seen = {}
    def hook(m, i, o):
        seen.setdefault("n", i[0].shape[1])
    model.transformer.register_forward_hook(hook)
    model(torch.randn(1, 1, 32, 128, 128))
    assert seen["n"] == (128 // 8) ** 2


def test_gradients_flow_to_input():
    model = VesuviusModel()
    x = torch.randn(1, 1, 32, 64, 64, requires_grad=True)
    model(x).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dice_loss_bounds_and_mask():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4] = 1
    perfect = (target * 20 - 10)  # confident logits matching target
    assert dice_loss(perfect, target).item() < 0.01
    assert dice_loss(-perfect, target).item() > 0.9
    zero_mask = torch.zeros_like(target)
    assert dice_loss(-perfect, target, mask=zero_mask).item() < 0.5  # masked-out pixels are ignored


def test_dataset_shapes_and_split(tmp_path):
    z, h, w = 32, 1400, 600
    vol = np.random.randint(0, 255, (z, h, w), dtype=np.uint8)
    lbl = np.zeros((h, w), np.uint8)
    lbl[100:300, 100:300] = 255
    msk = np.full((h, w), 255, np.uint8)
    paths = []
    for name, arr in (("v", vol), ("l", lbl), ("m", msk)):
        p = str(tmp_path / f"{name}.npy")
        np.save(p, arr)
        paths.append(p)

    tr = VesuviusDataset(*paths, tile_size=256, split="train", val_frac=0.2, augment=True)
    va = VesuviusDataset(*paths, tile_size=256, stride=256, split="val", val_frac=0.2)
    assert len(tr) > 0 and len(va) > 0
    tile, label, mask = tr[0]
    assert tile.shape == (1, 32, 256, 256) and label.shape == (1, 256, 256) and mask.shape == (1, 256, 256)
    assert set(np.unique(label.numpy())) <= {0.0, 1.0}
    split_y = int(round(h * 0.8))
    assert max(y + 256 for y, _ in tr.tiles) <= split_y <= min(y for y, _ in va.tiles)
