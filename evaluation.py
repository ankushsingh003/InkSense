"""Evaluate a trained checkpoint on real predictions (sliding window over a held-out region).

Steps: overlapping-tile inference (averaged) -> threshold sweep on the region -> morphological
post-processing -> metrics before/after post-processing. Nothing here uses synthetic predictions.

Honest-reporting note: if you pick the threshold on the same region you used for checkpoint
selection, the number is slightly optimistic. For an unbiased score, tune on the validation
fragment, then pass that value via --threshold when evaluating a different (test) fragment.
"""
import argparse
import json
import os

import numpy as np

from inksense_utils import (metrics_at_threshold, post_process, sweep_thresholds, tile_origins,
                            val_region_start)


def predict_region(predict_fn, volume, region, tile_size=256, stride=128):
    """Average overlapping tile predictions over region=(y0, y1, x0, x1). predict_fn: (Z,H,W)->(H,W) probs."""
    y0, y1, x0, x1 = region
    acc = np.zeros((y1 - y0, x1 - x0), np.float32)
    cnt = np.zeros_like(acc)
    for y, x in tile_origins(y0, y1, x0, x1, tile_size, stride):
        tile = np.asarray(volume[:, y:y + tile_size, x:x + tile_size], dtype=np.float32) / 255.0
        p = predict_fn(tile)
        acc[y - y0:y - y0 + tile_size, x - x0:x - x0 + tile_size] += p
        cnt[y - y0:y - y0 + tile_size, x - x0:x - x0 + tile_size] += 1
    if (cnt == 0).any():
        raise ValueError("Region smaller than one tile, or not fully covered.")
    return acc / cnt


def evaluate(predict_fn, volume, labels, mask, region, tile_size=256, stride=128,
             threshold=None, kernel_size=3):
    y0, y1, x0, x1 = region
    probs = predict_region(predict_fn, volume, region, tile_size, stride)
    lab = np.asarray(labels[y0:y1, x0:x1]) > 127
    valid = np.asarray(mask[y0:y1, x0:x1]) > 0

    if threshold is None:
        best, sweep = sweep_thresholds(probs, lab, valid, metric="f0.5")
        threshold = best["threshold"]
    else:
        sweep = [metrics_at_threshold(probs, lab, valid, threshold)]

    pred = post_process(probs, threshold, kernel_size)
    tp = metrics_at_threshold(probs, lab, valid, threshold)
    post = metrics_at_threshold(pred.astype(np.float32), lab, valid, 0.5)
    return {
        "region": [y0, y1, x0, x1], "threshold": float(threshold),
        "raw_threshold_only": tp, "after_morphology": post, "sweep": sweep,
        "probs": probs, "pred": pred,
    }


def make_torch_predict_fn(checkpoint, z_dim, device=None):
    import torch
    from model import VesuviusModel
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesuviusModel(z_dim=z_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    @torch.no_grad()
    def predict(tile):
        x = torch.from_numpy(tile)[None, None].to(device)
        return torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--data_dir", default="data/ink_detection_fragments/processed")
    ap.add_argument("--fragment", type=int, default=1)
    ap.add_argument("--region", choices=["val", "all"], default="val",
                    help="'val' = bottom band held out by train.py; 'all' = whole fragment (use for a test fragment)")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=None, help="Fixed threshold; skips the sweep")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--crop_h", type=int, default=None, help="Optional max height for evaluation window")
    ap.add_argument("--crop_w", type=int, default=None, help="Optional max width for evaluation window")
    a = ap.parse_args()

    p = lambda k: os.path.join(a.data_dir, f"fragment{a.fragment}_{k}.npy")
    volume, labels, mask = (np.load(p(k), mmap_mode="r") for k in ("volume", "labels", "mask"))
    h, w = mask.shape
    y0 = val_region_start(h, a.val_frac) if a.region == "val" else 0
    y1 = min(h, y0 + a.crop_h) if a.crop_h else h
    x1 = min(w, a.crop_w) if a.crop_w else w
    region = (y0, y1, 0, x1)

    predict_fn = make_torch_predict_fn(a.checkpoint, z_dim=volume.shape[0])
    res = evaluate(predict_fn, volume, labels, mask, region, a.tile_size, a.stride, a.threshold)

    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "probs.npy"), res["probs"].astype(np.float16))
    np.save(os.path.join(a.out_dir, "pred.npy"), res["pred"])
    summary = {k: v for k, v in res.items() if k not in ("probs", "pred")}
    summary.update({"fragment": a.fragment, "checkpoint": a.checkpoint, "region_mode": a.region})
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    r, m = res["raw_threshold_only"], res["after_morphology"]
    print(f"threshold={res['threshold']:.2f}")
    print(f"threshold only : Dice={r['dice']:.4f}  F0.5={r['f0.5']:.4f}  P={r['precision']:.3f}  R={r['recall']:.3f}")
    print(f"+ morphology   : Dice={m['dice']:.4f}  F0.5={m['f0.5']:.4f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    print(f"Saved {a.out_dir}/metrics.json, probs.npy, pred.npy")


if __name__ == "__main__":
    main()
