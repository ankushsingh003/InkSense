"""Figure from REAL model output: IR image, ground truth, probability map and a TP/FP/FN error overlay.

Run `python evaluation.py` first; this reads results/metrics.json, probs.npy and pred.npy.
"""
import argparse
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def error_overlay(pred, truth, valid):
    """RGB map: green=TP, red=FP (false ink), blue=FN (missed ink)."""
    pred, truth, valid = pred.astype(bool), truth.astype(bool), valid.astype(bool)
    img = np.zeros(pred.shape + (3,), np.uint8)
    img[pred & truth & valid] = (0, 200, 0)
    img[pred & ~truth & valid] = (220, 0, 0)
    img[~pred & truth & valid] = (0, 90, 255)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data_dir", default="data/ink_detection_fragments/processed")
    ap.add_argument("--fragment", type=int, default=1)
    ap.add_argument("--ir", default="image_for_eval/ir.png")
    ap.add_argument("--downsample", type=int, default=4, help="Shrink for a readable figure")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.results, "metrics.json")))
    y0, y1, x0, x1 = meta["region"]
    probs = np.load(os.path.join(a.results, "probs.npy")).astype(np.float32)
    pred = np.load(os.path.join(a.results, "pred.npy"))
    labels = np.load(os.path.join(a.data_dir, f"fragment{a.fragment}_labels.npy"), mmap_mode="r")[y0:y1, x0:x1] > 127
    mask = np.load(os.path.join(a.data_dir, f"fragment{a.fragment}_mask.npy"), mmap_mode="r")[y0:y1, x0:x1] > 0
    ir = cv2.imread(a.ir, 0)
    ir = ir[y0:y1, x0:x1] if ir is not None and ir.shape[0] >= y1 else np.zeros_like(probs)

    d = a.downsample
    sm = lambda x, interp=cv2.INTER_AREA: cv2.resize(np.ascontiguousarray(x), None, fx=1 / d, fy=1 / d, interpolation=interp)
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    panels = [("Infrared scan", sm(ir), "gray"), ("Ground-truth ink", sm(labels.astype(np.uint8) * 255), "gray"),
              ("Predicted probability", sm(probs), "magma"),
              ("Error map (green TP, red FP, blue FN)", sm(error_overlay(pred, labels, mask), cv2.INTER_NEAREST), None)]
    for axis, (title, img, cmap) in zip(ax, panels):
        axis.imshow(img, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    m = meta["after_morphology"]
    fig.suptitle(f"Held-out region | F0.5={m['f0.5']:.3f}  Dice={m['dice']:.3f}  threshold={meta['threshold']:.2f}")
    plt.tight_layout()
    out = os.path.join(a.results, "held_out_result.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
