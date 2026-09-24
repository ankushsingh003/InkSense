"""Convert a Vesuvius fragment (stack of surface-volume TIFF slices) into memory-mappable .npy files.

Fixes over the first version:
* 16-bit TIFF slices are scaled to uint8 (previously they silently wrapped around on assignment).
* The volume is written straight to disk with `open_memmap`, so RAM use stays at one slice.
* A missing slice is an error, not a silent all-zero slice.
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from inksense_utils import to_uint8


def preprocess_fragment(fragment_id, data_dir, output_dir, z_start=16, z_end=48):
    frag_path = os.path.join(data_dir, "train", str(fragment_id))
    surface_dir = os.path.join(frag_path, "surface_volume")

    mask = cv2.imread(os.path.join(frag_path, "mask.png"), 0)
    labels = cv2.imread(os.path.join(frag_path, "inklabels.png"), 0)
    if mask is None or labels is None:
        raise FileNotFoundError(f"mask.png / inklabels.png not found in {frag_path}")
    if mask.shape != labels.shape:
        raise ValueError(f"mask {mask.shape} and labels {labels.shape} differ in shape")

    h, w = mask.shape
    z_count = z_end - z_start
    print(f"--- Fragment {fragment_id}: {h}x{w} px, {z_count} slices (z={z_start}..{z_end - 1}) ---")

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"fragment{fragment_id}_volume.npy")
    volume = np.lib.format.open_memmap(out_file, mode="w+", dtype=np.uint8, shape=(z_count, h, w))

    for i, z in enumerate(tqdm(range(z_start, z_end), desc="Writing slices")):
        slice_path = os.path.join(surface_dir, f"{z:02d}.tif")
        img = cv2.imread(slice_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Missing slice: {slice_path}")
        if img.shape != (h, w):
            raise ValueError(f"Slice {slice_path} has shape {img.shape}, expected {(h, w)}")
        volume[i] = to_uint8(img)
    volume.flush()

    np.save(os.path.join(output_dir, f"fragment{fragment_id}_labels.npy"), labels)
    np.save(os.path.join(output_dir, f"fragment{fragment_id}_mask.npy"), mask)
    print(f"Saved {out_file} ({os.path.getsize(out_file) / 1e9:.2f} GB), plus labels and mask.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess a Vesuvius fragment")
    ap.add_argument("--fragment", type=int, default=1)
    ap.add_argument("--data_dir", type=str, default="data/ink_detection_fragments")
    ap.add_argument("--out_dir", type=str, default="data/ink_detection_fragments/processed")
    ap.add_argument("--z_start", type=int, default=16)
    ap.add_argument("--z_end", type=int, default=48)
    a = ap.parse_args()
    preprocess_fragment(a.fragment, a.data_dir, a.out_dir, a.z_start, a.z_end)
