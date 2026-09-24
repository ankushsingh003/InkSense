"""Train the InkSense 3D-CNN + Transformer on Vesuvius fragments.

Validation is a held-out *spatial band* of each training fragment (no pixel overlap with any
training tile) or, better, one or more whole held-out fragments (--val_fragments).
The best checkpoint is chosen by validation F0.5 (the competition metric).
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from inksense_utils import (confusion_counts, dice_from_counts, fbeta_from_counts,
                            precision_recall, spatial_split_tiles, tile_origins)
from model import VesuviusModel, count_parameters, dice_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VesuviusDataset(Dataset):
    """Tiles from one fragment. split: 'train' | 'val' (spatial band) | 'all' (whole fragment)."""

    def __init__(self, volume_path, label_path, mask_path, tile_size=256, stride=128,
                 split="all", val_frac=0.2, augment=False):
        self.volume = np.load(volume_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        self.mask = np.load(mask_path, mmap_mode="r")
        self.tile_size, self.augment = tile_size, augment

        h, w = self.mask.shape
        if split == "all":
            keep = lambda t: (self.mask[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size] > 0).mean() >= 0.05
            self.tiles = [t for t in tile_origins(0, h, 0, w, tile_size, stride, cover=stride < tile_size) if keep(t)]
        else:
            self.tiles = spatial_split_tiles(np.asarray(self.mask), tile_size, stride, val_frac)[split]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        y, x = self.tiles[idx]
        s = self.tile_size
        tile = np.asarray(self.volume[:, y:y + s, x:x + s], dtype=np.float32) / 255.0
        label = (np.asarray(self.labels[y:y + s, x:x + s], dtype=np.float32) > 127).astype(np.float32)
        mask = (np.asarray(self.mask[y:y + s, x:x + s]) > 0).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                tile, label, mask = tile[:, :, ::-1], label[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                tile, label, mask = tile[:, ::-1, :], label[::-1, :], mask[::-1, :]
            k = random.randint(0, 3)
            if k:
                tile, label, mask = np.rot90(tile, k, (1, 2)), np.rot90(label, k), np.rot90(mask, k)
            tile = np.clip(tile * random.uniform(0.9, 1.1), 0, 1)

        tile = torch.from_numpy(np.ascontiguousarray(tile)).unsqueeze(0)      # (1, Z, H, W)
        label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)    # (1, H, W)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)      # (1, H, W)
        return tile, label, mask


def masked_bce(logits, labels, mask):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (bce * mask).sum() / mask.sum().clamp(min=1.0)


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0.0
    for tiles, labels, mask in tqdm(loader, desc="train", leave=False):
        tiles, labels, mask = tiles.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(tiles)
        logits = logits.float()
        loss = 0.5 * masked_bce(logits, labels, mask) + 0.5 * dice_loss(logits, labels, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, threshold=0.5):
    """Pixel-level metrics from global TP/FP/FN counts over the whole validation set."""
    model.eval()
    tp = fp = fn = 0
    for tiles, labels, mask in tqdm(loader, desc="val", leave=False):
        probs = torch.sigmoid(model(tiles.to(device)).float()).cpu().numpy()
        a, b, c = confusion_counts(probs > threshold, labels.numpy() > 0.5, mask.numpy() > 0.5)
        tp, fp, fn = tp + a, fp + b, fn + c
    p, r = precision_recall(tp, fp, fn)
    return {"dice": dice_from_counts(tp, fp, fn), "f0.5": fbeta_from_counts(tp, fp, fn, 0.5),
            "precision": p, "recall": r}


def frag_paths(data_dir, fid):
    return [os.path.join(data_dir, f"fragment{fid}_{k}.npy") for k in ("volume", "labels", "mask")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/ink_detection_fragments/processed")
    ap.add_argument("--train_fragments", type=int, nargs="+", default=[1])
    ap.add_argument("--val_fragments", type=int, nargs="*", default=[],
                    help="Whole held-out fragments. If empty, a spatial band of the train fragments is held out.")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_train_tiles", type=int, default=None, help="Cap training tiles for fast runs / testing")
    ap.add_argument("--max_val_tiles", type=int, default=None, help="Cap validation tiles for fast runs / testing")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_sets, val_sets = [], []
    for fid in args.train_fragments:
        v, l, m = frag_paths(args.data_dir, fid)
        if args.val_fragments:
            train_sets.append(VesuviusDataset(v, l, m, args.tile_size, args.stride, "all", augment=True))
        else:
            train_sets.append(VesuviusDataset(v, l, m, args.tile_size, args.stride, "train", args.val_frac, augment=True))
            val_sets.append(VesuviusDataset(v, l, m, args.tile_size, args.tile_size, "val", args.val_frac))
    for fid in args.val_fragments:
        v, l, m = frag_paths(args.data_dir, fid)
        val_sets.append(VesuviusDataset(v, l, m, args.tile_size, args.tile_size, "all"))

    train_ds, val_ds = ConcatDataset(train_sets), ConcatDataset(val_sets)
    if args.max_train_tiles and len(train_ds) > args.max_train_tiles:
        train_ds = torch.utils.data.Subset(train_ds, list(range(args.max_train_tiles)))
    if args.max_val_tiles and len(val_ds) > args.max_val_tiles:
        val_ds = torch.utils.data.Subset(val_ds, list(range(args.max_val_tiles)))

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"device={device} | train tiles={len(train_ds)} | val tiles={len(val_ds)}")

    model = VesuviusModel(z_dim=train_sets[0].volume.shape[0]).to(device)
    print(f"parameters: {count_parameters(model) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="InkSense", config=vars(args))
        except Exception as e:  # wandb is optional
            print(f"wandb disabled ({e})")
            use_wandb = False

    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        m = validate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": loss, **{f"val_{k}": v for k, v in m.items()}}
        history.append(row)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if use_wandb:
            wandb.log(row)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))
        if best is None or m["f0.5"] > best["val_f0.5"]:
            best = row
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

    with open(os.path.join(args.out_dir, "train_summary.json"), "w") as f:
        json.dump({"args": vars(args), "best": best, "history": history}, f, indent=2)
    print(f"Done. Best epoch {best['epoch']}: val F0.5={best['val_f0.5']:.4f}, Dice={best['val_dice']:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
