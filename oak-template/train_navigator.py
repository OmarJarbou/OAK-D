#!/usr/bin/env python3
"""
Train TinyDepthNavigator on recorded depth + label pairs.

Run on a PC (or Colab) after record_navigation_data.py:

  python train_navigator.py --data data/nav_recordings --epochs 30

Saves models/navigator.pth by default.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from utils.tiny_depth_navigator import TinyDepthNavigator, CLASS_TO_IDX, NAV_CLASSES
from utils.ml_navigator import MLNavigator


class DepthNavDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        min_depth_mm: int = 300,
        max_depth_mm: int = 5000,
    ) -> None:
        self.files = sorted(glob.glob(os.path.join(data_dir, "frame_*_depth.npy")))
        if not self.files:
            raise FileNotFoundError(f"No frame_*_depth.npy in {data_dir}")
        self.min_mm = min_depth_mm
        self.max_mm = max_depth_mm

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        depth_path = self.files[idx]
        label_path = depth_path.replace("_depth.npy", "_label.txt")
        depth = np.load(depth_path).astype(np.float32)
        depth = MLNavigator.preprocess_depth(depth, self.min_mm, self.max_mm)
        label_name = Path(label_path).read_text(encoding="utf-8").strip().upper()
        if label_name not in CLASS_TO_IDX:
            raise ValueError(f"Unknown label '{label_name}' in {label_path}")
        x = torch.tensor(depth).unsqueeze(0)
        y = torch.tensor(CLASS_TO_IDX[label_name], dtype=torch.long)
        return x, y


def train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train depth navigator CNN")
    parser.add_argument("--data", default="data/nav_recordings")
    parser.add_argument("--out", default="models/navigator.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DepthNavDataset(args.data)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDepthNavigator().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Dataset: {len(dataset)} samples ({n_train} train / {n_val} val)")
    print(f"Classes: {NAV_CLASSES}")
    print(f"Device: {device}")

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.1%}  "
            f"val loss={va_loss:.4f} acc={va_acc:.1%}"
        )
        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), out_path)
            print(f"  → saved {out_path} (val acc={va_acc:.1%})")

    print(f"Done. Best val accuracy: {best_acc:.1%}")


if __name__ == "__main__":
    main()
