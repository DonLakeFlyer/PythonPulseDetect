#!/usr/bin/env python3
"""
Train a 1D CNN pulse detector from prebuilt dataset arrays.

Expected dataset files in --data-dir:
  - X_train.npy  [N, 1, T]
  - y_train.npy  [N]
  - X_val.npy    [M, 1, T]
  - y_val.npy    [M]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PulseCNN1D(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 1D CNN pulse detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="artifacts/processed/cnn_simple",
        help="Directory containing X/y train/val .npy files",
    )
    parser.add_argument("--output-dir", default="artifacts/models/cnn_simple", help="Directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Training device",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience on best val_f1",
    )
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    x_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    if x_train.ndim != 3 or x_val.ndim != 3:
        raise ValueError("X arrays must have shape [N, C, T]")
    if y_train.ndim != 1 or y_val.ndim != 1:
        raise ValueError("y arrays must be 1D")
    if x_train.shape[0] != y_train.shape[0] or x_val.shape[0] != y_val.shape[0]:
        raise ValueError("X/y sample count mismatch")

    return (
        x_train.astype(np.float32),
        y_train.astype(np.float32),
        x_val.astype(np.float32),
        y_val.astype(np.float32),
    )


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.float32)
    y_true = y_true.astype(np.float32)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())

    val_loss = float(np.mean(losses)) if losses else 0.0
    probs = np.concatenate(all_probs) if all_probs else np.empty((0,), dtype=np.float32)
    targets = np.concatenate(all_targets) if all_targets else np.empty((0,), dtype=np.float32)
    return val_loss, targets, probs


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val = load_dataset(data_dir)
    in_channels = int(x_train.shape[1])

    device = select_device(args.device)
    print(f"Using device: {device}")

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 512),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = PulseCNN1D(in_channels=in_channels, dropout=args.dropout).to(device)

    positives = float(np.sum(y_train == 1.0))
    negatives = float(np.sum(y_train == 0.0))
    if positives <= 0:
        raise RuntimeError("Training set has no positive samples")
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_f1 = -1.0
    best_epoch = -1
    epochs_since_best = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        val_loss, y_true, y_prob = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_prob, threshold=0.5)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={metrics['f1']:.4f} val_prec={metrics['precision']:.4f} val_rec={metrics['recall']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            epochs_since_best = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": best_f1,
                    "input_channels": in_channels,
                },
                out_dir / "best_model.pt",
            )
        else:
            epochs_since_best += 1

        if epochs_since_best >= args.patience:
            print(f"Early stopping at epoch {epoch} (no val_f1 improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - start_time

    with (out_dir / "train_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "epochs_ran": len(history),
        "elapsed_sec": elapsed,
        "device": str(device),
        "args": vars(args),
    }
    with (out_dir / "train_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val_f1: {best_f1:.4f}")
    print(f"Saved checkpoint: {out_dir / 'best_model.pt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
