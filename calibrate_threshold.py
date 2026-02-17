#!/usr/bin/env python3
"""
Calibrate detection threshold for a trained 1D CNN using validation data.

Inputs:
  - model checkpoint (best_model.pt from train_cnn1d.py)
  - validation arrays (X_val.npy, y_val.npy)

Outputs:
  - threshold_report.json (full sweep + chosen threshold)
  - prints recommended threshold to stdout
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class PulseCNN1D(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.0):
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
        return self.head(self.features(x)).squeeze(-1)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate detection threshold on validation set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="artifacts/models/cnn_simple/best_model.pt",
        help="Path to best_model.pt",
    )
    parser.add_argument(
        "--data-dir",
        default="artifacts/processed/cnn_simple",
        help="Directory containing X_val.npy and y_val.npy",
    )
    parser.add_argument("--output-dir", default="artifacts/models/cnn_simple", help="Directory for threshold report")
    parser.add_argument("--num-thresholds", type=int, default=201, help="Number of thresholds in [0,1]")
    parser.add_argument(
        "--criterion",
        default="f1",
        choices=["f1", "target_recall", "target_precision"],
        help="Selection criterion for recommended threshold",
    )
    parser.add_argument("--target-recall", type=float, default=0.95, help="Used when criterion=target_recall")
    parser.add_argument("--target-precision", type=float, default=0.90, help="Used when criterion=target_precision")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device",
    )
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(np.float32)
    y_pred = (y_prob >= threshold).astype(np.float32)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def predict_probs(model: nn.Module, x_val: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_val), batch_size):
            xb = torch.from_numpy(x_val[start:start + batch_size].astype(np.float32)).to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
    if not probs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(probs).astype(np.float32)


def select_threshold(
    rows: List[Dict[str, float]],
    criterion: str,
    target_recall: float,
    target_precision: float,
) -> Dict[str, float]:
    if criterion == "f1":
        return max(rows, key=lambda r: (r["f1"], r["precision"], r["recall"]))

    if criterion == "target_recall":
        candidates = [r for r in rows if r["recall"] >= target_recall]
        if candidates:
            return max(candidates, key=lambda r: (r["precision"], r["f1"]))
        return max(rows, key=lambda r: (r["recall"], r["f1"]))

    candidates = [r for r in rows if r["precision"] >= target_precision]
    if candidates:
        return max(candidates, key=lambda r: (r["recall"], r["f1"]))
    return max(rows, key=lambda r: (r["precision"], r["f1"]))


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    x_val = np.load(data_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(data_dir / "y_val.npy").astype(np.float32)

    if x_val.ndim != 3:
        raise ValueError(f"Expected X_val shape [N,C,T], got {x_val.shape}")
    if y_val.ndim != 1:
        raise ValueError(f"Expected y_val shape [N], got {y_val.shape}")
    if len(x_val) != len(y_val):
        raise ValueError("X_val and y_val size mismatch")
    if len(x_val) == 0:
        raise ValueError("Validation set is empty")

    device = select_device(args.device)

    checkpoint = torch.load(model_path, map_location=device)
    in_channels = int(checkpoint.get("input_channels", x_val.shape[1]))
    model = PulseCNN1D(in_channels=in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    y_prob = predict_probs(model, x_val, device)

    thresholds = np.linspace(0.0, 1.0, args.num_thresholds)
    rows = [compute_metrics(y_val, y_prob, float(t)) for t in thresholds]
    best = select_threshold(rows, args.criterion, args.target_recall, args.target_precision)

    report = {
        "criterion": args.criterion,
        "target_recall": args.target_recall,
        "target_precision": args.target_precision,
        "device": str(device),
        "num_samples": int(len(y_val)),
        "class_balance": {
            "positive": int(np.sum(y_val == 1)),
            "negative": int(np.sum(y_val == 0)),
        },
        "recommended": best,
        "sweep": rows,
    }

    report_path = out_dir / "threshold_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    print(f"Recommended threshold: {best['threshold']:.4f}")
    print(
        f"Metrics @ threshold={best['threshold']:.4f}: "
        f"f1={best['f1']:.4f} precision={best['precision']:.4f} recall={best['recall']:.4f}"
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
