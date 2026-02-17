#!/usr/bin/env python3
"""
Build 1D CNN datasets from captured IQ windows.

Reads one or more capture session directories created by capture_training_data.py,
loads windows referenced in manifest.jsonl, converts each IQ window to a phase-invariant
envelope feature, normalizes, balances classes, and writes:

- X_train.npy  shape: [N, 1, T]
- y_train.npy  shape: [N]
- X_val.npy    shape: [N, 1, T]
- y_val.npy    shape: [N]
- dataset_meta.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

import numpy as np


def robust_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    scale = 1.4826 * mad + eps
    return (x - median) / scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val arrays for 1D CNN from captured windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="One or more capture session directories (each containing manifest.jsonl)",
    )
    parser.add_argument(
        "--training-sessions-dir",
        default=None,
        help="Parent directory containing capture sessions; auto-discovers subfolders with manifest.jsonl",
    )
    parser.add_argument("--output-dir", default="artifacts/processed/cnn_simple", help="Output directory for dataset files")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction [0, 1)")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def discover_session_dirs(root_dir: Path) -> List[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"training-sessions-dir is not a valid directory: {root_dir}")

    manifests = sorted(root_dir.rglob("manifest.jsonl"))
    session_dirs = sorted({manifest.parent for manifest in manifests})
    if not session_dirs:
        raise RuntimeError(f"No sessions found under {root_dir} (expected manifest.jsonl files)")
    return session_dirs


def read_manifest_rows(session_dir: Path) -> List[Dict]:
    manifest = session_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    rows: List[Dict] = []
    with manifest.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest}")

    return rows


def load_feature_window(file_path: Path) -> np.ndarray:
    iq = np.load(file_path)
    if iq.ndim != 1:
        raise ValueError(f"Expected 1D IQ array in {file_path}, got shape {iq.shape}")
    env = np.abs(iq).astype(np.float32)
    return robust_zscore(env).astype(np.float32)


def collect_samples(session_dirs: List[Path]) -> Tuple[List[np.ndarray], List[int], Dict]:
    features: List[np.ndarray] = []
    labels: List[int] = []

    label_counts = {"positive": 0, "negative": 0}
    sample_rates = set()
    center_freqs = set()
    session_labels: Dict[str, Set[str]] = {}

    for session_dir in session_dirs:
        session_key = str(session_dir)
        session_labels[session_key] = set()
        rows = read_manifest_rows(session_dir)
        for row in rows:
            label_text = row.get("label")
            if label_text not in ("positive", "negative"):
                continue

            session_labels[session_key].add(label_text)

            file_name = row.get("file")
            if not file_name:
                continue

            file_path = session_dir / file_name
            if not file_path.exists():
                print(f"Warning: missing file referenced in manifest: {file_path}", file=sys.stderr)
                continue

            feature = load_feature_window(file_path)
            features.append(feature)
            labels.append(1 if label_text == "positive" else 0)
            label_counts[label_text] += 1

            if "sample_rate_hz" in row:
                sample_rates.add(int(row["sample_rate_hz"]))
            if "center_frequency_hz" in row:
                center_freqs.add(int(row["center_frequency_hz"]))

    meta = {
        "num_positive": label_counts["positive"],
        "num_negative": label_counts["negative"],
        "sample_rates_hz": sorted(sample_rates),
        "center_frequencies_hz": sorted(center_freqs),
        "session_label_summary": {
            "positive_sessions": int(sum(1 for values in session_labels.values() if "positive" in values)),
            "negative_sessions": int(sum(1 for values in session_labels.values() if "negative" in values)),
            "mixed_label_sessions": int(sum(1 for values in session_labels.values() if len(values) > 1)),
        },
    }
    return features, labels, meta


def build_quality_analysis(
    y_train: np.ndarray,
    y_val: np.ndarray,
    capture_meta: Dict,
    total_samples: int,
) -> Dict:
    train_positive = int(np.sum(y_train == 1))
    train_negative = int(np.sum(y_train == 0))
    val_positive = int(np.sum(y_val == 1))
    val_negative = int(np.sum(y_val == 0))

    warnings: List[str] = []
    recommendations: List[str] = []

    raw_positive = int(capture_meta.get("num_positive", 0))
    raw_negative = int(capture_meta.get("num_negative", 0))
    raw_min = min(raw_positive, raw_negative)
    raw_max = max(raw_positive, raw_negative)
    if raw_min == 0:
        warnings.append("Raw capture data has a missing class (only positive or only negative windows).")
        recommendations.append("Capture additional sessions for the missing class before training.")
    elif raw_max / raw_min >= 3.0:
        warnings.append(
            f"Raw capture data is highly imbalanced before balancing ({raw_positive} positive vs {raw_negative} negative)."
        )
        recommendations.append("Collect more windows for the minority class to reduce sampling bias.")

    session_summary = capture_meta.get("session_label_summary", {})
    positive_sessions = int(session_summary.get("positive_sessions", 0))
    negative_sessions = int(session_summary.get("negative_sessions", 0))
    mixed_sessions = int(session_summary.get("mixed_label_sessions", 0))

    if positive_sessions < 2:
        warnings.append(
            f"Only {positive_sessions} positive training session(s) detected; validation can look overly optimistic."
        )
        recommendations.append("Capture additional positive sessions from independent runs/times.")
    if negative_sessions < 2:
        warnings.append(
            f"Only {negative_sessions} negative training session(s) detected; background variability may be underrepresented."
        )
        recommendations.append("Capture additional negative sessions in varied environments.")
    if mixed_sessions > 0:
        warnings.append(f"Detected {mixed_sessions} session(s) containing mixed labels in manifest data.")
        recommendations.append("Review mixed-label sessions to ensure labeling consistency.")

    if len(y_val) < 50:
        warnings.append(f"Validation split is small ({len(y_val)} windows).")
        recommendations.append("Increase captured data or val set size for more stable validation metrics.")

    if val_positive == 0 or val_negative == 0:
        warnings.append("Validation split is missing at least one class.")
        recommendations.append("Regenerate dataset with both classes represented in validation split.")

    if total_samples < 500:
        warnings.append(f"Balanced dataset is relatively small ({total_samples} windows total).")
        recommendations.append("Capture more sessions to improve robustness and generalization.")

    train_pos_frac = train_positive / max(len(y_train), 1)
    val_pos_frac = val_positive / max(len(y_val), 1)
    if abs(train_pos_frac - val_pos_frac) > 0.20:
        warnings.append("Train/validation class mix differs substantially.")
        recommendations.append("Adjust split strategy or seed to keep train/val class distributions closer.")

    status = "ok" if not warnings else "warning"
    return {
        "status": status,
        "warnings": warnings,
        "recommendations": recommendations,
        "stats": {
            "total_balanced_samples": int(total_samples),
            "train_positive_fraction": float(train_pos_frac),
            "val_positive_fraction": float(val_pos_frac),
            "raw_positive": raw_positive,
            "raw_negative": raw_negative,
            "positive_sessions": positive_sessions,
            "negative_sessions": negative_sessions,
            "mixed_label_sessions": mixed_sessions,
        },
    }


def truncate_to_common_length(features: List[np.ndarray]) -> List[np.ndarray]:
    if not features:
        return features
    min_len = min(len(x) for x in features)
    return [x[:min_len] for x in features]


def balance_classes(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    max_per_class: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise RuntimeError("Need at least one positive and one negative sample")

    target = min(len(pos_idx), len(neg_idx))
    if max_per_class is not None:
        target = min(target, max_per_class)

    pos_pick = rng.choice(pos_idx, size=target, replace=False)
    neg_pick = rng.choice(neg_idx, size=target, replace=False)

    keep = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(keep)
    return x[keep], y[keep]


def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0, 1)")

    idx = np.arange(len(y))
    rng.shuffle(idx)
    x = x[idx]
    y = y[idx]

    n_val = int(round(len(y) * val_frac))
    if n_val == 0 and len(y) >= 2 and val_frac > 0:
        n_val = 1

    x_val = x[:n_val]
    y_val = y[:n_val]
    x_train = x[n_val:]
    y_train = y[n_val:]

    if len(y_train) == 0:
        raise RuntimeError("Train split is empty; decrease val_frac or add more samples")

    return x_train, y_train, x_val, y_val


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.max_per_class is not None and args.max_per_class <= 0:
        args.max_per_class = None

    if args.sessions and args.training_sessions_dir:
        print("Error: use either --sessions or --training-sessions-dir, not both", file=sys.stderr)
        return 1
    if not args.sessions and not args.training_sessions_dir:
        print("Error: provide --sessions or --training-sessions-dir", file=sys.stderr)
        return 1

    if args.training_sessions_dir:
        try:
            session_dirs = discover_session_dirs(Path(args.training_sessions_dir))
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Discovered {len(session_dirs)} session directories under {args.training_sessions_dir}")
    else:
        session_dirs = [Path(p) for p in args.sessions]

    for session_dir in session_dirs:
        if not session_dir.exists() or not session_dir.is_dir():
            print(f"Error: invalid session directory: {session_dir}", file=sys.stderr)
            return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, labels, meta = collect_samples(session_dirs)
    if not features:
        print("Error: no samples found", file=sys.stderr)
        return 1

    features = truncate_to_common_length(features)

    x = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)

    x, y = balance_classes(x, y, rng, max_per_class=args.max_per_class)

    x = x[:, None, :]

    x_train, y_train, x_val, y_val = split_train_val(x, y, args.val_frac, rng)

    np.save(out_dir / "X_train.npy", x_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", x_val)
    np.save(out_dir / "y_val.npy", y_val)

    dataset_meta = {
        "sessions": [str(p) for p in session_dirs],
        "seed": args.seed,
        "val_frac": args.val_frac,
        "max_per_class": args.max_per_class,
        "feature": "abs(iq) robust_zscore",
        "num_samples_total": int(len(y)),
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "window_samples": int(x.shape[-1]),
        "train_positive": int(np.sum(y_train == 1)),
        "train_negative": int(np.sum(y_train == 0)),
        "val_positive": int(np.sum(y_val == 1)),
        "val_negative": int(np.sum(y_val == 0)),
        "capture_meta": meta,
    }

    dataset_meta["quality_analysis"] = build_quality_analysis(
        y_train=y_train,
        y_val=y_val,
        capture_meta=meta,
        total_samples=int(len(y)),
    )

    with (out_dir / "dataset_meta.json").open("w") as handle:
        json.dump(dataset_meta, handle, indent=2)

    print(f"Saved dataset to {out_dir}")
    print(f"X_train: {x_train.shape} y_train: {y_train.shape}")
    print(f"X_val:   {x_val.shape} y_val:   {y_val.shape}")
    analysis = dataset_meta["quality_analysis"]
    if analysis["warnings"]:
        print("Dataset quality warnings:")
        for warning in analysis["warnings"]:
            print(f"- {warning}")
        print("Recommended follow-ups:")
        for recommendation in analysis["recommendations"]:
            print(f"- {recommendation}")
    else:
        print("Dataset quality analysis: no obvious issues detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
