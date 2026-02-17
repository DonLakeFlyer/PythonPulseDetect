#!/usr/bin/env python3
"""
Create inference runtime config JSON from threshold calibration output.

Reads threshold_report.json from calibrate_threshold.py, takes the recommended threshold,
and writes a runtime JSON consumable by infer_cnn1d.py via --runtime-config.
"""

import argparse
import json
from pathlib import Path

from config import PulseDetectConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate runtime config for infer_cnn1d.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--threshold-report", required=True, help="Path to threshold_report.json")
    parser.add_argument(
        "--model",
        default="artifacts/models/cnn_simple/best_model.pt",
        help="Path to best_model.pt",
    )
    parser.add_argument("--out", required=True, help="Output runtime config JSON path")
    parser.add_argument("--config", "-c", default="capture_config.json", help="Base capture JSON config file")
    parser.add_argument("--host", default="localhost", help="Decimator host")
    parser.add_argument("--port", type=int, default=None, help="Decimator output port")
    parser.add_argument("--hwm", type=int, default=None, help="ZeroMQ receive HWM")
    parser.add_argument("--window-ms", type=float, default=64.0, help="Inference window length")
    parser.add_argument("--hop-ms", type=float, default=16.0, help="Sliding hop length")
    parser.add_argument("--smooth-windows", type=int, default=3, help="Smoothing window count")
    parser.add_argument("--cooldown-ms", type=float, default=150.0, help="Detection cooldown")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    threshold_report_path = Path(args.threshold_report)
    model_path = Path(args.model)
    out_path = Path(args.out)

    if not threshold_report_path.exists():
        raise FileNotFoundError(f"Threshold report not found: {threshold_report_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with threshold_report_path.open("r") as handle:
        report = json.load(handle)

    recommended = report.get("recommended", {})
    if "threshold" not in recommended:
        raise ValueError("Invalid threshold report: missing recommended.threshold")

    config = PulseDetectConfig.from_file(args.config)
    port = args.port if args.port is not None else config.get_decimator_output_port()
    hwm = args.hwm if args.hwm is not None else config.get_zmq_hwm()

    runtime = {
        "model": str(model_path),
        "host": args.host,
        "port": int(port),
        "hwm": int(hwm),
        "window_ms": float(args.window_ms),
        "hop_ms": float(args.hop_ms),
        "threshold": float(recommended["threshold"]),
        "smooth_windows": int(args.smooth_windows),
        "cooldown_ms": float(args.cooldown_ms),
        "device": args.device,
        "threshold_source": str(threshold_report_path),
        "threshold_criterion": report.get("criterion"),
        "recommended_metrics": {
            "f1": recommended.get("f1"),
            "precision": recommended.get("precision"),
            "recall": recommended.get("recall"),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(runtime, handle, indent=2)

    print(f"Wrote runtime config: {out_path}")
    print(f"Recommended threshold: {runtime['threshold']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
