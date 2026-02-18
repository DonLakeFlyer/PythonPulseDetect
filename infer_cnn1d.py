#!/usr/bin/env python3
"""
Run real-time 1D CNN pulse inference on decimator IQ stream.

Input stream format (from decimator):
  - 8-byte little-endian sequence number
  - complex64 IQ payload
"""

import argparse
import json
import math
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import zmq

from submodules.AirspyTools.airspy_tools_config import AirspyToolsConfig


DEFAULT_MODEL_PATH = "artifacts/models/cnn_simple/best_model.pt"


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


def robust_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    scale = 1.4826 * mad + eps
    return (x - median) / scale


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


class LiveInfer:
    def __init__(
        self,
        config: AirspyToolsConfig,
        model_path: Path,
        host: str,
        port: int,
        hwm: int,
        window_ms: float,
        hop_ms: float,
        threshold: float,
        smooth_windows: int,
        cooldown_ms: float,
        device: torch.device,
        stream_logs: bool,
    ):
        self.config = config
        self.model_path = model_path
        self.host = host
        self.port = port
        self.hwm = hwm
        self.sample_rate_hz = config.get_decimator_output_sample_rate_hz()
        self.window_samples = max(1, int(round(window_ms * self.sample_rate_hz / 1000.0)))
        self.hop_samples = max(1, int(round(hop_ms * self.sample_rate_hz / 1000.0)))
        self.threshold = threshold
        self.smooth_windows = max(1, smooth_windows)
        self.cooldown_samples = max(1, int(round(cooldown_ms * self.sample_rate_hz / 1000.0)))
        self.device = device
        self.stream_logs = stream_logs

        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.running = False

        self.model: Optional[nn.Module] = None
        self.buffer = np.empty(0, dtype=np.complex64)
        self.buffer_start_index = 0
        self.total_samples = 0
        self.next_window_start = 0

        self.last_seq = None
        self.seq_drops = 0
        self.message_count = 0

        self.smooth_queue = deque(maxlen=self.smooth_windows)
        self.last_detection_sample = -10**12

    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        in_channels = int(checkpoint.get("input_channels", 1))

        model = PulseCNN1D(in_channels=in_channels)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

        val_f1 = checkpoint.get("val_f1")
        epoch = checkpoint.get("epoch")
        if val_f1 is not None and epoch is not None:
            print(f"Loaded model from epoch={epoch} val_f1={val_f1:.4f}")
        else:
            print("Loaded model checkpoint")

    def setup_zmq(self):
        endpoint = f"tcp://{self.host}:{self.port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, self.hwm)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(endpoint)
        print(f"Connected to {endpoint} (RCVHWM={self.hwm})")

    def cleanup(self):
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()

    def handle_signal(self, *_):
        self.running = False

    def _append_samples(self, samples: np.ndarray):
        self.buffer = np.concatenate([self.buffer, samples])
        self.total_samples += len(samples)

        max_buf = max(self.window_samples * 8, self.sample_rate_hz * 20)
        if len(self.buffer) > max_buf:
            trim = len(self.buffer) - max_buf
            self.buffer = self.buffer[trim:]
            self.buffer_start_index += trim

    def _get_window(self, start_index: int) -> Optional[np.ndarray]:
        if start_index < self.buffer_start_index:
            return None
        end_index = start_index + self.window_samples
        if end_index > self.total_samples:
            return None
        local_start = start_index - self.buffer_start_index
        local_end = local_start + self.window_samples
        return self.buffer[local_start:local_end]

    def _infer_prob(self, iq_window: np.ndarray) -> float:
        envelope = np.abs(iq_window).astype(np.float32)
        feature = robust_zscore(envelope)
        x = torch.from_numpy(feature[None, None, :]).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits)[0].item()
        return float(prob)

    def _estimate_snr_db(self, iq_window: np.ndarray) -> float:
        power = np.abs(iq_window).astype(np.float32) ** 2
        noise_power = float(np.median(power))
        signal_power = float(np.percentile(power, 95.0))
        eps = 1e-12
        return 10.0 * math.log10((signal_power + eps) / (noise_power + eps))

    def run(self) -> int:
        self.load_model()
        self.setup_zmq()

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        self.running = True

        print("Live inference started. Press Ctrl+C to stop.")
        print(
            f"window={self.window_samples} samples hop={self.hop_samples} samples "
            f"threshold={self.threshold:.3f} smooth={self.smooth_windows}"
        )

        start = time.time()
        try:
            while self.running:
                if self.socket is None:
                    raise RuntimeError("ZeroMQ socket not initialized")
                message = self.socket.recv()
                self.message_count += 1

                seq = int.from_bytes(message[:8], byteorder="little")
                if self.last_seq is not None and seq != self.last_seq + 1:
                    self.seq_drops += max(0, seq - self.last_seq - 1)
                self.last_seq = seq

                iq = np.frombuffer(message[8:], dtype=np.complex64)
                if len(iq) == 0:
                    continue

                self._append_samples(iq)

                while True:
                    window = self._get_window(self.next_window_start)
                    if window is None:
                        break

                    prob = self._infer_prob(window)
                    self.smooth_queue.append(prob)
                    prob_smooth = float(np.mean(self.smooth_queue))

                    center_index = self.next_window_start + self.window_samples // 2
                    since_last = center_index - self.last_detection_sample
                    if prob_smooth >= self.threshold and since_last >= self.cooldown_samples:
                        self.last_detection_sample = center_index
                        t_sec = center_index / self.sample_rate_hz
                        snr_db = self._estimate_snr_db(window)
                        print(
                            f"DETECT t={t_sec:9.3f}s prob={prob_smooth:.3f} "
                            f"(raw={prob:.3f}, snr_db={snr_db:.1f}, seq={seq})"
                        )

                    self.next_window_start += self.hop_samples

                if self.stream_logs and self.message_count % 200 == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    rate = self.total_samples / elapsed
                    print(
                        f"status msgs={self.message_count} drops={self.seq_drops} "
                        f"samples={self.total_samples} rate={rate:.1f} sps"
                    )

        finally:
            self.running = False
            self.cleanup()

        print("Inference stopped")
        print(f"Messages: {self.message_count}")
        print(f"Sequence drops: {self.seq_drops}")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time CNN inference from decimator IQ stream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", default="capture_config.json", help="JSON config file")
    parser.add_argument("--runtime-config", default=None, help="Optional runtime JSON (model + inference params)")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to best_model.pt (default: artifacts/models/cnn_simple/best_model.pt)",
    )
    parser.add_argument("--host", default=None, help="Decimator host")
    parser.add_argument("--port", type=int, default=None, help="Decimator port override")
    parser.add_argument("--hwm", type=int, default=None, help="ZeroMQ receive HWM override")
    parser.add_argument("--window-ms", type=float, default=None, help="Inference window length")
    parser.add_argument("--hop-ms", type=float, default=None, help="Sliding hop length")
    parser.add_argument("--threshold", type=float, default=None, help="Detection threshold")
    parser.add_argument("--smooth-windows", type=int, default=None, help="Moving average windows")
    parser.add_argument("--cooldown-ms", type=float, default=None, help="Detection cooldown")
    parser.add_argument(
        "--stream-logs",
        action="store_true",
        help="Enable periodic inference status logs while streaming (default: off)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device",
    )
    parser.add_argument("--dump-config", default=None, help="Optional output path to save resolved runtime config JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    training_meta = {}
    try:
        with Path(args.config).open("r") as handle:
            raw_config = json.load(handle)
        if isinstance(raw_config, dict):
            value = raw_config.get("training", {})
            if isinstance(value, dict):
                training_meta = value
    except Exception:
        training_meta = {}

    try:
        config = AirspyToolsConfig.from_file(args.config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    runtime = {}
    if args.runtime_config:
        runtime_path = Path(args.runtime_config)
        if not runtime_path.exists():
            print(f"Error: runtime config not found: {runtime_path}", file=sys.stderr)
            return 1
        try:
            with runtime_path.open("r") as handle:
                runtime = json.load(handle)
        except Exception as exc:
            print(f"Error loading runtime config: {exc}", file=sys.stderr)
            return 1

    model_value = args.model if args.model is not None else runtime.get("model", DEFAULT_MODEL_PATH)
    if not model_value:
        print("Error: model path could not be resolved", file=sys.stderr)
        return 1

    model_path = Path(model_value)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        return 1

    host = args.host if args.host is not None else runtime.get("host", "localhost")
    port = args.port if args.port is not None else int(runtime.get("port", config.get_decimator_output_port()))
    hwm = args.hwm if args.hwm is not None else int(runtime.get("hwm", config.get_zmq_hwm()))
    window_ms = args.window_ms if args.window_ms is not None else float(runtime.get("window_ms", 64.0))
    hop_ms = args.hop_ms if args.hop_ms is not None else float(runtime.get("hop_ms", 16.0))
    threshold = args.threshold if args.threshold is not None else float(runtime.get("threshold", 0.6))
    smooth_windows = (
        args.smooth_windows if args.smooth_windows is not None else int(runtime.get("smooth_windows", 3))
    )
    cooldown_ms = args.cooldown_ms if args.cooldown_ms is not None else float(runtime.get("cooldown_ms", 150.0))

    device_arg = args.device if args.device is not None else runtime.get("device", "auto")
    try:
        device = select_device(device_arg)
    except Exception as exc:
        print(f"Error selecting device: {exc}", file=sys.stderr)
        return 1

    if args.dump_config:
        resolved = {
            "model": str(model_path),
            "host": host,
            "port": port,
            "hwm": hwm,
            "sample_rate_hz": config.get_decimator_output_sample_rate_hz(),
            "window_ms": window_ms,
            "hop_ms": hop_ms,
            "threshold": threshold,
            "smooth_windows": smooth_windows,
            "cooldown_ms": cooldown_ms,
            "device": str(device),
        }
        with Path(args.dump_config).open("w") as handle:
            json.dump(resolved, handle, indent=2)

    criterion = training_meta.get("last_threshold_criterion")
    if criterion:
        prepared_utc = training_meta.get("last_prepared_utc")
        detail = f" (prepared={prepared_utc})" if prepared_utc else ""
        print(f"Calibration criterion from config: {criterion}{detail}")

    app = LiveInfer(
        config=config,
        model_path=model_path,
        host=host,
        port=port,
        hwm=hwm,
        window_ms=window_ms,
        hop_ms=hop_ms,
        threshold=threshold,
        smooth_windows=smooth_windows,
        cooldown_ms=cooldown_ms,
        device=device,
        stream_logs=args.stream_logs,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
