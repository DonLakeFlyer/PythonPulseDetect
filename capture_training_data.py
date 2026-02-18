#!/usr/bin/env python3
"""
Capture labeled training windows from the decimator ZeroMQ IQ stream.

This app subscribes to the decimator output stream (sequence number + complex64 IQ),
extracts fixed-length windows, and writes them to disk with JSONL metadata for CNN training.
"""

import argparse
import json
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, List

import numpy as np
import zmq

from submodules.AirspyTools.airspy_tools_config import AirspyToolsConfig


PROFILE_DEFAULTS = {
    "standard": {
        "positive": {
            "window_ms": 64.0,
            "hop_ms": 64.0,
            "duration": 0.0,
            "max_windows": 400,
            "trigger_z": 0.65,
            "trigger_refractory_ms": 80.0,
        },
        "negative": {
            "window_ms": 64.0,
            "hop_ms": 64.0,
            "duration": 0.0,
            "max_windows": 600,
            "trigger_z": None,
            "trigger_refractory_ms": 80.0,
        },
    },
    "quick": {
        "positive": {
            "window_ms": 64.0,
            "hop_ms": 64.0,
            "duration": 0.0,
            "max_windows": 200,
            "trigger_z": 0.65,
            "trigger_refractory_ms": 80.0,
        },
        "negative": {
            "window_ms": 64.0,
            "hop_ms": 64.0,
            "duration": 0.0,
            "max_windows": 300,
            "trigger_z": None,
            "trigger_refractory_ms": 80.0,
        },
    },
    "long": {
        "positive": {
            "window_ms": 64.0,
            "hop_ms": 64.0,
            "duration": 0.0,
            "max_windows": 1200,
            "trigger_z": 0.65,
            "trigger_refractory_ms": 80.0,
        },
        "negative": {
            "window_ms": 64.0,
            "hop_ms": 64.0,
            "duration": 0.0,
            "max_windows": 1800,
            "trigger_z": None,
            "trigger_refractory_ms": 80.0,
        },
    },
}


class TrainingDataCapture:
    """Capture labeled IQ windows for model training."""

    def __init__(
        self,
        output_dir: Path,
        label: str,
        sample_rate_hz: int,
        center_frequency_hz: int,
        pulse_period_ms: float,
        host: str,
        port: int,
        hwm: int,
        stream_logs: bool,
        trigger_debug: bool,
        window_ms: float,
        hop_ms: float,
        duration_s: float,
        max_windows: Optional[int],
        trigger_z: Optional[float],
        trigger_refractory_ms: float,
        profile_name: str,
        pulse_width_ms: float,
    ):
        self.output_dir = output_dir
        self.label = label
        self.sample_rate_hz = sample_rate_hz
        self.center_frequency_hz = center_frequency_hz
        self.pulse_period_ms = pulse_period_ms
        self.host = host
        self.port = port
        self.hwm = hwm
        self.stream_logs = stream_logs
        self.trigger_debug = trigger_debug
        self.window_ms = window_ms
        self.hop_ms = hop_ms
        self.duration_s = duration_s
        self.max_windows = max_windows
        self.trigger_z = trigger_z
        self.trigger_refractory_ms = trigger_refractory_ms
        self.profile_name = profile_name
        self.pulse_width_ms = pulse_width_ms

        self.window_samples = max(1, int(round(self.sample_rate_hz * self.window_ms / 1000.0)))
        self.hop_samples = max(1, int(round(self.sample_rate_hz * self.hop_ms / 1000.0)))
        self.refractory_samples = max(1, int(round(self.sample_rate_hz * self.trigger_refractory_ms / 1000.0)))
        self.pulse_period_samples = max(1, int(round(self.sample_rate_hz * self.pulse_period_ms / 1000.0)))
        self.pulse_width_samples = max(1, int(round(self.sample_rate_hz * self.pulse_width_ms / 1000.0)))
        self.min_pulse_samples = max(1, int(round(self.pulse_width_samples * 0.4)))
        self.max_pulse_samples = max(self.min_pulse_samples + 1, int(round(self.pulse_width_samples * 3.0)))
        self.trigger_release_z = max(1.0, (self.trigger_z or 3.0) * 0.5)
        self.rearm_z = max(0.5, self.trigger_release_z * 0.6)
        self.rearm_required_samples = max(3, self.pulse_width_samples)
        self.min_inter_pulse_samples = self.refractory_samples

        # Moving-average power + moving-std detector settings
        self.avg_power_window_samples = max(3, self.pulse_width_samples // 2)
        self.avg_power_alpha = 2.0 / (self.avg_power_window_samples + 1.0)
        self.std_window_samples = max(5, self.pulse_width_samples * 2)
        self.power_avg_ema = 0.0
        self.std_history = deque()
        self.std_sum = 0.0
        self.std_sumsq = 0.0

        self.context = None
        self.socket = None
        self.running = False

        self.buffer = np.empty(0, dtype=np.complex64)
        self.buffer_start_index = 0
        self.total_samples_received = 0

        self.pending_trigger_indices: List[int] = []
        self.next_window_start = 0

        self.in_pulse = False
        self.detector_armed = True
        self.rearm_below_count = 0
        self.current_pulse_start_index = 0
        self.current_pulse_peak_index = 0
        self.current_pulse_peak_z = -1e9

        self.seq_last = None
        self.seq_drops = 0
        self.messages = 0

        self.windows_saved = 0
        self.trigger_windows_saved = 0
        self.start_time = 0.0
        self.stop_reason = "completed"
        self.last_trigger_sample_index: Optional[int] = None
        self.last_trigger_wall_time_s: Optional[float] = None

        self.trigger_stat_max_z = float("-inf")
        self.trigger_stat_baseline_blocked = 0
        self.trigger_stat_unarmed_blocked = 0
        self.trigger_stat_trigger_crossings = 0
        self.trigger_stat_pulse_candidates = 0
        self.trigger_stat_refractory_blocked = 0

        self.ewma_mean = 0.0
        self.ewma_var = 1e-3
        self.ewma_alpha = 0.01
        self.baseline_samples = 0
        self.min_baseline_samples = max(self.std_window_samples * 4, int(self.sample_rate_hz * 0.5))

        self.max_buffer_samples = max(self.window_samples * 8, self.sample_rate_hz * 30)
        self.pre_trigger_samples = self.window_samples // 3
        self.post_trigger_samples = self.window_samples - self.pre_trigger_samples

        self.session_dir = self._make_session_dir()
        self.manifest_path = self.session_dir / "manifest.jsonl"
        self.status_interval_messages = 1000
        self.window_save_log_interval = 200
        self.eta_log_interval_s = 60.0
        self.next_eta_log_time_s = 0.0

    @staticmethod
    def _format_duration_s(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        minutes, sec = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}h {minutes:02d}m {sec:02d}s"
        return f"{minutes:d}m {sec:02d}s"

    def _log_time_remaining_if_due(self):
        now = time.time()
        if now < self.next_eta_log_time_s:
            return
        self.next_eta_log_time_s = now + self.eta_log_interval_s

        elapsed = max(now - self.start_time, 1e-6)
        estimates = []

        if self.duration_s > 0:
            rem_duration_s = max(0.0, self.duration_s - elapsed)
            estimates.append(rem_duration_s)

        if self.max_windows is not None and self.max_windows > 0:
            windows_remaining = max(0, self.max_windows - self.windows_saved)
            if windows_remaining == 0:
                estimates.append(0.0)
            elif self.windows_saved > 0:
                windows_per_s = self.windows_saved / elapsed
                if windows_per_s > 1e-9:
                    estimates.append(windows_remaining / windows_per_s)

        if not estimates:
            return

        eta_s = min(estimates)
        print(f"Time remaining: ~{self._format_duration_s(eta_s)}")

    def _planned_total_time_text(self) -> str:
        if self.duration_s > 0:
            return self._format_duration_s(self.duration_s)

        if self.max_windows is None or self.max_windows <= 0:
            return "n/a"

        if self.trigger_z is not None:
            est_seconds = self.max_windows * (self.pulse_period_ms / 1000.0)
            return f"~{self._format_duration_s(est_seconds)} (estimated from pulse period)"

        est_seconds = self.max_windows * (self.hop_samples / self.sample_rate_hz)
        return f"~{self._format_duration_s(est_seconds)} (estimated from hop/window target)"

    def _make_session_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode = "trigger" if self.trigger_z is not None else "sequential"
        directory = self.output_dir / f"{self.label}_{timestamp}_{mode}"
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    def setup_zmq(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, self.hwm)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        endpoint = f"tcp://{self.host}:{self.port}"
        self.socket.connect(endpoint)
        print(f"Connected to decimator stream at {endpoint}")
        print(f"RCVHWM={self.hwm}")

    def cleanup(self):
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()

    def handle_signal(self, *_):
        self.running = False

    def append_samples(self, samples: np.ndarray):
        self.buffer = np.concatenate([self.buffer, samples])
        self.total_samples_received += len(samples)

        if len(self.buffer) > self.max_buffer_samples:
            trim = len(self.buffer) - self.max_buffer_samples
            self.buffer = self.buffer[trim:]
            self.buffer_start_index += trim

    def _window_from_global_start(self, start_index: int) -> Optional[np.ndarray]:
        end_index = start_index + self.window_samples
        if start_index < self.buffer_start_index:
            return None
        if end_index > self.total_samples_received:
            return None
        local_start = start_index - self.buffer_start_index
        local_end = local_start + self.window_samples
        return self.buffer[local_start:local_end]

    def _save_window(self, iq_window: np.ndarray, trigger_sample_index: Optional[int]):
        timestamp_ns = time.time_ns()
        file_stem = (
            f"{self.label}_{int(self.center_frequency_hz)}_{self.sample_rate_hz}_"
            f"{self.windows_saved:07d}"
        )
        file_name = f"{file_stem}.npy"
        file_path = self.session_dir / file_name

        np.save(file_path, iq_window.astype(np.complex64))

        record = {
            "file": file_name,
            "label": self.label,
            "center_frequency_hz": self.center_frequency_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "window_ms": self.window_ms,
            "window_samples": self.window_samples,
            "capture_mode": "trigger" if self.trigger_z is not None else "sequential",
            "trigger_sample_index": trigger_sample_index,
            "timestamp_ns": timestamp_ns,
        }

        with self.manifest_path.open("a") as manifest:
            manifest.write(json.dumps(record) + "\n")

        self.windows_saved += 1
        if trigger_sample_index is not None:
            self.trigger_windows_saved += 1
            delta_last_pulse_s = None
            now = time.time()
            if self.last_trigger_sample_index is not None:
                delta_last_pulse_s = (trigger_sample_index - self.last_trigger_sample_index) / self.sample_rate_hz
            self.last_trigger_wall_time_s = now
            self.last_trigger_sample_index = trigger_sample_index

            delta_last_text = "n/a" if delta_last_pulse_s is None else f"{delta_last_pulse_s:.3f}s"
            wall_time_ms = int(round((now - self.start_time) * 1000.0))
            print(
                f"Pulse detected: wall_time_ms={wall_time_ms} count={self.trigger_windows_saved} dt_last_pulse={delta_last_text}"
            )
        if self.stream_logs and self.windows_saved % self.window_save_log_interval == 0:
            print(f"Saved {self.windows_saved} windows...")

    def _update_trigger_state(self, power_samples: np.ndarray, base_global_index: int):
        trigger_z = self.trigger_z if self.trigger_z is not None else 3.0
        for i, raw_power in enumerate(power_samples):
            self.power_avg_ema = (1.0 - self.avg_power_alpha) * self.power_avg_ema + self.avg_power_alpha * raw_power
            avg_power = self.power_avg_ema

            if len(self.std_history) >= self.std_window_samples:
                old = self.std_history.popleft()
                self.std_sum -= old
                self.std_sumsq -= old * old

            self.std_history.append(avg_power)
            self.std_sum += avg_power
            self.std_sumsq += avg_power * avg_power

            if len(self.std_history) < self.std_window_samples:
                continue

            n = len(self.std_history)
            mean_p = self.std_sum / n
            var_p = max(self.std_sumsq / n - mean_p * mean_p, 1e-12)
            moving_std = np.sqrt(var_p)
            trigger_metric = avg_power

            baseline_std = np.sqrt(max(self.ewma_var, 1e-12))
            z = (trigger_metric - self.ewma_mean) / baseline_std

            if not self.in_pulse:
                warmup_mode = self.baseline_samples < self.min_baseline_samples
                can_update_baseline = warmup_mode or (z < trigger_z)
                if can_update_baseline:
                    self.ewma_mean = (1.0 - self.ewma_alpha) * self.ewma_mean + self.ewma_alpha * trigger_metric
                    delta = trigger_metric - self.ewma_mean
                    self.ewma_var = (1.0 - self.ewma_alpha) * self.ewma_var + self.ewma_alpha * (delta * delta)
                    self.baseline_samples += 1
            if z > self.trigger_stat_max_z:
                self.trigger_stat_max_z = float(z)

            global_index = base_global_index + i

            if not self.in_pulse:
                if self.baseline_samples < self.min_baseline_samples:
                    self.trigger_stat_baseline_blocked += 1
                    continue

                if not self.detector_armed:
                    self.trigger_stat_unarmed_blocked += 1
                    if z <= self.rearm_z:
                        self.rearm_below_count += 1
                        if self.rearm_below_count >= self.rearm_required_samples:
                            self.detector_armed = True
                    else:
                        self.rearm_below_count = 0
                    continue

                if z >= trigger_z:
                    self.trigger_stat_trigger_crossings += 1
                    self.in_pulse = True
                    self.current_pulse_start_index = global_index
                    self.current_pulse_peak_index = global_index
                    self.current_pulse_peak_z = z
                continue

            if z > self.current_pulse_peak_z:
                self.current_pulse_peak_z = z
                self.current_pulse_peak_index = global_index

            pulse_len = global_index - self.current_pulse_start_index + 1
            pulse_ended = z <= self.trigger_release_z
            pulse_too_long = pulse_len >= self.max_pulse_samples

            if pulse_ended or pulse_too_long:
                valid_peak = self.current_pulse_peak_z >= trigger_z
                if valid_peak:
                    self.trigger_stat_pulse_candidates += 1
                    self.pending_trigger_indices.append(self.current_pulse_peak_index)
                    self.detector_armed = False
                    self.rearm_below_count = 0

                self.in_pulse = False
                self.current_pulse_peak_z = -1e9

    def _flush_pending_triggers(self):
        if not self.pending_trigger_indices:
            return

        remaining = []
        for center in self.pending_trigger_indices:
            if self.last_trigger_sample_index is not None:
                gap = center - self.last_trigger_sample_index
                if gap < self.min_inter_pulse_samples:
                    self.trigger_stat_refractory_blocked += 1
                    continue

            start = center - self.pre_trigger_samples
            end = center + self.post_trigger_samples

            if start < self.buffer_start_index:
                continue
            if end > self.total_samples_received:
                remaining.append(center)
                continue

            window = self._window_from_global_start(start)
            if window is not None and len(window) == self.window_samples:
                self._save_window(window, trigger_sample_index=center)

        self.pending_trigger_indices = remaining

    def _capture_sequential(self):
        while True:
            window = self._window_from_global_start(self.next_window_start)
            if window is None:
                break
            self._save_window(window, trigger_sample_index=None)
            self.next_window_start += self.hop_samples

    def run(self):
        self.setup_zmq()
        if self.socket is None:
            raise RuntimeError("ZeroMQ socket initialization failed")

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        self.running = True
        self.start_time = time.time()
        self.next_eta_log_time_s = self.start_time + self.eta_log_interval_s

        print("\nCapture started")
        print(f"Session directory: {self.session_dir}")
        print(f"Label: {self.label}")
        print(f"Training profile: {self.profile_name}")
        print(f"Sample rate: {self.sample_rate_hz} Hz")
        print(f"Window: {self.window_ms} ms ({self.window_samples} samples)")
        print(f"Total planned capture time: {self._planned_total_time_text()}")
        if self.trigger_z is not None:
            print(f"Mode: trigger (moving-std tolerance >= {self.trigger_z})")
            print(f"Trigger diagnostics logs: {'enabled' if self.trigger_debug else 'disabled'}")
            print(
                "Trigger width gate: disabled (accepting spikes of any width)"
            )
            print(
                f"Trigger spacing gate (refractory only): min_gap={self.min_inter_pulse_samples} samples "
                f"({1000.0 * self.min_inter_pulse_samples / self.sample_rate_hz:.1f}ms)"
            )
            print(
                f"Power detector windows: avg={self.avg_power_window_samples} samples, "
                f"std={self.std_window_samples} samples"
            )
            print("Pulse detection logs: printing every detected pulse")
        else:
            print(f"Mode: sequential (hop={self.hop_ms} ms)")
        print(f"Streaming progress logs: {'enabled' if self.stream_logs else 'disabled'}")
        if self.duration_s > 0:
            print(f"Stop guidance: target duration {self.duration_s:.0f}s")
        if self.max_windows is not None:
            print(f"Stop guidance: target windows {self.max_windows}")
        if self.duration_s <= 0 and self.max_windows is None:
            print("Stop guidance: no automatic stop target, stop manually with Ctrl+C")

        try:
            while self.running:
                if self.max_windows is not None and self.windows_saved >= self.max_windows:
                    self.stop_reason = f"target windows reached ({self.max_windows})"
                    print("✅ Recommended capture target reached. Stopping automatically.")
                    break
                if self.duration_s > 0 and (time.time() - self.start_time) >= self.duration_s:
                    self.stop_reason = f"target duration reached ({self.duration_s:.0f}s)"
                    print("✅ Recommended capture target reached. Stopping automatically.")
                    break

                if self.socket is None:
                    raise RuntimeError("ZeroMQ socket not initialized")
                message = self.socket.recv()
                self.messages += 1

                seq = int.from_bytes(message[:8], byteorder="little")
                if self.seq_last is not None and seq != self.seq_last + 1:
                    self.seq_drops += max(0, seq - self.seq_last - 1)
                self.seq_last = seq

                samples = np.frombuffer(message[8:], dtype=np.complex64)
                if len(samples) == 0:
                    continue

                base_global = self.total_samples_received
                self.append_samples(samples)

                if self.trigger_z is not None:
                    power = (np.abs(samples).astype(np.float32) ** 2)
                    self._update_trigger_state(power, base_global)
                    self._flush_pending_triggers()
                else:
                    self._capture_sequential()

                self._log_time_remaining_if_due()

                if (self.stream_logs or self.trigger_debug) and self.messages % self.status_interval_messages == 0:
                    elapsed = max(time.time() - self.start_time, 1e-6)
                    rate = self.total_samples_received / elapsed
                    progress_parts = []
                    if self.max_windows is not None and self.max_windows > 0:
                        win_pct = min(100.0, 100.0 * self.windows_saved / self.max_windows)
                        progress_parts.append(f"windows={self.windows_saved}/{self.max_windows} ({win_pct:.1f}%)")
                    else:
                        progress_parts.append(f"windows={self.windows_saved}")

                    if self.duration_s > 0:
                        elapsed = time.time() - self.start_time
                        dur_pct = min(100.0, 100.0 * elapsed / self.duration_s)
                        remaining = max(0.0, self.duration_s - elapsed)
                        progress_parts.append(f"time={elapsed:.0f}s/{self.duration_s:.0f}s ({dur_pct:.1f}%) rem={remaining:.0f}s")

                    trigger_diag = ""
                    if self.trigger_z is not None:
                        baseline_pct = min(100.0, 100.0 * self.baseline_samples / max(self.min_baseline_samples, 1))
                        max_z_text = "n/a" if not np.isfinite(self.trigger_stat_max_z) else f"{self.trigger_stat_max_z:.2f}"
                        trigger_diag = (
                            f" | trig: z_th={self.trigger_z:.2f} max_z={max_z_text} "
                            f"baseline={baseline_pct:.1f}% crossings={self.trigger_stat_trigger_crossings} "
                            f"candidates={self.trigger_stat_pulse_candidates} accepted={self.trigger_windows_saved} "
                            f"blocked_baseline={self.trigger_stat_baseline_blocked} "
                            f"blocked_unarmed={self.trigger_stat_unarmed_blocked} "
                            f"blocked_refractory={self.trigger_stat_refractory_blocked}"
                        )

                    print(
                        f"Messages={self.messages} {' | '.join(progress_parts)} "
                        f"drops={self.seq_drops} stream_rate={rate:.1f} sps{trigger_diag}"
                    )

        finally:
            self.running = False
            self.cleanup()

        elapsed = max(time.time() - self.start_time, 1e-6)
        print("\nCapture complete")
        print(f"Stop reason: {self.stop_reason}")
        print(f"Elapsed: {elapsed:.1f} s")
        print(f"Messages received: {self.messages}")
        print(f"Sequence drops: {self.seq_drops}")
        print(f"Samples received: {self.total_samples_received}")
        print(f"Windows saved: {self.windows_saved}")
        print(f"Detected pulses: {self.trigger_windows_saved}")
        if self.trigger_z is not None:
            max_z_text = "n/a" if not np.isfinite(self.trigger_stat_max_z) else f"{self.trigger_stat_max_z:.2f}"
            print(
                "Trigger diagnostics: "
                f"z_th={self.trigger_z:.2f}, max_z={max_z_text}, "
                f"crossings={self.trigger_stat_trigger_crossings}, "
                f"candidates={self.trigger_stat_pulse_candidates}, "
                f"accepted={self.trigger_windows_saved}, "
                f"blocked_baseline={self.trigger_stat_baseline_blocked}, "
                f"blocked_unarmed={self.trigger_stat_unarmed_blocked}, "
                f"blocked_refractory={self.trigger_stat_refractory_blocked}"
            )
        print(f"Manifest: {self.manifest_path}")

        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture labeled training windows from decimator IQ stream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", "-c", default="capture_config.json", help="JSON config file")
    parser.add_argument("--label", required=True, choices=["positive", "negative"], help="Dataset label")
    parser.add_argument("--output-dir", default="artifacts/training-sessions", help="Base output directory")
    parser.add_argument(
        "--training-profile",
        choices=["standard", "quick", "long"],
        default="standard",
        help="Capture profile preset; CLI args still override individual values",
    )
    parser.add_argument("--host", default="localhost", help="Decimator host")
    parser.add_argument("--port", type=int, default=None, help="Decimator output port")
    parser.add_argument("--hwm", type=int, default=None, help="ZeroMQ receive HWM")
    parser.add_argument(
        "--stream-logs",
        action="store_true",
        help="Enable periodic capture progress logs while streaming (default: off)",
    )
    parser.add_argument(
        "--trigger-debug",
        action="store_true",
        help="Enable periodic trigger diagnostics logs while streaming (default: off)",
    )

    parser.add_argument("--window-ms", type=float, default=None, help="Window length in milliseconds")
    parser.add_argument("--hop-ms", type=float, default=None, help="Sequential mode hop in milliseconds")

    parser.add_argument("--duration", type=float, default=None, help="Capture duration in seconds (0 = run until Ctrl+C)")
    parser.add_argument("--max-windows", type=int, default=None, help="Optional stop after saving N windows")

    parser.add_argument(
        "--trigger-z",
        type=float,
        default=None,
        help="Optional moving-std tolerance threshold for trigger mode",
    )
    parser.add_argument(
        "--trigger-refractory-ms",
        type=float,
        default=None,
        help="Minimum gap between accepted triggers in milliseconds",
    )

    return parser.parse_args()


def resolve_capture_params(args: argparse.Namespace) -> dict:
    profile_values = PROFILE_DEFAULTS[args.training_profile][args.label]

    window_ms = args.window_ms if args.window_ms is not None else profile_values["window_ms"]
    hop_ms = args.hop_ms if args.hop_ms is not None else profile_values["hop_ms"]

    duration = args.duration if args.duration is not None else profile_values["duration"]
    max_windows = args.max_windows if args.max_windows is not None else profile_values["max_windows"]

    trigger_z = args.trigger_z if args.trigger_z is not None else profile_values["trigger_z"]
    trigger_refractory_ms = (
        args.trigger_refractory_ms
        if args.trigger_refractory_ms is not None
        else profile_values["trigger_refractory_ms"]
    )

    return {
        "window_ms": window_ms,
        "hop_ms": hop_ms,
        "duration": duration,
        "max_windows": max_windows,
        "trigger_z": trigger_z,
        "trigger_refractory_ms": trigger_refractory_ms,
    }


def load_config(config_path: str) -> AirspyToolsConfig:
    return AirspyToolsConfig.from_file(config_path)


def main():
    args = parse_args()

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Error loading configuration: {exc}", file=sys.stderr)
        return 1

    port = args.port if args.port is not None else config.get_decimator_output_port()
    hwm = args.hwm if args.hwm is not None else config.get_zmq_hwm()

    sample_rate_hz = config.get_decimator_output_sample_rate_hz()
    center_frequency_hz = config.get_frequency_hz()
    pulse_width_ms = config.get_pulse_width_ms()
    pulse_period_ms = config.get_pulse_period_ms()
    resolved = resolve_capture_params(args)

    capture = TrainingDataCapture(
        output_dir=Path(args.output_dir),
        label=args.label,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=center_frequency_hz,
        pulse_period_ms=pulse_period_ms,
        host=args.host,
        port=port,
        hwm=hwm,
        stream_logs=args.stream_logs,
        trigger_debug=args.trigger_debug,
        window_ms=resolved["window_ms"],
        hop_ms=resolved["hop_ms"],
        duration_s=resolved["duration"],
        max_windows=resolved["max_windows"],
        trigger_z=resolved["trigger_z"],
        trigger_refractory_ms=resolved["trigger_refractory_ms"],
        profile_name=args.training_profile,
        pulse_width_ms=pulse_width_ms,
    )

    return capture.run()


if __name__ == "__main__":
    sys.exit(main())
