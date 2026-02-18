#!/usr/bin/env python3
"""
Run the full reader -> decimator -> capture pipeline.

This script launches:
    1) submodules/AirspyTools/airspy_hf_reader.py
    2) submodules/AirspyTools/decimator.py
  3) capture_training_data.py

When capture exits (for example, after hitting window targets), this script gracefully
shuts down decimator and reader.
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch reader/decimator/capture pipeline and auto-shutdown on capture completion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="capture_config.json",
        help="JSON config file used by reader/decimator/capture",
    )
    parser.add_argument("--label", required=True, choices=["positive", "negative"], help="Capture label")
    parser.add_argument(
        "--output-dir",
        default="artifacts/training-sessions",
        help="Base output directory for captured windows",
    )

    parser.add_argument(
        "--training-profile",
        choices=["standard", "quick", "long"],
        default="standard",
        help="Capture profile passed to capture_training_data.py",
    )

    parser.add_argument("--window-ms", type=float, default=None)
    parser.add_argument("--hop-ms", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--trigger-z", type=float, default=None)
    parser.add_argument("--trigger-refractory-ms", type=float, default=None)
    parser.add_argument(
        "--capture-stream-logs",
        action="store_true",
        help="Enable periodic capture streaming logs (default: off)",
    )
    parser.add_argument(
        "--trigger-debug",
        action="store_true",
        help="Enable periodic trigger diagnostics logs in capture stage (default: off)",
    )

    parser.add_argument(
        "--decimator-stages",
        nargs="+",
        type=int,
        default=[8, 8, 6],
        help="Decimator stages passed to submodule decimator --stages",
    )
    parser.add_argument(
        "--decimator-stream-logs",
        action="store_true",
        help="Enable periodic decimator streaming logs (default: off)",
    )
    parser.add_argument(
        "--reader-stream-logs",
        action="store_true",
        help="Enable periodic reader streaming logs (default: off)",
    )

    parser.add_argument("--reader-warmup-sec", type=float, default=1.5, help="Delay after reader start")
    parser.add_argument("--decimator-warmup-sec", type=float, default=1.5, help="Delay after decimator start")
    parser.add_argument("--shutdown-timeout-sec", type=float, default=8.0, help="Graceful shutdown wait per process")

    return parser.parse_args()


def build_cmd(base_dir: Path, args: argparse.Namespace):
    python_exe = sys.executable
    tools_dir = base_dir / "submodules" / "AirspyTools"

    reader_cmd = [python_exe, "-u", str(tools_dir / "airspy_hf_reader.py"), "-c", args.config]
    if args.reader_stream_logs:
        reader_cmd.append("--stream-logs")
    decimator_cmd = [python_exe, "-u", str(tools_dir / "decimator.py"), "-c", args.config, "--stages"]
    decimator_cmd.extend(str(s) for s in args.decimator_stages)
    if args.decimator_stream_logs:
        decimator_cmd.append("--stream-logs")

    capture_cmd: List[str] = [
        python_exe,
        "-u",
        str(base_dir / "capture_training_data.py"),
        "--config",
        args.config,
        "--label",
        args.label,
        "--output-dir",
        args.output_dir,
        "--training-profile",
        args.training_profile,
    ]

    optional_map = {
        "--window-ms": args.window_ms,
        "--hop-ms": args.hop_ms,
        "--duration": args.duration,
        "--max-windows": args.max_windows,
        "--trigger-z": args.trigger_z,
        "--trigger-refractory-ms": args.trigger_refractory_ms,
    }
    for flag, value in optional_map.items():
        if value is not None:
            capture_cmd.extend([flag, str(value)])

    if args.capture_stream_logs:
        capture_cmd.append("--stream-logs")
    if args.trigger_debug:
        capture_cmd.append("--trigger-debug")

    return reader_cmd, decimator_cmd, capture_cmd


def start_process(name: str, cmd: List[str]) -> subprocess.Popen:
    print(f"Starting {name}: {' '.join(shlex.quote(x) for x in cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"{name} started with pid={proc.pid}")
    return proc


def stream_process_output(name: str, proc: subprocess.Popen) -> threading.Thread:
    def _pump():
        if proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                print(f"[{name}] {line.rstrip()}")
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    return thread


def stop_process(proc: Optional[subprocess.Popen], name: str, timeout_sec: float):
    if proc is None:
        return
    if proc.poll() is not None:
        print(f"{name} already exited with code {proc.returncode}")
        return

    print(f"Stopping {name} (SIGINT)...")
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=timeout_sec)
        print(f"{name} stopped")
        return
    except subprocess.TimeoutExpired:
        pass

    print(f"{name} did not stop in time; sending SIGTERM...")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=max(2.0, timeout_sec / 2.0))
        print(f"{name} stopped after SIGTERM")
        return
    except subprocess.TimeoutExpired:
        pass

    print(f"{name} still running; sending SIGKILL...")
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=2.0)
    print(f"{name} killed")


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    tools_dir = base_dir / "submodules" / "AirspyTools"

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1
    if not (tools_dir / "airspy_hf_reader.py").exists() or not (tools_dir / "decimator.py").exists():
        print(
            "Error: AirspyTools submodule scripts not found. Run: git submodule update --init --recursive",
            file=sys.stderr,
        )
        return 1

    reader_cmd, decimator_cmd, capture_cmd = build_cmd(base_dir, args)

    reader_proc = None
    decimator_proc = None
    capture_proc = None
    output_threads: List[threading.Thread] = []

    abnormal_exit = False
    capture_rc = 1

    try:
        reader_proc = start_process("reader", reader_cmd)
        output_threads.append(stream_process_output("reader", reader_proc))
        time.sleep(max(0.0, args.reader_warmup_sec))

        if reader_proc.poll() is not None:
            print(f"Error: reader exited early with code {reader_proc.returncode}", file=sys.stderr)
            return 1

        decimator_proc = start_process("decimator", decimator_cmd)
        output_threads.append(stream_process_output("decimator", decimator_proc))
        time.sleep(max(0.0, args.decimator_warmup_sec))

        if decimator_proc.poll() is not None:
            print(f"Error: decimator exited early with code {decimator_proc.returncode}", file=sys.stderr)
            return 1

        capture_proc = start_process("capture-training", capture_cmd)
        output_threads.append(stream_process_output("capture-training", capture_proc))
        print("Pipeline running concurrently (reader + decimator + capture-training). Capture controls pipeline lifetime.")

        while True:
            capture_state = capture_proc.poll()
            if capture_state is not None:
                capture_rc = capture_state
                break

            if reader_proc.poll() is not None:
                print(f"Error: reader exited unexpectedly with code {reader_proc.returncode}", file=sys.stderr)
                abnormal_exit = True
                break

            if decimator_proc.poll() is not None:
                print(f"Error: decimator exited unexpectedly with code {decimator_proc.returncode}", file=sys.stderr)
                abnormal_exit = True
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down pipeline...")
        abnormal_exit = True
    finally:
        stop_process(capture_proc, "capture-training", args.shutdown_timeout_sec)
        stop_process(decimator_proc, "decimator", args.shutdown_timeout_sec)
        stop_process(reader_proc, "reader", args.shutdown_timeout_sec)
        for thread in output_threads:
            thread.join(timeout=1.0)

    if abnormal_exit:
        return 1

    print(f"Capture exited with code {capture_rc}")
    return int(capture_rc)


if __name__ == "__main__":
    raise SystemExit(main())
