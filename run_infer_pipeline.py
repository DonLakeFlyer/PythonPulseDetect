#!/usr/bin/env python3
"""
Run the full reader -> decimator -> inference pipeline.

This script launches:
    1) submodules/AirspyTools/airspy_hf_reader.py
    2) submodules/AirspyTools/decimator.py
  3) infer_cnn1d.py

When inference exits (for example, Ctrl+C), this script gracefully shuts down
decimator and reader.
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
        description="Launch reader/decimator/infer pipeline and auto-shutdown on inference completion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="capture_config.json",
        help="JSON config file used by reader/decimator/infer",
    )

    parser.add_argument(
        "--runtime-config",
        default="artifacts/models/cnn_simple/infer_runtime.json",
        help="Runtime JSON for infer_cnn1d.py",
    )
    parser.add_argument("--model", default=None, help="Optional model override passed to infer_cnn1d.py")
    parser.add_argument("--host", default=None, help="Optional decimator host override")
    parser.add_argument("--port", type=int, default=None, help="Optional decimator port override")
    parser.add_argument("--hwm", type=int, default=None, help="Optional ZeroMQ receive HWM override")
    parser.add_argument("--window-ms", type=float, default=None, help="Inference window length")
    parser.add_argument("--hop-ms", type=float, default=None, help="Sliding hop length")
    parser.add_argument("--threshold", type=float, default=None, help="Detection threshold")
    parser.add_argument("--smooth-windows", type=int, default=None, help="Moving average windows")
    parser.add_argument("--cooldown-ms", type=float, default=None, help="Detection cooldown")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device override",
    )
    parser.add_argument(
        "--dump-config",
        default=None,
        help="Optional output path for resolved runtime config produced by infer_cnn1d.py",
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
    parser.add_argument(
        "--infer-stream-logs",
        action="store_true",
        help="Enable periodic inference streaming logs (default: off)",
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

    infer_cmd: List[str] = [
        python_exe,
        "-u",
        str(base_dir / "infer_cnn1d.py"),
        "--config",
        args.config,
    ]

    if args.runtime_config:
        infer_cmd.extend(["--runtime-config", args.runtime_config])

    optional_map = {
        "--model": args.model,
        "--host": args.host,
        "--port": args.port,
        "--hwm": args.hwm,
        "--window-ms": args.window_ms,
        "--hop-ms": args.hop_ms,
        "--threshold": args.threshold,
        "--smooth-windows": args.smooth_windows,
        "--cooldown-ms": args.cooldown_ms,
        "--device": args.device,
        "--dump-config": args.dump_config,
    }
    for flag, value in optional_map.items():
        if value is not None:
            infer_cmd.extend([flag, str(value)])

    if args.infer_stream_logs:
        infer_cmd.append("--stream-logs")

    return reader_cmd, decimator_cmd, infer_cmd


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

    reader_cmd, decimator_cmd, infer_cmd = build_cmd(base_dir, args)

    reader_proc = None
    decimator_proc = None
    infer_proc = None
    output_threads: List[threading.Thread] = []

    abnormal_exit = False
    infer_rc = 1

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

        infer_proc = start_process("infer", infer_cmd)
        output_threads.append(stream_process_output("infer", infer_proc))
        print("Pipeline running concurrently (reader + decimator + infer). Inference controls pipeline lifetime.")

        while True:
            infer_state = infer_proc.poll()
            if infer_state is not None:
                infer_rc = infer_state
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
        stop_process(infer_proc, "infer", args.shutdown_timeout_sec)
        stop_process(decimator_proc, "decimator", args.shutdown_timeout_sec)
        stop_process(reader_proc, "reader", args.shutdown_timeout_sec)
        for thread in output_threads:
            thread.join(timeout=1.0)

    if abnormal_exit:
        return 1

    print(f"Inference exited with code {infer_rc}")
    return int(infer_rc)


if __name__ == "__main__":
    raise SystemExit(main())
