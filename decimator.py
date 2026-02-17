#!/usr/bin/env python3
"""
IQ Data Decimator

Receives IQ data from airspy_zmq_capture.py, decimates it in three stages (8x8x6 = 384x),
and outputs the decimated data via ZeroMQ.
"""

import zmq
import numpy as np
import argparse
import sys
import time
from typing import Optional
from scipy import signal
from config import PulseDetectConfig


class Decimator:
    """Multi-stage decimation processor."""

    def __init__(
        self,
        config: PulseDetectConfig,
        output_port: Optional[int] = None,
        stages: tuple = (8, 8, 6),
        stream_logs: bool = False,
    ):
        """
        Initialize the decimator.

        Args:
            config: PulseDetectConfig instance
            output_port: Port for output ZeroMQ publisher (if None, uses config)
            stages: Tuple of decimation factors (1-3 stages, default: (8, 8, 6))
        """
        self.config = config
        self.output_port = output_port if output_port is not None else config.get_decimator_output_port()
        self.stream_logs = stream_logs

        # Validate stages
        if not isinstance(stages, (tuple, list)) or len(stages) < 1 or len(stages) > 3:
            raise ValueError("stages must be a tuple/list of 1-3 decimation factors")

        self.stages = tuple(stages)
        self.total_decimation = np.prod(stages)

        # Input ZeroMQ
        self.input_context = None
        self.input_socket = None

        # Output ZeroMQ
        self.output_context = None
        self.output_socket = None

        self.running = False

        # Statistics
        self.input_samples = 0
        self.output_samples = 0
        self.input_sequence = None
        self.output_sequence = 0
        self.dropped_input = 0
        self.output_overflows = 0
        self.last_drop_log_count = 0
        self.last_overflow_log_count = 0

        # Calculate output sample rate
        self.input_sample_rate = config.get_sample_rate_hz()
        self.output_sample_rate = self.input_sample_rate // self.total_decimation

        # Stateful per-stage FIR decimation to preserve exact long-run sample ratio
        self.stage_taps = []
        self.stage_filter_state = []
        self.stage_mod = []
        for factor in self.stages:
            numtaps = max(31, factor * 12)
            if numtaps % 2 == 0:
                numtaps += 1
            cutoff = min(0.49, 0.45 / factor)
            taps = signal.firwin(numtaps, cutoff=cutoff, window='hamming')
            self.stage_taps.append(taps)
            self.stage_filter_state.append(np.zeros(numtaps - 1, dtype=np.complex128))
            self.stage_mod.append(0)

    def _decimate_stage_stateful(self, samples: np.ndarray, stage_index: int, factor: int) -> np.ndarray:
        if len(samples) == 0:
            return np.array([], dtype=np.complex64)

        taps = self.stage_taps[stage_index]
        zi = self.stage_filter_state[stage_index]
        filtered, zf = signal.lfilter(taps, [1.0], samples, zi=zi)
        self.stage_filter_state[stage_index] = zf

        # Keep every Nth sample with continuity across message boundaries.
        # This indexing yields floor(total_samples/factor) over time.
        mod = self.stage_mod[stage_index]
        start = (factor - 1 - mod) % factor
        decimated = filtered[start::factor] if start < len(filtered) else np.array([], dtype=filtered.dtype)
        self.stage_mod[stage_index] = (mod + len(filtered)) % factor

        return decimated.astype(np.complex64)

    def setup_input(self):
        """Setup input ZeroMQ subscriber."""
        self.input_context = zmq.Context()
        self.input_socket = self.input_context.socket(zmq.SUB)

        # Set high water mark from config
        hwm = self.config.get_zmq_hwm()
        self.input_socket.setsockopt(zmq.RCVHWM, hwm)

        input_address = f"tcp://localhost:{self.config.get_reader_output_port()}"
        self.input_socket.connect(input_address)
        self.input_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"Input: Connected to {input_address}")
        print(f"Input HWM: {hwm}")

    def setup_output(self):
        """Setup output ZeroMQ publisher."""
        self.output_context = zmq.Context()
        self.output_socket = self.output_context.socket(zmq.PUB)

        # Set high water mark
        hwm = self.config.get_zmq_hwm()
        self.output_socket.setsockopt(zmq.SNDHWM, hwm)
        self.output_socket.setsockopt(zmq.LINGER, 0)

        output_address = f"tcp://*:{self.output_port}"
        self.output_socket.bind(output_address)

        print(f"Output: Bound to {output_address}")
        print(f"Output HWM: {hwm}")
        print(f"Waiting for subscribers to connect...")
        time.sleep(2.0)

    def decimate_multistage(self, samples: np.ndarray) -> np.ndarray:
        """
        Perform multi-stage decimation with anti-aliasing filters using resample_poly.

        Args:
            samples: Input complex samples

        Returns:
            Decimated complex samples
        """
        # Validate input samples
        if len(samples) == 0:
            return np.array([], dtype=np.complex64)

        # Check for and sanitize non-finite values in input
        if not np.all(np.isfinite(samples)):
            print(f"⚠️  Warning: Non-finite values in input samples! Sanitizing...")
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

        # Check input magnitude
        input_max = np.max(np.abs(samples))
        if input_max > 10.0:
            print(f"⚠️  Warning: Large input magnitude {input_max:.2e}, normalizing...")
            samples = samples / input_max

        result = samples.copy()

        for stage, factor in enumerate(self.stages, 1):
            if len(result) == 0:
                break
            try:
                result = self._decimate_stage_stateful(result, stage - 1, factor)

                # Check for non-finite values after resampling
                if not np.all(np.isfinite(result)):
                    print(f"⚠️  Stage {stage}: Non-finite values after stage decimation! Sanitizing...")
                    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

                # Check for overflow
                result_max = np.max(np.abs(result))
                if result_max > 1e10:
                    print(f"⚠️  Stage {stage}: Overflow detected ({result_max:.2e}), normalizing...")
                    result = result / result_max

            except Exception as e:
                print(f"⚠️  Error in decimation stage {stage}: {e}")
                # On error, just decimate without filtering (simple downsampling)
                result = result[::factor]

        if len(result) == 0:
            return np.array([], dtype=np.complex64)

        # Final validation and normalization
        if not np.all(np.isfinite(result)):
            print(f"⚠️  Warning: Non-finite values in decimated output! Sanitizing...")
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize output to reasonable range (avoid downstream overflow)
        result_max = np.max(np.abs(result))
        if result_max > 1.0:
            result = result / result_max

        return result.astype(np.complex64)

    def process_loop(self):
        """Main processing loop."""
        self.running = True

        print("\nDecimation Configuration:")
        print(f"  Input Sample Rate: {self.input_sample_rate/1e3:.1f} kHz")
        print(f"  Decimation Stages: {' x '.join(map(str, self.stages))} = {self.total_decimation}x total")
        print(f"  Output Sample Rate: {self.output_sample_rate/1e3:.1f} kHz")
        print(f"\nStarting decimation...\n")

        try:
            while self.running:
                if self.input_socket is None or self.output_socket is None:
                    raise RuntimeError("ZeroMQ sockets are not initialized")

                # Receive input message
                message_bytes = self.input_socket.recv()

                # Extract sequence number and data
                input_seq = int.from_bytes(message_bytes[:8], byteorder='little')
                data_bytes = message_bytes[8:]

                # Check for dropped input messages
                if self.input_sequence is not None:
                    expected = self.input_sequence + 1
                    if input_seq != expected:
                        dropped = input_seq - expected
                        self.dropped_input += dropped
                        print(
                            f"⚠️  INPUT: Dropped {dropped} messages! "
                            f"(expected seq={expected}, got seq={input_seq}) "
                            f"Total dropped: {self.dropped_input}"
                        )
                self.input_sequence = input_seq

                # Convert to complex samples
                samples = np.frombuffer(data_bytes, dtype=np.complex64)
                self.input_samples += len(samples)

                # Perform multi-stage decimation
                decimated = self.decimate_multistage(samples)
                if len(decimated) == 0:
                    continue
                self.output_samples += len(decimated)

                # Prepare output message with sequence number
                self.output_sequence += 1
                seq_bytes = self.output_sequence.to_bytes(8, byteorder='little')
                output_message = seq_bytes + decimated.tobytes()

                # Send output (non-blocking)
                try:
                    self.output_socket.send(output_message, zmq.NOBLOCK)
                except zmq.Again:
                    self.output_overflows += 1
                    if self.output_overflows <= 3 or (self.output_overflows - self.last_overflow_log_count) >= 25:
                        self.last_overflow_log_count = self.output_overflows
                        print(f"⚠️  OUTPUT: Send buffer full! Dropped output seq={self.output_sequence}. "
                              f"Total overflows: {self.output_overflows}")

                # Print status periodically
                if self.stream_logs and self.output_sequence % 1000 == 0:
                    status = f"Processed: In={self.input_samples/1e6:.1f}M samples, "
                    status += f"Out={self.output_samples/1e6:.1f}M samples, "
                    status += f"Seq={self.output_sequence}"
                    if self.dropped_input > 0:
                        status += f" | 🔴 Input drops: {self.dropped_input}"
                    if self.output_overflows > 0:
                        status += f" | 🔴 Output overflows: {self.output_overflows}"
                    print(status)

        except KeyboardInterrupt:
            print("\nStopping decimator...")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.running = False

        if self.input_socket:
            self.input_socket.close()
        if self.input_context:
            self.input_context.term()

        if self.output_socket:
            self.output_socket.close()
        if self.output_context:
            self.output_context.term()

        print("\nDecimator Statistics:")
        print(f"  Input samples: {self.input_samples}")
        print(f"  Output samples: {self.output_samples}")
        print(f"  Output messages: {self.output_sequence}")
        print(f"  Decimation ratio: {self.input_samples/max(self.output_samples, 1):.1f}x")
        if self.dropped_input > 0:
            print(f"  Input drops: {self.dropped_input}")
        if self.output_overflows > 0:
            print(f"  Output overflows: {self.output_overflows}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Decimate IQ data stream in multiple stages (1-3 stages)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-c', '--config',
        help='JSON configuration file (for input settings)',
        default='capture_config.json'
    )

    parser.add_argument(
        '-o', '--output-port',
        type=int,
        default=None,
        help='Output ZeroMQ port (default: from config or 5556)'
    )

    parser.add_argument(
        '--stages',
        type=int,
        nargs='+',
        default=[8, 8, 6],
        metavar='FACTOR',
        help='Decimation factors for 1-3 stages (default: 8 8 6). Examples: --stages 10 (1 stage), --stages 8 8 (2 stages), --stages 8 8 6 (3 stages)'
    )

    parser.add_argument(
        '--stream-logs',
        action='store_true',
        help='Enable periodic decimator status logs while streaming (default: off)'
    )

    args = parser.parse_args()

    # Validate number of stages
    if len(args.stages) < 1 or len(args.stages) > 3:
        print(f"Error: Must specify 1-3 decimation stages, got {len(args.stages)}")
        return 1

    # Load configuration
    try:
        config = PulseDetectConfig.from_file(args.config)
        print(f"Loaded configuration from {args.config}\n")
    except Exception as e:
        print(f"Error loading config file: {e}")
        return 1

    # Create and run decimator
    decimator = Decimator(config, args.output_port, tuple(args.stages), stream_logs=args.stream_logs)

    try:
        decimator.setup_input()
        decimator.setup_output()
        decimator.process_loop()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
