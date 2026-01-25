#!/usr/bin/env python3
"""
Airspy HF+ Data Simulator with ZeroMQ Output

This program simulates an Airspy HF+ device by generating synthetic IQ data
and streaming it via ZeroMQ for downstream processing. It mimics the behavior
of the airspy_hf_reader.py without requiring actual hardware.
"""

import argparse
import sys
import numpy as np
import zmq
import time
from typing import Optional

from config import PulseDetectConfig


class AirspyHFSimulator:
    """Simulates Airspy HF+ device and ZeroMQ streaming."""

    def __init__(self, config: PulseDetectConfig, signal_mode: str = 'noise'):
        """
        Initialize the simulator.

        Args:
            config: PulseDetectConfig instance with device and ZeroMQ settings
            signal_mode: Type of signal to generate ('noise', 'tone', 'pulse', 'mixed')
        """
        self.config = config
        self.signal_mode = signal_mode
        self.zmq_context = None
        self.zmq_socket = None
        self.running = False
        self.sample_count = 0
        self.overflow_count = 0
        self.sequence_number = 0
        self.send_attempts = 0
        self.send_failures = 0

        # Simulation parameters
        self.sample_rate = config.get_sample_rate_hz()
        self.center_freq = config.get_frequency_hz()
        self.samples_per_block = 262144  # Match typical AirSpy HF+ block size
        self.phase_accumulator = 0.0

        # Pulse generation parameters (for 'pulse' and 'mixed' modes)
        self.pulse_interval = config.get_pulse_period_ms() / 1000.0  # Convert ms to seconds
        self.pulse_duration = config.get_pulse_width_ms() / 1000.0  # Convert ms to seconds
        self.last_pulse_time = 0.0

    def setup_zmq(self):
        """Initialize ZeroMQ publisher socket."""
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)

        # Allow address reuse to avoid "Address already in use" errors
        self.zmq_socket.setsockopt(zmq.LINGER, 0)

        # Set high water mark from config
        hwm = self.config.get_zmq_hwm()
        self.zmq_socket.setsockopt(zmq.SNDHWM, hwm)

        zmq_address = f"tcp://*:{self.config.get_reader_output_port()}"

        try:
            self.zmq_socket.bind(zmq_address)
            print(f"ZeroMQ publisher bound to {zmq_address}")
            print(f"Send HWM set to {hwm} - overflow detection enabled")
        except zmq.error.ZMQError as e:
            print(f"Error: Failed to bind ZeroMQ socket to {zmq_address}")
            print(f"  {e}")
            print(f"\nTroubleshooting:")
            print(f"  1. Check if port {self.config.get_reader_output_port()} is already in use:")
            print(f"     lsof -i :{self.config.get_reader_output_port()}")
            print(f"  2. Use a different port with: --port <port_number>")
            print(f"  3. Kill the process using the port or wait for it to close")
            raise

        # Give subscribers time to connect (important for PUB/SUB)
        print("Waiting for subscribers to connect...")
        time.sleep(2.0)

    def generate_samples(self) -> np.ndarray:
        """
        Generate synthetic IQ samples based on signal mode.

        Returns:
            Complex numpy array of IQ samples
        """
        if self.signal_mode == 'noise':
            # Generate Gaussian noise
            i_samples = np.random.normal(0, 0.1, self.samples_per_block).astype(np.float32)
            q_samples = np.random.normal(0, 0.1, self.samples_per_block).astype(np.float32)
            samples = i_samples + 1j * q_samples

        elif self.signal_mode == 'tone':
            # Generate a single tone at 10 kHz offset from center
            tone_freq = 10000.0  # 10 kHz
            t = np.arange(self.samples_per_block, dtype=np.float32) / self.sample_rate
            t += self.phase_accumulator
            phase = 2 * np.pi * tone_freq * t
            samples = (0.3 * np.exp(1j * phase)).astype(np.complex64)

            # Add small amount of noise
            noise = np.random.normal(0, 0.02, self.samples_per_block).astype(np.float32) + \
                    1j * np.random.normal(0, 0.02, self.samples_per_block).astype(np.float32)
            samples += noise

            # Update phase accumulator for continuity
            self.phase_accumulator = t[-1] + 1.0 / self.sample_rate

        elif self.signal_mode == 'pulse':
            # Generate pulses at regular intervals
            current_time = self.sample_count / self.sample_rate

            # Start with noise floor
            i_samples = np.random.normal(0, 0.05, self.samples_per_block).astype(np.float32)
            q_samples = np.random.normal(0, 0.05, self.samples_per_block).astype(np.float32)
            samples = i_samples + 1j * q_samples

            # Add pulse if it's time
            if current_time - self.last_pulse_time >= self.pulse_interval:
                pulse_samples = int(self.pulse_duration * self.sample_rate)
                if pulse_samples < self.samples_per_block:
                    # Create a rectangular pulse envelope
                    pulse_env = np.zeros(self.samples_per_block, dtype=np.float32)
                    pulse_env[:pulse_samples] = 1.0  # Flat rectangular pulse

                    # Add baseband pulse (no carrier)
                    tone_freq = 0.0  # Baseband pulse
                    t = np.arange(self.samples_per_block, dtype=np.float32) / self.sample_rate
                    phase = 2 * np.pi * tone_freq * t
                    pulse_signal = 0.5 * pulse_env * np.exp(1j * phase)
                    samples += pulse_signal.astype(np.complex64)

                    self.last_pulse_time = current_time

        elif self.signal_mode == 'mixed':
            # Mix of continuous tone + periodic pulses + noise
            # Continuous tone at 10 kHz
            tone_freq = 10000.0
            t = np.arange(self.samples_per_block, dtype=np.float32) / self.sample_rate
            t += self.phase_accumulator
            phase = 2 * np.pi * tone_freq * t
            samples = (0.2 * np.exp(1j * phase)).astype(np.complex64)

            # Add noise
            noise = np.random.normal(0, 0.05, self.samples_per_block).astype(np.float32) + \
                    1j * np.random.normal(0, 0.05, self.samples_per_block).astype(np.float32)
            samples += noise

            # Add pulse if it's time
            current_time = self.sample_count / self.sample_rate
            if current_time - self.last_pulse_time >= self.pulse_interval:
                pulse_samples = int(self.pulse_duration * self.sample_rate)
                if pulse_samples < self.samples_per_block:
                    pulse_env = np.zeros(self.samples_per_block, dtype=np.float32)
                    pulse_env[:pulse_samples] = 1.0  # Flat rectangular pulse

                    pulse_freq = 0.0  # Baseband pulse
                    t2 = np.arange(self.samples_per_block, dtype=np.float32) / self.sample_rate
                    phase2 = 2 * np.pi * pulse_freq * t2
                    pulse_signal = 0.4 * pulse_env * np.exp(1j * phase2)
                    samples += pulse_signal.astype(np.complex64)

                    self.last_pulse_time = current_time

            self.phase_accumulator = t[-1] + 1.0 / self.sample_rate

        else:
            raise ValueError(f"Unknown signal mode: {self.signal_mode}")

        # Ensure samples are complex64 (matching Airspy HF+ output format)
        return samples.astype(np.complex64)

    def stream_samples(self):
        """Generate and stream samples continuously."""
        self.running = True
        print(f"\nSimulating Airspy HF+ device...")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Center frequency: {self.center_freq} Hz ({self.center_freq/1e6:.3f} MHz)")
        print(f"Signal mode: {self.signal_mode}")
        print(f"Samples per block: {self.samples_per_block}")
        print("\nStarting data simulator...")
        print("Press Ctrl+C to stop\n")

        try:
            while self.running:
                # Generate samples
                samples = self.generate_samples()

                # Validate samples
                if not np.all(np.isfinite(samples)):
                    print(f"⚠️  Warning: Generated {np.sum(~np.isfinite(samples))} non-finite samples! Skipping this buffer.")
                    continue

                self.sample_count += len(samples)
                self.sequence_number += 1

                # Create a message with sequence number prepended
                # Format: 8-byte sequence number + sample data
                seq_bytes = self.sequence_number.to_bytes(8, byteorder='little')
                message = seq_bytes + samples.tobytes()

                # Send via ZeroMQ - non-blocking
                self.send_attempts += 1
                try:
                    self.zmq_socket.send(message, zmq.NOBLOCK)
                except zmq.Again:
                    # Buffer full - data dropped
                    self.send_failures += 1
                    self.overflow_count += 1
                    print(f"⚠️  ZeroMQ send buffer FULL! Dropped message #{self.sequence_number} "
                          f"({len(samples)} samples). Total overflows: {self.overflow_count}")

                # Print status every 1M samples
                if self.sample_count % 1000000 < len(samples):
                    status = f"Streamed {self.sample_count / 1e6:.1f}M samples (seq={self.sequence_number})..."
                    if self.overflow_count > 0:
                        drop_rate = (self.send_failures / self.send_attempts) * 100
                        status += f" | 🔴 Overflows: {self.overflow_count} ({drop_rate:.1f}% drop rate)"
                    print(status)

                # Calculate timing to match real-time sample rate
                block_duration = self.samples_per_block / self.sample_rate
                time.sleep(block_duration)

        except KeyboardInterrupt:
            print("\nStopping simulator...")

    def stop_streaming(self):
        """Stop generating data."""
        self.running = False

    def cleanup(self):
        """Clean up resources."""
        if self.zmq_socket:
            self.zmq_socket.close()

        if self.zmq_context:
            self.zmq_context.term()
            print("ZeroMQ context terminated")

    def run(self):
        """Main execution method."""
        try:
            self.setup_zmq()
            self.stream_samples()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
        finally:
            self.stop_streaming()
            self.cleanup()

        print(f"\nTotal samples generated: {self.sample_count}")
        print(f"Total send attempts: {self.send_attempts}")
        if self.overflow_count > 0:
            drop_rate = (self.send_failures / self.send_attempts) * 100
            print(f"Total buffer overflows: {self.overflow_count} ({drop_rate:.1f}% drop rate)")
            print("Note: Overflows indicate subscribers are not keeping up with data rate")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Simulate Airspy HF+ device data and stream via ZeroMQ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-c', '--config',
        help='JSON configuration file',
        default=None
    )

    parser.add_argument(
        '-f', '--frequency',
        type=int,
        help='Center frequency in Hz (e.g., 146000000 for 146 MHz)',
        default=None
    )

    parser.add_argument(
        '-s', '--sample-rate',
        type=int,
        help='Sample rate in Hz (e.g., 768000)',
        default=None
    )

    parser.add_argument(
        '-p', '--port',
        type=int,
        help='ZeroMQ port number',
        default=None
    )

    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['noise', 'tone', 'pulse', 'mixed'],
        default='pulse',
        help='Signal generation mode: noise (white noise), tone (single carrier), '
             'pulse (periodic pulses), mixed (tone + pulses + noise)'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        try:
            config = PulseDetectConfig.from_file(args.config)
            print(f"Loaded configuration from {args.config}\n")
        except Exception as e:
            print(f"Error loading config file: {e}")
            return 1
    else:
        config = PulseDetectConfig()
        print("Using default configuration\n")

    # Override with command-line arguments
    overrides = {'airspy': {}, 'zmq': {}}
    if args.frequency is not None:
        overrides['airspy']['center_frequency_hz'] = args.frequency
    if args.sample_rate is not None:
        overrides['airspy']['sample_rate_hz'] = args.sample_rate
    if args.port is not None:
        overrides['zmq']['reader_output_port'] = args.port

    # Only update if there are overrides
    if overrides['airspy'] or overrides['zmq']:
        try:
            config.update(overrides)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Print configuration
    print(config)
    print()

    # Run simulator
    simulator = AirspyHFSimulator(config, signal_mode=args.mode)
    return simulator.run()


if __name__ == '__main__':
    sys.exit(main())
