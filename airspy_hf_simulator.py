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
        if self.signal_mode == 'no-pulse':
            # Generate Gaussian noise only
            i_samples = np.random.normal(0, 0.1, self.samples_per_block).astype(np.float32)
            q_samples = np.random.normal(0, 0.1, self.samples_per_block).astype(np.float32)
            samples = i_samples + 1j * q_samples

        elif self.signal_mode == 'dirty-pulse':
            # Generate pulses at regular intervals with noise
            # Start with noise floor
            i_samples = np.random.normal(0, 0.05, self.samples_per_block).astype(np.float32)
            q_samples = np.random.normal(0, 0.05, self.samples_per_block).astype(np.float32)
            samples = i_samples + 1j * q_samples

            # Calculate pulse timing for each sample in this block
            pulse_samples_total = int(self.pulse_duration * self.sample_rate)

            for i in range(self.samples_per_block):
                current_time = (self.sample_count + i) / self.sample_rate
                time_since_last_pulse = current_time - self.last_pulse_time

                # Check if we should start a new pulse
                if time_since_last_pulse >= self.pulse_interval:
                    self.last_pulse_time = current_time

                # Check if we're within a pulse
                time_in_pulse = current_time - self.last_pulse_time
                if 0 <= time_in_pulse < self.pulse_duration:
                    # Add pulse signal (baseband, no carrier)
                    samples[i] += 0.5

        elif self.signal_mode == 'clean-pulse':
            # Generate pulses at regular intervals with no noise
            # Start with zero (no noise)
            samples = np.zeros(self.samples_per_block, dtype=np.complex64)

            # Calculate pulse timing for each sample in this block
            pulse_samples_total = int(self.pulse_duration * self.sample_rate)

            for i in range(self.samples_per_block):
                current_time = (self.sample_count + i) / self.sample_rate
                time_since_last_pulse = current_time - self.last_pulse_time

                # Check if we should start a new pulse
                if time_since_last_pulse >= self.pulse_interval:
                    self.last_pulse_time = current_time

                # Check if we're within a pulse
                time_in_pulse = current_time - self.last_pulse_time
                if 0 <= time_in_pulse < self.pulse_duration:
                    # Add pulse signal (baseband, no carrier)
                    samples[i] += 0.5

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
        default='capture_config.json'
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
        choices=['no-pulse', 'dirty-pulse', 'clean-pulse'],
        default='dirty-pulse',
        help='Signal generation mode: no-pulse (noise only), '
             'dirty-pulse (pulse + noise), clean-pulse (pulse only, no noise)'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = PulseDetectConfig.from_file(args.config)
        print(f"Loaded configuration from {args.config}\n")
    except Exception as e:
        print(f"Error loading config file: {e}")
        return 1

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
