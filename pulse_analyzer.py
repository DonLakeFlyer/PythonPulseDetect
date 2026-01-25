#!/usr/bin/env python3
"""
Pulse Analyzer

This program receives decimated IQ data via ZeroMQ and analyzes individual
pulses to measure their frequency, width, period, and jitter.
"""

import argparse
import sys
import numpy as np
import zmq
import time
from typing import Optional, List, Tuple
from collections import deque

from config import PulseDetectConfig


class PulseAnalyzer:
    """Analyzes individual pulses in IQ data stream."""

    def __init__(self, config: PulseDetectConfig,
                 decimated_sample_rate: Optional[int] = None,
                 threshold_sigma: float = 3.0,
                 history_size: int = 100):
        """
        Initialize the pulse analyzer.

        Args:
            config: PulseDetectConfig instance
            decimated_sample_rate: Sample rate after decimation (if None, uses config)
            threshold_sigma: Detection threshold in standard deviations above noise
            history_size: Number of pulses to keep in history for statistics
        """
        self.config = config
        self.threshold_sigma = threshold_sigma
        self.history_size = history_size

        # Use decimated sample rate if provided, otherwise use config
        self.sample_rate = decimated_sample_rate if decimated_sample_rate else config.get_decimator_output_sample_rate_hz()

        # ZeroMQ setup
        self.zmq_context = None
        self.zmq_socket = None
        self.running = False

        # Statistics
        self.messages_received = 0
        self.last_sequence = 0
        self.missing_sequences = 0

        # Pulse detection state
        self.in_pulse = False
        self.pulse_start_sample = 0
        self.pulse_peak_power = 0.0
        self.pulse_samples = []

        # Background noise estimation
        self.noise_buffer = deque(maxlen=1000)
        self.noise_mean = 0.0
        self.noise_std = 0.0
        self.detection_threshold = 0.0

        # Pulse history
        self.pulse_widths = deque(maxlen=history_size)  # in seconds
        self.pulse_periods = deque(maxlen=history_size)  # in seconds
        self.pulse_powers = deque(maxlen=history_size)
        self.last_pulse_end_sample = None

        # Global sample counter
        self.sample_count = 0
        self.pulse_count = 0

        # Statistics update
        self.last_stats_time = time.time()
        self.stats_interval = 5.0  # seconds

    def setup_zmq(self):
        """Initialize ZeroMQ subscriber socket."""
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)

        # Subscribe to all messages
        self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b'')

        # Set receive timeout
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout

        # Connect to decimator output
        zmq_address = f"tcp://localhost:{self.config.get_decimator_output_port()}"
        self.zmq_socket.connect(zmq_address)

        print(f"Connected to decimator output: {zmq_address}")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Detection threshold: {self.threshold_sigma}σ above noise")
        print(f"Pulse history size: {self.history_size}")

    def update_noise_estimate(self, power: float):
        """
        Update background noise estimate.

        Args:
            power: Power sample to add to noise estimate
        """
        self.noise_buffer.append(power)

        if len(self.noise_buffer) >= 100:
            self.noise_mean = np.mean(self.noise_buffer)
            self.noise_std = np.std(self.noise_buffer)
            self.detection_threshold = self.noise_mean + self.threshold_sigma * self.noise_std

    def process_samples(self, samples: np.ndarray):
        """
        Process a block of IQ samples to detect and analyze pulses.

        Args:
            samples: Complex numpy array of IQ samples
        """
        # Compute power (magnitude squared) for each sample
        power = np.abs(samples) ** 2

        for i, p in enumerate(power):
            current_sample = self.sample_count + i

            # Update noise estimate when not in pulse
            if not self.in_pulse:
                self.update_noise_estimate(p)

            # Check if we have enough noise samples to set threshold
            if len(self.noise_buffer) < 100:
                continue

            # Pulse detection state machine
            if not self.in_pulse:
                # Check for pulse start
                if p > self.detection_threshold:
                    self.in_pulse = True
                    self.pulse_start_sample = current_sample
                    self.pulse_peak_power = p
                    self.pulse_samples = [p]
            else:
                # In pulse - accumulate samples
                self.pulse_samples.append(p)
                if p > self.pulse_peak_power:
                    self.pulse_peak_power = p

                # Check for pulse end
                if p < self.detection_threshold:
                    # Pulse ended
                    self.process_pulse(current_sample)
                    self.in_pulse = False

        # Update global sample counter
        self.sample_count += len(samples)

    def process_pulse(self, pulse_end_sample: int):
        """
        Process a detected pulse and extract measurements.

        Args:
            pulse_end_sample: Sample index where pulse ended
        """
        self.pulse_count += 1

        # Calculate pulse width at -3dB points (half power) for more accurate measurement
        # This reduces the effect of filter ringing at pulse edges
        if len(self.pulse_samples) > 0:
            pulse_power_array = np.array(self.pulse_samples)
            peak_power = np.max(pulse_power_array)
            half_power = peak_power / 2.0

            # Find samples above half power
            above_half = pulse_power_array >= half_power
            if np.any(above_half):
                # Find first and last sample above half power
                indices = np.where(above_half)[0]
                pulse_width_samples = indices[-1] - indices[0] + 1
            else:
                # Fallback to full width if no samples above half power
                pulse_width_samples = len(self.pulse_samples)
        else:
            pulse_width_samples = len(self.pulse_samples)

        pulse_width_sec = pulse_width_samples / self.sample_rate
        self.pulse_widths.append(pulse_width_sec)

        # Calculate pulse period if we have a previous pulse
        if self.last_pulse_end_sample is not None:
            period_samples = pulse_end_sample - self.last_pulse_end_sample
            period_sec = period_samples / self.sample_rate
            self.pulse_periods.append(period_sec)

        # Store pulse power
        self.pulse_powers.append(self.pulse_peak_power)

        self.last_pulse_end_sample = pulse_end_sample

        # Print individual pulse detection (optional - can be commented out for less verbose output)
        # print(f"Pulse #{self.pulse_count}: width={pulse_width_sec*1000:.2f}ms, peak_power={self.pulse_peak_power:.3e}")

    def get_statistics(self) -> dict:
        """Get current pulse statistics."""
        stats = {
            'pulse_count': self.pulse_count,
            'sample_count': self.sample_count,
            'noise_mean': self.noise_mean,
            'noise_std': self.noise_std,
            'detection_threshold': self.detection_threshold,
        }

        if len(self.pulse_widths) > 0:
            widths_ms = np.array(self.pulse_widths) * 1000
            stats['pulse_width_mean_ms'] = float(np.mean(widths_ms))
            stats['pulse_width_std_ms'] = float(np.std(widths_ms))
            stats['pulse_width_min_ms'] = float(np.min(widths_ms))
            stats['pulse_width_max_ms'] = float(np.max(widths_ms))

        if len(self.pulse_periods) > 0:
            periods_ms = np.array(self.pulse_periods) * 1000
            stats['pulse_period_mean_ms'] = float(np.mean(periods_ms))
            stats['pulse_period_std_ms'] = float(np.std(periods_ms))
            stats['pulse_period_min_ms'] = float(np.min(periods_ms))
            stats['pulse_period_max_ms'] = float(np.max(periods_ms))

            # Calculate jitter as percentage of mean period
            if stats['pulse_period_mean_ms'] > 0:
                stats['pulse_period_jitter_pct'] = float(stats['pulse_period_std_ms'] / stats['pulse_period_mean_ms'] * 100)

            # Calculate pulse repetition frequency
            stats['pulse_rate_hz'] = float(1000.0 / stats['pulse_period_mean_ms'])

        if len(self.pulse_powers) > 0:
            stats['pulse_power_mean'] = float(np.mean(self.pulse_powers))
            stats['pulse_power_std'] = float(np.std(self.pulse_powers))

        return stats

    def print_statistics(self):
        """Print current pulse analysis statistics."""
        stats = self.get_statistics()

        print(f"\n{'='*70}")
        print(f"PULSE ANALYSIS STATISTICS")
        print(f"{'='*70}")
        print(f"  Samples processed: {stats['sample_count']:,}")
        print(f"  Pulses detected: {stats['pulse_count']}")
        print(f"\nNoise Statistics:")
        print(f"  Mean power: {stats['noise_mean']:.6e}")
        print(f"  Std dev: {stats['noise_std']:.6e}")
        print(f"  Detection threshold: {stats['detection_threshold']:.6e} ({self.threshold_sigma}σ)")

        if 'pulse_width_mean_ms' in stats:
            print(f"\nPulse Width:")
            print(f"  Mean: {stats['pulse_width_mean_ms']:.3f} ms")
            print(f"  Std dev: {stats['pulse_width_std_ms']:.3f} ms")
            print(f"  Range: {stats['pulse_width_min_ms']:.3f} - {stats['pulse_width_max_ms']:.3f} ms")

        if 'pulse_period_mean_ms' in stats:
            print(f"\nPulse Period:")
            print(f"  Mean: {stats['pulse_period_mean_ms']:.3f} ms")
            print(f"  Std dev: {stats['pulse_period_std_ms']:.3f} ms")
            print(f"  Jitter: {stats['pulse_period_jitter_pct']:.2f}%")
            print(f"  Range: {stats['pulse_period_min_ms']:.3f} - {stats['pulse_period_max_ms']:.3f} ms")
            print(f"  Pulse rate: {stats['pulse_rate_hz']:.3f} Hz")

        if 'pulse_power_mean' in stats:
            print(f"\nPulse Power:")
            print(f"  Mean: {stats['pulse_power_mean']:.6e}")
            print(f"  Std dev: {stats['pulse_power_std']:.6e}")

        print(f"{'='*70}\n")

    def receive_and_process(self):
        """Main processing loop."""
        self.running = True
        print("\nStarting pulse analysis...")
        print("Collecting noise samples for threshold estimation...")
        print("Press Ctrl+C to stop\n")

        try:
            while self.running:
                try:
                    # Receive message from ZeroMQ
                    message = self.zmq_socket.recv()

                    # Parse message: 8-byte sequence number + sample data
                    if len(message) < 8:
                        print(f"⚠️  Warning: Received short message ({len(message)} bytes)")
                        continue

                    sequence = int.from_bytes(message[:8], byteorder='little')
                    sample_bytes = message[8:]

                    # Convert bytes to complex64 samples
                    samples = np.frombuffer(sample_bytes, dtype=np.complex64)

                    # Check for missing sequences
                    if self.messages_received > 0:
                        expected_seq = self.last_sequence + 1
                        if sequence != expected_seq:
                            missed = sequence - expected_seq
                            self.missing_sequences += missed
                            print(f"⚠️  Missing {missed} sequence(s) (expected {expected_seq}, got {sequence})")

                    self.last_sequence = sequence
                    self.messages_received += 1

                    # Process samples through pulse analyzer
                    self.process_samples(samples)

                    # Print statistics periodically
                    current_time = time.time()
                    if current_time - self.last_stats_time >= self.stats_interval:
                        self.print_statistics()
                        self.last_stats_time = current_time

                except zmq.Again:
                    # Timeout - no data received
                    if self.messages_received == 0:
                        print("⏳ Waiting for data from decimator...")
                    continue

        except KeyboardInterrupt:
            print("\n\nStopping pulse analyzer...")

    def cleanup(self):
        """Clean up resources."""
        if self.zmq_socket:
            self.zmq_socket.close()

        if self.zmq_context:
            self.zmq_context.term()

    def run(self):
        """Main execution method."""
        try:
            self.setup_zmq()
            self.receive_and_process()
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # Print final statistics
            print("\nFinal Statistics:")
            self.print_statistics()

            self.cleanup()

        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze pulse characteristics (frequency, width, period, jitter)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-c', '--config',
        help='JSON configuration file',
        default=None
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        help='Decimated sample rate in Hz (overrides config)',
        default=None
    )

    parser.add_argument(
        '--port',
        type=int,
        help='ZeroMQ decimator output port (overrides config)',
        default=None
    )

    parser.add_argument(
        '--threshold',
        type=float,
        help='Detection threshold in sigma (standard deviations above noise)',
        default=3.0
    )

    parser.add_argument(
        '--history',
        type=int,
        help='Number of pulses to keep in history for statistics',
        default=100
    )

    parser.add_argument(
        '--stats-interval',
        type=float,
        help='Interval in seconds between statistics updates',
        default=5.0
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
    overrides = {'decimator': {}}
    if args.port is not None:
        overrides['decimator']['output_port'] = args.port

    # Only update if there are overrides
    if overrides['decimator']:
        try:
            config.update(overrides)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Print configuration
    print(config)
    print()

    # Run analyzer
    analyzer = PulseAnalyzer(
        config,
        decimated_sample_rate=args.sample_rate,
        threshold_sigma=args.threshold,
        history_size=args.history
    )

    # Set stats interval
    analyzer.stats_interval = args.stats_interval

    return analyzer.run()


if __name__ == '__main__':
    sys.exit(main())
