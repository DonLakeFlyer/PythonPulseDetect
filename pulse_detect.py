#!/usr/bin/env python3
"""
Pulse Detection with Folding Engine

This program receives decimated IQ data via ZeroMQ and applies a folding
algorithm to detect periodic pulses. It uses phase binning with triangular
weighting to build up a folded pulse profile over many periods.
"""

import argparse
import sys
import numpy as np
import zmq
import time
from typing import Optional, Tuple
import json

from submodules.AirspyTools.airspy_tools_config import AirspyToolsConfig


class FoldingEngine:
    """
    Implements a pulse folding algorithm with phase binning and triangular weighting.
    """

    def __init__(self, fold_period_ms: float, sample_rate_hz: int, n_bins: int = 256):
        """
        Initialize the folding engine.

        Args:
            fold_period_ms: Expected pulse period in milliseconds
            sample_rate_hz: Sample rate of incoming data in Hz
            n_bins: Number of phase bins per fold (default: 256)
        """
        self.fold_period_ms = fold_period_ms
        self.sample_rate_hz = sample_rate_hz
        self.n_bins = n_bins

        # Folding frequency (Hz) = 1 / period
        self.f_fold = 1000.0 / fold_period_ms  # Convert ms to seconds

        # Accumulator array for phase bins
        self.bins = np.zeros(n_bins, dtype=np.float64)

        # Count of samples accumulated in each bin (for normalization)
        self.bin_counts = np.zeros(n_bins, dtype=np.float64)

        # Sample index counter
        self.sample_index = 0

        # Fold counter
        self.fold_count = 0

        # Statistics
        self.total_samples = 0
        self.total_power = 0.0

    def reset(self):
        """Reset the accumulator arrays."""
        self.bins.fill(0.0)
        self.bin_counts.fill(0.0)
        self.sample_index = 0
        self.fold_count = 0
        self.total_samples = 0
        self.total_power = 0.0

    def process_samples(self, samples: np.ndarray):
        """
        Process a block of IQ samples through the folding engine.

        Args:
            samples: Complex numpy array of IQ samples
        """
        # Compute power (magnitude squared) for each sample
        power = np.abs(samples) ** 2

        # Update statistics
        self.total_samples += len(samples)
        self.total_power += np.sum(power)

        # Process each sample
        for i, p in enumerate(power):
            # Step 2: Phase binning
            # Compute phase position in units of bins
            phase_position = (self.sample_index * self.f_fold * self.n_bins / self.sample_rate_hz)

            # Get the bin index (modulo n_bins for wrapping)
            bin_float = phase_position % self.n_bins
            bin_left = int(np.floor(bin_float))
            bin_right = (bin_left + 1) % self.n_bins

            # Step 3: Triangular weighting
            # Compute fractional position within the bin
            frac = bin_float - bin_left
            weight_left = 1.0 - frac
            weight_right = frac

            # Step 4: Accumulation
            # Add weighted power to adjacent bins
            self.bins[bin_left] += weight_left * p
            self.bins[bin_right] += weight_right * p

            # Track bin counts for normalization
            self.bin_counts[bin_left] += weight_left
            self.bin_counts[bin_right] += weight_right

            # Increment sample index
            self.sample_index += 1

            # Check if we completed a fold
            samples_per_fold = int(self.sample_rate_hz / self.f_fold)
            if self.sample_index >= (self.fold_count + 1) * samples_per_fold:
                self.fold_count += 1

    def get_profile(self, normalized: bool = True) -> np.ndarray:
        """
        Get the current folded pulse profile.

        Args:
            normalized: If True, normalize to SNR (subtract mean, divide by std)
                       If False, return average power per bin

        Returns:
            Array of power in each phase bin
        """
        # Always normalize by bin counts to get average power per bin
        if np.sum(self.bin_counts) > 0:
            profile = np.divide(self.bins, self.bin_counts,
                               out=np.zeros_like(self.bins),
                               where=self.bin_counts > 0)
        else:
            profile = self.bins.copy()

        # Optionally normalize to SNR
        if normalized:
            if len(profile) > 0 and np.std(profile) > 0:
                profile = (profile - np.mean(profile)) / np.std(profile)

        return profile

    def get_statistics(self) -> dict:
        """Get folding statistics."""
        mean_power = self.total_power / self.total_samples if self.total_samples > 0 else 0.0
        profile = self.get_profile(normalized=False)
        profile_snr = self.get_profile(normalized=True)

        return {
            'fold_count': self.fold_count,
            'total_samples': self.total_samples,
            'mean_power': mean_power,
            'fold_period_ms': self.fold_period_ms,
            'f_fold_hz': self.f_fold,
            'peak_bin': int(np.argmax(profile)) if len(profile) > 0 else 0,
            'peak_snr': float(np.max(profile_snr)) if len(profile_snr) > 0 else 0.0,
        }


class PulseDetector:
    """Main pulse detector application."""

    def __init__(self, config: AirspyToolsConfig, n_bins: int = 256,
                 decimated_sample_rate: Optional[int] = None,
                 folds_per_integration: int = 5,
                 display_mode: str = 'snr'):
        """
        Initialize the pulse detector.

        Args:
            config: AirspyToolsConfig instance
            n_bins: Number of phase bins per fold
            decimated_sample_rate: Sample rate after decimation (if None, uses config sample rate)
            folds_per_integration: Number of folds to integrate before outputting results (default: 5)
            display_mode: Profile display mode ('snr' or 'power', default: 'snr')
        """
        self.config = config
        self.n_bins = n_bins
        self.folds_per_integration = folds_per_integration
        self.display_mode = display_mode

        # Use decimated sample rate if provided, otherwise use decimator output sample rate from config
        self.sample_rate = decimated_sample_rate if decimated_sample_rate else config.get_decimator_output_sample_rate_hz()

        # Get pulse parameters from config
        pulse_period_ms = config.get_pulse_period_ms()

        # Initialize folding engine
        self.folding_engine = FoldingEngine(
            fold_period_ms=pulse_period_ms,
            sample_rate_hz=self.sample_rate,
            n_bins=n_bins
        )

        # ZeroMQ setup
        self.zmq_context = None
        self.zmq_socket = None
        self.running = False

        # Statistics
        self.messages_received = 0
        self.last_sequence = 0
        self.missing_sequences = 0

        # Integration tracking
        self.integration_start_folds = 0
        self.integration_count = 0

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
        print(f"Pulse period: {self.config.get_pulse_period_ms()} ms")
        print(f"Phase bins: {self.n_bins}")
        print(f"Fold frequency: {self.folding_engine.f_fold:.3f} Hz")
        print(f"Folds per integration: {self.folds_per_integration}")

    def receive_and_process(self):
        """Main processing loop."""
        self.running = True
        print("\nStarting pulse detection...")
        print("Press Ctrl+C to stop\n")

        try:
            while self.running:
                try:
                    # Receive message from ZeroMQ
                    if self.zmq_socket is None:
                        raise RuntimeError("ZeroMQ socket not initialized")
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

                    # Process samples through folding engine
                    self.folding_engine.process_samples(samples)

                    # Check if we've integrated enough folds
                    folds_since_start = self.folding_engine.fold_count - self.integration_start_folds
                    if folds_since_start >= self.folds_per_integration:
                        # Output detection results
                        self.integration_count += 1
                        self.output_detection_results()

                        # Reset for next integration period
                        self.integration_start_folds = self.folding_engine.fold_count
                        self.folding_engine.reset()

                except zmq.Again:
                    # Timeout - no data received
                    if self.messages_received == 0:
                        print("⏳ Waiting for data from decimator...")
                    continue

        except KeyboardInterrupt:
            print("\n\nStopping pulse detector...")

    def output_detection_results(self):
        """Output detection results after integration period."""
        stats = self.folding_engine.get_statistics()

        print(f"\n{'='*70}")
        print(f"DETECTION RESULTS - Integration #{self.integration_count}")
        print(f"{'='*70}")
        print(f"  Folds integrated: {self.folds_per_integration}")
        print(f"  Total samples: {stats['total_samples']:,}")
        print(f"  Mean power: {stats['mean_power']:.6e}")
        print(f"  Fold period: {stats['fold_period_ms']:.3f} ms ({stats['f_fold_hz']:.3f} Hz)")
        print(f"  Peak bin: {stats['peak_bin']} / {self.n_bins} (phase: {stats['peak_bin']/self.n_bins:.3f})")
        print(f"  Peak SNR: {stats['peak_snr']:.2f} σ")

        # Detection threshold (typically 3-5 sigma)
        if stats['peak_snr'] >= 5.0:
            print(f"  ✓ PULSE DETECTED (SNR ≥ 5σ)")
        elif stats['peak_snr'] >= 3.0:
            print(f"  ⚠ WEAK DETECTION (3σ ≤ SNR < 5σ)")
        else:
            print(f"  ✗ NO DETECTION (SNR < 3σ)")

        print(f"{'='*70}")

        # Display profile
        self.display_profile()

    def print_statistics(self):
        """Print current folding statistics."""
        stats = self.folding_engine.get_statistics()

        print(f"\n{'='*70}")
        print(f"Folding Statistics:")
        print(f"  Messages received: {self.messages_received}")
        if self.missing_sequences > 0:
            print(f"  Missing sequences: {self.missing_sequences}")
        print(f"  Folds completed: {stats['fold_count']}")
        print(f"  Total samples: {stats['total_samples']:,}")
        print(f"  Mean power: {stats['mean_power']:.6e}")
        print(f"  Fold period: {stats['fold_period_ms']:.3f} ms ({stats['f_fold_hz']:.3f} Hz)")
        print(f"  Peak bin: {stats['peak_bin']} / {self.n_bins}")
        print(f"  Peak SNR: {stats['peak_snr']:.2f}")
        print(f"{'='*70}\n")

    def display_profile(self):
        """Display the current folded pulse profile."""
        # Get profile based on display mode
        if self.display_mode == 'power':
            profile = self.folding_engine.get_profile(normalized=False)
            profile_label = "Folded Pulse Profile (Power)"
        else:
            profile = self.folding_engine.get_profile(normalized=True)
            profile_label = "Folded Pulse Profile (SNR)"

        if len(profile) == 0:
            return

        print(f"\n{profile_label}:")

        # Simple ASCII plot
        height = 20
        width = min(self.n_bins, 80)

        # Downsample profile if needed
        if len(profile) > width:
            # Average bins together
            factor = len(profile) // width
            # Trim profile to be evenly divisible by factor
            trim_len = (len(profile) // factor) * factor
            profile_trimmed = profile[:trim_len]
            profile_display = np.mean(profile_trimmed.reshape(-1, factor), axis=1)
        else:
            profile_display = profile

        # Normalize to plot height
        if np.max(profile_display) > np.min(profile_display):
            profile_norm = (profile_display - np.min(profile_display)) / (np.max(profile_display) - np.min(profile_display))
        else:
            profile_norm = np.zeros_like(profile_display)

        # Print ASCII plot
        for row in range(height, 0, -1):
            line = ""
            threshold = row / height
            for val in profile_norm:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            print(f"  {line}")
        print(f"  {'-' * len(profile_display)}")
        print(f"  Phase: 0{'.' * (len(profile_display) - 10)}1.0")
        print()

    def save_profile(self, filename: str):
        """
        Save the folded profile to a file.

        Args:
            filename: Output filename
        """
        profile = self.folding_engine.get_profile(normalized=True)
        stats = self.folding_engine.get_statistics()

        # Create output data
        output = {
            'profile': profile.tolist(),
            'statistics': stats,
            'n_bins': self.n_bins,
            'sample_rate_hz': self.sample_rate,
            'fold_period_ms': self.config.get_pulse_period_ms(),
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Profile saved to: {filename}")

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
            self.display_profile()

            self.cleanup()

        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Detect periodic pulses using folding algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-c', '--config',
        help='JSON configuration file',
        default='capture_config.json'
    )

    parser.add_argument(
        '--bins',
        type=int,
        help='Number of phase bins per fold',
        default=256
    )

    parser.add_argument(
        '--folds',
        type=int,
        help='Number of folds to integrate before outputting results',
        default=5
    )

    parser.add_argument(
        '--display',
        type=str,
        choices=['snr', 'power'],
        help='Profile display mode: snr (normalized) or power (raw)',
        default='snr'
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        help='Decimated sample rate in Hz (overrides config)',
        default=None
    )

    parser.add_argument(
        '--period',
        type=float,
        help='Pulse period in milliseconds (overrides config)',
        default=None
    )

    parser.add_argument(
        '--port',
        type=int,
        help='ZeroMQ decimator output port (overrides config)',
        default=None
    )

    parser.add_argument(
        '-o', '--output',
        help='Save final profile to JSON file',
        default=None
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = AirspyToolsConfig.from_file(args.config)
        print(f"Loaded configuration from {args.config}\n")
    except Exception as e:
        print(f"Error loading config file: {e}")
        return 1

    # Override with command-line arguments
    overrides = {'decimator': {}, 'tag': {}}
    if args.port is not None:
        overrides['decimator']['output_port'] = args.port
    if args.period is not None:
        overrides['tag']['pulse_period_ms'] = args.period

    # Only update if there are overrides
    if overrides['decimator'] or overrides['tag']:
        try:
            config.update(overrides)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Print configuration
    print(config)
    print()

    # Run detector
    detector = PulseDetector(
        config,
        n_bins=args.bins,
        decimated_sample_rate=args.sample_rate,
        folds_per_integration=args.folds,
        display_mode=args.display
    )
    result = detector.run()

    # Save profile if requested
    if args.output:
        detector.save_profile(args.output)

    return result


if __name__ == '__main__':
    sys.exit(main())
