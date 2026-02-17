#!/usr/bin/env python3
"""
Airspy HF+ Data Reader with ZeroMQ Output

This program reads IQ data from an Airspy HF+ device and streams it
via ZeroMQ for downstream processing.
"""

import argparse
import sys
import numpy as np
import zmq
import time

# Import our ctypes wrapper for libairspyhf
import airspyhf_wrapper as airspy
from config import PulseDetectConfig


class AirspyHFReader:
    """Manages Airspy HF+ device and ZeroMQ streaming."""

    def __init__(self, config: PulseDetectConfig, stream_logs: bool = False):
        """
        Initialize the reader system.

        Args:
            config: PulseDetectConfig instance with device and ZeroMQ settings
        """
        self.config = config
        self.stream_logs = stream_logs
        self.device = None
        self.zmq_context = None
        self.zmq_socket = None
        self.running = False
        self.sample_count = 0
        self.overflow_count = 0
        self.sequence_number = 0
        self.send_attempts = 0
        self.send_failures = 0
        self.last_overflow_log_count = 0

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

    def setup_device(self):
        """Initialize and configure the Airspy HF+ device."""
        # Open device
        result, self.device = airspy.open()
        if result != airspy.AIRSPYHF_SUCCESS:
            raise RuntimeError(f"Failed to open Airspy HF+ device: {result}")

        print("Airspy HF+ device opened successfully")

        # Set sample rate
        sample_rate = self.config.get_sample_rate_hz()
        result = airspy.set_samplerate(self.device, sample_rate)
        if result != airspy.AIRSPYHF_SUCCESS:
            raise RuntimeError(f"Failed to set sample rate: {result}")
        print(f"Sample rate set to: {sample_rate} Hz")

        # Set frequency
        frequency = self.config.get_frequency_hz()
        result = airspy.set_freq(self.device, frequency)
        if result != airspy.AIRSPYHF_SUCCESS:
            raise RuntimeError(f"Failed to set frequency: {result}")
        print(f"Frequency set to: {frequency} Hz ({frequency/1e6:.3f} MHz)")

        # Set LNA gain (0 = off, 1 = on)
        lna_gain = self.config.get_lna_gain()
        result = airspy.set_hf_lna(self.device, lna_gain)
        if result != airspy.AIRSPYHF_SUCCESS:
            print(f"Warning: Failed to set LNA gain: {result}")
        else:
            print(f"LNA gain: {'ON' if lna_gain else 'OFF'}")

        # Set AGC (0 = off, 1 = on)
        agc = self.config.get_agc()
        result = airspy.set_hf_agc(self.device, agc)
        if result != airspy.AIRSPYHF_SUCCESS:
            print(f"Warning: Failed to set AGC: {result}")
        else:
            print(f"AGC: {'ON' if agc else 'OFF'}")

        # Set attenuation (0-48 dB in 6 dB steps)
        atten = self.config.get_attenuation()
        result = airspy.set_hf_att(self.device, atten // 6)
        if result != airspy.AIRSPYHF_SUCCESS:
            print(f"Warning: Failed to set attenuation: {result}")
        else:
            print(f"Attenuation: {atten} dB")

    def callback(self, transfer):
        """
        Callback function for receiving samples from the device.

        Args:
            transfer: Data transfer object from airspy library
        """
        if not self.running:
            return -1  # Stop streaming

        try:
            if self.zmq_socket is None:
                return -1

            # Get IQ samples as float32 array (interleaved I/Q)
            # Airspy HF+ provides float32 samples, 2 values per complex sample (I, Q)
            float_count = transfer.contents.sample_count * 2
            samples_float = np.ctypeslib.as_array(
                transfer.contents.samples,
                shape=(float_count,)
            )

            # Convert interleaved I/Q floats to complex samples
            samples = samples_float[::2] + 1j * samples_float[1::2]

            # Validate samples before sending
            if not np.all(np.isfinite(samples)):
                print(f"⚠️  Warning: Airspy produced {np.sum(~np.isfinite(samples))} non-finite samples! Dropping this buffer.")
                return 0  # Continue but skip this buffer

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
                if self.overflow_count <= 3 or (self.overflow_count - self.last_overflow_log_count) >= 25:
                    self.last_overflow_log_count = self.overflow_count
                    print(f"⚠️  ZeroMQ send buffer FULL! Dropped message #{self.sequence_number} "
                          f"({len(samples)} samples). Total overflows: {self.overflow_count}")

            # Optional status output while streaming
            if self.stream_logs and self.sample_count % 10000000 < len(samples):
                status = f"Streamed {self.sample_count / 1e6:.1f}M samples (seq={self.sequence_number})..."
                if self.overflow_count > 0:
                    drop_rate = (self.send_failures / self.send_attempts) * 100
                    status += f" | 🔴 Overflows: {self.overflow_count} ({drop_rate:.1f}% drop rate)"
                print(status)

        except Exception as e:
            print(f"Error in callback: {e}")
            return -1

        return 0  # Continue streaming

    def start_streaming(self):
        """Start reading and streaming data."""
        self.running = True
        print("\nStarting data reader...")
        print("Press Ctrl+C to stop\n")

        # Start receiving samples
        result = airspy.start(self.device, self.callback)
        if result != airspy.AIRSPYHF_SUCCESS:
            raise RuntimeError(f"Failed to start streaming: {result}")

        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping capture...")

    def stop_streaming(self):
        """Stop reading data."""
        self.running = False
        if self.device:
            airspy.stop(self.device)
            print("Streaming stopped")

    def cleanup(self):
        """Clean up resources."""
        if self.device:
            airspy.close(self.device)
            print("Device closed")

        if self.zmq_socket:
            self.zmq_socket.close()

        if self.zmq_context:
            self.zmq_context.term()
            print("ZeroMQ context terminated")

    def run(self):
        """Main execution method."""
        try:
            self.setup_zmq()
            self.setup_device()
            self.start_streaming()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        finally:
            self.stop_streaming()
            self.cleanup()

        print(f"\nTotal samples captured: {self.sample_count}")
        print(f"Total send attempts: {self.send_attempts}")
        if self.overflow_count > 0:
            drop_rate = (self.send_failures / self.send_attempts) * 100
            print(f"Total buffer overflows: {self.overflow_count} ({drop_rate:.1f}% drop rate)")
            print("Note: Overflows indicate subscribers are not keeping up with data rate")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Read data from Airspy HF+ and stream via ZeroMQ',
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
        '--lna',
        type=int,
        choices=[0, 1],
        help='LNA gain (0=off, 1=on)',
        default=None
    )

    parser.add_argument(
        '--agc',
        type=int,
        choices=[0, 1],
        help='AGC (0=off, 1=on)',
        default=None
    )

    parser.add_argument(
        '--attenuation',
        type=int,
        choices=[0, 6, 12, 18, 24, 30, 36, 42, 48],
        help='Attenuation in dB (0-48 in 6 dB steps)',
        default=None
    )

    parser.add_argument(
        '--stream-logs',
        action='store_true',
        help='Enable periodic status logs while streaming (default: off)',
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
    if args.lna is not None:
        overrides['airspy']['lna_gain'] = args.lna
    if args.agc is not None:
        overrides['airspy']['agc'] = args.agc
    if args.attenuation is not None:
        overrides['airspy']['attenuation'] = args.attenuation

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

    # Run reader
    reader = AirspyHFReader(config, stream_logs=args.stream_logs)
    return reader.run()


if __name__ == '__main__':
    sys.exit(main())
