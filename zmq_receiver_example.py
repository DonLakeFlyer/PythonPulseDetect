#!/usr/bin/env python3
"""
Example ZeroMQ receiver for Airspy HF+ data stream.

This demonstrates how to receive and process the data sent by airspy_hf_reader.py
"""

import zmq
import numpy as np
import argparse
import sys
import time


class DataReceiver:
    """Receives and processes data from ZeroMQ stream."""

    def __init__(self, host: str, port: int, hwm: int = 10, slow_delay: float = 0.0):
        """
        Initialize the receiver.

        Args:
            host: Host address to connect to
            port: Port number to connect to
            hwm: High water mark for receive queue
            slow_delay: Delay in seconds to simulate slow consumer (default: 0.0)
        """
        self.host = host
        self.port = port
        self.hwm = hwm
        self.slow_delay = slow_delay
        self.context = None
        self.socket = None
        self.running = False
        self.total_samples = 0
        self.last_sequence = None
        self.dropped_count = 0

    def connect(self):
        """Connect to ZeroMQ publisher."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        # Set high water mark
        self.socket.setsockopt(zmq.RCVHWM, self.hwm)

        address = f"tcp://{self.host}:{self.port}"
        self.socket.connect(address)

        # Subscribe to all messages
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"Connected to {address}")
        print(f"Receive HWM set to {self.hwm}")
        if self.slow_delay > 0:
            print(f"Running in SLOW mode with {self.slow_delay}s delay per message")
            print(f"Expect overflow warnings on producer side!")
        print("Waiting for data...\n")

    def receive_loop(self):
        """Main receive loop."""
        self.running = True

        try:
            while self.running:
                # Receive single-part message (sequence number + data)
                if self.socket is None:
                    raise RuntimeError("ZeroMQ socket not initialized")
                message_bytes = self.socket.recv()

                # Extract sequence number (first 8 bytes)
                sequence = int.from_bytes(message_bytes[:8], byteorder='little')
                data_bytes = message_bytes[8:]

                # Check for dropped messages
                if self.last_sequence is not None:
                    expected = self.last_sequence + 1
                    if sequence != expected:
                        dropped = sequence - expected
                        self.dropped_count += dropped
                        print(f"⚠️  DROPPED {dropped} messages! "
                              f"(expected seq={expected}, got seq={sequence}) "
                              f"Total dropped: {self.dropped_count}")
                self.last_sequence = sequence

                # Convert data back to complex numpy array
                samples = np.frombuffer(data_bytes, dtype=np.complex64)

                # Validate received samples
                if not np.all(np.isfinite(samples)):
                    non_finite = np.sum(~np.isfinite(samples))
                    print(f"⚠️  WARNING: {non_finite}/{len(samples)} non-finite samples in message {sequence}")

                # Check for unreasonably large values
                max_mag = np.max(np.abs(samples))
                if max_mag > 100.0:
                    print(f"⚠️  WARNING: Very large sample magnitude: {max_mag:.2e} in message {sequence}")

                self.total_samples += len(samples)

                # Process the data (example: compute power spectrum)
                power = np.abs(samples) ** 2
                avg_power = np.mean(power)
                max_power = np.max(power)

                # Print status
                status = f"Received seq={sequence} | {len(samples)} samples | "
                status += f"Avg Power: {avg_power:.6f} | "
                status += f"Max Power: {max_power:.6f} | "
                status += f"Total: {self.total_samples/1e6:.1f}M"
                if self.dropped_count > 0:
                    status += f" | 🔴 DROPPED: {self.dropped_count}"
                print(status)

                # Simulate slow consumer if requested
                if self.slow_delay > 0:
                    time.sleep(self.slow_delay)

                # Here you would add your signal processing code
                # For example:
                # - FFT analysis
                # - Pulse detection
                # - Demodulation
                # - Recording to file

        except KeyboardInterrupt:
            print("\nStopping receiver...")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("Disconnected")
        print(f"Total samples received: {self.total_samples}")
        if self.dropped_count > 0:
            print(f"Total messages dropped: {self.dropped_count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Receive and process Airspy HF+ data via ZeroMQ'
    )

    parser.add_argument(
        '--host',
        default='localhost',
        help='Host to connect to (default: localhost)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5556,
        help='Port to connect to (default: 5556)'
    )

    parser.add_argument(
        '--hwm',
        type=int,
        default=10,
        help='Receive high water mark (default: 10)'
    )

    parser.add_argument(
        '--slow',
        type=float,
        default=0.0,
        metavar='SECONDS',
        help='Delay in seconds per message to simulate slow consumer (default: 0.0)'
    )

    args = parser.parse_args()

    receiver = DataReceiver(args.host, args.port, args.hwm, args.slow)
    receiver.connect()
    receiver.receive_loop()


if __name__ == '__main__':
    main()
