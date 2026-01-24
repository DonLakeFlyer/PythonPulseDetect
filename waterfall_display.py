#!/usr/bin/env python3
"""
Real-time Waterfall Display for Decimated IQ Data

Displays a scrolling waterfall spectrogram of the signal from the decimator output.
"""

import zmq
import numpy as np
import argparse
import sys
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal as sp_signal


class WaterfallDisplay:
    """Real-time waterfall spectrogram display."""

    def __init__(self, host: str, port: int, history_seconds: float = 3.0,
                 update_interval: int = 50, sample_rate: float = 2000.0,
                 target_frames: int = 300):
        """
        Initialize the waterfall display.

        Args:
            host: Host address to connect to
            port: Port number to connect to
            history_seconds: Time history to display in seconds (default: 3.0)
            update_interval: Update interval in milliseconds (default: 50)
            sample_rate: Sample rate in Hz (default: 2000.0 - decimator output at 768kHz/384)
            target_frames: Target number of frames for display (default: 300)
        """
        self.host = host
        self.port = port
        self.update_interval = update_interval
        self.sample_rate = sample_rate
        self.history_size = target_frames

        # Calculate time per frame from history and target frames
        # time_per_frame = history_seconds / target_frames
        self.time_per_frame = history_seconds / target_frames

        # Calculate FFT size from time per frame
        # fft_size = time_per_frame * sample_rate
        self.fft_size = int(self.time_per_frame * sample_rate)

        # Ensure FFT size is at least a power of 2 for efficiency
        if self.fft_size < 64:
            self.fft_size = 64
        # Round to nearest power of 2
        self.fft_size = 2 ** int(np.log2(self.fft_size) + 0.5)

        # Recalculate actual time per frame with rounded FFT size
        self.time_per_frame = self.fft_size / sample_rate

        # Calculate history size in frames from desired time
        self.history_size = int(history_seconds / self.time_per_frame)
        if self.history_size < 10:
            self.history_size = 10  # Minimum 10 frames

        # ZeroMQ
        self.context = None
        self.socket = None

        # Data buffers
        self.waterfall_data = deque(maxlen=self.history_size)
        self.sample_buffer = np.array([], dtype=np.complex64)

        # Statistics
        self.last_sequence = None
        self.dropped_count = 0
        self.frame_count = 0

        # Matplotlib figure and axes
        self.fig = None
        self.ax = None
        self.im = None
        self.colorbar = None

    def connect(self):
        """Connect to ZeroMQ publisher."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        # Increase high water mark to buffer more messages
        self.socket.setsockopt(zmq.RCVHWM, 1000)

        # Set non-blocking receive with timeout
        self.socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

        address = f"tcp://{self.host}:{self.port}"
        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"Connected to {address}")
        print(f"Sample Rate: {self.sample_rate:.1f} Hz")
        print(f"Time per frame: {self.time_per_frame:.3f} seconds")
        print(f"FFT Size: {self.fft_size} (calculated)")
        print(f"Frequency Resolution: {self.sample_rate/self.fft_size:.3f} Hz/bin")
        print(f"History: {self.history_size} frames ({self.history_size * self.time_per_frame:.1f} seconds)")
        print(f"Receive HWM: 1000")
        print("Starting waterfall display...\n")

    def compute_fft(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute FFT and return power spectrum in dB.

        Args:
            samples: Complex IQ samples

        Returns:
            Power spectrum in dB
        """
        try:
            # Validate input
            if len(samples) == 0:
                # Return zeros if no samples
                return np.full(self.fft_size, -100.0, dtype=np.float32)

            # Make a copy to avoid modifying the original
            samples = samples.copy()

            # Ensure we have the right number of samples
            if len(samples) != self.fft_size:
                # Pad or truncate to FFT size
                if len(samples) < self.fft_size:
                    samples = np.pad(samples, (0, self.fft_size - len(samples)), mode='constant')
                else:
                    samples = samples[:self.fft_size]

            # Ensure samples are complex64 and sanitize
            samples = np.asarray(samples, dtype=np.complex64)

            # Check for and fix non-finite values BEFORE windowing
            if not np.all(np.isfinite(samples)):
                print("Warning: Non-finite values in input samples - sanitizing")
                samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

            # Create and validate window function
            window = sp_signal.windows.hann(self.fft_size, sym=False)
            window = np.asarray(window, dtype=np.float32)

            # Validate window
            if len(window) != self.fft_size:
                raise ValueError(f"Window size {len(window)} != FFT size {self.fft_size}")
            if not np.all(np.isfinite(window)):
                raise ValueError("Window contains non-finite values")

            # Apply window function
            windowed = samples * window

            # Double-check windowed data (shouldn't be needed now)
            if not np.all(np.isfinite(windowed)):
                print("Warning: Non-finite values after windowing - sanitizing")
                windowed = np.nan_to_num(windowed, nan=0.0, posinf=0.0, neginf=0.0)

            # Compute FFT
            fft_result = np.fft.fft(windowed, n=self.fft_size)

            # Shift zero frequency to center
            fft_result = np.fft.fftshift(fft_result)

            # Compute power spectrum in dB with overflow protection
            magnitude = np.abs(fft_result)

            # Clip magnitude to prevent overflow (reasonable max for signal power)
            magnitude = np.clip(magnitude, 0, 1e6)

            # Use float64 for power calculation to avoid overflow
            power = magnitude.astype(np.float64) ** 2

            # Convert to dB with floor to avoid log(0)
            power_db = 10 * np.log10(np.maximum(power, 1e-20))

            # Convert back to float32 and clip to reasonable dB range
            power_db = np.clip(power_db, -120.0, 100.0).astype(np.float32)

            # Final validation
            if not np.all(np.isfinite(power_db)):
                power_db = np.nan_to_num(power_db, nan=-100.0, posinf=0.0, neginf=-100.0)

            return power_db

        except Exception as e:
            print(f"Error in compute_fft: {e}")
            import traceback
            traceback.print_exc()
            # Return zeros on error
            return np.full(self.fft_size, -100.0, dtype=np.float32)

    def update_data(self):
        """Fetch new data from ZeroMQ and update buffers."""
        # Process multiple messages per update to keep up with data rate
        messages_processed = 0
        max_messages_per_update = 10  # Process up to 10 messages per animation frame

        while messages_processed < max_messages_per_update:
            try:
                # Try to receive data (non-blocking)
                message_bytes = self.socket.recv(zmq.NOBLOCK)

                # Extract sequence number
                sequence = int.from_bytes(message_bytes[:8], byteorder='little')
                data_bytes = message_bytes[8:]

                # Check for dropped messages
                if self.last_sequence is not None:
                    expected = self.last_sequence + 1
                    if sequence != expected:
                        dropped = sequence - expected
                        self.dropped_count += dropped
                        print(f"⚠️  Dropped {dropped} messages! Total: {self.dropped_count}")
                self.last_sequence = sequence

                # Convert to complex samples
                samples = np.frombuffer(data_bytes, dtype=np.complex64)

                # Add to buffer
                self.sample_buffer = np.concatenate([self.sample_buffer, samples])

                messages_processed += 1

            except zmq.Again:
                # No more messages available
                break

        # Process complete FFT frames
        while len(self.sample_buffer) >= self.fft_size:
            # Take FFT_size samples
            frame = self.sample_buffer[:self.fft_size]
            self.sample_buffer = self.sample_buffer[self.fft_size:]

            # Compute FFT
            power_db = self.compute_fft(frame)

            # Add to waterfall
            self.waterfall_data.append(power_db)
            self.frame_count += 1

    def init_plot(self):
        """Initialize matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Initialize with zeros
        initial_data = np.zeros((self.history_size, self.fft_size))

        # Calculate frequency extent in kHz
        freq_min = -self.sample_rate / 2000.0  # Convert to kHz
        freq_max = self.sample_rate / 2000.0

        # Calculate time extent in seconds
        time_span = self.history_size * self.time_per_frame

        # Create image
        self.im = self.ax.imshow(
            initial_data,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
            vmin=-80,
            vmax=-20,
            extent=[freq_min, freq_max, 0, time_span],
            origin='lower'  # Time increases upward
        )

        # Labels and title
        self.ax.set_xlabel('Frequency (kHz)')
        self.ax.set_ylabel('Time (seconds)')
        self.ax.set_title(f'Waterfall Spectrogram - {self.sample_rate:.1f} Hz Sample Rate')

        # Colorbar
        self.colorbar = self.fig.colorbar(self.im, ax=self.ax, label='Power (dB)')

        # Grid
        self.ax.grid(True, alpha=0.3)

        plt.tight_layout()

    def animate(self, frame):
        """
        Animation update function.

        Args:
            frame: Frame number (from FuncAnimation)

        Returns:
            Updated artist objects
        """
        # Update data
        self.update_data()

        # Convert deque to numpy array
        if len(self.waterfall_data) > 0:
            data = np.array(list(self.waterfall_data))

            # Update image
            self.im.set_data(data)

            # Auto-adjust color limits based on data
            if frame % 10 == 0:  # Update every 10 frames
                vmin = np.percentile(data, 1)
                vmax = np.percentile(data, 99)
                self.im.set_clim(vmin=vmin, vmax=vmax)

            # Calculate elapsed time
            elapsed_time = self.frame_count * self.time_per_frame

            # Update title with stats
            self.ax.set_title(
                f'Waterfall Spectrogram - {self.sample_rate:.1f} Hz | '
                f'Elapsed: {elapsed_time:.1f}s | Frames: {self.frame_count} | Dropped: {self.dropped_count}'
            )

        return [self.im]

    def run(self):
        """Start the waterfall display."""
        try:
            self.connect()
            self.init_plot()

            # Create animation
            anim = animation.FuncAnimation(
                self.fig,
                self.animate,
                interval=self.update_interval,
                blit=True,
                cache_frame_data=False
            )

            plt.show()

        except KeyboardInterrupt:
            print("\nStopping waterfall display...")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("Disconnected")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Real-time waterfall display for decimated IQ data'
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
        help='Port to connect to (default: 5556 - decimator output)'
    )

    parser.add_argument(
        '--sample-rate',
        type=float,
        default=2000.0,
        help='Sample rate in Hz (default: 2000.0 - decimator output at 768kHz/384)'
    )

    parser.add_argument(
        '--history',
        type=float,
        default=3.0,
        help='Time history to display in seconds (default: 3.0)'
    )

    parser.add_argument(
        '--frames',
        type=int,
        default=300,
        help='Target number of frames for display (default: 300)'
    )

    parser.add_argument(
        '--update-interval',
        type=int,
        default=50,
        help='Update interval in milliseconds (default: 50)'
    )

    args = parser.parse_args()

    display = WaterfallDisplay(
        args.host,
        args.port,
        args.history,
        args.update_interval,
        args.sample_rate,
        args.frames
    )

    display.run()


if __name__ == '__main__':
    main()
