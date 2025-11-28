from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from circular_iq_buffer import CircularIQBuffer
from mini_gain_profile import MiniGainProfile

SUPPORTED_MINI_SAMPLE_RATE = 3_000_000

__all__ = [
    "AirspyMiniBackend",
    "AirspyMiniReader",
    "SUPPORTED_MINI_SAMPLE_RATE",
]


class AirspyMiniBackend(Protocol):
    """Minimal backend surface needed to talk to an Airspy Mini device."""

    def start_stream(
        self,
        *,
        sample_rate_hz: int,
        center_frequency_hz: float,
        gain: MiniGainProfile,
        high_accuracy: bool,
        callback: Callable[[np.ndarray], None],
    ) -> None:
        ...

    def stop_stream(self) -> None:
        ...


class AirspyMiniReader:
    """Read float32 IQ samples from an Airspy Mini in 3 MSPS high-accuracy mode."""

    def __init__(
        self,
        *,
        sample_rate_hz: int,
        center_frequency_hz: float,
        gain: MiniGainProfile,
        backend: AirspyMiniBackend,
        buffer: CircularIQBuffer | None = None,
    ) -> None:
        if sample_rate_hz != SUPPORTED_MINI_SAMPLE_RATE:
            raise ValueError("Only 3 MSPS high-accuracy mode is supported")
        if center_frequency_hz <= 0:
            raise ValueError("center_frequency_hz must be positive")
        if backend is None:
            raise ValueError("backend is required")

        self._sample_rate_hz = sample_rate_hz
        self._center_frequency_hz = center_frequency_hz
        self._gain = gain
        self._backend = backend
        self._buffer = buffer if buffer is not None else CircularIQBuffer(capacity_samples=2_000_000)
        self._running = False
        self._dropped_samples = 0
        self._high_accuracy = True

    @property
    def dropped_samples(self) -> int:
        return self._dropped_samples

    def start(self) -> None:
        if self._running:
            return
        self._backend.start_stream(
            sample_rate_hz=self._sample_rate_hz,
            center_frequency_hz=self._center_frequency_hz,
            gain=self._gain,
            high_accuracy=self._high_accuracy,
            callback=self._handle_samples,
        )
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._backend.stop_stream()
        self._running = False

    def read(self, count: int, *, block: bool = True) -> np.ndarray:
        return self._buffer.pop(count, block=block)

    def _handle_samples(self, samples: np.ndarray) -> None:
        if samples.dtype != np.float32:
            raise ValueError("Incoming samples must be float32")
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("Incoming samples must be shaped (N, 2)")

        accepted = self._buffer.push(samples, block=False)
        dropped = samples.shape[0] - accepted
        if dropped > 0:
            self._dropped_samples += dropped
