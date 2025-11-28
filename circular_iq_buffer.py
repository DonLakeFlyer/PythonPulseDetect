from __future__ import annotations

import threading
from typing import Iterable

import numpy as np

__all__ = ["CircularIQBuffer"]


class CircularIQBuffer:
    """Thread-safe circular buffer for interleaved float32 IQ samples."""

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self._capacity = int(capacity_samples)
        self._storage = np.zeros(self._capacity * 2, dtype=np.float32)
        self._head = 0  # next sample index to read
        self._tail = 0  # next slot to write
        self._size = 0  # buffered IQ samples
        self._condition = threading.Condition()

    @property
    def capacity(self) -> int:
        """Maximum number of IQ samples the buffer can hold."""

        return self._capacity

    def __len__(self) -> int:
        with self._condition:
            return self._size

    def push(self, samples: np.ndarray | Iterable[float], *, block: bool = True) -> int:
        """Insert IQ samples into the buffer.

        Args:
            samples: Array-like of interleaved float32 IQ values. Accepts shape
                (N, 2) or flat view of length 2*N.
            block: When True, wait until all samples are written. When False,
                write as many samples as fit and return immediately.

        Returns:
            Number of IQ samples actually written.
        """

        data = _normalize_iq_samples(samples)
        total = data.size // 2
        written = 0

        with self._condition:
            while written < total:
                space = self._capacity - self._size
                if space == 0:
                    if not block:
                        break
                    self._condition.wait()
                    continue

                chunk = min(space, total - written)
                self._write_chunk(data[written * 2 : (written + chunk) * 2], chunk)
                written += chunk
                self._size += chunk
                self._condition.notify_all()

        return written

    def pop(self, count: int, *, block: bool = True) -> np.ndarray:
        """Remove IQ samples from the buffer.

        Args:
            count: Number of IQ samples requested.
            block: When True, wait until `count` samples are available. When False,
                return immediately with whatever is buffered.

        Returns:
            A NumPy array shaped (M, 2) where M<=count if block=False.
        """

        if count <= 0:
            raise ValueError("count must be positive")

        with self._condition:
            target = count
            while self._size < target:
                if not block:
                    target = min(target, self._size)
                    break
                self._condition.wait()

            if target == 0:
                return np.empty((0, 2), dtype=np.float32)

            data = self._read_chunk(target)
            self._condition.notify_all()
            return data

    def clear(self) -> None:
        """Drop all buffered samples."""

        with self._condition:
            self._head = 0
            self._tail = 0
            self._size = 0
            self._condition.notify_all()

    def _write_chunk(self, data: np.ndarray, count: int) -> None:
        first = min(count, self._capacity - self._tail)
        self._storage[self._tail * 2 : (self._tail + first) * 2] = data[: first * 2]
        remaining = count - first
        if remaining:
            self._storage[0 : remaining * 2] = data[first * 2 :]
        self._tail = (self._tail + count) % self._capacity

    def _read_chunk(self, count: int) -> np.ndarray:
        out = np.empty(count * 2, dtype=np.float32)
        first = min(count, self._capacity - self._head)
        out[: first * 2] = self._storage[self._head * 2 : (self._head + first) * 2]
        remaining = count - first
        if remaining:
            out[first * 2 :] = self._storage[0 : remaining * 2]
        self._head = (self._head + count) % self._capacity
        self._size -= count
        return out.reshape(-1, 2)


def _normalize_iq_samples(samples: np.ndarray | Iterable[float]) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[1] != 2:
            raise ValueError("Expected shape (N, 2) for IQ pairs")
        array = array.reshape(-1)
    elif array.ndim == 1:
        if array.size % 2:
            raise ValueError("Flat IQ data must contain an even number of floats")
    else:
        raise ValueError("IQ data must be 1-D or 2-D array-like")
    return np.ascontiguousarray(array, dtype=np.float32)
