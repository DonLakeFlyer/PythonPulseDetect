from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from circular_iq_buffer import CircularIQBuffer

__all__ = ["FileIQReader"]


class FileIQReader:
    """Read float32 IQ samples from a file into a circular buffer."""

    def __init__(
        self,
        *,
        file_path: str | Path,
        chunk_samples: int = 131_072,
        loop: bool = False,
        buffer: CircularIQBuffer | None = None,
    ) -> None:
        self._path = Path(file_path)
        if not self._path.is_file():
            raise FileNotFoundError(self._path)
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")

        self._chunk_samples = int(chunk_samples)
        self._loop = loop
        self._buffer = buffer if buffer is not None else CircularIQBuffer(capacity_samples=2_000_000)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._eof = False
        self._error: Exception | None = None

    @property
    def eof(self) -> bool:
        return self._eof

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._eof = False
        self._thread = threading.Thread(target=self._run, name="FileIQReader", daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join()
        self._running = False
        self._raise_if_error()

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        self._raise_if_error()

    def read(self, count: int, *, block: bool = True) -> np.ndarray:
        return self._buffer.pop(count, block=block)

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._stream_file()
                if not self._loop:
                    break
        except Exception as exc:  # pragma: no cover - surfaced via join()
            self._error = exc
        finally:
            self._running = False
            self._eof = True

    def _stream_file(self) -> None:
        with self._path.open("rb") as handle:
            while not self._stop_event.is_set():
                floats = np.fromfile(handle, dtype=np.float32, count=self._chunk_samples * 2)
                if floats.size == 0:
                    break
                if floats.size % 2:
                    raise ValueError("File contains an incomplete IQ pair")
                samples = floats.reshape(-1, 2)
                self._buffer.push(samples)

    def _raise_if_error(self) -> None:
        if self._error is not None:
            err = self._error
            self._error = None
            raise err
