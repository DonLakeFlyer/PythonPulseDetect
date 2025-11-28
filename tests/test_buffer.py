import threading
import time

import numpy as np

from circular_iq_buffer import CircularIQBuffer


def _iq_sequence(samples: int, *, start: float = 0.0) -> np.ndarray:
    data = np.arange(start, start + samples * 2, dtype=np.float32)
    return data.reshape(samples, 2)


def test_push_pop_roundtrip():
    buf = CircularIQBuffer(capacity_samples=8)
    payload = _iq_sequence(4)
    written = buf.push(payload)
    assert written == 4
    out = buf.pop(4)
    np.testing.assert_array_equal(out, payload)


def test_wraparound_behavior():
    buf = CircularIQBuffer(capacity_samples=4)
    first = _iq_sequence(3)
    second = _iq_sequence(3, start=100.0)
    buf.push(first)
    np.testing.assert_array_equal(buf.pop(2), first[:2])
    buf.push(second)
    assert len(buf) == 4
    expected = np.vstack([first[2:], second[:3]])
    np.testing.assert_array_equal(buf.pop(4), expected)


def test_non_blocking_push_and_pop():
    buf = CircularIQBuffer(capacity_samples=2)
    buf.push(_iq_sequence(2))
    assert buf.push(_iq_sequence(1), block=False) == 0
    buf.pop(2)
    out = buf.pop(1, block=False)
    assert out.shape == (0, 2)


def test_blocking_pop_waits_for_data():
    buf = CircularIQBuffer(capacity_samples=4)
    result: list[np.ndarray] = []

    def consumer() -> None:
        result.append(buf.pop(2))

    thread = threading.Thread(target=consumer)
    thread.start()
    time.sleep(0.05)
    buf.push(_iq_sequence(2))
    thread.join(timeout=1)
    assert thread.is_alive() is False
    np.testing.assert_array_equal(result[0], _iq_sequence(2))