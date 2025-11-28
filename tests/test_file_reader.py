import numpy as np
import pytest

from circular_iq_buffer import CircularIQBuffer
from file_iq_reader import FileIQReader


def _write_iq_file(path, samples):
    data = np.arange(samples * 2, dtype=np.float32)
    data.tofile(path)
    return data.reshape(samples, 2)


def test_file_reader_streams_into_buffer(tmp_path):
    iq = _write_iq_file(tmp_path / "iq.bin", samples=8)
    buffer = CircularIQBuffer(capacity_samples=16)
    reader = FileIQReader(file_path=tmp_path / "iq.bin", chunk_samples=3, buffer=buffer)

    reader.start()
    reader.join()

    out = reader.read(8)
    np.testing.assert_array_equal(out, iq)
    assert reader.eof is True


def test_file_reader_validates_inputs(tmp_path):
    bogus = tmp_path / "missing.bin"
    with pytest.raises(FileNotFoundError):
        FileIQReader(file_path=bogus)

    path = tmp_path / "bad.bin"
    floats = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    floats.tofile(path)
    reader = FileIQReader(file_path=path, chunk_samples=1)
    reader.start()
    with pytest.raises(ValueError):
        reader.join()