import numpy as np
import pytest

from airspy_mini_reader import AirspyMiniReader
from circular_iq_buffer import CircularIQBuffer
from mini_gain_profile import MiniGainProfile


class FakeMiniBackend:
    def __init__(self) -> None:
        self.started = False
        self.kwargs = None

    def start_stream(self, **kwargs):
        self.started = True
        self.kwargs = kwargs

    def stop_stream(self):
        self.started = False


def test_sample_rate_validation():
    backend = FakeMiniBackend()
    with pytest.raises(ValueError):
        AirspyMiniReader(
            sample_rate_hz=6_000_000,
            center_frequency_hz=433_920_000,
            gain=MiniGainProfile(lna_gain=1, mixer_gain=1, vga_gain=1),
            backend=backend,
        )


def test_gain_validation():
    with pytest.raises(ValueError):
        MiniGainProfile(lna_gain=20, mixer_gain=1, vga_gain=1)


def test_linearity_gain_profile_maps_to_manual_values():
    profile = MiniGainProfile.linearity(0)
    assert profile.mode == "linearity"
    assert profile.stage_gains() == (14, 12, 13)


def test_sensitivity_gain_profile_maps_to_manual_values():
    profile = MiniGainProfile.sensitivity(21)
    assert profile.mode == "sensitivity"
    assert profile.stage_gains() == (0, 0, 4)


def test_mixed_manual_and_preset_is_not_allowed():
    with pytest.raises(ValueError):
        MiniGainProfile(lna_gain=1, mixer_gain=1, vga_gain=1, linearity_gain=0)


def test_partial_manual_values_are_rejected():
    with pytest.raises(ValueError):
        MiniGainProfile(lna_gain=1, mixer_gain=1)


def test_stream_flow_and_drop():
    backend = FakeMiniBackend()
    buffer = CircularIQBuffer(capacity_samples=4)
    reader = AirspyMiniReader(
        sample_rate_hz=3_000_000,
        center_frequency_hz=433_920_000,
        gain=MiniGainProfile(lna_gain=5, mixer_gain=6, vga_gain=7),
        backend=backend,
        buffer=buffer,
    )

    reader.start()
    assert backend.started is True
    assert backend.kwargs is not None
    assert backend.kwargs["sample_rate_hz"] == 3_000_000
    assert backend.kwargs["center_frequency_hz"] == 433_920_000
    assert backend.kwargs["high_accuracy"] is True

    samples = np.arange(12, dtype=np.float32).reshape(6, 2)
    backend.kwargs["callback"](samples)
    out = reader.read(4)
    np.testing.assert_array_equal(out, samples[:4])

    # Buffer holds 4 samples, so pushing 6 drops 2
    assert reader.dropped_samples == 2

    reader.stop()
    assert backend.started is False
