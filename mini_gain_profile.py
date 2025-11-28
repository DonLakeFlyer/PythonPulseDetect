from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LNA_GAIN_RANGE = range(0, 15)  # 0-14 inclusive
MIXER_GAIN_RANGE = range(0, 16)  # 0-15 inclusive
VGA_GAIN_RANGE = range(0, 16)  # 0-15 inclusive

_LINEARITY_LNA_GAINS = (
    14,
    14,
    14,
    13,
    12,
    10,
    9,
    9,
    8,
    9,
    8,
    6,
    5,
    3,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
)
_LINEARITY_MIXER_GAINS = (
    12,
    12,
    11,
    9,
    8,
    7,
    6,
    6,
    5,
    0,
    0,
    1,
    0,
    0,
    2,
    2,
    1,
    1,
    1,
    1,
    0,
    0,
)
_LINEARITY_VGA_GAINS = (
    13,
    12,
    11,
    11,
    11,
    11,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    9,
    8,
    7,
    6,
    5,
    4,
)
_SENSITIVITY_LNA_GAINS = (
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    13,
    12,
    12,
    9,
    9,
    8,
    7,
    6,
    5,
    3,
    2,
    1,
    0,
)
_SENSITIVITY_MIXER_GAINS = (
    12,
    12,
    12,
    12,
    11,
    10,
    10,
    9,
    9,
    8,
    7,
    4,
    4,
    4,
    3,
    2,
    2,
    1,
    0,
    0,
    0,
    0,
)
_SENSITIVITY_VGA_GAINS = (
    13,
    12,
    11,
    10,
    9,
    8,
    7,
    6,
    5,
    5,
    5,
    5,
    5,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
)
_LINEARITY_PRESETS = tuple(
    zip(_LINEARITY_LNA_GAINS, _LINEARITY_MIXER_GAINS, _LINEARITY_VGA_GAINS)
)
_SENSITIVITY_PRESETS = tuple(
    zip(_SENSITIVITY_LNA_GAINS, _SENSITIVITY_MIXER_GAINS, _SENSITIVITY_VGA_GAINS)
)
PRESET_GAIN_RANGE = range(0, len(_LINEARITY_PRESETS))

__all__ = ["MiniGainProfile", "LNA_GAIN_RANGE", "MIXER_GAIN_RANGE", "VGA_GAIN_RANGE"]


@dataclass(frozen=True, slots=True)
class MiniGainProfile:
    """Airspy Mini gain selection supporting manual, linearity, or sensitivity modes."""

    lna_gain: int | None = None
    mixer_gain: int | None = None
    vga_gain: int | None = None
    linearity_gain: int | None = None
    sensitivity_gain: int | None = None

    def __post_init__(self) -> None:
        manual_values = (self.lna_gain, self.mixer_gain, self.vga_gain)
        manual_specified = any(value is not None for value in manual_values)
        preset_flags = [self.linearity_gain is not None, self.sensitivity_gain is not None]
        preset_count = sum(preset_flags)

        if preset_count and manual_specified:
            raise ValueError(
                "Manual stage gains cannot be combined with linearity/sensitivity presets"
            )
        if preset_count > 1:
            raise ValueError("linearity_gain and sensitivity_gain are mutually exclusive")

        if preset_count == 0:
            if not all(value is not None for value in manual_values):
                raise ValueError(
                    "lna_gain, mixer_gain, and vga_gain must all be provided for manual mode"
                )
            assert self.lna_gain is not None
            assert self.mixer_gain is not None
            assert self.vga_gain is not None
            _validate_gain("lna_gain", self.lna_gain, LNA_GAIN_RANGE)
            _validate_gain("mixer_gain", self.mixer_gain, MIXER_GAIN_RANGE)
            _validate_gain("vga_gain", self.vga_gain, VGA_GAIN_RANGE)
        else:
            preset_name = "linearity_gain" if self.linearity_gain is not None else "sensitivity_gain"
            preset_value = self.linearity_gain if self.linearity_gain is not None else self.sensitivity_gain
            assert preset_value is not None
            _validate_gain(preset_name, preset_value, PRESET_GAIN_RANGE)
            object.__setattr__(self, preset_name, int(preset_value))

    @classmethod
    def linearity(cls, index: int) -> MiniGainProfile:
        """Factory for the airspy_rx-style linearity preset ladder (0-21)."""

        return cls(linearity_gain=index)

    @classmethod
    def sensitivity(cls, index: int) -> MiniGainProfile:
        """Factory for the airspy_rx-style sensitivity preset ladder (0-21)."""

        return cls(sensitivity_gain=index)

    @property
    def mode(self) -> Literal["manual", "linearity", "sensitivity"]:
        if self.linearity_gain is not None:
            return "linearity"
        if self.sensitivity_gain is not None:
            return "sensitivity"
        return "manual"

    def stage_gains(self) -> tuple[int, int, int]:
        """Return explicit (lna, mixer, vga) gains for the configured mode."""

        if self.mode == "manual":
            assert self.lna_gain is not None
            assert self.mixer_gain is not None
            assert self.vga_gain is not None
            return self.lna_gain, self.mixer_gain, self.vga_gain

        index = self.linearity_gain if self.mode == "linearity" else self.sensitivity_gain
        assert index is not None
        table = _LINEARITY_PRESETS if self.mode == "linearity" else _SENSITIVITY_PRESETS
        return table[index]


def _validate_gain(name: str, value: int, allowed: range) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value not in allowed:
        raise ValueError(f"{name} must be within {allowed.start}-{allowed.stop - 1}")
