#!/usr/bin/env python3
"""
Configuration module for Airspy HF+ data capture system.

Manages loading and validation of configuration from JSON files.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path


class PulseDetectConfig:
    """Configuration for Airspy HF+ capture system."""

    # Default configuration values
    DEFAULTS = {
        'airspy': {
            'sample_rate_hz': 768000,       # 768 kHz
            'center_frequency_hz': 146000000,  # 146 MHz
            'lna_gain': 1,                  # LNA on (0=off, 1=on)
            'agc': 0,                       # AGC off (0=off, 1=on)
            'attenuation': 0,               # 0 dB (0-48 in 6 dB steps)
        },
        'zmq': {
            'reader_output_port': 5555,     # Airspy reader output port
            'decimator_output_port': 5556,  # Decimator output port
            'hwm': 10,                      # High water mark for queue
        }
    }

    # Valid sample rates for Airspy HF+ in Hz
    VALID_SAMPLE_RATES_HZ = [
        192000, 256000, 384000, 456000, 512000, 768000, 912000
    ]

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Optional dictionary with configuration values
        """
        # Deep copy defaults
        self.config = {
            'airspy': self.DEFAULTS['airspy'].copy(),
            'zmq': self.DEFAULTS['zmq'].copy()
        }
        if config_dict:
            self.update(config_dict)

    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with values from dictionary.

        Args:
            config_dict: Dictionary with configuration values to update
        """
        # Update airspy settings
        if 'airspy' in config_dict:
            self.config['airspy'].update(config_dict['airspy'])
        # Update zmq settings
        if 'zmq' in config_dict:
            self.config['zmq'].update(config_dict['zmq'])
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        # Validate sample rate
        sample_rate = self.config['airspy']['sample_rate_hz']
        if sample_rate not in self.VALID_SAMPLE_RATES_HZ:
            valid_rates = ', '.join([f"{r}" for r in self.VALID_SAMPLE_RATES_HZ])
            raise ValueError(
                f"Invalid airspy.sample_rate_hz: {sample_rate}. "
                f"Valid rates (Hz): {valid_rates}"
            )

        # Validate LNA gain
        if self.config['airspy']['lna_gain'] not in [0, 1]:
            raise ValueError(f"Invalid airspy.lna_gain: {self.config['airspy']['lna_gain']}. Must be 0 or 1")

        # Validate AGC
        if self.config['airspy']['agc'] not in [0, 1]:
            raise ValueError(f"Invalid airspy.agc: {self.config['airspy']['agc']}. Must be 0 or 1")

        # Validate attenuation
        atten = self.config['airspy']['attenuation']
        if atten not in range(0, 49, 6):
            raise ValueError(
                f"Invalid airspy.attenuation: {atten}. "
                f"Must be 0-48 in 6 dB steps (0, 6, 12, 18, 24, 30, 36, 42, 48)"
            )

        # Validate frequency range (9 kHz to 31 MHz, 60-260 MHz for Airspy HF+)
        freq_hz = self.config['airspy']['center_frequency_hz']
        if not ((9000 <= freq_hz <= 31000000) or (60000000 <= freq_hz <= 260000000)):
            raise ValueError(
                f"Invalid airspy.center_frequency_hz: {freq_hz}. "
                f"Must be 9 kHz-31 MHz or 60-260 MHz"
            )

        # Validate ZeroMQ reader output port
        reader_port = self.config['zmq']['reader_output_port']
        if not (1024 <= reader_port <= 65535):
            raise ValueError(f"Invalid zmq.reader_output_port: {reader_port}. Must be 1024-65535")

        # Validate ZeroMQ decimator output port
        decimator_port = self.config['zmq']['decimator_output_port']
        if not (1024 <= decimator_port <= 65535):
            raise ValueError(f"Invalid zmq.decimator_output_port: {decimator_port}. Must be 1024-65535")

        # Validate HWM
        hwm = self.config['zmq']['hwm']
        if not (1 <= hwm <= 10000):
            raise ValueError(f"Invalid zmq.hwm: {hwm}. Must be 1-10000")

    @classmethod
    def from_file(cls, filepath: str) -> 'PulseDetectConfig':
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            PulseDetectConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
            ValueError: If config values are invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(config_dict)

    def to_file(self, filepath: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to save JSON configuration file
        """
        path = Path(filepath)
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get_sample_rate_hz(self) -> int:
        """Get sample rate in Hz."""
        return self.config['airspy']['sample_rate_hz']

    def get_frequency_hz(self) -> int:
        """Get center frequency in Hz."""
        return self.config['airspy']['center_frequency_hz']

    def get_lna_gain(self) -> int:
        """Get LNA gain setting (0 or 1)."""
        return self.config['airspy']['lna_gain']

    def get_agc(self) -> int:
        """Get AGC setting (0 or 1)."""
        return self.config['airspy']['agc']

    def get_attenuation(self) -> int:
        """Get attenuation in dB."""
        return self.config['airspy']['attenuation']

    def get_zmq_port(self) -> int:
        """Get ZeroMQ reader output port number (for backward compatibility)."""
        return self.config['zmq']['reader_output_port']

    def get_reader_output_port(self) -> int:
        """Get ZeroMQ reader output port number."""
        return self.config['zmq']['reader_output_port']

    def get_decimator_output_port(self) -> int:
        """Get ZeroMQ decimator output port number."""
        return self.config['zmq']['decimator_output_port']

    def get_zmq_hwm(self) -> int:
        """Get ZeroMQ high water mark."""
        return self.config['zmq']['hwm']

    def __str__(self) -> str:
        """Return string representation of configuration."""
        airspy = self.config['airspy']
        zmq = self.config['zmq']
        lines = [
            "Capture Configuration:",
            "  Airspy HF+ Settings:",
            f"    Sample Rate: {airspy['sample_rate_hz']/1e6:.3f} MHz ({airspy['sample_rate_hz']} Hz)",
            f"    Center Frequency: {airspy['center_frequency_hz']/1e6:.3f} MHz ({airspy['center_frequency_hz']} Hz)",
            f"    LNA Gain: {'ON' if airspy['lna_gain'] else 'OFF'}",
            f"    AGC: {'ON' if airspy['agc'] else 'OFF'}",
            f"    Attenuation: {airspy['attenuation']} dB",
            "  ZeroMQ Settings:",
            f"    Reader Output Port: {zmq['reader_output_port']}",
            f"    Decimator Output Port: {zmq['decimator_output_port']}",
            f"    HWM: {zmq['hwm']}",
        ]
        return '\n'.join(lines)

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"PulseDetectConfig({self.config})"


def create_default_config_file(filepath: str) -> None:
    """
    Create a default configuration file.

    Args:
        filepath: Path where to create the config file
    """
    config = PulseDetectConfig()
    config.to_file(filepath)
    print(f"Created default configuration file: {filepath}")


if __name__ == '__main__':
    # Example usage and testing
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'create':
        # Create default config file
        filename = sys.argv[2] if len(sys.argv) > 2 else 'capture_config.json'
        create_default_config_file(filename)
    else:
        # Test configuration
        print("Testing configuration module...\n")

        # Test default config
        config = PulseDetectConfig()
        print(config)
        print()

        # Test custom config
        custom = PulseDetectConfig({
            'airspy': {
                'sample_rate_hz': 912000,
                'center_frequency_hz': 100000000,
                'lna_gain': 0,
            }
        })
        print(custom)
