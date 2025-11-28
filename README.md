# airspytools

Experimental utilities for working with Airspy SDR captures. The current
building blocks are:

- `CircularIQBuffer`: a NumPy-backed, thread-safe circular buffer for
	interleaved float32 IQ samples.
- `AirspyMiniReader`: a high-level wrapper around Airspy Mini streams locked to
	3 MSPS high-accuracy mode.
- `FileIQReader`: a helper that replays float32 IQ recordings into a
	`CircularIQBuffer` using the same API shape as the hardware reader.

## Requirements

- Python 3.12+
- NumPy 1.26 or newer

## Installation

```bash
pip install airspytools
```

To work on the project locally:

```bash
pip install -e .[dev]
```

### Local virtual environment helper scripts

If you prefer working in an isolated environment, the repository ships with
simple helpers that create and activate a `.venv` at the project root:

```bash
# create or refresh the environment and install project deps
bash scripts/setup_venv.sh

# activate it in your current shell session
source scripts/activate_venv.sh
```

Both scripts take care of resolving the project root, so you can run them from
any directory inside the repo. Pass a custom interpreter via
`PYTHON_BIN=/path/to/python bash scripts/setup_venv.sh` if you need a specific
Python build.

## Usage

```python
import numpy as np
from airspy_mini_reader import AirspyMiniReader
from circular_iq_buffer import CircularIQBuffer
from mini_gain_profile import MiniGainProfile

class FakeBackend:
	def start_stream(self, **kwargs):
		self.callback = kwargs["callback"]

	def stop_stream(self):
		pass

backend = FakeBackend()
reader = AirspyMiniReader(
	sample_rate_hz=3_000_000,
	center_frequency_hz=433_920_000,
	gain=MiniGainProfile(lna_gain=10, mixer_gain=5, vga_gain=8),
	backend=backend,
	buffer=CircularIQBuffer(capacity_samples=4096),
)
reader.start()

iq = np.random.standard_normal((1024, 2)).astype(np.float32)
backend.callback(iq)
captured = reader.read(1024)

# Use the airspy_rx-style linearity presets instead of per-stage values
linearity_gain = MiniGainProfile.linearity(10)
```

The reader enforces Airspy Mini-friendly configuration: only 3 MSPS high
accuracy mode is allowed and gains can be supplied either per-stage or via the
`MiniGainProfile.linearity()` / `.sensitivity()` helpers that mirror the
`airspy_rx` simplified presets. Swap in `FileIQReader` (imported from
`file_iq_reader`) with the same buffer if you need to play back recorded
float32 IQ captures rather than stream from hardware.
