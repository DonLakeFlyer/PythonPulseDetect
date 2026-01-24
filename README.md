# Python Pulse Detect

Python-based signal processing system for capturing IQ data from an Airspy HF+ software-defined radio, decimating it, and detecting pulses via ZeroMQ streaming.

## Features

- **Real-time data capture** from Airspy HF+ device
- **Multi-stage decimation** (8x8x6 = 384x) with anti-aliasing filters
- **Real-time waterfall spectrogram display**
- **ZeroMQ streaming** for distributed processing
- **Sequence number tracking** to detect dropped messages
- **Flexible configuration** via JSON files or command-line arguments
- **Configurable RF parameters**:
  - Center frequency
  - Sample rate
  - LNA gain
  - AGC (Automatic Gain Control)
  - Attenuation

## Installation

### Prerequisites

1. **Install Airspy HF+ driver and library:**

   ```bash
   # macOS (using Homebrew)
   brew install airspyhf

   # Linux (Ubuntu/Debian)
   sudo apt-get install libairspyhf-dev
   ```

2. **Set up Python virtual environment and install dependencies:**

   ```bash
   # Run the setup script
   ./setup_venv.sh
   ```

   Or manually:
   ```bash
   # Create virtual environment
   python3 -m venv venv

   # Activate it
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Install Python bindings for Airspy HF+:**

   Since there's no official PyPI package, you'll need to use the ctypes-based wrapper or install from source:

   ```bash
   # Option 1: Install from GitHub (if available)
   pip install git+https://github.com/rsp-sermons/pyairspyhf.git

   # Option 2: The program uses ctypes to interface with the native library
   # Ensure libairspyhf is installed and accessible in your system library path
   ```

## Usage

### 1. Create Configuration File

Create a JSON configuration file (or use default settings):

```bash
python3 config.py create my_config.json
```

Example configuration (`my_config.json`):
```json
{
  "airspy": {
    "sample_rate_hz": 768000,
    "center_frequency_hz": 146000000,
    "lna_gain": 1,
    "agc": 0,
    "attenuation": 0
  },
  "zmq": {
    "port": 5555,
    "hwm": 10
  }
}
```

### 2. Start the Airspy HF+ Reader

```bash
python3 airspy_hf_reader.py -c my_config.json
```

This reads data from the Airspy HF+ at 768 kHz and publishes it on port 5555.

### 3. Start the Decimator (Optional)

```bash
python3 decimator.py -c my_config.json -o 5556
```

This decimates the 768 kHz stream by 384x (to 2 kHz) and publishes it on port 5556.

### 4. Monitor the Data Stream

```bash
# Monitor the raw 768 kHz stream
python3 zmq_receiver_example.py --port 5555

# Monitor the decimated 2 kHz stream
python3 zmq_receiver_example.py --port 5556

# Display real-time waterfall spectrogram
python3 waterfall_display.py --port 5556
```

## Configuration Options

### Airspy Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate_hz` | int | 768000 | Sample rate in Hz |
| `center_frequency_hz` | int | 146000000 | Center frequency in Hz |
| `lna_gain` | int (0/1) | 1 | LNA gain (0=off, 1=on) |
| `agc` | int (0/1) | 0 | AGC (0=off, 1=on) |
| `attenuation` | int | 0 | Attenuation in dB (0-48 in 6 dB steps) |

### ZeroMQ Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port` | int | 5555 | ZeroMQ publisher port |
| `hwm` | int | 10 | High water mark (queue size) |

### Supported Sample Rates

The Airspy HF+ supports these sample rates (Hz):
- 912000 (912 kHz)
- 768000 (768 kHz)
- 512000 (512 kHz)
- 456000 (456 kHz)
- 384000 (384 kHz)
- 256000 (256 kHz)
- 192000 (192 kHz)

### Frequency Range

- 9 kHz to 31 MHz (HF mode)
- 60 MHz to 260 MHz (VHF mode)

## Command-Line Options

### airspy_hf_reader.py

```bash
python3 airspy_hf_reader.py -c my_config.json [options]

Options:
  -c, --config FILE         JSON configuration file
  -f, --frequency HZ        Override center frequency
  -s, --sample-rate HZ      Override sample rate
  -p, --port PORT           Override ZeroMQ port
  --lna {0,1}              Override LNA gain
  --agc {0,1}              Override AGC
  --attenuation {0..48}    Override attenuation (6 dB steps)
```

### decimator.py

```bash
python3 decimator.py -c my_config.json [options]

Options:
  -c, --config FILE         JSON configuration file (required)
  -o, --output-port PORT    Output ZeroMQ port (default: 5556)
  --stages S1 S2 S3        Decimation stages (default: 8 8 6)
```

### zmq_receiver_example.py

```bash
python3 zmq_receiver_example.py [options]

Options:
  -c, --config FILE         JSON configuration file
  --host HOST              Host to connect to (default: localhost)
  --port PORT              Port to connect to
  --hwm HWM                High water mark
  --slow SECONDS           Delay per message (testing slow consumer)
```

### waterfall_display.py

```bash
python3 waterfall_display.py [options]

Options:
  --host HOST              Host to connect to (default: localhost)
  --port PORT              Port to connect to (default: 5556)
  --fft-size SIZE          FFT size (default: 512)
  --history FRAMES         Number of FFT frames to display (default: 200)
  --update-interval MS     Update interval in milliseconds (default: 50)
```

## Data Format

All ZeroMQ messages use this format:
1. **Sequence Number** (8 bytes, little-endian): Message sequence for drop detection
2. **IQ Data** (variable length): Complex64 numpy array (complex float32 samples)

### Example Receiver Code

```python
import zmq
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

last_seq = None
while True:
    message = socket.recv()

    # Extract sequence number
    seq = int.from_bytes(message[:8], byteorder='little')

    # Check for drops
    if last_seq is not None and seq != last_seq + 1:
        print(f"Dropped {seq - last_seq - 1} messages!")
    last_seq = seq

    # Extract IQ samples
    samples = np.frombuffer(message[8:], dtype=np.complex64)

    # Process samples here
    print(f"Seq {seq}: {len(samples)} samples")
```

## Use Cases

- **Pulse detection** and analysis
- **Signal monitoring** and recording
- **Real-time spectrum visualization**
- **Distributed signal processing**
- **Real-time spectrum analysis**
- **Radio astronomy**
- **Amateur radio applications**

## Architecture

```
┌─────────────────┐      ┌──────────────────────┐      ┌──────────────┐
│   Airspy HF+    │─────▶│  airspy_hf_reader    │─────▶│   ZeroMQ     │
│     Device      │ USB  │   (768 kHz)          │ TCP  │  (Port 5555) │
└─────────────────┘      └──────────────────────┘      └──────┬───────┘
                                                                │
                                                                ▼
                                                        ┌───────────────┐
                                                        │   decimator   │
                                                        │  (8x8x6=384x) │
                                                        │   (2 kHz)     │
                                                        └───────┬───────┘
                                                                │
                                                                ▼
                                                        ┌───────────────┐
                                                        │   ZeroMQ      │
                                                        │ (Port 5556)   │
                                                        └───────┬───────┘
                                                                │
                              ┌─────────────────────────────────┴─────────────┐
                              │                                               │
                              ▼                                               ▼
                      ┌────────────────┐                            ┌────────────────┐
                      │ waterfall      │                            │  Subscriber N  │
                      │ display        │                            │  (Analysis)    │
                      │ (Spectrogram)  │                            │                │
                      └────────────────┘                            └────────────────┘
```

## Troubleshooting

### Device Not Found

Ensure the Airspy HF+ is connected and drivers are installed:
```bash
# List USB devices (macOS)
system_profiler SPUSBDataType | grep -i airspy

# Test with airspyhf_info tool
airspyhf_info
```

### Permission Issues (Linux)

Add udev rules for Airspy:
```bash
sudo cp /usr/local/share/airspyhf/60-airspyhf.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### ZeroMQ Connection Issues

Check if the port is available:
```bash
# Check if port is in use
lsof -i :5555
```

### Dropped Messages

If you see dropped message warnings:
- Increase the `hwm` (high water mark) in configuration
- Check CPU usage - processing may be too slow
- Use `--slow` option on receiver to test backpressure

### Testing Slow Consumer

Simulate a slow consumer to test overflow detection:
```bash
python3 zmq_receiver_example.py --port 5555 --slow 0.5
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
