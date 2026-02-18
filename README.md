# Python Pulse Detect

Python-based signal processing system for capturing IQ data from an Airspy HF+ software-defined radio, decimating it, and detecting pulses via ZeroMQ streaming.

## Features

- **Real-time data capture** from Airspy HF+ device
- **Multi-stage decimation** (8x8x6 = 384x) with anti-aliasing filters
- **Real-time waterfall spectrogram display**
- **ZeroMQ streaming** for distributed processing
- **Sequence number tracking** to detect dropped messages
- **Labeled training data capture** from decimator IQ stream for ML datasets
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
  # Initialize submodules (required for AirspyTools)
  git submodule update --init --recursive

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

Pipeline convenience scripts:
- `./run_capture_pipeline.sh` wraps `run_capture_pipeline.py`
- `./run_infer_pipeline.sh` wraps `run_infer_pipeline.py`

### 1. Create Configuration File

Create a JSON configuration file (or use default settings):

```bash
python3 submodules/AirspyTools/airspy_tools_config.py create capture_config.json
```

Example configuration (`capture_config.json`):
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
    "reader_output_port": 5555,
    "hwm": 10
  },
  "decimator": {
    "output_port": 5556,
    "output_sample_rate_hz": 2000
  },
  "tag": {
    "pulse_width_ms": 15,
    "pulse_period_ms": 2000
  }
}
```

For complete defaults, validation ranges, and field meanings, see `submodules/AirspyTools/airspy_tools_config.py` (`AirspyToolsConfig.DEFAULTS` and `_validate`).

### 2. Capture Training Data

Start here after creating config. This runs reader + decimator + capture together and auto-shuts down when capture targets are reached.

```bash
# Quick positive capture
./run_capture_pipeline.sh \
  --label positive \
  --training-profile quick

# Quick negative capture
./run_capture_pipeline.sh \
  --label negative \
  --training-profile quick
```

Optional stream logs:

```bash
./run_capture_pipeline.sh \
  --label positive \
  --training-profile quick \
  --reader-stream-logs \
  --decimator-stream-logs \
  --capture-stream-logs
```

### 3. Prepare Runtime Config for Inference based on Training Data

After training a model, run calibration and generate runtime config in one command:

```bash
# Existing processed dataset flow (all artifacts defaults)
./run_prepare_infer_pipeline.sh

# Optional: custom criterion while keeping default model/sessions paths
./run_prepare_infer_pipeline.sh \
  --config capture_config.json \
  --criterion target_recall

# Optional: reuse an existing model and skip training
./run_prepare_infer_pipeline.sh --skip-train

# Show script options/help
./run_prepare_infer_pipeline.sh --help
```

`--criterion` controls how the threshold is selected during calibration:
- `f1`: best balance of precision/recall.
- `target_recall`: highest threshold that still meets recall target (fewer misses).
- `target_precision`: lowest threshold that still meets precision target (fewer false alarms).

By default, this helper uses `artifacts/models/cnn_simple/best_model.pt` for `--model` and `artifacts/training-sessions` for `--training-sessions-dir`; dataset auto-build runs only when session folders are found under `artifacts/training-sessions`.

This helper runs `train_cnn1d.py` each time before threshold calibration. Use `--skip-train` to reuse an existing model checkpoint at `--model`.

`--skip-train` is useful when you already trust the current model and want faster reruns for calibration/runtime-config work (for example, trying different `--criterion` settings) without waiting for retraining. Avoid `--skip-train` after adding/changing training sessions, because you generally want a freshly trained model that reflects the new data.

This helper does **not** start live inference; it prepares `threshold_report.json` and `infer_runtime.json` for `infer_cnn1d.py`.

### 4. Run the Live Inference Pipeline

Use this to launch reader + decimator + live inference together:

```bash
./run_infer_pipeline.sh

# Optional: enable reader/decimator/infer stream logs
./run_infer_pipeline.sh \
  --reader-stream-logs \
  --decimator-stream-logs \
  --infer-stream-logs
```

## Detailed Usage

Use this section when you want to run individual apps manually.

### A. Start the Airspy HF+ Reader

```bash
python3 submodules/AirspyTools/airspy_hf_reader.py -c capture_config.json

# Optional: periodic reader progress logs
python3 submodules/AirspyTools/airspy_hf_reader.py -c capture_config.json --stream-logs
```

### B. Start the Decimator

```bash
python3 submodules/AirspyTools/decimator.py -c capture_config.json -o 5556

# Optional: periodic decimator progress logs
python3 submodules/AirspyTools/decimator.py -c capture_config.json -o 5556 --stream-logs
```

This decimates the 768 kHz stream by 384x (to 2 kHz) and publishes it on port 5556.

### C. Monitor the Data Stream

```bash
# Monitor the raw 768 kHz stream
python3 zmq_receiver_example.py --port 5555

# Monitor the decimated 2 kHz stream
python3 zmq_receiver_example.py --port 5556

# Display real-time waterfall spectrogram
python3 waterfall_display.py --port 5556
```

### D. Capture Training Data Windows

Use the decimator output stream (default 2 kHz) to save labeled IQ windows for CNN training.

```bash
# Positive windows (collar transmitting): trigger-focused capture
python3 capture_training_data.py \
  --config capture_config.json \
  --label positive \
  --training-profile quick \
  --window-ms 64

# Optional: periodic capture progress logs
python3 capture_training_data.py \
  --config capture_config.json \
  --label positive \
  --training-profile quick \
  --window-ms 64 \
  --stream-logs

# Negative windows (no collar/interference background): sequential capture
python3 capture_training_data.py \
  --config capture_config.json \
  --label negative \
  --training-profile quick \
  --window-ms 64
```

Each capture session creates a timestamped folder with `.npy` windows and `manifest.jsonl` metadata.

Profiles:
- `standard`: balanced default capture (`positive` target ~400 windows, `negative` target ~800 windows; window-count-driven)
- `quick`: short captures for fast testing (`positive` target ~200 windows, `negative` target ~400 windows; window-count-driven)
- `long`: longer captures for real training (`positive` target ~1200 windows, `negative` target ~2400 windows; window-count-driven)

### E. Build 1D CNN Train/Validation Arrays

```bash
# Auto-discover all sessions under a parent folder
python3 build_cnn_dataset.py \
  --training-sessions-dir artifacts/training-sessions \
  --val-frac 0.2

# Or explicitly pass session folders
python3 build_cnn_dataset.py \
  --sessions artifacts/training-sessions/positive_20260217_120000_trigger artifacts/training-sessions/negative_20260217_121000_sequential \
  --val-frac 0.2
```

Output files:
- `X_train.npy`, `y_train.npy`
- `X_val.npy`, `y_val.npy`
- `dataset_meta.json`

### F. Train the 1D CNN

```bash
python3 train_cnn1d.py \
  --epochs 30 \
  --batch-size 256
```

Training outputs:
- `best_model.pt`
- `train_history.json`
- `train_summary.json`

### G. Calibrate Detection Threshold

Pick a threshold from validation data instead of manually guessing.

Criterion options:
- `f1`: chooses the threshold with the best F1 score (balanced precision/recall).
- `target_recall`: chooses the highest threshold that still meets your recall target (`--target-recall`, default `0.95`). Use this when missing real pulses is costly.
- `target_precision`: chooses the lowest threshold that still meets your precision target (`--target-precision`, default `0.90`). Use this when false alarms are costly.

```bash
python3 calibrate_threshold.py \
  --criterion f1
```

### H. Run Live CNN Inference

```bash
python3 infer_cnn1d.py \
  --config capture_config.json \
  --window-ms 64 \
  --hop-ms 16 \
  --threshold 0.6
```

Detection lines are printed as:
- `DETECT t=<seconds> prob=<smoothed_prob> (raw=<raw_prob>, snr_db=<snr_db>, seq=<message_seq>)`

Or use calibrated runtime config:

```bash
# Create runtime inference config with recommended threshold
python3 make_infer_runtime_config.py \
  --config capture_config.json \
  --threshold-report artifacts/models/cnn_simple/threshold_report.json \
  --out artifacts/models/cnn_simple/infer_runtime.json

# Run inference directly from runtime config
python3 infer_cnn1d.py \
  --config capture_config.json \
  --runtime-config artifacts/models/cnn_simple/infer_runtime.json
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
| `reader_output_port` | int | 5555 | Airspy reader ZeroMQ publisher port |
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

### submodules/AirspyTools/airspy_hf_reader.py

```bash
python3 submodules/AirspyTools/airspy_hf_reader.py -c capture_config.json [options]

Options:
  -c, --config FILE         JSON configuration file
  -f, --frequency HZ        Override center frequency
  -s, --sample-rate HZ      Override sample rate
  -p, --port PORT           Override ZeroMQ port
  --lna {0,1}              Override LNA gain
  --agc {0,1}              Override AGC
  --attenuation {0..48}    Override attenuation (6 dB steps)
  --stream-logs            Enable periodic status logs while streaming (default: off)
```

### submodules/AirspyTools/decimator.py

```bash
python3 submodules/AirspyTools/decimator.py -c capture_config.json [options]

Options:
  -c, --config FILE         JSON configuration file (default: capture_config.json)
  -o, --output-port PORT    Output ZeroMQ port (default: 5556)
  --stages S1 S2 S3        Decimation stages (default: 8 8 6)
  --stream-logs            Enable periodic decimator status logs while streaming (default: off)
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

### capture_training_data.py

```bash
python3 capture_training_data.py [options]

Options:
  -c, --config FILE         JSON configuration file (default: capture_config.json)
  --label {positive,negative}
                            Label for saved windows (required)
  --training-profile {standard,quick,long}
                            Capture profile preset (default: standard)
  --output-dir DIR          Base output directory (default: artifacts/training-sessions)
  --host HOST               Decimator host (default: localhost)
  --port PORT               Decimator output port (default: from config)
  --window-ms MS            Window length in milliseconds (default: 64)
  --hop-ms MS               Sequential hop in milliseconds (default: 64)
  --stream-logs             Enable periodic capture progress logs while streaming (default: off)
  --duration SEC            Capture duration seconds, 0=until Ctrl+C (default: 120)
  --max-windows N           Stop after saving N windows
  --trigger-z Z             Enable trigger mode using envelope z-score threshold
  --trigger-refractory-ms MS
                            Minimum trigger spacing in milliseconds (default: 80)
```

### run_capture_pipeline.py

```bash
./run_capture_pipeline.sh [options]

Options:
  -c, --config FILE         JSON config file (default: capture_config.json)
  --label {positive,negative}
                            Capture label (required)
  --output-dir DIR          Base output directory (default: artifacts/training-sessions)
  --training-profile {standard,quick,long}
                            Capture profile for capture stage (default: standard)
  --decimator-stages S1 [S2 ...]
                            Decimator stages (default: 8 8 6)
  --reader-stream-logs      Enable reader periodic stream logs (default: off)
  --decimator-stream-logs   Enable decimator periodic stream logs (default: off)
  --capture-stream-logs     Enable capture periodic stream logs (default: off)
  --reader-warmup-sec SEC   Delay after starting reader (default: 1.5)
  --decimator-warmup-sec SEC
                            Delay after starting decimator (default: 1.5)
  --shutdown-timeout-sec SEC
                            Graceful shutdown timeout per process (default: 8)
```

### run_infer_pipeline.py

```bash
./run_infer_pipeline.sh [options]

Options:
  -c, --config FILE         JSON config file (default: capture_config.json)
  --runtime-config FILE     Runtime JSON for infer_cnn1d.py (default: artifacts/models/cnn_simple/infer_runtime.json)
  --model FILE              Optional model override passed to infer_cnn1d.py
  --host HOST               Optional decimator host override
  --port PORT               Optional decimator port override
  --hwm HWM                 Optional ZeroMQ receive HWM override
  --window-ms MS            Inference window length override
  --hop-ms MS               Sliding hop length override
  --threshold FLOAT         Detection threshold override
  --smooth-windows N        Moving-average window override
  --cooldown-ms MS          Detection cooldown override
  --device {auto,cpu,mps,cuda}
                            Inference device override
  --dump-config FILE        Optional output path for resolved runtime config
  --decimator-stages S1 [S2 ...]
                            Decimator stages (default: 8 8 6)
  --reader-stream-logs      Enable reader periodic stream logs (default: off)
  --decimator-stream-logs   Enable decimator periodic stream logs (default: off)
  --infer-stream-logs       Enable periodic inference stream logs (default: off)
  --reader-warmup-sec SEC   Delay after starting reader (default: 1.5)
  --decimator-warmup-sec SEC
                            Delay after starting decimator (default: 1.5)
  --shutdown-timeout-sec SEC
                            Graceful shutdown timeout per process (default: 8)
```

### build_cnn_dataset.py

```bash
python3 build_cnn_dataset.py [options]

Options:
  --sessions DIR [DIR ...]  One or more capture session directories
  --training-sessions-dir DIR
                            Parent directory to auto-discover session folders recursively
  --output-dir DIR          Output directory for dataset files (default: artifacts/processed/cnn_simple)
  --val-frac F              Validation fraction [0,1) (default: 0.2)
  --max-per-class N         Optional cap per class before split
  --seed N                  Random seed (default: 42)
```

### train_cnn1d.py

```bash
python3 train_cnn1d.py [options]

Options:
  --data-dir DIR            Directory with X/y train/val .npy files (default: artifacts/processed/cnn_simple)
  --output-dir DIR          Output directory for model/metrics (default: artifacts/models/cnn_simple)
  --epochs N                Number of epochs (default: 30)
  --batch-size N            Batch size (default: 256)
  --lr FLOAT                Learning rate (default: 1e-3)
  --weight-decay FLOAT      AdamW weight decay (default: 1e-4)
  --dropout FLOAT           Model dropout (default: 0.2)
  --device {auto,cpu,mps,cuda}
                            Training device selection (default: auto)
  --patience N              Early-stop patience on val_f1 (default: 8)
```

### infer_cnn1d.py

```bash
python3 infer_cnn1d.py [options]

Options:
  -c, --config FILE         JSON configuration file (default: capture_config.json)
  --runtime-config FILE     Optional runtime JSON with model + inference params
  --model FILE              Path to trained best_model.pt (default: artifacts/models/cnn_simple/best_model.pt)
  --host HOST               Decimator host (default: localhost)
  --port PORT               Decimator output port override
  --window-ms MS            Inference window length (default: 64)
  --hop-ms MS               Sliding hop length (default: 16)
  --threshold FLOAT         Detection threshold (default: 0.6)
  --smooth-windows N        Moving-average smoothing length (default: 3)
  --cooldown-ms MS          Minimum spacing between detections (default: 150)
  --stream-logs             Enable periodic inference status logs while streaming (default: off)
  --device {auto,cpu,mps,cuda}
                            Inference device selection (default: auto)
```

### calibrate_threshold.py

```bash
python3 calibrate_threshold.py [options]

Options:
  --model FILE              Path to trained best_model.pt (default: artifacts/models/cnn_simple/best_model.pt)
  --data-dir DIR            Directory with X_val.npy and y_val.npy (default: artifacts/processed/cnn_simple)
  --output-dir DIR          Output directory for threshold_report.json (default: artifacts/models/cnn_simple)
  --num-thresholds N        Threshold sweep count in [0,1] (default: 201)
  --criterion {f1,target_recall,target_precision}
                            Threshold selection strategy (default: f1)
  --target-recall FLOAT     Recall target for target_recall mode (default: 0.95)
  --target-precision FLOAT  Precision target for target_precision mode (default: 0.90)
  --device {auto,cpu,mps,cuda}
                            Inference device selection (default: auto)
```

### make_infer_runtime_config.py

```bash
python3 make_infer_runtime_config.py [options]

Options:
  --threshold-report FILE   threshold_report.json from calibrate_threshold.py (required)
  --model FILE              Path to trained best_model.pt (default: artifacts/models/cnn_simple/best_model.pt)
  --out FILE                Output runtime config JSON (required)
  -c, --config FILE         Base capture JSON config (default: capture_config.json)
  --host HOST               Decimator host (default: localhost)
  --port PORT               Decimator output port override
  --hwm HWM                 ZeroMQ receive HWM override
  --window-ms MS            Inference window length (default: 64)
  --hop-ms MS               Sliding hop length (default: 16)
  --smooth-windows N        Moving-average smoothing length (default: 3)
  --cooldown-ms MS          Minimum spacing between detections (default: 150)
  --device {auto,cpu,mps,cuda}
                            Inference device selection (default: auto)
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
