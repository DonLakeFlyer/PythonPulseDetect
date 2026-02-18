#!/bin/bash
# Setup script for Airspy HF+ ZeroMQ Capture
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "================================================"
echo "Python Pulse Detect - Environment Setup"
echo "================================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python: $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment."
    fi
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo "Ensuring git submodules are initialized..."
git submodule update --init --recursive

echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the full capture pipeline:"
echo "  python run_capture_pipeline.py --config capture_config.json --label positive --training-profile quick"
echo ""
echo "To train the 1D CNN after capture/dataset build:"
echo "  python train_cnn1d.py --data-dir artifacts/processed/cnn_simple"
echo ""
echo "Note: PyTorch is included in requirements for training/inference and can take longer to install."
echo ""
echo "Note: You still need to install the Airspy HF+ native library:"
echo "  macOS:  brew install airspyhf"
echo "  Linux:  sudo apt-get install libairspyhf-dev"
echo ""
