#!/bin/bash
# Setup script for Airspy HF+ ZeroMQ Capture
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "================================================"
echo "Airspy HF+ ZeroMQ Capture - Environment Setup"
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
echo "To run the capture program:"
echo "  python airspy_zmq_capture.py --help"
echo ""
echo "Note: You still need to install the Airspy HF+ native library:"
echo "  macOS:  brew install airspyhf"
echo "  Linux:  sudo apt-get install libairspyhf-dev"
echo ""
