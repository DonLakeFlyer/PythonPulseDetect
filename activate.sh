#!/bin/bash
# Quick activation script for the virtual environment

if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found."
    echo "Run './setup_venv.sh' first to create it."
    exit 1
fi

source venv/bin/activate
echo "Virtual environment activated!"
echo "Python: $(which python)"
echo "Run 'deactivate' to exit the virtual environment."
