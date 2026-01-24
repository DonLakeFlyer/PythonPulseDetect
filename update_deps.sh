#!/bin/bash
# Update dependencies in the virtual environment

set -e

if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found."
    echo "Run './setup_venv.sh' first to create it."
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Updating dependencies from requirements.txt..."
pip install --upgrade -r requirements.txt

echo ""
echo "Current installed packages:"
pip list

echo ""
echo "Update complete!"
