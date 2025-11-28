#!/usr/bin/env bash
# Bootstraps a local Python virtual environment for PythonPulseDetect.
# Usage: bash scripts/setup_venv.sh [python-executable]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${1:-${PYTHON_BIN:-python3}}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "error: python executable '${PYTHON_BIN}' not found" >&2
    exit 1
fi

if [ ! -d "${VENV_PATH}" ]; then
    echo "Creating virtual environment at ${VENV_PATH}" >&2
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
else
    echo "Reusing existing virtual environment at ${VENV_PATH}" >&2
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo "Virtual environment ready. Activate it with:"
echo "  source .venv/bin/activate"
