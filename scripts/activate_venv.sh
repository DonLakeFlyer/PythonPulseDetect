#!/usr/bin/env bash
# Helper script to activate the local Python virtual environment.
# Usage: source scripts/activate_venv.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

if [ ! -d "${VENV_PATH}" ]; then
    cat >&2 <<'EOF'
error: no virtual environment found at .venv
Run 'scripts/setup_venv.sh' first to create one.
EOF
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
