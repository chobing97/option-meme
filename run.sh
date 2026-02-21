#!/usr/bin/env bash
# option-meme pipeline launcher
# Usage: ./run.sh <stage> [options]
# Run ./run.sh --help for details

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Error: Virtual environment not found at ${VENV_DIR}" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
python "${SCRIPT_DIR}/run_pipeline.py" "$@"
