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

# Log directory: log/yyyy-MM-dd/
LOG_DATE="$(date +%Y-%m-%d)"
LOG_DIR="${SCRIPT_DIR}/log/${LOG_DATE}"
mkdir -p "${LOG_DIR}"

# Log file: run.sh-{stage}-HHMMSS.log
STAGE="${1:-unknown}"
LOG_TIME="$(date +%H%M%S)"
LOG_FILE="${LOG_DIR}/run.sh-${STAGE}-${LOG_TIME}.log"

source "${VENV_DIR}/bin/activate"

# Dashboard: launch Streamlit directly (not a pipeline stage)
if [ "${STAGE}" = "dashboard" ]; then
    exec streamlit run "${SCRIPT_DIR}/src/dashboard/app.py" --server.headless true "${@:2}"
fi

echo "Logging to: ${LOG_FILE}"
python "${SCRIPT_DIR}/src/run_pipeline.py" "$@" 2>&1 | tee "${LOG_FILE}"
