#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec .venv/bin/streamlit run dashboard/app.py --server.headless true "$@"
