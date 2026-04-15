#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: repo-local Python not found at $PYTHON_BIN"
  echo "Please create the virtual environment first."
  read -r -p "Press Enter to close..." _
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/live_overlay_demo.py" --source 0 "$@"
