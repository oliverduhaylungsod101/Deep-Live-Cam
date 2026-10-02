#!/usr/bin/env bash
# Run from Git Bash / MSYS — use this instead of run-cuda.bat syntax.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT/ffmpeg/bin:$PATH"
export DLC_GUI_LAUNCH=1
exec "$ROOT/venv/Scripts/python.exe" "$ROOT/run.py" --execution-provider cuda
