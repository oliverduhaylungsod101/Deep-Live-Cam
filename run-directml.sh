#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT/ffmpeg/bin:$PATH"
export DLC_GUI_LAUNCH=1
exec "$ROOT/venv/Scripts/python.exe" "$ROOT/run.py" --execution-provider dml
