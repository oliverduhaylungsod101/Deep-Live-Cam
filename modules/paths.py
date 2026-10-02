"""Shared path constants for the Deep-Live-Cam project.

Works for both source runs and PyInstaller frozen deploys (run.exe).
"""

from __future__ import annotations

import os
import sys


def get_app_root() -> str:
    """Directory that contains run.exe / run.py, models/, ffmpeg/, locales/."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT_DIR = get_app_root()
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FFMPEG_BIN_DIR = os.path.join(ROOT_DIR, "ffmpeg", "bin")
LOCALES_DIR = os.path.join(ROOT_DIR, "locales")
BROWSER_PROFILE_DIR = os.path.join(ROOT_DIR, "browser_profile")
