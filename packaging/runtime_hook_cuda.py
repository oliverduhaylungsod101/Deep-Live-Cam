# PyInstaller runtime hook: ensure CUDA provider when launched as run.exe
import os
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# If frozen and no execution-provider was passed, default to CUDA (matches run-cuda.bat).
if getattr(sys, "frozen", False):
    args = sys.argv[1:]
    if not any(a.startswith("--execution-provider") for a in args):
        sys.argv[1:1] = ["--execution-provider", "cuda"]
