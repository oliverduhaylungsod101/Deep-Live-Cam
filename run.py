#!/usr/bin/env python3

import os
import sys


def _hide_gui_launch_console() -> None:
    """Detach any console when started from launch.vbs / run-cuda.bat."""
    if sys.platform != "win32" or getattr(sys, "frozen", False):
        return
    if os.environ.get("DLC_GUI_LAUNCH") != "1":
        return
    try:
        import ctypes

        devnull = open(os.devnull, "w", encoding="utf-8", errors="replace")
        sys.stdout = devnull
        sys.stderr = devnull
        ctypes.windll.kernel32.FreeConsole()
    except Exception:
        pass


_hide_gui_launch_console()

# Skip albumentations' PyPI version ping — it can hang startup for 30–90s when offline.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
# Quieter TensorFlow logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Add bundled ffmpeg and project root to PATH so ffmpeg/ffprobe are found
import sys as _sys

if getattr(_sys, "frozen", False):
    project_root = os.path.dirname(os.path.abspath(_sys.executable))
else:
    project_root = os.path.dirname(os.path.abspath(__file__))
_ffmpeg_bin = os.path.join(project_root, "ffmpeg", "bin")
for _path_dir in (_ffmpeg_bin, project_root):
    if os.path.isdir(_path_dir):
        os.environ["PATH"] = _path_dir + os.pathsep + os.environ.get("PATH", "")

# On Windows, register NVIDIA CUDA DLL directories so onnxruntime-gpu can
# find cuDNN/cublas. Python 3.8+ ignores PATH for extension-module native deps —
# os.add_dll_directory() is required. Also keep PATH for child processes/ffmpeg.
if sys.platform == "win32":
    _candidate_roots = []
    if getattr(sys, "frozen", False):
        # PyInstaller onedir: nvidia/* lives under _internal/
        _meipass = getattr(sys, "_MEIPASS", None)
        if _meipass:
            _candidate_roots.append(_meipass)
        _candidate_roots.append(os.path.join(project_root, "_internal"))
    _candidate_roots.extend(
        [
            os.path.join(sys.prefix, "Lib", "site-packages"),
            os.path.join(project_root, "venv", "Lib", "site-packages"),
        ]
    )
    for _sp in _candidate_roots:
        if not os.path.isdir(_sp):
            continue
        _candidate_dirs = []
        _torch_lib = os.path.join(_sp, "torch", "lib")
        if os.path.isdir(_torch_lib):
            _candidate_dirs.append(_torch_lib)
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _pkg in os.listdir(_nvidia_dir):
                _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
                if os.path.isdir(_bin_dir):
                    _candidate_dirs.append(_bin_dir)
        for _d in _candidate_dirs:
            os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(_d)
            except (OSError, AttributeError):
                pass

# On Linux, pre-load NVIDIA shared libraries (cuDNN, cuBLAS, nvrtc...) shipped
# inside the venv via pip wheels (nvidia-cudnn-cu12, etc.). LD_LIBRARY_PATH
# cannot be set after Python starts, so we use ctypes.CDLL with RTLD_GLOBAL
# instead. This makes symbols available to onnxruntime when it dlopens its
# CUDA provider.
if sys.platform.startswith("linux"):
    import ctypes
    import glob
    _py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _site_packages_candidates = [
        os.path.join(project_root, "venv", "lib", _py_lib, "site-packages"),
        os.path.join(sys.prefix, "lib", _py_lib, "site-packages"),
    ]
    for _sp in _site_packages_candidates:
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if not os.path.isdir(_nvidia_dir):
            continue
        for _pkg in os.listdir(_nvidia_dir):
            _lib_dir = os.path.join(_nvidia_dir, _pkg, "lib")
            if not os.path.isdir(_lib_dir):
                continue
            # Also expose the directory to child processes, without
            # duplicating an entry that is already present.
            _ldp = os.environ.get("LD_LIBRARY_PATH", "")
            if _lib_dir not in _ldp.split(os.pathsep):
                os.environ["LD_LIBRARY_PATH"] = (
                    _lib_dir + (os.pathsep + _ldp if _ldp else "")
                )
            for _so in sorted(glob.glob(os.path.join(_lib_dir, "lib*.so*"))):
                try:
                    ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        break

# Preload CUDA/cuDNN DLLs before any onnxruntime session is created.
try:
    import onnxruntime as _ort
    if hasattr(_ort, "preload_dlls"):
        _ort.preload_dlls()
except ImportError:
    pass

print("Starting Swapface Live…", flush=True)

from modules import platform_info
platform_info.print_banner()

print("Loading UI…", flush=True)

from modules import core

print("Opening window…", flush=True)

if __name__ == '__main__':
    core.run()
