"""Send live swapped frames to a system virtual camera (OBS on Windows)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_CAMERA = None


def start(width: int, height: int, fps: int) -> str:
    """Open the virtual camera device. Returns the device name."""
    global _CAMERA
    stop()
    import pyvirtualcam

    _CAMERA = pyvirtualcam.Camera(
        width=int(width),
        height=int(height),
        fps=max(1, int(round(fps))),
    )
    return _CAMERA.device


def send_bgr(frame: np.ndarray) -> None:
    """Publish a BGR frame without blocking the Qt/UI thread.

    Capture is already paced upstream, so sleep_until_next_frame() here only
    stalls rendering and causes processed frames to queue up and be discarded.
    """
    if _CAMERA is None:
        return
    h, w = frame.shape[:2]
    if w != _CAMERA.width or h != _CAMERA.height:
        frame = cv2.resize(frame, (_CAMERA.width, _CAMERA.height))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _CAMERA.send(rgb)


def stop() -> None:
    global _CAMERA
    if _CAMERA is not None:
        _CAMERA.close()
        _CAMERA = None


def is_active() -> bool:
    return _CAMERA is not None
