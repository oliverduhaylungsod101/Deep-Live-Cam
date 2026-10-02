"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import base64
import os
import platform
import queue
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
    QRect,
    QBuffer,
    QIODevice,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QFontMetrics,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules import face_library
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.hotkeys import GlobalHotkeyManager, HotkeyCaptureButton, default_hotkeys
from modules.paths import ROOT_DIR
from modules.processors.frame.core import get_frame_processors_modules
from modules.model_registry import (
    available_enhancer_choices,
    available_swap_choices,
    enhancer_fp_key_for_label,
    enhancer_label_for_fp_key,
)
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import virtual_camera
from modules import imread_unicode
from modules.video_capture import VideoCapturer

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

ROOT_HEIGHT = 1000
ROOT_WIDTH = 1400
SIDEBAR_WIDTH = 68
RIGHT_PANEL_WIDTH = 420
AI_BROWSER_WIDTH = 440
CAPTION_PANEL_MIN_WIDTH = 220
CAPTION_PANEL_MAX_WIDTH = 480
AI_PANEL_MIN_WIDTH = 400
AI_PANEL_WIDTH_RATIO = 0.64
CONTROLS_PANEL_MAX_HEIGHT = 200
IMAGE_OUTPUT_RESOLUTIONS = (
    ("1280 × 720", 1280, 720),
    ("1920 × 1080", 1920, 1080),
    ("640 × 480", 640, 480),
)
PROFILE_CARD_W = 260
PROFILE_CARD_H = 92
CARDS_PER_PAGE_FALLBACK = 2
AVATAR_SIZE = 52
FACE_GRID_COLS = 3
LIVE_PREVIEW_HEIGHT = 220

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 200


# ─── Swapface premium dark theme (Voicebox-inspired) ─────────────────────

QSS = """
QMainWindow, QDialog {
    background-color: #0c0c0c;
    color: #f0f0f0;
}
QWidget {
    color: #f0f0f0;
    font-family: "Segoe UI Variable Text", "Segoe UI", Arial, sans-serif;
    font-size: 10pt;
}
QWidget#contentRoot, QScrollArea, QScrollArea > QWidget > QWidget {
    background: transparent;
}

QFrame#sidebar {
    background-color: #080808;
    border-right: 1px solid #1e1e1e;
}
QPushButton#navBtn {
    background: transparent;
    border: none;
    border-radius: 12px;
    color: #666666;
    font-size: 18pt;
    padding: 10px;
    min-width: 44px;
    min-height: 44px;
}
QPushButton#navBtn:hover {
    background-color: #1a1a1a;
    color: #aaaaaa;
}
QPushButton#navBtn[active="true"] {
    background-color: rgba(201, 168, 76, 0.18);
    color: #c9a84c;
}
QLabel#sidebarLogo {
    color: #c9a84c;
    font-size: 22pt;
    font-weight: 700;
}

/* Segmented tab bar (below-section navigation) */
QPushButton#tabBtn {
    background-color: #161616;
    border: 1px solid #252525;
    border-radius: 10px;
    color: #999999;
    font-size: 10pt;
    font-weight: 700;
    padding: 8px 20px;
    min-height: 18px;
}
QPushButton#tabBtn:hover {
    background-color: #1e1e1e;
    color: #d0d0d0;
}
QPushButton#tabBtn[active="true"] {
    background-color: rgba(201, 168, 76, 0.16);
    border: 1px solid #c9a84c;
    color: #e0bc62;
}
QPushButton#collapseBtn {
    background-color: #161616;
    border: 1px solid #252525;
    border-radius: 8px;
    color: #c9a84c;
    font-size: 11pt;
    font-weight: 700;
    padding: 0;
    min-width: 30px;
    max-width: 30px;
    min-height: 30px;
    max-height: 30px;
}
QPushButton#collapseBtn:hover {
    background-color: #1e1e1e;
    border-color: #c9a84c;
}

/* Card pager arrows */
QPushButton#pagerArrow {
    background-color: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 22px;
    color: #c9a84c;
    font-size: 15pt;
    font-weight: 700;
    min-width: 44px;
    max-width: 44px;
    min-height: 44px;
    max-height: 44px;
    padding: 0;
}
QPushButton#pagerArrow:hover {
    background-color: #242424;
    border-color: #c9a84c;
    color: #e0bc62;
}
QPushButton#pagerArrow:disabled {
    background-color: #131313;
    border-color: #1e1e1e;
    color: #3a3a3a;
}
QLabel#pagerLabel {
    color: #777777;
    font-size: 9pt;
    font-weight: 600;
}
QLabel#pagerEmpty {
    color: #555555;
    font-size: 10pt;
}

QLabel#pageTitle {
    font-size: 22pt;
    font-weight: 700;
    color: #ffffff;
}
QLabel#pageSubtitle {
    font-size: 9.5pt;
    color: #666666;
}

QPushButton {
    background-color: #c9a84c;
    color: #0c0c0c;
    border: none;
    border-radius: 10px;
    padding: 9px 20px;
    font-weight: 700;
    min-height: 20px;
}
QPushButton:hover  { background-color: #e0bc62; }
QPushButton:pressed{ background-color: #a88938; }
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #555555;
}
QPushButton#secondary {
    background-color: transparent;
    color: #d0d0d0;
    border: 1px solid #333333;
}
QPushButton#secondary:hover {
    background-color: #1a1a1a;
    border-color: #444444;
}
QPushButton#danger {
    background-color: transparent;
    color: #cc6666;
    border: 1px solid #442222;
}
QPushButton#danger:hover {
    background-color: #331111;
    color: #ff8888;
}
QPushButton#iconBtn {
    background: transparent;
    border: none;
    border-radius: 8px;
    color: #666666;
    padding: 4px 8px;
    font-size: 11pt;
    min-width: 28px;
    min-height: 28px;
}
QPushButton#iconBtn:hover {
    background-color: #252525;
    color: #c9a84c;
}
QPushButton#liveBtn {
    background-color: #c9a84c;
    color: #0c0c0c;
    border-radius: 12px;
    padding: 10px 28px;
    font-size: 11pt;
    font-weight: 800;
    min-width: 100px;
}

QFrame#profileCard {
    background-color: #181818;
    border: 1px solid #252525;
    border-radius: 14px;
}
QFrame#profileCard:hover {
    background-color: #1e1e1e;
    border-color: #333333;
}
QFrame#profileCard[selected="true"] {
    border: 2px solid #c9a84c;
    background-color: rgba(201, 168, 76, 0.08);
}
QFrame#addProfileCard {
    background-color: #121212;
    border: 1px dashed #333333;
    border-radius: 14px;
}
QFrame#addProfileCard:hover {
    border-color: #c9a84c;
    background-color: rgba(201, 168, 76, 0.05);
}
QLabel#avatar {
    background-color: #252525;
    border: 2px solid #333333;
    border-radius: 26px;
    color: #555555;
}
QLabel#profileName {
    font-size: 11pt;
    font-weight: 700;
    color: #f0f0f0;
}
QLabel#profileMeta {
    font-size: 9pt;
    color: #666666;
}
QLabel#profileTag {
    background-color: #252525;
    color: #888888;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 8pt;
}

QFrame#previewPlaceholder {
    background-color: #0a0a0a;
    border: 1px solid #252525;
    border-radius: 16px;
}
QFrame#rightPanel {
    background-color: #080808;
    border-left: 1px solid #1e1e1e;
}
QFrame#activeCard {
    background-color: #181818;
    border: 1px solid #252525;
    border-radius: 14px;
}
QLabel#panelTitle {
    font-size: 9pt;
    font-weight: 700;
    color: #666666;
    letter-spacing: 1px;
}
QLabel#activePreview {
    background-color: #121212;
    border: 1px solid #252525;
    border-radius: 12px;
    color: #444444;
}
QFrame#statusPill {
    background-color: rgba(201, 168, 76, 0.12);
    border: 1px solid rgba(201, 168, 76, 0.35);
    border-radius: 14px;
}
QLabel#statusPillText {
    color: #c9a84c;
    font-size: 9pt;
    font-weight: 700;
}
QLabel#heroDot { color: #6ccb5f; font-size: 10pt; }

QFrame#floatingDock {
    background-color: #141414;
    border: 1px solid #252525;
    border-radius: 16px;
}
QLabel#dockLabel {
    font-size: 8.5pt;
    color: #666666;
    font-weight: 600;
}

/* Embedded live preview (spectrum-bar position) */
QFrame#livePreviewPanel {
    background-color: #0a0a0a;
    border: 1px solid #252525;
    border-radius: 16px;
}
QLabel#livePreviewVideo {
    background-color: #0a0a0a;
    border-radius: 12px;
    color: #444444;
}
QLabel#livePreviewHint {
    color: #555555;
    font-size: 10pt;
}
QLabel#livePreviewBadge {
    color: #c9a84c;
    font-size: 9pt;
    font-weight: 700;
}
QProgressBar#loadProgress {
    background-color: #1a1a1a;
    border: 1px solid #252525;
    border-radius: 8px;
    height: 22px;
    text-align: center;
    color: #f0f0f0;
    font-size: 9pt;
    font-weight: 600;
}
QProgressBar#loadProgress::chunk {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0, stop:0 #a88938, stop:1 #e0bc62
    );
    border-radius: 7px;
}
QLabel#loadStage {
    color: #c9a84c;
    font-size: 9.5pt;
    font-weight: 600;
}

/* Live captions — transparent overlay text (no fill), over the preview */
QLabel#captionOverlay {
    background: transparent;
    background-color: transparent;
    border: none;
    color: #ffffff;
    font-size: 15pt;
    font-weight: 650;
    padding: 8px 16px;
    qproperty-alignment: AlignLeft;
}
QTextEdit#aiResponseOverlay {
    background: transparent;
    background-color: rgba(0, 0, 0, 0);
    border: none;
    color: #ffffff;
    font-size: 14pt;
    font-weight: 600;
    padding: 10px 16px;
    selection-background-color: #c9a84c;
    selection-color: #141414;
}
QTextEdit#aiResponseOverlay[waiting="true"] {
    color: #c9a84c;
}
QFrame#captionAiDivider {
    background-color: rgba(201, 168, 76, 190);
    border: none;
}
QTextEdit#captionHistory {
    background-color: #141414;
    color: #e8e8e8;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 8px;
    font-size: 9.5pt;
    selection-background-color: #c9a84c;
    selection-color: #141414;
}
QFrame#aiBrowserFrame {
    background-color: #121212;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
}
QPushButton#aiBrowserNav {
    background-color: #1e1e1e;
    color: #e8e8e8;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 6px 10px;
    font-weight: 600;
}
QPushButton#aiBrowserNav:hover {
    border-color: #c9a84c;
    color: #c9a84c;
}
QPushButton#aiShowStrip {
    background-color: #1a1a1a;
    color: #c9a84c;
    border: 1px solid #333333;
    border-radius: 10px;
    font-weight: 700;
    font-size: 9pt;
    padding: 8px 4px;
}
QPushButton#aiShowStrip:hover {
    background-color: #c9a84c;
    color: #141414;
    border-color: #c9a84c;
}
QPushButton#chatgptSubmit {
    background-color: #c9a84c;
    color: #141414;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 700;
}
QPushButton#chatgptSubmit:hover { background-color: #d4b65a; }

QGroupBox {
    background-color: #181818;
    border: 1px solid #252525;
    border-radius: 14px;
    margin-top: 14px;
    padding: 18px 14px 14px 14px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: #888888;
    font-size: 9pt;
    font-weight: 700;
}
QComboBox {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 8px 12px;
    min-height: 20px;
    color: #f0f0f0;
}
QComboBox:hover { border-color: #c9a84c; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background-color: #1e1e1e;
    color: #f0f0f0;
    selection-background-color: rgba(201, 168, 76, 0.25);
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 4px;
}
QCheckBox { spacing: 10px; color: #d0d0d0; }
QCheckBox::indicator {
    width: 40px; height: 20px;
    border-radius: 10px;
    background-color: #2a2a2a;
    border: 1px solid #444444;
}
QCheckBox::indicator:checked {
    background-color: #c9a84c;
    border: 1px solid #c9a84c;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #333333;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #c9a84c;
    width: 18px; height: 18px;
    margin: -7px 0;
    border-radius: 9px;
    border: 3px solid #141414;
}
QSlider::sub-page:horizontal {
    background: #c9a84c;
    border-radius: 2px;
}
QLabel#statusLabel {
    color: #888888;
    font-size: 9pt;
}
QScrollBar:vertical {
    width: 8px;
    background: transparent;
}
QScrollBar::handle:vertical {
    background: #333333;
    border-radius: 4px;
    min-height: 32px;
}
QScrollBar::handle:vertical:hover { background: #c9a84c; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QMenu {
    background-color: #1e1e1e;
    color: #f0f0f0;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 4px 0;
}
QMenu::item {
    background-color: transparent;
    color: #f0f0f0;
    padding: 8px 28px 8px 20px;
}
QMenu::item:selected {
    background-color: #2a2a2a;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background-color: #333333;
    margin: 4px 10px;
}
QMenu#trayMenu {
    background-color: #1e1e1e;
    color: #f0f0f0;
}
QMenu#trayMenu::item {
    color: #f0f0f0;
    min-height: 24px;
}
QToolTip {
    background-color: #1e1e1e;
    color: #f0f0f0;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 6px 10px;
}
QInputDialog, QMessageBox { background-color: #141414; }
"""


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None


# ─── stealth / screen-share exclusion (Windows) ───────────────────────────

# WDA_EXCLUDEFROMCAPTURE (0x11): window is omitted from screen share & screenshots
# on Windows 10 2004+ / Windows 11. Falls back to WDA_MONITOR (black box) on older builds.
_WDA_NONE = 0x00
_WDA_MONITOR = 0x01
_WDA_EXCLUDEFROMCAPTURE = 0x11


def set_window_stealth(widget: QWidget, enabled: bool) -> bool:
    """Exclude (or include) a top-level Qt window from Windows screen capture.

    Only uses WDA_EXCLUDEFROMCAPTURE. Never falls back to WDA_MONITOR — that
    affinity blanks the window on several NVIDIA + Windows 11 setups, which
    looks like \"the UI never opened\".
    """
    if platform.system() != "Windows":
        return False
    try:
        import ctypes

        hwnd = int(widget.winId())
        if not hwnd:
            return False
        user32 = ctypes.windll.user32
        affinity = _WDA_EXCLUDEFROMCAPTURE if enabled else _WDA_NONE
        ok = bool(user32.SetWindowDisplayAffinity(hwnd, affinity))
        if not ok:
            # Restore normal visibility if exclude-from-capture is unsupported.
            user32.SetWindowDisplayAffinity(hwnd, _WDA_NONE)
            print(
                "[stealth] Exclude-from-capture unsupported on this Windows build; "
                "window stays visible.",
                flush=True,
            )
        return ok
    except Exception as exc:
        print(f"[stealth] SetWindowDisplayAffinity failed: {exc}", flush=True)
        return False


def apply_stealth_to_all_windows(enabled: Optional[bool] = None) -> None:
    """Re-apply stealth affinity to every top-level app window."""
    if enabled is None:
        enabled = bool(getattr(modules.globals, "stealth_mode", False))
    targets: List[QWidget] = []
    if _MAIN is not None:
        targets.append(_MAIN)
    if _PREVIEW is not None:
        targets.append(_PREVIEW)
    if _MAPPER is not None:
        targets.append(_MAPPER)
    if _LIVE_MAPPER is not None:
        targets.append(_LIVE_MAPPER)
    if _APP is not None:
        for w in _APP.topLevelWidgets():
            if w not in targets and w.isWindow() and w.isVisible():
                targets.append(w)
    for w in targets:
        try:
            set_window_stealth(w, enabled)
        except Exception:
            pass


def stealth_effective_for(widget: Optional[QWidget] = None) -> bool:
    """Whether capture-exclusion should apply (stealth toggle or ghost overlay)."""
    if bool(getattr(modules.globals, "stealth_mode", False)):
        return True
    if widget is _MAIN and bool(getattr(modules.globals, "always_on_top", False)):
        return True
    return False


def set_win_taskbar_visible(widget: QWidget, visible: bool) -> bool:
    """Show or hide a top-level window from the Windows taskbar."""
    if platform.system() != "Windows":
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = int(widget.winId())
        if not hwnd:
            return False
        gwl_exstyle = -20
        ws_ex_toolwindow = 0x00000080
        ws_ex_appwindow = 0x00040000
        style = user32.GetWindowLongW(hwnd, gwl_exstyle)
        if visible:
            style = (style & ~ws_ex_toolwindow) | ws_ex_appwindow
        else:
            style = (style | ws_ex_toolwindow) & ~ws_ex_appwindow
        user32.SetWindowLongW(hwnd, gwl_exstyle, style)
        swp_flags = 0x0002 | 0x0001 | 0x0004 | 0x0020
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, swp_flags)
        return True
    except Exception:
        return False


def tray_app_hides_taskbar() -> bool:
    """Whether tray mode keeps app windows off the Windows taskbar."""
    if not QSystemTrayIcon.isSystemTrayAvailable():
        return False
    return bool(getattr(modules.globals, "minimize_to_tray", True))


def apply_tray_taskbar_hiding(widget: Optional[QWidget]) -> None:
    """Apply or restore taskbar visibility for a top-level window."""
    if widget is None or not widget.isWindow():
        return
    set_win_taskbar_visible(widget, not tray_app_hides_taskbar())


def apply_tray_taskbar_hiding_to_all_windows() -> None:
    """Re-apply taskbar policy to every visible top-level app window."""
    for widget in (_MAIN, _PREVIEW, _MAPPER, _LIVE_MAPPER):
        if widget is not None and widget.isVisible():
            apply_tray_taskbar_hiding(widget)


def set_win_click_through(widget: QWidget, enabled: bool) -> bool:
    """Make a top-level window ignore mouse hits (Windows WS_EX_TRANSPARENT)."""
    if platform.system() != "Windows":
        return False
    try:
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_TRANSPARENT = 0x00000020
        SWP_NOSIZE = 0x0001
        SWP_NOMOVE = 0x0002
        SWP_NOZORDER = 0x0004
        SWP_FRAMECHANGED = 0x0020
        SWP_NOACTIVATE = 0x0010

        hwnd = int(widget.winId())
        if not hwnd:
            return False
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if enabled:
            new_style = style | WS_EX_TRANSPARENT
        else:
            new_style = style & ~WS_EX_TRANSPARENT
        if new_style == style:
            return True
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
        user32.SetWindowPos(
            hwnd,
            0,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOACTIVATE,
        )
        return True
    except Exception as exc:
        print(f"[click-through] SetWindowLong failed: {exc}", flush=True)
        return False


# ─── glass effects ─────────────────────────────────────────────────────────


def _apply_glass_shadow(
    widget: QWidget,
    blur: int = 18,
    color: Tuple[int, int, int, int] = (0, 0, 0, 90),
    y_offset: int = 2,
) -> None:
    """Subtle Fluent-style elevation shadow for cards/panels."""
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setColor(QColor(*color))
    effect.setOffset(0, y_offset)
    widget.setGraphicsEffect(effect)


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def letterbox_image_to_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale image to fit inside width×height and center on a black canvas."""
    fitted = fit_image_to_size(image, width, height)
    fh, fw = fitted.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = max(0, (width - fw) // 2)
    y = max(0, (height - fh) // 2)
    canvas[y : y + fh, x : x + fw] = fitted
    return canvas


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "swap_model_choice": modules.globals.swap_model_choice,
        "show_fps": modules.globals.show_fps,
        "stealth_mode": modules.globals.stealth_mode,
        "live_captions": modules.globals.live_captions,
        "ai_browser_url": getattr(
            modules.globals, "ai_browser_url", "https://chatgpt.com"
        ),
        "ai_browser_visible": getattr(modules.globals, "ai_browser_visible", True),
        "window_opacity": getattr(modules.globals, "window_opacity", 1.0),
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "always_on_top": getattr(modules.globals, "always_on_top", False),
        "code_test_mode": getattr(modules.globals, "code_test_mode", False),
        "controls_collapsed": getattr(modules.globals, "controls_collapsed", False),
        "minimize_to_tray": getattr(modules.globals, "minimize_to_tray", True),
        "image_output_path": getattr(modules.globals, "image_output_path", None),
        "image_output_width": getattr(modules.globals, "image_output_width", 1280),
        "image_output_height": getattr(modules.globals, "image_output_height", 720),
        "hotkeys": dict(getattr(modules.globals, "hotkeys", default_hotkeys())),
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        modules.globals.fp_ui = state.get("fp_ui", {"face_enhancer": False})
        modules.globals.swap_model_choice = state.get("swap_model_choice", "auto")
        modules.globals.show_fps = state.get("show_fps", False)
        modules.globals.stealth_mode = state.get("stealth_mode", False)
        modules.globals.live_captions = state.get("live_captions", True)
        modules.globals.ai_browser_url = state.get(
            "ai_browser_url", "https://chatgpt.com"
        )
        modules.globals.ai_browser_visible = state.get("ai_browser_visible", True)
        try:
            modules.globals.window_opacity = float(
                state.get("window_opacity", 1.0)
            )
        except (TypeError, ValueError):
            modules.globals.window_opacity = 1.0
        modules.globals.window_opacity = max(
            0.30, min(1.0, modules.globals.window_opacity)
        )
        # Mouth mask always starts disabled (slider at 0) on launch,
        # regardless of the persisted value — enable it explicitly each session.
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
        modules.globals.always_on_top = bool(state.get("always_on_top", False))
        modules.globals.code_test_mode = bool(state.get("code_test_mode", False))
        modules.globals.controls_collapsed = bool(
            state.get("controls_collapsed", False)
        )
        modules.globals.minimize_to_tray = bool(
            state.get("minimize_to_tray", True)
        )
        saved_image_output = state.get("image_output_path")
        modules.globals.image_output_path = (
            str(saved_image_output)
            if saved_image_output and os.path.isfile(str(saved_image_output))
            else None
        )
        try:
            modules.globals.image_output_width = int(
                state.get("image_output_width", 1280)
            )
            modules.globals.image_output_height = int(
                state.get("image_output_height", 720)
            )
        except (TypeError, ValueError):
            modules.globals.image_output_width = 1280
            modules.globals.image_output_height = 720
        saved_hotkeys = state.get("hotkeys")
        if isinstance(saved_hotkeys, dict):
            base = default_hotkeys()
            base.update({k: str(v) for k, v in saved_hotkeys.items() if v is not None})
            # Migrate broken Win+Arrow combos (Ctrl+Meta+Down fails on Windows).
            migrated = False
            for action, bad in (
                ("ai_chat_scroll_up", "Ctrl+Meta+Up"),
                ("ai_chat_scroll_down", "Ctrl+Meta+Down"),
            ):
                cur = (base.get(action) or "").strip()
                if not cur or cur.lower() == bad.lower():
                    base[action] = default_hotkeys()[action]
                    migrated = True
            modules.globals.hotkeys = base
            if migrated:
                try:
                    # Persist fixed defaults so Settings shows the working shortcuts.
                    with open("switch_states.json", "r") as f:
                        st = json.load(f)
                    st["hotkeys"] = base
                    with open("switch_states.json", "w") as f:
                        json.dump(st, f)
                except (OSError, json.JSONDecodeError, TypeError):
                    pass
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


_VIRTUAL_CAMERA_NAME_HINTS = (
    "virtual",
    "obs",
    "swapface",
    "usb 3.0 pc camera",
    "unity capture",
    "unity video",
)


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            indices: List[int] = []
            names: List[str] = []
            for i, name in enumerate(devices or []):
                lower = str(name).lower()
                if any(hint in lower for hint in _VIRTUAL_CAMERA_NAME_HINTS):
                    continue
                indices.append(i)
                names.append(name)
            if names:
                return indices, names
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices = []
    names = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)


class ProfileCard(QFrame):
    """Horizontal profile card — Voicebox-style face identity tile."""

    selected = Signal(dict)
    deleted = Signal(dict)

    def __init__(self, entry: dict, kind: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.entry = entry
        self.kind = kind
        self.setObjectName("profileCard")
        self.setFixedSize(PROFILE_CARD_W, PROFILE_CARD_H)
        self.setProperty("selected", False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 12, 14, 12)
        row.setSpacing(12)

        self._avatar = QLabel()
        self._avatar.setObjectName("avatar")
        self._avatar.setFixedSize(AVATAR_SIZE, AVATAR_SIZE)
        self._avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        path = entry.get("path")
        if path and os.path.isfile(path):
            try:
                pix = render_image_preview(path, (AVATAR_SIZE, AVATAR_SIZE))
                self._avatar.setPixmap(
                    pix.scaled(
                        AVATAR_SIZE, AVATAR_SIZE,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            except Exception:
                self._avatar.setText("?")
        else:
            self._avatar.setText("?")
        row.addWidget(self._avatar)

        info = QVBoxLayout()
        info.setSpacing(2)
        name = QLabel(entry.get("name", ""))
        name.setObjectName("profileName")
        info.addWidget(name)
        meta = QLabel(_("Source face") if kind == "source" else _("Target face"))
        meta.setObjectName("profileMeta")
        info.addWidget(meta)
        tag = QLabel(kind[:3].upper())
        tag.setObjectName("profileTag")
        tag.setFixedWidth(32)
        info.addWidget(tag)
        info.addStretch(1)
        row.addLayout(info, 1)

        actions = QVBoxLayout()
        actions.setSpacing(4)
        btn_use = QPushButton("\u2713")
        btn_use.setObjectName("iconBtn")
        btn_use.setToolTip(_("Use this face"))
        btn_use.clicked.connect(lambda: self.selected.emit(self.entry))
        btn_del = QPushButton("\u2715")
        btn_del.setObjectName("iconBtn")
        btn_del.setToolTip(_("Delete"))
        btn_del.clicked.connect(lambda: self.deleted.emit(self.entry))
        actions.addWidget(btn_use)
        actions.addWidget(btn_del)
        row.addLayout(actions)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.entry)
        super().mousePressEvent(event)

    def set_selected(self, active: bool) -> None:
        self.setProperty("selected", active)
        self.style().unpolish(self)
        self.style().polish(self)


class AddProfileCard(QFrame):
    """Dashed add-new tile in the profile grid."""

    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("addProfileCard")
        self.setFixedSize(PROFILE_CARD_W, PROFILE_CARD_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        plus = QLabel("+")
        plus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plus.setFixedSize(AVATAR_SIZE, AVATAR_SIZE)
        plus.setStyleSheet(
            "font-size: 28pt; font-weight: 300; color: #c9a84c; background: transparent;"
        )
        layout.addWidget(plus)

        col = QVBoxLayout()
        col.setSpacing(2)
        title = QLabel(_("Add Face"))
        title.setObjectName("profileName")
        col.addWidget(title)
        hint = QLabel(_("Import from image"))
        hint.setObjectName("profileMeta")
        col.addWidget(hint)
        col.addStretch(1)
        layout.addLayout(col, 1)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _CardPager(QWidget):
    """Container that re-renders its page of cards when resized."""

    resized = Signal()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()


class CaptionOverlay(QLabel):
    """Stable multi-line live captions over the preview.

    Finished lines are never rewritten. New words only append to the last line;
    when that line is full, a new line is created. When MAX_LINES is exceeded,
    the oldest line is dropped. Width is 3/4 of the preview, left-aligned.
    """

    MAX_LINES = 4
    FIXED_WIDTH = 450

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("captionOverlay")
        self.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.setWordWrap(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(14)
        shadow.setColor(QColor(0, 0, 0, 240))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        self._lines: list[str] = []
        self._source_words: list[str] = []
        # How many leading source words have scrolled off the visible lines.
        self._word_offset = 0
        self._line_width_px = 400
        self._host_width = 800
        # Next non-empty caption replaces tracking (new utterance after pause).
        self._expect_new_utterance = False
        self.hide()

    def clear_lines(self) -> None:
        self._lines = []
        self._source_words = []
        self._word_offset = 0
        self._expect_new_utterance = False
        self.setText("")
        self.hide()

    def begin_new_utterance(self) -> None:
        """Allow the next caption through even if it is shorter than sticky text."""
        self._expect_new_utterance = True
        # Keep on-screen text until the next caption arrives — only reset tracking.
        self._source_words = []
        self._word_offset = 0

    def show_caption(self, text: str) -> None:
        text = (text or "").strip()
        # Empty updates happen when speech pauses — keep the last captions visible.
        if not text:
            return

        lower = text.lower()
        if lower.startswith("listening") or lower.startswith("caption"):
            # Don't wipe real captions with a status tip if we already have text.
            if self._lines and self._source_words:
                return
            self._source_words = []
            self._word_offset = 0
            self._lines = [text]
            self._expect_new_utterance = False
            self._render()
            return

        words = text.split()
        if self._expect_new_utterance:
            self._expect_new_utterance = False
            self._lines = []
            self._word_offset = 0
            for word in words:
                self._append_word(word)
            self._source_words = words
            self._render()
            return

        old = self._source_words

        # Speech stopped / partial cleared: incoming text is only a prefix of
        # what is already on screen — do not shrink or hide captions.
        if old and len(words) <= len(old) and old[: len(words)] == words:
            return

        # Pure append — freeze full lines; only the last line may grow.
        if old and len(words) >= len(old) and words[: len(old)] == old:
            for word in words[len(old) :]:
                self._append_word(word)
            self._source_words = words
            self._render()
            return

        if not old:
            self._lines = []
            self._word_offset = 0
            for word in words:
                self._append_word(word)
            self._source_words = words
            self._render()
            return

        # Worker dropped leading words, or Vosk revised a hypothesis.
        overlap = 0
        max_check = min(len(old), len(words))
        for k in range(max_check, 0, -1):
            if old[-k:] == words[:k]:
                overlap = k
                break

        if overlap > 0 and len(old) > overlap:
            # Leading words removed from the stream — scroll them off display.
            removed = old[: len(old) - overlap]
            self._drop_leading_words(len(removed))
            for word in words[overlap:]:
                self._append_word(word)
            self._source_words = words
            self._render()
            return

        # Shared prefix from the start of both strings.
        common = 0
        limit = min(len(old), len(words))
        while common < limit and old[common] == words[common]:
            common += 1

        # New utterance / big revision with little shared start — replace instead
        # of keeping sticky old text (this was freezing live captions).
        if common < 2 and (len(words) < len(old) or common == 0):
            self._lines = []
            self._word_offset = 0
            for word in words:
                self._append_word(word)
            self._source_words = words
            self._render()
            return

        # In-place revision that would drop words: keep the longer on-screen text.
        if len(words) < len(old):
            return

        # In-place revision: keep lines fully covered by the common prefix.
        self._rebuild_from_word_index(words, common)
        self._source_words = words
        self._render()

    def _text_width(self, text: str) -> int:
        return int(QFontMetrics(self.font()).horizontalAdvance(text))

    def _append_word(self, word: str) -> None:
        """Append one word. Never modifies already-full (non-last) lines."""
        max_w = max(80, self._line_width_px)
        if not self._lines:
            self._lines = [word]
            return

        last = self._lines[-1]
        candidate = f"{last} {word}"
        if self._text_width(candidate) <= max_w:
            self._lines[-1] = candidate
            return

        # Last line is full — freeze it and start a new line.
        self._lines.append(word)
        while len(self._lines) > self.MAX_LINES:
            removed = self._lines.pop(0)
            self._word_offset += len(removed.split())

    def _drop_leading_words(self, count: int) -> None:
        """Remove the first `count` laid-out words from the visible lines."""
        if count <= 0:
            return
        remaining = count
        new_lines: list[str] = []
        for line in self._lines:
            parts = line.split()
            if remaining >= len(parts):
                remaining -= len(parts)
                self._word_offset += len(parts)
                continue
            if remaining > 0:
                parts = parts[remaining:]
                self._word_offset += remaining
                remaining = 0
            if parts:
                new_lines.append(" ".join(parts))
        self._lines = new_lines

    def _rebuild_from_word_index(self, words: list[str], keep_words: int) -> None:
        """Keep visible lines fully covered by words[offset:keep_words]; refill rest."""
        local_keep = keep_words - self._word_offset
        if local_keep < 0:
            self._word_offset = max(0, keep_words)
            self._lines = []
            for word in words[self._word_offset :]:
                self._append_word(word)
            return

        kept: list[str] = []
        counted = 0
        for line in self._lines:
            wc = len(line.split())
            if counted + wc <= local_keep:
                kept.append(line)
                counted += wc
            else:
                break

        self._lines = kept
        for word in words[self._word_offset + counted :]:
            self._append_word(word)

    def _render(self) -> None:
        body = "\n".join(self._lines)
        self.setText(body)
        if body:
            self.show()
            self.raise_()
            self._sync_geometry(reflow=False)
        else:
            self.hide()

    def _sync_geometry(self, reflow: bool = False) -> None:
        host = self.parentWidget()
        if host is None:
            return
        self._host_width = max(1, host.width())
        width = min(self.FIXED_WIDTH, max(200, self._host_width - 24))
        new_line_w = max(80, width - 40)
        width_changed = abs(new_line_w - self._line_width_px) > 2
        self._line_width_px = new_line_w
        top = 10  # captions live in the top quarter of the preview
        x = max(0, (self._host_width - width) // 2)

        if reflow and width_changed and self._source_words:
            stored = self._source_words[self._word_offset :]
            self._lines = []
            # Keep offset; rebuild only the still-visible tail.
            saved_offset = self._word_offset
            for word in stored:
                self._append_word(word)
            # _append_word may advance offset when dropping lines — OK.
            # Restore source words unchanged.
            self._source_words = self._source_words  # no-op clarity
            self.setText("\n".join(self._lines))

        self.setFixedWidth(width)
        fm = QFontMetrics(self.font())
        line_h = max(fm.height() + 4, 22)
        # Stay within the top 1/4 of the preview (captions band).
        band_h = max(48, self.parentWidget().height() // 4) if self.parentWidget() else 120
        height = min(max(line_h, line_h * max(1, len(self._lines)) + 16), band_h - 8)
        self.setGeometry(x, top, width, max(line_h, height))
        self.raise_()

    def reposition(self, host: QWidget) -> None:
        """Recalc width/position; reflow only when the line width actually changes."""
        self._sync_geometry(reflow=True)


class AIResponseOverlay(QTextEdit):
    """Caption-styled AI reply text in the bottom 3/4 of the preview."""

    FIXED_WIDTH = 450

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("aiResponseOverlay")
        self.setReadOnly(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(14)
        shadow.setColor(QColor(0, 0, 0, 240))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        self._last_text = ""
        self._waiting = False
        self.hide()

    def set_waiting(self, message: str) -> None:
        """Show a waiting alarm without treating it as a scanned reply."""
        message = (message or "").strip() or "Waiting for ChatGPT response…"
        self._waiting = True
        self.setProperty("waiting", "true")
        self.style().unpolish(self)
        self.style().polish(self)
        self.setPlainText(message)
        self.show()
        self.raise_()
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)

    def clear_waiting(self) -> None:
        if not self._waiting:
            return
        self._waiting = False
        self.setProperty("waiting", "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_response(self, text: str) -> None:
        text = (text or "").strip()
        if text == self._last_text and not self._waiting:
            return
        self.clear_waiting()
        self._last_text = text
        if not text:
            self.hide()
            return
        self._set_grammar_rich_text(text)
        self.show()
        self.raise_()
        # Keep the start of the reply visible so the user can read from the top
        # (streaming updates previously jumped the view to the bottom).
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)
        bar = self.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.minimum())

    def _set_grammar_rich_text(self, text: str) -> None:
        """Render the reply as one flowing sentence with colored chunks + '·' marks."""
        import html as html_lib

        try:
            from modules.grammar_parse import parse_grammar

            segments = parse_grammar(text)
        except Exception:
            self.setPlainText(text)
            return

        if not segments:
            self.setPlainText(text)
            return

        color_hex = {
            "main_clause": "#ffffff",
            "dependent_clause": "#9ec9ff",
            "relative_clause": "#e8c56b",
            "prepositional_phrase": "#8fd4a8",
            "noun_phrase": "#c9b0ff",
            "verb_phrase": "#ffb38a",
            "participial_phrase": "#f0a0c0",
            "infinitive_phrase": "#7edce8",
        }
        sep = '<span style="color:#c9a84c;font-weight:700;">&nbsp;·&nbsp;</span>'
        parts: list[str] = ['<div style="line-height:1.5;">']
        need_sep = False
        for seg in segments:
            label = seg.get("label", "main_clause")
            piece = (seg.get("text") or "").strip()
            if label == "paragraph_break":
                parts.append("<br/><br/>")
                need_sep = False
                continue
            if label == "sentence_break":
                # New sentence on its own line; no separator across sentences.
                parts.append("<br/>")
                need_sep = False
                continue
            if not piece:
                continue
            if need_sep:
                parts.append(sep)
            color = color_hex.get(label, "#ffffff")
            parts.append(
                f'<span style="color:{color};">{html_lib.escape(piece)}</span>'
            )
            need_sep = True
        parts.append("</div>")

        self.setHtml("".join(parts) if need_sep or len(parts) > 2 else html_lib.escape(text))

    def reposition(self, host: QWidget) -> None:
        host_w = max(1, host.width())
        host_h = max(1, host.height())
        width = min(self.FIXED_WIDTH, max(200, host_w - 24))
        x = max(0, (host_w - width) // 2)
        top = host_h // 4
        height = max(80, host_h - top - 12)
        self.setGeometry(x, top, width, height)
        self.raise_()


# JS: pull the latest assistant reply from ChatGPT / Claude / Gemini DOMs.
# Preserves paragraph / line breaks (does not squash all whitespace).
_AI_EXTRACT_JS = r"""
(() => {
  const clean = (t) => (t || "")
    .replace(/\r\n?/g, "\n")
    .replace(/[ \t\f\v]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  const pickLast = (nodes) => {
    const texts = [];
    nodes.forEach((el) => {
      const t = clean(el.innerText || el.textContent || "");
      if (t && t.length > 8) texts.push(t);
    });
    return texts.length ? texts[texts.length - 1] : "";
  };

  // ChatGPT
  let nodes = document.querySelectorAll('[data-message-author-role="assistant"]');
  let text = pickLast(nodes);
  if (text) return text;

  // Claude
  nodes = document.querySelectorAll(
    '[data-is-streaming="true"], .font-claude-message, [class*="AssistantMessage"], div[class*="assistant"] .standard-markdown'
  );
  text = pickLast(nodes);
  if (text) return text;

  nodes = document.querySelectorAll("div.font-claude-response, .prose");
  text = pickLast(nodes);
  if (text) return text;

  // Gemini
  nodes = document.querySelectorAll(
    "model-response, .model-response-text, message-content.model-response-text, .response-container"
  );
  text = pickLast(nodes);
  if (text) return text;

  nodes = document.querySelectorAll("[data-message-author-role], [aria-live='polite']");
  text = pickLast(nodes);
  if (text) return text;

  return "";
})()
"""

# Install a MutationObserver that keeps window.__DLC_AI_REPLY fresh.
_AI_SCAN_INSTALL_JS = r"""
(() => {
  const clean = (t) => (t || "")
    .replace(/\r\n?/g, "\n")
    .replace(/[ \t\f\v]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  const pickLast = (nodes) => {
    const texts = [];
    nodes.forEach((el) => {
      const t = clean(el.innerText || el.textContent || "");
      if (t && t.length > 8) texts.push(t);
    });
    return texts.length ? texts[texts.length - 1] : "";
  };
  const extract = () => {
    let nodes = document.querySelectorAll('[data-message-author-role="assistant"]');
    let text = pickLast(nodes);
    if (text) return text;
    nodes = document.querySelectorAll(
      '[data-is-streaming="true"], .font-claude-message, [class*="AssistantMessage"], div[class*="assistant"] .standard-markdown'
    );
    text = pickLast(nodes);
    if (text) return text;
    nodes = document.querySelectorAll("div.font-claude-response, .prose");
    text = pickLast(nodes);
    if (text) return text;
    nodes = document.querySelectorAll(
      "model-response, .model-response-text, message-content.model-response-text, .response-container"
    );
    text = pickLast(nodes);
    if (text) return text;
    nodes = document.querySelectorAll("[data-message-author-role], [aria-live='polite']");
    return pickLast(nodes);
  };
  const update = () => {
    try {
      const t = extract();
      if (t) window.__DLC_AI_REPLY = t;
    } catch (e) {}
  };
  if (!window.__DLC_AI_SCAN_INSTALLED) {
    window.__DLC_AI_SCAN_INSTALLED = true;
    window.__DLC_AI_REPLY = window.__DLC_AI_REPLY || "";
    update();
    const obs = new MutationObserver(() => {
      if (window.__DLC_AI_SCAN_TIMER) return;
      window.__DLC_AI_SCAN_TIMER = setTimeout(() => {
        window.__DLC_AI_SCAN_TIMER = null;
        update();
      }, 40);
    });
    obs.observe(document.documentElement || document.body, {
      childList: true,
      subtree: true,
      characterData: true
    });
    window.__DLC_AI_SCAN_OBS = obs;
  } else {
    update();
  }
  return window.__DLC_AI_REPLY || "";
})()
"""

_AI_SCAN_READ_JS = r"""
(() => {
  try {
    return window.__DLC_AI_REPLY || "";
  } catch (e) {
    return "";
  }
})()
"""

# Back-compat alias used by older call sites.
_AI_SCAN_JS = _AI_EXTRACT_JS

_AI_FOCUS_INPUT_JS = r"""
(() => {
  const selectors = [
    '#prompt-textarea',
    'div.ProseMirror[contenteditable="true"]',
    'textarea[data-testid]',
    'rich-textarea .ql-editor',
    '.ProseMirror[contenteditable="true"]',
    '[contenteditable="true"][role="textbox"]',
    'textarea',
    '[contenteditable="true"]'
  ];
  for (const selector of selectors) {
    const matches = Array.from(document.querySelectorAll(selector));
    const input = matches.find((el) => {
      const r = el.getBoundingClientRect();
      return r.width > 0 && r.height > 0 && !el.disabled;
    });
    if (input) {
      input.focus();
      input.click();
      return true;
    }
  }
  return false;
})()
"""

_AI_PASTE_IMAGE_JS = r"""
((b64, name, token) => {
  try {
    // One token per screenshot: the same capture can never attach twice,
    // but every new screenshot adds another image alongside existing ones.
    window.__DLC_ATTACHED_TOKENS = window.__DLC_ATTACHED_TOKENS || {};
    if (window.__DLC_ATTACHED_TOKENS[token]) {
      return ('skip:duplicate');
    }
    window.__DLC_ATTACHED_TOKENS[token] = true;

    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const file = new File([bytes], name, { type: 'image/png' });
    const dt = new DataTransfer();
    dt.items.add(file);

    const root =
      document.querySelector('form[data-type="unified-composer"]') ||
      document.querySelector('form') ||
      document.querySelector('footer') ||
      document.body;
    let inputs = Array.from(
      (root || document).querySelectorAll('input[type="file"]')
    );
    if (!inputs.length) {
      inputs = Array.from(document.querySelectorAll('input[type="file"]'));
    }
    const preferred = inputs.filter((el) => {
      const accept = (el.getAttribute('accept') || '').toLowerCase();
      return !accept || accept.includes('image') || accept.includes('*') || accept.includes('png');
    });
    const fileInput = (preferred.length ? preferred : inputs)[0];
    if (!fileInput) {
      delete window.__DLC_ATTACHED_TOKENS[token];
      return ('err:file input not found');
    }

    try { fileInput.value = ''; } catch (e) {}
    try {
      fileInput.files = dt.files;
    } catch (e) {
      delete window.__DLC_ATTACHED_TOKENS[token];
      return ('err:cannot assign file');
    }

    // Exactly one change event — input+change starts two uploads on ChatGPT.
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    return ('ok:file');
  } catch (e) {
    return ('err:' + (e && e.message ? e.message : e));
  }
})
"""

# Ignore repeat screenshot / paste triggers for this many milliseconds.
_SCREENSHOT_DEBOUNCE_MS = 800

# Injected into the AI browser so scroll hotkeys work without focusing the app.
# Do NOT require overflow:auto — ChatGPT/Claude often scroll containers that
# still report overflow:hidden/visible while scrollHeight > clientHeight.
_AI_SCROLL_HELPER_JS = r"""
(() => {
  if (window.__DLC_SCROLL__) return 'exists';
  window.__DLC_SCROLL__ = function(direction) {
    // direction: +1 down, -1 up. Distance = half the target pane's visible height.
    const dir = Number(direction) || 0;
    if (!dir) return 'zero';
    const canScroll = (el) => {
      if (!el) return false;
      try {
        const sh = el.scrollHeight || 0;
        const ch = el.clientHeight || 0;
        return (sh - ch) > 16;
      } catch (e) {
        return false;
      }
    };
    const deltaFor = (el) => {
      const ch = Math.max(el.clientHeight || 0, 1);
      // Quarter-page step (half of the previous half-page jump).
      return Math.round(ch * 0.25) * (dir > 0 ? 1 : -1);
    };
    const apply = (el) => {
      if (!canScroll(el)) return false;
      const delta = deltaFor(el);
      const before = el.scrollTop;
      el.scrollTop = before + delta;
      if (el.scrollTop === before) {
        try { el.scrollBy(0, delta); } catch (e) {}
      }
      return el.scrollTop !== before;
    };
    // 1) Walk from elements under typical chat viewport points.
    const pts = [
      [window.innerWidth * 0.50, window.innerHeight * 0.40],
      [window.innerWidth * 0.50, window.innerHeight * 0.55],
      [window.innerWidth * 0.58, window.innerHeight * 0.45],
      [window.innerWidth * 0.42, window.innerHeight * 0.45],
    ];
    for (const [x, y] of pts) {
      let n = null;
      try { n = document.elementFromPoint(x, y); } catch (e) {}
      while (n) {
        // Only the first scrollable ancestor — avoid stacking deltas.
        if (apply(n)) return 'point:' + Math.round(n.scrollTop);
        n = n.parentElement;
      }
    }
    // 2) Prefer largest visible scrollable region (chat pane).
    let best = null;
    let bestScore = 0;
    const consider = (el) => {
      if (!canScroll(el)) return;
      const area = Math.max(el.clientWidth || 0, 1) * Math.max(el.clientHeight || 0, 1);
      const room = (el.scrollHeight || 0) - (el.clientHeight || 0);
      const score = area * Math.min(room, 4000);
      if (score > bestScore) {
        best = el;
        bestScore = score;
      }
    };
    consider(document.scrollingElement);
    consider(document.documentElement);
    consider(document.body);
    try {
      document.querySelectorAll('div, main, section, article, [role="main"]').forEach(consider);
    } catch (e) {}
    if (best && apply(best)) return 'best:' + Math.round(best.scrollTop);
    // 3) Last resort: nudge every scrollable node.
    let n = 0;
    try {
      document.querySelectorAll('*').forEach((el) => { if (apply(el)) n += 1; });
    } catch (e) {}
    if (n > 0) return 'all:' + n;
    try {
      window.scrollBy(0, Math.round(window.innerHeight * 0.25) * (dir > 0 ? 1 : -1));
    } catch (e) {}
    return 'window';
  };
  return 'installed';
})()
"""


class MainWindow(QMainWindow):
    _PAGE_SOURCE = 0
    _PAGE_TARGET = 1
    _PAGE_IMAGE_OUTPUT = 2
    _PAGE_SETTINGS = 3

    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        modules.globals.map_faces = False
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb
        self._force_quit = False
        self._tray: Optional[QSystemTrayIcon] = None
        self._face_cards: dict[str, list[ProfileCard]] = {"source": [], "target": []}
        self._current_kind = "source"
        # Card-pager state (per kind).
        self._page_index: dict[str, int] = {"source": 0, "target": 0}
        self._face_entries: dict[str, list[dict]] = {"source": [], "target": []}
        self._cards_row: dict[str, QHBoxLayout] = {}
        self._cards_container: dict[str, _CardPager] = {}
        self._pager_prev: dict[str, QPushButton] = {}
        self._pager_next: dict[str, QPushButton] = {}
        self._pager_label: dict[str, QLabel] = {}

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
        )
        self.setMinimumSize(960, 640)
        self.resize(ROOT_WIDTH, ROOT_HEIGHT)
        self._apply_window_opacity(
            getattr(modules.globals, "window_opacity", 1.0)
        )
        self._apply_always_on_top(
            getattr(modules.globals, "always_on_top", False)
        )

        shell = QWidget()
        self.setCentralWidget(shell)
        shell_row = QHBoxLayout(shell)
        shell_row.setContentsMargins(0, 0, 0, 0)
        shell_row.setSpacing(0)

        # ── Center column ─────────────────────────────────────────────
        center = QWidget()
        center.setObjectName("contentRoot")
        center_col = QVBoxLayout(center)
        center_col.setContentsMargins(24, 20, 24, 16)
        center_col.setSpacing(0)

        # Top center: captions + ChatGPT scan display | AI browser.
        # (Live camera preview lives in the right panel.)
        preview_wrap = QWidget()
        preview_wrap.setObjectName("previewChatWrap")
        preview_row = QHBoxLayout(preview_wrap)
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(10)

        self._preview_placeholder = QFrame()
        self._preview_placeholder.setObjectName("previewPlaceholder")
        self._preview_placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._preview_placeholder.setMinimumWidth(CAPTION_PANEL_MIN_WIDTH)
        self._preview_placeholder.setMaximumWidth(CAPTION_PANEL_MAX_WIDTH)
        preview_row.addWidget(self._preview_placeholder, 34)

        # Always-visible control so the browser can be shown again after Hide.
        self._btn_show_ai_strip = QPushButton("◀\n" + _("Show\nAI"))
        self._btn_show_ai_strip.setObjectName("aiShowStrip")
        self._btn_show_ai_strip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_show_ai_strip.setFixedWidth(44)
        self._btn_show_ai_strip.setToolTip(_("Show the AI browser panel"))
        self._btn_show_ai_strip.clicked.connect(self._show_ai_browser)
        self._btn_show_ai_strip.hide()
        preview_row.addWidget(self._btn_show_ai_strip, 0)

        self._ai_browser_panel = self._build_ai_browser_panel()
        self._ai_browser_panel.setMinimumWidth(AI_PANEL_MIN_WIDTH)
        self._ai_browser_panel.setMinimumHeight(420)
        self._ai_browser_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        preview_row.addWidget(self._ai_browser_panel, 66)

        self._preview_wrap = preview_wrap
        self._preview_wrap.setMinimumHeight(420)
        self._center_widget = center
        self._center_col = center_col

        # Compact bar shown only in code-test mode (Exit + return to full UI).
        self._code_test_bar = QWidget()
        bar_row = QHBoxLayout(self._code_test_bar)
        bar_row.setContentsMargins(0, 0, 0, 6)
        bar_row.setSpacing(8)
        code_test_lbl = QLabel(_("Code test mode"))
        code_test_lbl.setObjectName("panelTitle")
        bar_row.addWidget(code_test_lbl)
        bar_row.addStretch(1)
        self._btn_code_test_full = QPushButton(_("Full UI"))
        self._btn_code_test_full.setObjectName("aiBrowserNav")
        self._btn_code_test_full.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_code_test_full.setToolTip(_("Show face-swap controls and settings again"))
        self._btn_code_test_full.clicked.connect(
            lambda: self._set_code_test_mode(False)
        )
        bar_row.addWidget(self._btn_code_test_full)
        self._btn_code_test_exit = QPushButton(_("Exit"))
        self._btn_code_test_exit.setObjectName("danger")
        self._btn_code_test_exit.clicked.connect(self.quit_app)
        bar_row.addWidget(self._btn_code_test_exit)
        self._code_test_bar.hide()
        center_col.addWidget(self._code_test_bar)
        center_col.addWidget(preview_wrap, 5)

        self._controls_panel = QWidget()
        controls_col = QVBoxLayout(self._controls_panel)
        controls_col.setContentsMargins(0, 0, 0, 0)
        controls_col.setSpacing(0)

        self._page_header_widget = QWidget()
        self._page_header_widget.setLayout(self._build_page_header())
        controls_col.addWidget(self._page_header_widget)

        self._controls_body = QWidget()
        body_col = QVBoxLayout(self._controls_body)
        body_col.setContentsMargins(0, 0, 0, 0)
        body_col.setSpacing(0)

        self._header_gap = QWidget()
        self._header_gap.setFixedHeight(6)
        body_col.addWidget(self._header_gap)

        self._stack = QStackedWidget()
        self._stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        self._stack.setMaximumHeight(CONTROLS_PANEL_MAX_HEIGHT)
        self._stack.addWidget(self._build_faces_page("source"))
        self._stack.addWidget(self._build_faces_page("target"))
        self._stack.addWidget(self._build_image_output_page())
        self._stack.addWidget(self._build_settings_page())
        body_col.addWidget(self._stack)
        controls_col.addWidget(self._controls_body)
        center_col.addWidget(self._controls_panel, 0)

        self._dock_gap = QWidget()
        self._dock_gap.setFixedHeight(10)
        center_col.addWidget(self._dock_gap)
        self._floating_dock = self._build_floating_dock()
        center_col.addWidget(self._floating_dock)
        shell_row.addWidget(center, 1)

        # ── Right panel (camera preview only + captions history helpers) ─
        self._right_panel = self._build_right_panel()
        shell_row.addWidget(self._right_panel)

        # Captions + ChatGPT scan stay in the center display area (original place).
        self._caption_host = self._preview_placeholder
        self._caption_overlay = CaptionOverlay(self._preview_placeholder)
        self._caption_ai_divider = QFrame(self._preview_placeholder)
        self._caption_ai_divider.setObjectName("captionAiDivider")
        self._caption_ai_divider.setFixedHeight(2)
        self._caption_ai_divider.hide()
        self._ai_response_overlay = AIResponseOverlay(self._preview_placeholder)
        self._caption_worker: Optional[Any] = None
        self._shell = shell

        self._ai_scan_timer = QTimer(self)
        self._ai_scan_timer.setInterval(150)
        self._ai_scan_timer.timeout.connect(self._scan_ai_browser_response)
        self._ai_scan_timer.start()
        self._ai_scan_busy = False
        self._ai_scan_installed = False
        self._ai_browser_parked = False
        self._ai_browser_keeper = None
        self._ai_waiting_for_response = False
        self._ai_wait_baseline = ""
        self._ai_wait_dots = 0
        self._ai_wait_timer = QTimer(self)
        self._ai_wait_timer.setInterval(450)
        self._ai_wait_timer.timeout.connect(self._tick_ai_wait_alarm)

        self._apply_ai_browser_visibility(
            bool(getattr(modules.globals, "ai_browser_visible", True))
        )

        self._switch_page(self._PAGE_SOURCE)
        QTimer.singleShot(0, self._refresh_face_library)
        # Apply after the native HWND exists.
        QTimer.singleShot(0, lambda: set_window_stealth(
            self, bool(getattr(modules.globals, "stealth_mode", False))
        ))
        QTimer.singleShot(0, self._post_window_init)

        self._session_notes_path = os.path.join(ROOT_DIR, "session_notes.txt")
        self._screenshots_dir = os.path.join(ROOT_DIR, "notes", "screenshots")
        self._notes_save_timer = QTimer(self)
        self._notes_save_timer.setSingleShot(True)
        self._notes_save_timer.setInterval(1200)
        self._notes_save_timer.timeout.connect(self._save_session_notes)

        self._hotkey_manager = GlobalHotkeyManager(self)
        self._hotkey_manager.set_host(self)
        self._hotkey_manager.actionTriggered.connect(self._on_hotkey_action)
        self._screenshot_busy = False
        self._screenshot_paste_busy = False
        self._setup_system_tray()

    def _post_window_init(self) -> None:
        self._load_session_notes()
        self._reload_hotkeys()
        self._sync_live_captions()
        self._apply_controls_collapsed(
            bool(getattr(modules.globals, "controls_collapsed", False))
        )
        self._apply_code_test_mode(
            bool(getattr(modules.globals, "code_test_mode", False))
        )
        QTimer.singleShot(0, self._layout_preview_chat_sizes)
        QTimer.singleShot(0, lambda: apply_tray_taskbar_hiding(self))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_caption_overlay()
        self._layout_preview_chat_sizes()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        set_window_stealth(self, stealth_effective_for(self))
        self._reposition_caption_overlay()
        apply_tray_taskbar_hiding(self)
        if getattr(modules.globals, "always_on_top", False):
            QTimer.singleShot(0, self._apply_stay_on_top_mouse_pass_through)

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        if getattr(modules.globals, "always_on_top", False):
            set_win_click_through(self, False)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        if getattr(modules.globals, "always_on_top", False):
            QTimer.singleShot(0, self._apply_stay_on_top_mouse_pass_through)

    def _reposition_caption_overlay(self) -> None:
        host = getattr(self, "_caption_host", None)
        if host is None:
            return
        overlay = getattr(self, "_caption_overlay", None)
        if overlay is not None:
            overlay.reposition(host)
        ai_overlay = getattr(self, "_ai_response_overlay", None)
        if ai_overlay is not None:
            ai_overlay.reposition(host)
        divider = getattr(self, "_caption_ai_divider", None)
        if divider is not None:
            width = min(450, max(200, host.width() - 24))
            x = max(0, (host.width() - width) // 2)
            y = max(1, host.height() // 4)
            divider.setGeometry(x, y, width, 2)
            divider.show()
            divider.raise_()
        # Captions must stay above the AI band / divider.
        if overlay is not None and overlay.isVisible():
            overlay.raise_()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        from PySide6.QtCore import QEvent

        if event.type() in (
            QEvent.Type.WindowStateChange,
            QEvent.Type.ActivationChange,
        ):
            set_window_stealth(self, stealth_effective_for(self))
            if getattr(modules.globals, "always_on_top", False):
                QTimer.singleShot(0, self._apply_stay_on_top_mouse_pass_through)

    # ── page header ───────────────────────────────────────────────────

    def _build_page_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self._btn_controls_toggle = QPushButton("\u25be")
        self._btn_controls_toggle.setObjectName("collapseBtn")
        self._btn_controls_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_controls_toggle.setToolTip(_("Collapse face-swap controls"))
        self._btn_controls_toggle.clicked.connect(self._toggle_controls_collapsed)
        row.addWidget(self._btn_controls_toggle)

        # Segmented tab navigation (moved here from the old left sidebar).
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        nav_specs = [
            (_("Source"), self._PAGE_SOURCE),
            (_("Target"), self._PAGE_TARGET),
            (_("Image Output"), self._PAGE_IMAGE_OUTPUT),
            (_("Settings"), self._PAGE_SETTINGS),
        ]
        self._nav_buttons: list[QPushButton] = []
        for label, page_idx in nav_specs:
            btn = QPushButton(label)
            btn.setObjectName("tabBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("active", False)
            btn.clicked.connect(lambda checked=False, p=page_idx: self._switch_page(p))
            self._nav_group.addButton(btn)
            row.addWidget(btn)
            self._nav_buttons.append(btn)

        row.addStretch(1)

        self.btn_import = QPushButton(_("Import Face"))
        self.btn_import.setObjectName("secondary")
        self.btn_import.setToolTip(_("Import a face image from disk"))
        self.btn_import.clicked.connect(self._on_import_face)
        row.addWidget(self.btn_import)

        self.btn_add = QPushButton(_("+ Add Face"))
        self.btn_add.setToolTip(_("Add a new face to the library"))
        self.btn_add.clicked.connect(self._on_add_face_header)
        row.addWidget(self.btn_add)

        self.btn_exit = QPushButton(_("Exit"))
        self.btn_exit.setObjectName("danger")
        self.btn_exit.setToolTip(_("Stop live and close the application"))
        self.btn_exit.clicked.connect(self.quit_app)
        row.addWidget(self.btn_exit)
        return row

    def _switch_page(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_buttons):
            active = i == index
            btn.setChecked(active)
            btn.setProperty("active", active)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if index == self._PAGE_SOURCE:
            self._current_kind = "source"
        elif index == self._PAGE_TARGET:
            self._current_kind = "target"
        else:
            self._current_kind = None

        # Import/Add only make sense on the face pages.
        show_face_actions = index in (self._PAGE_SOURCE, self._PAGE_TARGET)
        if hasattr(self, "btn_import"):
            self.btn_import.setVisible(show_face_actions)
            self.btn_add.setVisible(show_face_actions)

    # ── face grid pages ───────────────────────────────────────────────

    def _build_faces_page(self, kind: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ── card row: [◄]  [ card card card ]  [►] ──
        row = QHBoxLayout()
        row.setSpacing(12)

        btn_prev = QPushButton("\u2039")
        btn_prev.setObjectName("pagerArrow")
        btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_prev.setToolTip(_("Previous"))
        btn_prev.clicked.connect(lambda: self._page_step(kind, -1))
        row.addWidget(btn_prev, 0, Qt.AlignmentFlag.AlignVCenter)
        self._pager_prev[kind] = btn_prev

        container = _CardPager()
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        cards_row = QHBoxLayout(container)
        cards_row.setContentsMargins(0, 0, 0, 0)
        cards_row.setSpacing(12)
        cards_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container.resized.connect(lambda k=kind: self._render_face_page(k))
        self._cards_row[kind] = cards_row
        self._cards_container[kind] = container
        row.addWidget(container, 1)

        btn_next = QPushButton("\u203a")
        btn_next.setObjectName("pagerArrow")
        btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_next.setToolTip(_("Next"))
        btn_next.clicked.connect(lambda: self._page_step(kind, 1))
        row.addWidget(btn_next, 0, Qt.AlignmentFlag.AlignVCenter)
        self._pager_next[kind] = btn_next

        layout.addLayout(row, 0)

        # ── page indicator ──
        label = QLabel("")
        label.setObjectName("pagerLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pager_label[kind] = label
        layout.addWidget(label, 0)
        return page

    def _clear_layout_items(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _cards_per_page(self, kind: str) -> int:
        container = self._cards_container.get(kind)
        if container is None or container.width() <= 0:
            return CARDS_PER_PAGE_FALLBACK
        per = (container.width() + 12) // (PROFILE_CARD_W + 12)
        return max(1, int(per))

    def _page_step(self, kind: str, delta: int) -> None:
        self._page_index[kind] += delta
        self._render_face_page(kind)

    def _render_face_page(self, kind: str) -> None:
        """Render the current page of cards for a kind (Add card is item 0)."""
        cards_row = self._cards_row.get(kind)
        if cards_row is None:
            return
        self._clear_layout_items(cards_row)
        self._face_cards[kind] = []

        # Item list = [Add] + entries.
        entries = self._face_entries.get(kind, [])
        total_items = len(entries) + 1
        per = self._cards_per_page(kind)
        pages = max(1, (total_items + per - 1) // per)

        # Clamp current page.
        self._page_index[kind] = max(0, min(self._page_index[kind], pages - 1))
        page = self._page_index[kind]

        start = page * per
        end = min(total_items, start + per)
        active_id = face_library.get_active(kind)

        for item_idx in range(start, end):
            if item_idx == 0:
                add_card = AddProfileCard()
                add_card.clicked.connect(
                    lambda checked=False, k=kind: self._add_face(k)
                )
                cards_row.addWidget(add_card)
            else:
                entry = entries[item_idx - 1]
                card = ProfileCard(entry, kind)
                card.set_selected(entry.get("id") == active_id)
                card.selected.connect(
                    lambda e, k=kind: self._select_saved_face(k, e)
                )
                card.deleted.connect(
                    lambda e, k=kind: self._delete_saved_face(k, e)
                )
                cards_row.addWidget(card)
                self._face_cards[kind].append(card)
        cards_row.addStretch(1)

        # Update arrows + label.
        self._pager_prev[kind].setEnabled(page > 0)
        self._pager_next[kind].setEnabled(page < pages - 1)
        self._pager_label[kind].setText(f"{page + 1} / {pages}")

    # ── image output page ─────────────────────────────────────────────

    def _build_image_output_page(self) -> QWidget:
        page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(0, 0, 8, 8)
        col.setSpacing(10)

        title = QLabel(_("IMAGE OUTPUT SELECT"))
        title.setObjectName("panelTitle")
        col.addWidget(title)

        tip = QLabel(
            _(
                "Pick any image and stream it to the virtual camera — "
                "no webcam and no face swap. Separate from Source/Target faces."
            )
        )
        tip.setObjectName("statusLabel")
        tip.setWordWrap(True)
        col.addWidget(tip)

        self._image_output_preview = QLabel(_("No image selected"))
        self._image_output_preview.setObjectName("activePreview")
        self._image_output_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_output_preview.setFixedSize(200, 200)
        col.addWidget(self._image_output_preview, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._image_output_path_label = QLabel("")
        self._image_output_path_label.setObjectName("statusLabel")
        self._image_output_path_label.setWordWrap(True)
        col.addWidget(self._image_output_path_label)

        btn_row = QHBoxLayout()
        self._btn_choose_image_output = QPushButton(_("Choose image"))
        self._btn_choose_image_output.setObjectName("secondary")
        self._btn_choose_image_output.clicked.connect(self._on_choose_image_output)
        btn_row.addWidget(self._btn_choose_image_output)

        res_col = QVBoxLayout()
        res_col.setSpacing(4)
        res_col.addWidget(self._dock_label(_("Output resolution")))
        self.cb_image_output_resolution = QComboBox()
        saved_w = getattr(modules.globals, "image_output_width", 1280)
        saved_h = getattr(modules.globals, "image_output_height", 720)
        selected_idx = 0
        for i, (label, w, h) in enumerate(IMAGE_OUTPUT_RESOLUTIONS):
            self.cb_image_output_resolution.addItem(label, (w, h))
            if w == saved_w and h == saved_h:
                selected_idx = i
        self.cb_image_output_resolution.setCurrentIndex(selected_idx)
        self.cb_image_output_resolution.currentIndexChanged.connect(
            self._on_image_output_resolution_change
        )
        res_col.addWidget(self.cb_image_output_resolution)
        btn_row.addLayout(res_col, 1)
        col.addLayout(btn_row)

        self._btn_image_output = QPushButton("\u25b6  " + _("Output image"))
        self._btn_image_output.setObjectName("liveBtn")
        self._btn_image_output.setToolTip(
            _("Stream the selected image to the virtual camera")
        )
        self._btn_image_output.clicked.connect(self._on_image_output)
        col.addWidget(self._btn_image_output)

        col.addStretch(1)
        scroll.setWidget(inner)

        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._refresh_image_output_preview()
        return page

    def _on_image_output_resolution_change(self, index: int) -> None:
        if index < 0:
            return
        data = self.cb_image_output_resolution.itemData(index)
        if not data:
            return
        w, h = data
        modules.globals.image_output_width = int(w)
        modules.globals.image_output_height = int(h)
        save_switch_states()

    def _on_choose_image_output(self) -> None:
        global _RECENT_OUTPUT_DIR
        recent = _RECENT_OUTPUT_DIR or ""
        path, _filter = QFileDialog.getOpenFileName(
            self,
            _("Choose image for camera output"),
            recent,
            "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp)",
        )
        if not path:
            return
        if not is_image(path):
            update_status("Please choose an image file")
            return
        _RECENT_OUTPUT_DIR = os.path.dirname(path)
        modules.globals.image_output_path = path
        save_switch_states()
        self._refresh_image_output_preview()
        update_status(f"Image output: {os.path.basename(path)}")

    def _refresh_image_output_preview(self) -> None:
        path = getattr(modules.globals, "image_output_path", None)
        preview = getattr(self, "_image_output_preview", None)
        label = getattr(self, "_image_output_path_label", None)
        if preview is None:
            return
        if not path or not os.path.isfile(path):
            preview.setText(_("No image selected"))
            preview.setPixmap(QPixmap())
            if label is not None:
                label.setText("")
            return
        try:
            preview.setPixmap(
                render_image_preview(path, (200, 200))
            )
            preview.setText("")
        except Exception:
            preview.setText(_("Preview unavailable"))
        if label is not None:
            label.setText(path)

    def _set_image_output_button(self, running: bool) -> None:
        if not hasattr(self, "_btn_image_output"):
            return
        if running:
            self._btn_image_output.setText("\u25a0  " + _("Stop output"))
            self._btn_image_output.setToolTip(_("Stop streaming image to virtual camera"))
        else:
            self._btn_image_output.setText("\u25b6  " + _("Output image"))
            self._btn_image_output.setToolTip(
                _("Stream the selected image to the virtual camera")
            )

    def _on_image_output(self) -> None:
        preview = getattr(self, "_live_preview", None)
        if preview is None:
            return
        if preview.is_static_output():
            preview.stop()
            self._set_image_output_button(False)
            self.set_status_pill(_("Ready"), live=False)
            update_status("Image output stopped")
            return

        path = getattr(modules.globals, "image_output_path", None)
        if not path or not os.path.isfile(path):
            update_status("Choose an image in Image Output first")
            self._switch_page(self._PAGE_IMAGE_OUTPUT)
            return

        if preview.is_face_swap_live():
            preview.stop()
            self._set_live_button(False)

        idx = self.cb_image_output_resolution.currentIndex()
        data = self.cb_image_output_resolution.itemData(idx)
        if data:
            width, height = data
        else:
            width = getattr(modules.globals, "image_output_width", 1280)
            height = getattr(modules.globals, "image_output_height", 720)

        self._btn_image_output.setEnabled(False)
        try:
            started = preview.start_static_image(path, width, height)
            if started:
                self._set_image_output_button(True)
                self.set_status_pill(_("Image out"), live=True)
                update_status("Streaming image to virtual camera")
            else:
                self._set_image_output_button(False)
        finally:
            self._btn_image_output.setEnabled(True)

    # ── settings page ─────────────────────────────────────────────────

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(0, 0, 8, 8)
        col.setSpacing(12)
        col.addWidget(self._build_options_card())
        col.addWidget(self._build_hotkeys_card())
        col.addWidget(self._build_ai_browser_settings_card())
        col.addWidget(self._build_sliders_card())
        col.addStretch(1)
        scroll.setWidget(inner)

        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return page

    # ── floating dock ─────────────────────────────────────────────────

    def _build_floating_dock(self) -> QFrame:
        dock = QFrame()
        dock.setObjectName("floatingDock")
        _apply_glass_shadow(dock, blur=32, color=(0, 0, 0, 140), y_offset=4)
        row = QHBoxLayout(dock)
        row.setContentsMargins(18, 14, 18, 14)
        row.setSpacing(16)

        cam_col = QVBoxLayout()
        cam_col.setSpacing(4)
        cam_col.addWidget(self._dock_label(_("Camera")))
        self._camera_indices, self._camera_names = get_available_cameras()
        self.cb_camera = QComboBox()
        cam_ok = bool(self._camera_names) and self._camera_names[0] != "No cameras found"
        if cam_ok:
            self.cb_camera.addItems(self._camera_names)
        else:
            self.cb_camera.addItem(_("No cameras found"))
            self.cb_camera.setEnabled(False)
        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        cam_col.addWidget(self.cb_camera)
        row.addLayout(cam_col, 2)

        swap_col = QVBoxLayout()
        swap_col.setSpacing(4)
        swap_col.addWidget(self._dock_label(_("Swap Model")))
        self.cb_swap_model = QComboBox()
        self._populate_swap_model_choices()
        self.cb_swap_model.currentIndexChanged.connect(self._on_swap_model_change)
        swap_col.addWidget(self.cb_swap_model)
        row.addLayout(swap_col, 2)

        enh_col = QVBoxLayout()
        enh_col.setSpacing(4)
        enh_col.addWidget(self._dock_label(_("Enhancer")))
        self.cb_enhancer = QComboBox()
        self._populate_enhancer_choices()
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        enh_col.addWidget(self.cb_enhancer)
        row.addLayout(enh_col, 2)

        row.addStretch(1)

        self.btn_live = QPushButton("\u25b6  " + _("Go Live"))
        self.btn_live.setObjectName("liveBtn")
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)
        row.addWidget(self.btn_live, 0, Qt.AlignmentFlag.AlignBottom)

        return dock

    def _set_live_button(self, live: bool) -> None:
        if live:
            self.btn_live.setText("\u25a0  " + _("Stop"))
            self.btn_live.setToolTip(_("Stop live camera preview"))
        else:
            self.btn_live.setText("\u25b6  " + _("Go Live"))
            self.btn_live.setToolTip(_("Start real-time face swap using webcam"))

    def is_live(self) -> bool:
        preview = getattr(self, "_live_preview", None)
        return bool(preview and preview.is_face_swap_live())

    def is_image_output_running(self) -> bool:
        preview = getattr(self, "_live_preview", None)
        return bool(preview and preview.is_static_output())

    def _dock_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("dockLabel")
        return label

    # ── right panel ───────────────────────────────────────────────────

    def _build_right_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("rightPanel")
        panel.setFixedWidth(RIGHT_PANEL_WIDTH)
        col = QVBoxLayout(panel)
        col.setContentsMargins(12, 16, 12, 16)
        col.setSpacing(10)

        status_row = QHBoxLayout()
        self._status_pill = QFrame()
        self._status_pill.setObjectName("statusPill")
        pill_inner = QHBoxLayout(self._status_pill)
        pill_inner.setContentsMargins(10, 6, 12, 6)
        pill_inner.setSpacing(6)
        self._status_pill_dot = QLabel("\u25cf")
        self._status_pill_dot.setObjectName("heroDot")
        self._status_pill_text = QLabel(_("Ready"))
        self._status_pill_text.setObjectName("statusPillText")
        pill_inner.addWidget(self._status_pill_dot)
        pill_inner.addWidget(self._status_pill_text)
        status_row.addWidget(self._status_pill)
        status_row.addStretch(1)
        col.addLayout(status_row)

        # Live camera preview (moved here from the center column).
        col.addWidget(self._panel_section_title(_("PREVIEW")))
        self._live_preview = LivePreviewPanel()
        self._live_preview.setMinimumHeight(LIVE_PREVIEW_HEIGHT)
        col.addWidget(self._live_preview, 3)

        col.addWidget(self._panel_section_title(_("STATUS")))
        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._status_label.setMaximumHeight(56)
        col.addWidget(self._status_label)

        # Session notes — captions, screenshots, typed notes, and more.
        col.addWidget(self._panel_section_title(_("SESSION NOTES")))
        self._caption_history = QTextEdit()
        self._caption_history.setObjectName("captionHistory")
        self._caption_history.setReadOnly(False)
        self._caption_history.setPlaceholderText(
            _(
                "Live captions, screenshots, and your notes accumulate here. "
                "Edit freely — everything is saved automatically."
            )
        )
        self._caption_history.setMinimumHeight(140)
        self._caption_history.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._caption_history.textChanged.connect(self._on_caption_history_user_edit)
        col.addWidget(self._caption_history, 2)
        self._caption_history_base = ""
        self._caption_history_partial = ""
        self._caption_history_awaiting_reset = False
        self._caption_history_programmatic = False

        notes_btn_row = QHBoxLayout()
        notes_btn_row.setSpacing(8)
        self._btn_insert_screenshot = QPushButton(_("Screenshot → notes"))
        self._btn_insert_screenshot.setObjectName("aiBrowserNav")
        self._btn_insert_screenshot.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_insert_screenshot.setToolTip(
            _("Capture the screen, copy to clipboard, and add a note here")
        )
        self._btn_insert_screenshot.clicked.connect(self._capture_screenshot_to_clipboard)
        notes_btn_row.addWidget(self._btn_insert_screenshot)

        self._btn_clear_notes = QPushButton(_("Clear notes"))
        self._btn_clear_notes.setObjectName("aiBrowserNav")
        self._btn_clear_notes.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_clear_notes.clicked.connect(self._clear_session_notes)
        notes_btn_row.addWidget(self._btn_clear_notes)
        notes_btn_row.addStretch(1)
        col.addLayout(notes_btn_row)

        # Helper for the embedded AI browser (no API key — use the site login).
        col.addWidget(self._panel_section_title(_("AI CHAT")))
        tip = QLabel(
            _(
                "Camera preview is in this right panel. "
                "Live captions and ChatGPT replies show in the center display. "
                "Use Hide on the AI browser, then Show AI to bring it back."
            )
        )
        tip.setObjectName("statusLabel")
        tip.setWordWrap(True)
        col.addWidget(tip)

        self._btn_copy_captions = QPushButton(_("Send notes to AI"))
        self._btn_copy_captions.setObjectName("aiBrowserNav")
        self._btn_copy_captions.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_copy_captions.setToolTip(
            _(
                "Send session notes and/or attached photos to the AI chat. "
                "Works with text only, photos only, or both. "
                "Photos: take a Screenshot first, then Send."
            )
        )
        self._btn_copy_captions.clicked.connect(self._copy_captions_to_clipboard)
        col.addWidget(self._btn_copy_captions)

        return panel

    def _panel_section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("panelTitle")
        return label

    def set_status_pill(self, text: str, live: bool = False) -> None:
        if getattr(self, "_status_pill_text", None) is None:
            return
        self._status_pill_text.setText(text)
        dot_color = "#6ccb5f" if live else "#c9a84c"
        self._status_pill_dot.setStyleSheet(
            f"color: {dot_color}; font-size: 10pt; background: transparent;"
        )

    def _update_active_previews(self) -> None:
        # Active Source / Target cards were removed from the right panel.
        return

    # ── options card ─────────────────────────────────────────────────────

    def _build_options_card(self) -> QGroupBox:
        card = QGroupBox(_("Live Options"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(8)

        def make(field, label, tip):
            sw = _Switch(_(label), getattr(modules.globals, field), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_many_faces = make(
            "many_faces", "Many faces",
            "Swap every detected face, not just the primary one",
        )
        self.sw_poisson = make(
            "poisson_blend", "Poisson Blend",
            "Blend face edges smoothly using Poisson blending",
        )
        self.sw_color_fix = make(
            "color_correction", "Fix Blueish Cam",
            "Fix blue/green color cast from some webcams",
        )
        self.sw_show_fps = make(
            "show_fps", "Show FPS",
            "Display frames-per-second counter on the live preview",
        )
        self.sw_stealth = _Switch(
            _("Stealth mode"),
            getattr(modules.globals, "stealth_mode", False),
            _(
                "Hide this app from screen sharing and screenshots "
                "(Windows Graphics Capture). Virtual camera still works."
            ),
        )
        self.sw_stealth.toggled.connect(self._on_stealth_toggle)

        self.sw_captions = _Switch(
            _("Live captions"),
            getattr(modules.globals, "live_captions", False),
            _(
                "Show live speech-to-text at the top of the preview "
                "(system audio, streaming — updates many times per second)."
            ),
        )
        self.sw_captions.toggled.connect(self._on_captions_toggle)

        self.sw_always_on_top = _Switch(
            _("Stay on top (ghost)"),
            getattr(modules.globals, "always_on_top", False),
            _(
                "Float above other apps. When unfocused, the window is invisible "
                "to the mouse and screen capture — use Focus window hotkey or "
                "Alt+Tab to interact. Controls work only while focused."
            ),
        )
        self.sw_always_on_top.toggled.connect(self._on_always_on_top_toggle)

        self.sw_code_test = _Switch(
            _("Code test mode"),
            getattr(modules.globals, "code_test_mode", False),
            _(
                "Collapse face-swap controls — keep live captions, AI replies, "
                "and the ChatGPT browser only"
            ),
        )
        self.sw_code_test.toggled.connect(self._on_code_test_toggle)

        self.sw_minimize_to_tray = _Switch(
            _("Close to tray"),
            getattr(modules.globals, "minimize_to_tray", True),
            _(
                "Keep running in the system tray when the window is closed. "
                "Use the tray icon to show the window again or exit."
            ),
        )
        self.sw_minimize_to_tray.toggled.connect(self._on_minimize_to_tray_toggle)

        items = [
            self.sw_many_faces, self.sw_show_fps,
            self.sw_poisson, self.sw_color_fix,
            self.sw_stealth, self.sw_captions,
            self.sw_always_on_top, self.sw_code_test,
            self.sw_minimize_to_tray,
        ]
        for i, w in enumerate(items):
            grid.addWidget(w, i // 2, i % 2)
        return card

    def _on_stealth_toggle(self, enabled: bool) -> None:
        modules.globals.stealth_mode = enabled
        save_switch_states()
        apply_stealth_to_all_windows(enabled)
        update_status(
            "Stealth mode on — app window hidden from screen share"
            if enabled
            else "Stealth mode off — app window visible in screen share"
        )

    def _on_captions_toggle(self, enabled: bool) -> None:
        modules.globals.live_captions = enabled
        save_switch_states()
        self._sync_live_captions()
        update_status(
            "Live captions on"
            if enabled
            else "Live captions off"
        )

    def _on_always_on_top_toggle(self, enabled: bool) -> None:
        modules.globals.always_on_top = enabled
        save_switch_states()
        self._apply_always_on_top(enabled)
        update_status(
            "Ghost overlay on — unfocused clicks pass through; Alt+Tab or "
            "Focus window hotkey to control"
            if enabled
            else "Ghost overlay off"
        )

    def _apply_always_on_top(self, enabled: bool) -> None:
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, bool(enabled))
        self.show()
        set_window_stealth(self, stealth_effective_for(self))
        apply_tray_taskbar_hiding(self)
        QTimer.singleShot(0, self._apply_stay_on_top_mouse_pass_through)

    def _toggle_controls_collapsed(self) -> None:
        self._set_controls_collapsed(
            not bool(getattr(modules.globals, "controls_collapsed", False))
        )

    def _set_controls_collapsed(self, collapsed: bool) -> None:
        modules.globals.controls_collapsed = bool(collapsed)
        save_switch_states()
        self._apply_controls_collapsed(collapsed)

    def _apply_controls_collapsed(self, collapsed: bool) -> None:
        body = getattr(self, "_controls_body", None)
        gap = getattr(self, "_header_gap", None)
        btn = getattr(self, "_btn_controls_toggle", None)
        stack = getattr(self, "_stack", None)

        if body is not None:
            body.setVisible(not collapsed)
        if gap is not None:
            gap.setVisible(not collapsed)
        if btn is not None:
            btn.setText("\u25b8" if collapsed else "\u25be")
            btn.setToolTip(
                _("Expand face-swap controls")
                if collapsed
                else _("Collapse face-swap controls")
            )
        if stack is not None and not collapsed:
            stack.setMaximumHeight(CONTROLS_PANEL_MAX_HEIGHT)

    def _on_code_test_toggle(self, enabled: bool) -> None:
        self._set_code_test_mode(enabled)

    def _on_minimize_to_tray_toggle(self, enabled: bool) -> None:
        modules.globals.minimize_to_tray = bool(enabled)
        save_switch_states()
        apply_tray_taskbar_hiding_to_all_windows()
        QTimer.singleShot(0, apply_tray_taskbar_hiding_to_all_windows)
        update_status(
            "Close button will hide to system tray"
            if enabled
            else "Close button will exit the app"
        )

    def _set_code_test_mode(self, enabled: bool) -> None:
        modules.globals.code_test_mode = bool(enabled)
        if getattr(self, "sw_code_test", None) is not None:
            self.sw_code_test.blockSignals(True)
            self.sw_code_test.setChecked(enabled)
            self.sw_code_test.blockSignals(False)
        save_switch_states()
        self._apply_code_test_mode(enabled)
        update_status(
            "Code test mode — AI chat only"
            if enabled
            else "Full UI restored"
        )

    def _apply_code_test_mode(self, enabled: bool) -> None:
        """Hide face-swap UI; keep caption overlay + embedded AI browser."""
        bar = getattr(self, "_code_test_bar", None)
        if bar is not None:
            bar.setVisible(enabled)

        panel_ctrl = getattr(self, "_controls_panel", None)
        if panel_ctrl is not None:
            panel_ctrl.setVisible(not enabled)

        for w in (
            getattr(self, "_dock_gap", None),
            getattr(self, "_floating_dock", None),
            getattr(self, "_right_panel", None),
        ):
            if w is not None:
                w.setVisible(not enabled)

        center = getattr(self, "_center_widget", None)
        if center is not None:
            margins = (8, 8, 8, 8) if enabled else (24, 20, 24, 16)
            center.layout().setContentsMargins(*margins)

        panel = getattr(self, "_ai_browser_panel", None)
        placeholder = getattr(self, "_preview_placeholder", None)
        if panel is not None:
            if enabled:
                panel.setMinimumWidth(AI_PANEL_MIN_WIDTH)
                panel.setMaximumWidth(16777215)
                if placeholder is not None:
                    placeholder.setMaximumWidth(CAPTION_PANEL_MAX_WIDTH)
                self._apply_ai_browser_visibility(True)
                self._btn_show_ai_strip.hide()
                self.setMinimumSize(680, 420)
            else:
                panel.setMinimumWidth(AI_PANEL_MIN_WIDTH)
                panel.setMaximumWidth(16777215)
                if placeholder is not None:
                    placeholder.setMaximumWidth(CAPTION_PANEL_MAX_WIDTH)
                self.setMinimumSize(960, 640)

        if enabled:
            self._reposition_caption_overlay()
        QTimer.singleShot(0, self._layout_preview_chat_sizes)

    def _layout_preview_chat_sizes(self) -> None:
        """Keep the AI browser wider than the caption panel."""
        wrap = getattr(self, "_preview_wrap", None)
        panel = getattr(self, "_ai_browser_panel", None)
        placeholder = getattr(self, "_preview_placeholder", None)
        strip = getattr(self, "_btn_show_ai_strip", None)
        if wrap is None or panel is None or placeholder is None:
            return

        total = max(520, wrap.width())
        strip_w = strip.width() if strip is not None and strip.isVisible() else 0
        spacing = 10 if strip_w else 0
        avail = max(400, total - strip_w - spacing)

        browser_visible = (
            getattr(modules.globals, "ai_browser_visible", True)
            and panel.isVisible()
        )
        code_test = bool(getattr(modules.globals, "code_test_mode", False))

        if browser_visible:
            ratio = 0.68 if code_test else AI_PANEL_WIDTH_RATIO
            ai_w = max(AI_PANEL_MIN_WIDTH, int(avail * ratio))
            caption_w = max(CAPTION_PANEL_MIN_WIDTH, avail - ai_w)
            if ai_w <= caption_w:
                ai_w = max(AI_PANEL_MIN_WIDTH, int(avail * 0.58) + 1)
                caption_w = max(CAPTION_PANEL_MIN_WIDTH, avail - ai_w)
            panel.setMinimumWidth(ai_w)
            panel.setMaximumWidth(ai_w)
            placeholder.setMinimumWidth(caption_w)
            placeholder.setMaximumWidth(caption_w)
        else:
            panel.setMinimumWidth(0)
            panel.setMaximumWidth(0)
            placeholder.setMinimumWidth(CAPTION_PANEL_MIN_WIDTH)
            placeholder.setMaximumWidth(16777215)

    def _apply_stay_on_top_mouse_pass_through(self) -> None:
        """When ghost overlay is on and unfocused, ignore all mouse input."""
        if not getattr(modules.globals, "always_on_top", False):
            set_win_click_through(self, False)
            return
        set_win_click_through(self, not self.isActiveWindow())

    def _focus_main_window(self) -> None:
        """Bring the app to the front and allow mouse interaction."""
        self.show_from_tray()

    def _setup_system_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            return
        app = QApplication.instance()
        if app is not None:
            app.setQuitOnLastWindowClosed(False)

        icon_path = os.path.join(ROOT_DIR, "media", "Download.png")
        icon = (
            QIcon(icon_path)
            if os.path.isfile(icon_path)
            else self.windowIcon()
        )
        self._tray = QSystemTrayIcon(icon, self)
        self._tray_menu = QMenu(self)
        self._tray_menu.setObjectName("trayMenu")
        show_action = QAction(_("Show window"), self)
        show_action.triggered.connect(self.show_from_tray)
        self._tray_menu.addAction(show_action)
        hide_action = QAction(_("Hide to tray"), self)
        hide_action.triggered.connect(self.hide_to_tray)
        self._tray_menu.addAction(hide_action)
        self._tray_menu.addSeparator()
        quit_action = QAction(_("Exit"), self)
        quit_action.triggered.connect(self.quit_app)
        self._tray_menu.addAction(quit_action)
        self._tray.setContextMenu(self._tray_menu)
        self._tray.setToolTip(self.windowTitle())
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self.isVisible() and not self.isMinimized():
                self.hide_to_tray()
            else:
                self.show_from_tray()

    def show_from_tray(self) -> None:
        set_win_click_through(self, False)
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        apply_tray_taskbar_hiding(self)
        self.raise_()
        self.activateWindow()
        if getattr(modules.globals, "always_on_top", False):
            QTimer.singleShot(0, self._apply_stay_on_top_mouse_pass_through)

    def hide_to_tray(self) -> None:
        if self._tray is None:
            self.hide()
            return
        apply_tray_taskbar_hiding(self)
        self.hide()
        self._tray.showMessage(
            self.windowTitle(),
            _("Running in the system tray — double-click the icon to show."),
            QSystemTrayIcon.MessageIcon.Information,
            2500,
        )

    def quit_app(self) -> None:
        if self._force_quit:
            return
        self._force_quit = True
        if self._tray is not None:
            self._tray.hide()
        self._save_session_notes()
        if getattr(self, "_hotkey_manager", None) is not None:
            self._hotkey_manager.clear()
        self._stop_live_captions()
        if getattr(self, "_live_preview", None) is not None:
            self._live_preview.stop()
        self._destroy_cb()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _build_hotkeys_card(self) -> QGroupBox:
        card = QGroupBox(_("Hotkeys"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        tip = QLabel(
            _(
                "Click a button, then press a key combination. "
                "Shortcuts work globally while this app is running (Windows)."
            )
        )
        tip.setObjectName("statusLabel")
        tip.setWordWrap(True)
        grid.addWidget(tip, 0, 0, 1, 2)

        self._hotkey_buttons: dict[str, HotkeyCaptureButton] = {}
        specs = [
            (
                "screenshot",
                _("Screenshot to clipboard"),
                _("Capture the screen, copy it, and paste the image into the AI chat"),
            ),
            (
                "send_notes_to_ai",
                _("Send notes to AI"),
                _(
                    "Send notes text and/or attached photos to the AI chat "
                    "(text only, photos only, or both)"
                ),
            ),
            (
                "toggle_always_on_top",
                _("Toggle ghost overlay"),
                _("Turn stay-on-top ghost mode on or off"),
            ),
            (
                "focus_window",
                _("Focus window"),
                _("Activate this app so controls work (required in ghost mode)"),
            ),
            (
                "focus_session_notes",
                _("Focus session notes"),
                _("Activate the app and jump to the session notes box"),
            ),
            (
                "ai_chat_scroll_up",
                _("AI chat scroll up"),
                _("Scroll the embedded AI chat up (works globally, without focusing the app)"),
            ),
            (
                "ai_chat_scroll_down",
                _("AI chat scroll down"),
                _("Scroll the embedded AI chat down (works globally, without focusing the app)"),
            ),
        ]
        hotkeys = getattr(modules.globals, "hotkeys", default_hotkeys())
        for row, (action, label, help_text) in enumerate(specs, start=1):
            lbl = QLabel(label)
            lbl.setToolTip(help_text)
            btn = HotkeyCaptureButton(hotkeys.get(action, ""))
            btn.setObjectName("aiBrowserNav")
            btn.setToolTip(help_text)
            btn.hotkeyChanged.connect(
                lambda seq, a=action: self._on_hotkey_changed(a, seq)
            )
            self._hotkey_buttons[action] = btn
            grid.addWidget(lbl, row, 0)
            grid.addWidget(btn, row, 1)

        return card

    def _on_hotkey_changed(self, action: str, sequence: str) -> None:
        hotkeys = dict(getattr(modules.globals, "hotkeys", default_hotkeys()))
        hotkeys[action] = (sequence or "").strip()
        modules.globals.hotkeys = hotkeys
        save_switch_states()
        self._reload_hotkeys()

    def _reload_hotkeys(self) -> None:
        bindings = dict(getattr(modules.globals, "hotkeys", default_hotkeys()))
        errors = self._hotkey_manager.apply_bindings(bindings)
        if errors:
            update_status("Hotkey warning: " + "; ".join(errors[:2]))

    def _on_hotkey_action(self, action: str) -> None:
        if action == "screenshot":
            self._capture_screenshot_to_clipboard()
        elif action == "send_notes_to_ai":
            self._copy_captions_to_clipboard()
        elif action == "toggle_always_on_top":
            enabled = not bool(getattr(modules.globals, "always_on_top", False))
            if getattr(self, "sw_always_on_top", None) is not None:
                self.sw_always_on_top.blockSignals(True)
                self.sw_always_on_top.setChecked(enabled)
                self.sw_always_on_top.blockSignals(False)
            self._on_always_on_top_toggle(enabled)
        elif action == "focus_window":
            self._focus_main_window()
        elif action == "focus_session_notes":
            self._focus_main_window()
            box = getattr(self, "_caption_history", None)
            if box is not None:
                def _focus_notes() -> None:
                    box.setFocus()
                    box.moveCursor(QTextCursor.MoveOperation.End)

                QTimer.singleShot(120, _focus_notes)
        elif action == "ai_chat_scroll_up":
            print("[hotkey] ai_chat_scroll_up", flush=True)
            self._scroll_ai_chat(-1)
        elif action == "ai_chat_scroll_down":
            print("[hotkey] ai_chat_scroll_down", flush=True)
            self._scroll_ai_chat(1)

    def _install_ai_scroll_helper(self) -> None:
        """Ensure window.__DLC_SCROLL__ exists in the AI browser page."""
        browser = getattr(self, "_ai_browser", None)
        if browser is None:
            return
        page = browser.page()
        if page is None:
            return
        try:
            page.runJavaScript(_AI_SCROLL_HELPER_JS)
        except Exception as exc:
            print(f"[ai-scroll] helper install failed: {exc}", flush=True)

    def _scroll_ai_chat(self, direction: int) -> None:
        """Scroll the embedded AI browser without requiring window focus.

        direction: -1 = up, +1 = down
        """
        browser = getattr(self, "_ai_browser", None)
        if browser is None:
            update_status("AI chat scroll: browser unavailable")
            return
        self._keep_ai_browser_alive()
        page = browser.page()
        if page is None:
            update_status("AI chat scroll: page unavailable")
            return

        amount = int(direction)  # +1 down / -1 up; distance computed in JS as half-page
        # Install helper then scroll. Callback confirms movement for debugging.
        # Force-redefine helper so step size updates without a full page reload.
        call_js = (
            "window.__DLC_SCROLL__ = undefined;\n"
            f"{_AI_SCROLL_HELPER_JS};\n"
            f"window.__DLC_SCROLL__({amount});"
        )

        def _on_result(result: Any) -> None:
            text = str(result) if result is not None else "null"
            print(f"[ai-scroll] {text}", flush=True)
            if text in ("null", "None", "window", "zero"):
                update_status(f"AI chat scroll weak ({text}) — try clicking the chat once")
            # else silent success (avoid status spam while scrolling)

        try:
            page.runJavaScript(call_js, _on_result)
        except Exception as exc:
            print(f"[ai-scroll] JS failed: {exc}", flush=True)
            update_status(f"AI chat scroll failed: {exc}")
            return

        # Also fire wheel events at Chromium's render widget (helps some sites).
        try:
            from PySide6.QtGui import QWheelEvent
            from PySide6.QtCore import QPoint, QPointF

            targets = []
            proxy = browser.focusProxy() if hasattr(browser, "focusProxy") else None
            if proxy is not None:
                targets.append(proxy)
            targets.append(browser)
            for child in browser.findChildren(QWidget):
                name = child.metaObject().className()
                if "RenderWidget" in name or "DelegateWidget" in name or "OpenGL" in name:
                    targets.append(child)
            angle = QPoint(0, -60 * int(direction))
            pixel = QPoint(0, -20 * int(direction))
            for target in targets:
                if target is None or not target.isVisible():
                    continue
                pos = QPointF(max(1, target.width()) / 2, max(1, target.height()) / 2)
                global_pos = QPointF(target.mapToGlobal(pos.toPoint()))
                wheel = QWheelEvent(
                    pos,
                    global_pos,
                    pixel,
                    angle,
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.ScrollUpdate,
                    False,
                )
                QApplication.sendEvent(target, wheel)
        except Exception as exc:
            print(f"[ai-scroll] wheel fallback failed: {exc}", flush=True)

    def _ensure_ai_browser_ready(self) -> bool:
        browser = getattr(self, "_ai_browser", None)
        if browser is None or browser.page() is None:
            update_status("AI browser is unavailable")
            return False
        if not getattr(modules.globals, "ai_browser_visible", True):
            self._show_ai_browser()
        self._keep_ai_browser_alive()
        return True

    def _image_bytes_for_ai(self, image: QImage) -> bytes:
        upload = image
        if upload.width() > 1920:
            upload = upload.scaledToWidth(
                1920, Qt.TransformationMode.SmoothTransformation
            )
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        upload.save(buffer, "PNG")
        return bytes(buffer.data())

    def _paste_image_to_ai_browser(
        self, image: QImage, filename: str = "screenshot.png"
    ) -> None:
        """Attach a screenshot image to the embedded AI chat composer."""
        if getattr(self, "_screenshot_paste_busy", False):
            return
        if not self._ensure_ai_browser_ready():
            return

        self._screenshot_paste_busy = True
        browser = self._ai_browser
        page = browser.page()
        png_bytes = self._image_bytes_for_ai(image)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        token = uuid.uuid4().hex
        script = (
            f"{_AI_PASTE_IMAGE_JS}"
            f"({json.dumps(b64)}, {json.dumps(filename)}, {json.dumps(token)})"
        )

        def _release_paste_busy() -> None:
            self._screenshot_paste_busy = False

        def _after_js(result: Any) -> None:
            status = result if isinstance(result, str) else ""
            print(f"[screenshot-attach] {status}", flush=True)
            if status.startswith("ok") or status.startswith("skip"):
                # Clear the system clipboard so a focused ChatGPT composer
                # cannot paste this screenshot a second time.
                try:
                    QApplication.clipboard().clear()
                except Exception:
                    pass
                update_status(
                    "Screenshot attached to AI chat"
                    if status.startswith("ok")
                    else "Screenshot skipped (duplicate)"
                )
                QTimer.singleShot(
                    _SCREENSHOT_DEBOUNCE_MS, _release_paste_busy
                )
                return

            # Do not fall back to clipboard Paste — that duplicates attachments.
            update_status(
                "Screenshot saved to disk, but attach failed"
                + (f": {status[4:]}" if status.startswith("err:") else "")
                + ". Click + in ChatGPT and pick the file manually."
            )
            _release_paste_busy()

        try:
            page.runJavaScript(script, _after_js)
        except Exception as exc:
            update_status(f"Screenshot saved — attach failed: {exc}")
            _release_paste_busy()

    def _grab_desktop_image(self) -> QImage:
        app = QApplication.instance()
        if app is None:
            return QImage()
        screens = app.screens()
        if not screens:
            return QImage()
        if len(screens) == 1:
            return screens[0].grabWindow(0).toImage()

        union = QRect()
        for screen in screens:
            union = union.united(screen.geometry())
        result = QImage(union.size(), QImage.Format.Format_RGB32)
        result.fill(QColor(0, 0, 0).rgb())
        painter = QPainter(result)
        for screen in screens:
            geom = screen.geometry()
            shot = screen.grabWindow(0)
            offset = geom.topLeft() - union.topLeft()
            painter.drawPixmap(offset, shot)
        painter.end()
        return result

    def _capture_screenshot_to_clipboard(self) -> None:
        if getattr(self, "_screenshot_busy", False):
            return
        self._screenshot_busy = True

        image = self._grab_desktop_image()
        if image.isNull():
            self._screenshot_busy = False
            update_status("Screenshot failed — no display")
            return

        os.makedirs(self._screenshots_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{stamp}_{int(time.time() * 1000) % 1000:03d}.png"
        path = os.path.join(self._screenshots_dir, filename)
        image.save(path)

        self._paste_image_to_ai_browser(image, filename)
        update_status("Screenshot copied — pasting into AI chat…")

        QTimer.singleShot(
            _SCREENSHOT_DEBOUNCE_MS,
            lambda: setattr(self, "_screenshot_busy", False),
        )

    def _append_session_note_line(self, line: str) -> None:
        line = (line or "").strip()
        if not line:
            return
        base = getattr(self, "_caption_history_base", "").rstrip()
        self._caption_history_base = f"{base}\n{line}".strip() if base else line
        self._caption_history_partial = ""
        self._render_caption_history()
        self._schedule_session_notes_save()

    def _load_session_notes(self) -> None:
        path = getattr(self, "_session_notes_path", "")
        if not path or not os.path.isfile(path):
            return
        try:
            text = open(path, encoding="utf-8").read().strip()
        except OSError:
            return
        if not text:
            return
        self._caption_history_base = text
        self._caption_history_partial = ""
        self._render_caption_history()

    def _schedule_session_notes_save(self) -> None:
        timer = getattr(self, "_notes_save_timer", None)
        if timer is not None:
            timer.start()

    def _save_session_notes(self) -> None:
        box = getattr(self, "_caption_history", None)
        path = getattr(self, "_session_notes_path", "")
        if box is None or not path:
            return
        text = box.toPlainText()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass

    def _clear_session_notes(self) -> None:
        self._caption_history_base = ""
        self._caption_history_partial = ""
        box = getattr(self, "_caption_history", None)
        if box is not None:
            self._caption_history_programmatic = True
            box.clear()
            self._caption_history_programmatic = False
        self._save_session_notes()
        update_status("Session notes cleared")

    def _sync_live_captions(self) -> None:
        """Start/stop the STT worker to match the live_captions toggle."""
        enabled = bool(getattr(modules.globals, "live_captions", False))
        if enabled:
            self._start_live_captions()
        else:
            self._stop_live_captions()
            if getattr(self, "_caption_overlay", None) is not None:
                self._caption_overlay.clear_lines()

    def _start_live_captions(self) -> None:
        from modules.live_captions import LiveCaptionWorker, captions_available

        ok, msg = captions_available()
        if not ok:
            update_status(msg)
            modules.globals.live_captions = False
            save_switch_states()
            if getattr(self, "sw_captions", None) is not None:
                self.sw_captions.setChecked(False)
            return

        if self._caption_worker is not None and self._caption_worker.isRunning():
            return

        self._caption_worker = LiveCaptionWorker(self)
        self._caption_worker.captionReady.connect(self._on_caption_text)
        self._caption_worker.captionPartial.connect(self._on_caption_partial)
        self._caption_worker.captionFinal.connect(self._on_caption_final)
        self._caption_worker.utteranceBegan.connect(self._on_caption_utterance_began)
        self._caption_worker.historyReset.connect(self._on_caption_history_reset)
        self._caption_worker.statusChanged.connect(
            lambda t: update_status(t) if t else None
        )
        self._caption_worker.errorOccurred.connect(self._on_caption_error)
        self._caption_worker.start()
        # Show a visible placeholder so the overlay is obvious while listening.
        if getattr(self, "_caption_overlay", None) is not None:
            self._caption_overlay.show_caption(_("Listening to system audio..."))
            self._reposition_caption_overlay()

    def _stop_live_captions(self) -> None:
        worker = getattr(self, "_caption_worker", None)
        if worker is None:
            return
        worker.stop()
        worker.wait(2500)
        self._caption_worker = None

    def _on_caption_history_user_edit(self) -> None:
        """Keep typed/edited text as the history base (don't fight the user)."""
        if getattr(self, "_caption_history_programmatic", False):
            return
        box = getattr(self, "_caption_history", None)
        if box is None:
            return
        self._caption_history_base = box.toPlainText()
        self._caption_history_partial = ""
        self._schedule_session_notes_save()

    def _append_caption_history(self, text: str) -> None:
        """Add a finalized phrase onto the editable history base."""
        text = (text or "").strip()
        if not text:
            return
        base = getattr(self, "_caption_history_base", "").rstrip()
        self._caption_history_base = f"{base}\n{text}".strip() if base else text
        self._caption_history_partial = ""
        self._render_caption_history()
        self._schedule_session_notes_save()

    def _render_caption_history(self) -> None:
        """Show editable base text plus the newest live partial phrase."""
        box = getattr(self, "_caption_history", None)
        if box is None:
            return
        base = getattr(self, "_caption_history_base", "")
        partial = getattr(self, "_caption_history_partial", "").strip()
        if partial:
            text = f"{base.rstrip()}\n{partial}".strip() if base.strip() else partial
        else:
            text = base
        if box.toPlainText() == text:
            return
        self._caption_history_programmatic = True
        box.setPlainText(text)
        box.moveCursor(QTextCursor.MoveOperation.End)
        box.ensureCursorVisible()
        self._caption_history_programmatic = False

    def _on_caption_text(self, text: str) -> None:
        """Live overlay update (partials stream in several times per second)."""
        if not getattr(modules.globals, "live_captions", False):
            return
        text = (text or "").strip()
        if not text:
            return
        overlay = getattr(self, "_caption_overlay", None)
        if overlay is None:
            return
        # Live overlay is driven only by the STT stream (not history sync).
        overlay.show_caption(text)

    def _on_caption_partial(self, text: str) -> None:
        """Update unfinalized words in history as soon as Vosk recognizes them."""
        if not getattr(modules.globals, "live_captions", False):
            return
        if getattr(self, "_caption_history_awaiting_reset", False):
            return
        self._caption_history_partial = (text or "").strip()
        self._render_caption_history()

    def _on_caption_final(self, text: str) -> None:
        """Append finalized phrases to the permanent history panel."""
        if not getattr(modules.globals, "live_captions", False):
            return
        if getattr(self, "_caption_history_awaiting_reset", False):
            return
        text = (text or "").strip()
        if not text:
            return
        self._append_caption_history(text)

    def _clear_caption_history(self) -> None:
        # Wipe the box immediately, then reset the STT recognizer so leftover
        # partials/finals from the submitted utterance cannot refill history.
        # Ignore worker signals until historyReset confirms the wipe.
        self._caption_history_awaiting_reset = True
        self._caption_history_base = ""
        self._caption_history_partial = ""
        box = getattr(self, "_caption_history", None)
        if box is not None:
            self._caption_history_programmatic = True
            box.clear()
            self._caption_history_programmatic = False
        overlay = getattr(self, "_caption_overlay", None)
        if overlay is not None:
            overlay.begin_new_utterance()
        worker = getattr(self, "_caption_worker", None)
        if worker is not None and worker.isRunning():
            worker.request_reset()
            # Safety: never leave history gated forever if reset is delayed.
            QTimer.singleShot(1500, self._release_caption_history_reset_gate)
        else:
            self._caption_history_awaiting_reset = False

    def _release_caption_history_reset_gate(self) -> None:
        if getattr(self, "_caption_history_awaiting_reset", False):
            self._caption_history_awaiting_reset = False

    def _on_caption_utterance_began(self) -> None:
        """New speech after pause — let the live overlay accept a fresh string."""
        overlay = getattr(self, "_caption_overlay", None)
        if overlay is not None:
            overlay.begin_new_utterance()

    def _on_caption_history_reset(self) -> None:
        """STT recognizer is clean — accept new speech into history again."""
        self._caption_history_awaiting_reset = False
        self._caption_history_base = ""
        self._caption_history_partial = ""
        overlay = getattr(self, "_caption_overlay", None)
        if overlay is not None:
            overlay.begin_new_utterance()
        self._render_caption_history()

    def _on_caption_error(self, message: str) -> None:
        update_status(message)
        if getattr(self, "_caption_overlay", None) is not None:
            if not self._caption_overlay.text() or self._caption_overlay.text().startswith(
                "Listening"
            ):
                self._caption_overlay.show_caption(message)
                self._reposition_caption_overlay()

    _AI_SITES = (
        ("ChatGPT", "https://chatgpt.com"),
        ("Claude", "https://claude.ai"),
        ("Gemini", "https://gemini.google.com"),
    )

    def _build_ai_browser_panel(self) -> QFrame:
        """Embedded Chromium panel to the right of preview (hideable; scanned for replies)."""
        frame = QFrame()
        frame.setObjectName("aiBrowserFrame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel(_("AI CHAT"))
        title.setObjectName("panelTitle")
        header.addWidget(title)
        header.addStretch(1)

        self._ai_site_buttons: dict[str, QPushButton] = {}
        for label, url in self._AI_SITES:
            btn = QPushButton(label)
            btn.setObjectName("aiBrowserNav")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, u=url: self._open_ai_site(u))
            header.addWidget(btn)
            self._ai_site_buttons[url] = btn

        reload_btn = QPushButton(_("Reload"))
        reload_btn.setObjectName("aiBrowserNav")
        reload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reload_btn.clicked.connect(self._reload_ai_browser)
        header.addWidget(reload_btn)

        hide_btn = QPushButton(_("Hide"))
        hide_btn.setObjectName("aiBrowserNav")
        hide_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        hide_btn.setToolTip(_("Hide the AI browser panel"))
        hide_btn.clicked.connect(self._hide_ai_browser)
        header.addWidget(hide_btn)

        layout.addLayout(header)

        hint = QLabel(
            _("Log in here. Replies are scanned and shown on the preview (not this page chrome).")
        )
        hint.setObjectName("statusLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._ai_browser = None
        self._ai_browser_error = None
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
            from PySide6.QtWebEngineWidgets import QWebEngineView

            from modules.paths import BROWSER_PROFILE_DIR

            profile_dir = BROWSER_PROFILE_DIR
            os.makedirs(profile_dir, exist_ok=True)
            profile = QWebEngineProfile("ai-chat", frame)
            self._ai_browser_profile = profile
            profile.setPersistentStoragePath(profile_dir)
            profile.setCachePath(os.path.join(profile_dir, "cache"))
            profile.setPersistentCookiesPolicy(
                QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies
            )
            # Persist scroll helper across navigations (ChatGPT SPA included).
            try:
                from PySide6.QtWebEngineCore import QWebEngineScript

                scroll_script = QWebEngineScript()
                scroll_script.setName("dlc-ai-scroll-helper")
                scroll_script.setSourceCode(_AI_SCROLL_HELPER_JS)
                scroll_script.setInjectionPoint(
                    QWebEngineScript.InjectionPoint.DocumentReady
                )
                scroll_script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
                scroll_script.setRunsOnSubFrames(True)
                profile.scripts().insert(scroll_script)
            except Exception as exc:
                print(f"[ai-scroll] profile script inject failed: {exc}", flush=True)

            self._ai_browser = QWebEngineView(frame)
            page = QWebEnginePage(profile, self._ai_browser)
            self._ai_browser.setPage(page)
            self._ai_browser.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            layout.addWidget(self._ai_browser, 1)
            self._ai_scan_installed = False
            self._ai_browser_parked = False
            page.loadFinished.connect(self._on_ai_browser_load_finished)
            if hasattr(page, "lifecycleStateChanged"):
                page.lifecycleStateChanged.connect(self._on_ai_page_lifecycle_changed)

            start_url = getattr(
                modules.globals, "ai_browser_url", "https://chatgpt.com"
            ) or "https://chatgpt.com"
            self._set_ai_site_active(start_url)
            self._ai_browser.setUrl(QUrl(start_url))
            # Off-screen host used while the panel is "hidden" so Chromium
            # keeps running JS/DOM (zero-size / setVisible(False) freezes it).
            self._ensure_ai_browser_keeper()
        except Exception as exc:
            self._ai_browser_error = QLabel(
                _(
                    "Embedded browser unavailable ({error}).\n"
                    "Install with: pip install PySide6-WebEngine"
                ).format(error=exc)
            )
            self._ai_browser_error.setObjectName("statusLabel")
            self._ai_browser_error.setWordWrap(True)
            self._ai_browser_error.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._ai_browser_error, 1)

        return frame

    def _ensure_ai_browser_keeper(self) -> QWidget:
        """Always-visible off-screen host so WebEngine stays Active while hidden."""
        keeper = getattr(self, "_ai_browser_keeper", None)
        if keeper is not None:
            return keeper
        keeper = QWidget(self)
        keeper.setObjectName("aiBrowserKeeper")
        # Real size matters — Chromium freezes views with 0×0 geometry.
        keeper.setFixedSize(480, 720)
        keeper.move(-20000, -20000)
        keeper.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        keeper.show()
        self._ai_browser_keeper = keeper
        return keeper

    def _park_ai_browser_offscreen(self) -> None:
        """Move the WebEngine out of the collapsed panel into the off-screen host."""
        browser = getattr(self, "_ai_browser", None)
        if browser is None:
            return
        keeper = self._ensure_ai_browser_keeper()
        panel = getattr(self, "_ai_browser_panel", None)
        if panel is not None:
            lay = panel.layout()
            if lay is not None:
                lay.removeWidget(browser)
        browser.setParent(keeper)
        browser.setGeometry(0, 0, keeper.width(), keeper.height())
        browser.show()
        self._ai_browser_parked = True
        self._keep_ai_browser_alive()

    def _restore_ai_browser_to_panel(self) -> None:
        """Put the WebEngine back into the visible AI chat panel."""
        browser = getattr(self, "_ai_browser", None)
        panel = getattr(self, "_ai_browser_panel", None)
        if browser is None or panel is None:
            return
        if browser.parentWidget() is panel and not getattr(
            self, "_ai_browser_parked", False
        ):
            return
        lay = panel.layout()
        browser.setParent(panel)
        if lay is not None:
            # Avoid duplicate inserts if still in the layout.
            lay.removeWidget(browser)
            lay.addWidget(browser, 1)
        browser.show()
        self._ai_browser_parked = False
        self._keep_ai_browser_alive()

    def _apply_ai_browser_visibility(self, visible: bool) -> None:
        modules.globals.ai_browser_visible = bool(visible)
        panel = getattr(self, "_ai_browser_panel", None)
        strip = getattr(self, "_btn_show_ai_strip", None)

        if visible:
            if panel is not None:
                panel.setMinimumWidth(AI_PANEL_MIN_WIDTH)
                panel.setMaximumWidth(16777215)
                panel.show()
            self._restore_ai_browser_to_panel()
            if strip is not None:
                strip.hide()
        else:
            # Park browser off-screen FIRST (real size, still shown), then hide
            # the panel chrome. Hiding/zero-sizing the WebEngine freezes scan JS.
            self._park_ai_browser_offscreen()
            if panel is not None:
                panel.hide()
            if strip is not None:
                strip.show()

        self._keep_ai_browser_alive()
        QTimer.singleShot(0, self._layout_preview_chat_sizes)
        # Center caption/AI display area resizes when the browser shows/hides.
        QTimer.singleShot(0, self._reposition_caption_overlay)
        # Keep scanning immediately after hide/show transitions.
        QTimer.singleShot(50, self._scan_ai_browser_response)

    def _keep_ai_browser_alive(self) -> None:
        """Prevent Chromium from freezing the parked/collapsed AI page."""
        browser = getattr(self, "_ai_browser", None)
        if browser is None:
            return
        try:
            browser.show()
            if getattr(self, "_ai_browser_parked", False):
                keeper = getattr(self, "_ai_browser_keeper", None)
                if keeper is not None:
                    keeper.show()
                    browser.setGeometry(0, 0, keeper.width(), keeper.height())
        except Exception:
            pass
        page = browser.page()
        if page is None:
            return
        try:
            from PySide6.QtWebEngineCore import QWebEnginePage

            if hasattr(page, "setLifecycleState"):
                page.setLifecycleState(QWebEnginePage.LifecycleState.Active)
        except Exception:
            pass

    def _on_ai_page_lifecycle_changed(self, state: Any) -> None:
        """Force Active whenever Chromium tries to freeze the parked page."""
        browser = getattr(self, "_ai_browser", None)
        if browser is None:
            return
        page = browser.page()
        if page is None:
            return
        try:
            from PySide6.QtWebEngineCore import QWebEnginePage

            if state != QWebEnginePage.LifecycleState.Active:
                page.setLifecycleState(QWebEnginePage.LifecycleState.Active)
        except Exception:
            pass

    def _hide_ai_browser(self) -> None:
        if getattr(modules.globals, "code_test_mode", False):
            return
        if not getattr(modules.globals, "ai_browser_visible", True):
            return
        self._apply_ai_browser_visibility(False)
        save_switch_states()
        update_status("AI browser hidden — scanning continues; click Show AI to restore")

    def _show_ai_browser(self) -> None:
        if getattr(modules.globals, "ai_browser_visible", True):
            return
        self._apply_ai_browser_visibility(True)
        save_switch_states()
        update_status("AI browser shown")
        QTimer.singleShot(0, self._scan_ai_browser_response)

    def _toggle_ai_browser(self) -> None:
        """Legacy toggle helper (only Show AI strip + header Hide are exposed in UI)."""
        if getattr(modules.globals, "ai_browser_visible", True):
            self._hide_ai_browser()
        else:
            self._show_ai_browser()

    def _on_ai_browser_load_finished(self, ok: bool) -> None:
        self._ai_scan_installed = False
        if ok:
            self._keep_ai_browser_alive()
            self._install_ai_scroll_helper()
            self._install_ai_scan_observer()

    def _install_ai_scan_observer(self) -> None:
        browser = getattr(self, "_ai_browser", None)
        if browser is None:
            return
        page = browser.page()
        if page is None:
            return
        self._keep_ai_browser_alive()

        def _on_installed(result: Any) -> None:
            self._ai_scan_installed = True
            text = (result or "").strip() if isinstance(result, str) else ""
            if text:
                self._apply_ai_scan_text(text)

        try:
            page.runJavaScript(_AI_SCAN_INSTALL_JS, _on_installed)
        except Exception:
            pass

    def _start_ai_response_wait(self) -> None:
        """Show a waiting alarm until a new ChatGPT reply is scanned."""
        overlay = getattr(self, "_ai_response_overlay", None)
        self._ai_waiting_for_response = True
        self._ai_wait_baseline = (
            getattr(overlay, "_last_text", "") if overlay is not None else ""
        )
        self._ai_wait_dots = 0
        self._tick_ai_wait_alarm()
        if not self._ai_wait_timer.isActive():
            self._ai_wait_timer.start()
        update_status("Waiting for ChatGPT response…")

    def _tick_ai_wait_alarm(self) -> None:
        if not getattr(self, "_ai_waiting_for_response", False):
            self._ai_wait_timer.stop()
            return
        self._ai_wait_dots = (self._ai_wait_dots + 1) % 4
        dots = "." * self._ai_wait_dots
        message = _("Waiting for ChatGPT response") + dots
        overlay = getattr(self, "_ai_response_overlay", None)
        if overlay is not None:
            overlay.set_waiting(message)
            host = getattr(self, "_caption_host", None) or self._live_preview
            overlay.reposition(host)

    def _stop_ai_response_wait(self, arrived: bool = False) -> None:
        was_waiting = getattr(self, "_ai_waiting_for_response", False)
        self._ai_waiting_for_response = False
        self._ai_wait_timer.stop()
        overlay = getattr(self, "_ai_response_overlay", None)
        if overlay is not None:
            overlay.clear_waiting()
        if arrived and was_waiting:
            try:
                import winsound

                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                try:
                    from PySide6.QtWidgets import QApplication

                    QApplication.beep()
                except Exception:
                    pass
            update_status("ChatGPT response received")

    def _apply_ai_scan_text(self, text: str) -> None:
        text = (text or "").strip()
        overlay = getattr(self, "_ai_response_overlay", None)
        if overlay is None or not text:
            return

        if getattr(self, "_ai_waiting_for_response", False):
            baseline = getattr(self, "_ai_wait_baseline", "") or ""
            # Still the previous reply — keep waiting.
            if text == baseline:
                return
            # Ignore trivial shrinks of the old reply while streaming resets.
            if (
                baseline
                and len(text) < len(baseline)
                and baseline.startswith(text)
            ):
                return
            self._stop_ai_response_wait(arrived=True)

        overlay.set_response(text)
        host = getattr(self, "_caption_host", None) or self._live_preview
        overlay.reposition(host)

    def _scan_ai_browser_response(self) -> None:
        """Read assistant text from the embedded page and show it on the preview."""
        browser = getattr(self, "_ai_browser", None)
        if browser is None:
            return
        page = browser.page()
        if page is None:
            return
        if getattr(self, "_ai_scan_busy", False):
            return
        self._ai_scan_busy = True
        self._keep_ai_browser_alive()

        def _on_result(result: Any) -> None:
            self._ai_scan_busy = False
            text = (result or "").strip() if isinstance(result, str) else ""
            if text:
                self._apply_ai_scan_text(text)
            elif not getattr(self, "_ai_scan_installed", False):
                self._install_ai_scan_observer()

        try:
            # Idempotent: installs MutationObserver once, then re-extracts.
            page.runJavaScript(_AI_SCAN_INSTALL_JS, _on_result)
        except Exception:
            self._ai_scan_busy = False

    def _kick_ai_response_scan(self) -> None:
        """Poll aggressively right after a prompt is submitted."""
        for delay_ms in (50, 150, 300, 500, 800, 1200, 2000, 3000, 5000, 8000):
            QTimer.singleShot(delay_ms, self._scan_ai_browser_response)

    def _set_ai_site_active(self, url: str) -> None:
        for site_url, btn in getattr(self, "_ai_site_buttons", {}).items():
            btn.setProperty("active", "true" if site_url.rstrip("/") == url.rstrip("/") else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _open_ai_site(self, url: str) -> None:
        url = (url or "https://chatgpt.com").strip()
        modules.globals.ai_browser_url = url
        save_switch_states()
        self._set_ai_site_active(url)
        browser = getattr(self, "_ai_browser", None)
        if browser is not None:
            from PySide6.QtCore import QUrl

            browser.setUrl(QUrl(url))
            update_status(f"AI browser: {url}")
        # Ensure browser is visible when switching sites.
        if not getattr(modules.globals, "ai_browser_visible", True):
            self._apply_ai_browser_visibility(True)
            save_switch_states()

    def _reload_ai_browser(self) -> None:
        browser = getattr(self, "_ai_browser", None)
        if browser is not None:
            browser.reload()

    def _copy_captions_to_clipboard(self) -> None:
        """Send session notes and/or already-attached photos to the AI chat.

        Supports:
        - text only (notes field)
        - photos only (already pasted into the AI composer via Screenshot)
        - both text + photos
        """
        box = getattr(self, "_caption_history", None)
        text = box.toPlainText().strip() if box is not None else ""

        browser = getattr(self, "_ai_browser", None)
        if browser is None or browser.page() is None:
            update_status("AI browser is unavailable")
            return
        if not self._ensure_ai_browser_ready():
            return

        page = browser.page()
        self._btn_copy_captions.setEnabled(False)
        self._btn_copy_captions.setText(_("Sending…"))

        if text:
            # Always leave a clipboard copy as a fallback.
            QApplication.clipboard().setText(text)
            update_status("Sending text to AI chat…")
            self._fill_ai_prompt_then_submit(page, text)
            return

        # No notes text — allow photo-only send if the composer already has an attachment.
        update_status("No notes text — checking for attached photos…")
        self._submit_ai_prompt_allowing_attachments(page)

    def _composer_has_attachment_js(self) -> str:
        return r"""
(() => {
  try {
    const selectors = [
      '[data-testid="file-thumbnail"]',
      '[data-testid="image-thumbnail"]',
      'button[aria-label*="Remove" i]',
      'button[aria-label*="remove file" i]',
      'img[alt*="Uploaded" i]',
      'img[alt*="attachment" i]',
      '[class*="attachment"] img',
      'form[data-type="unified-composer"] img',
      'form img[src^="blob:"]',
      'form img[src^="data:"]'
    ];
    for (const selector of selectors) {
      const matches = Array.from(document.querySelectorAll(selector));
      const hit = matches.find((el) => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      });
      if (hit) return 'yes';
    }
    return 'no';
  } catch (e) {
    return 'err:' + (e && e.message ? e.message : e);
  }
})()
"""

    def _submit_ai_prompt_allowing_attachments(self, page: Any) -> None:
        """Submit when the composer has photos (and optionally no text)."""

        def _after_check(result: Any) -> None:
            status = result if isinstance(result, str) else ""
            if status == "yes":
                update_status("Photo attached — submitting…")
                self._submit_ai_prompt(page, attempts_left=16)
                return
            if status.startswith("err:"):
                self._finish_caption_send(False, status[4:])
                return
            self._finish_caption_send(
                False,
                "Nothing to send — add notes text and/or take a Screenshot first",
            )

        try:
            page.runJavaScript(self._composer_has_attachment_js(), _after_check)
        except Exception as exc:
            self._finish_caption_send(False, str(exc))

    def _fill_ai_prompt_then_submit(self, page: Any, text: str) -> None:
        prompt_json = json.dumps(text)
        fill_script = f"""
(() => {{
  try {{
    const text = {prompt_json};
    const selectors = [
      '#prompt-textarea',
      'div.ProseMirror[contenteditable="true"]',
      'textarea[data-testid]',
      'rich-textarea .ql-editor',
      '.ProseMirror[contenteditable="true"]',
      '[contenteditable="true"][role="textbox"]',
      'textarea',
      '[contenteditable="true"]'
    ];
    let input = null;
    for (const selector of selectors) {{
      const matches = Array.from(document.querySelectorAll(selector));
      input = matches.find((el) => {{
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0 && !el.disabled;
      }});
      if (input) break;
    }}
    if (!input) return 'err:AI input box was not found';

    input.focus();
    if (input instanceof HTMLTextAreaElement || input instanceof HTMLInputElement) {{
      const proto = input instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
      const existing = input.value || '';
      // Keep any existing content; append notes after a blank line when needed.
      const next = existing.trim()
        ? (existing.replace(/\\s+$/, '') + '\\n\\n' + text)
        : text;
      setter.call(input, next);
    }} else {{
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(input);
      selection.removeAllRanges();
      selection.addRange(range);
      // Move caret to end so we append rather than wipe attachments/UI chrome.
      selection.collapseToEnd();
      const existing = (input.innerText || input.textContent || '').trim();
      const toInsert = existing ? ('\\n\\n' + text) : text;
      const inserted = document.execCommand('insertText', false, toInsert);
      if (!inserted) {{
        const cur = (input.innerText || input.textContent || '').trim();
        input.textContent = cur ? (cur + '\\n\\n' + text) : text;
      }}
    }}
    input.dispatchEvent(new InputEvent('input', {{
      bubbles: true, inputType: 'insertText', data: text
    }}));
    input.dispatchEvent(new Event('change', {{bubbles: true}}));

    const current = (input.value || input.innerText || input.textContent || '').trim();
    return current ? 'ok' : 'err:Input stayed empty after fill';
  }} catch (e) {{
    return 'err:' + (e && e.message ? e.message : e);
  }}
}})()
"""

        def _after_fill(result: Any) -> None:
            status = result if isinstance(result, str) else ""
            if status == "ok":
                self._submit_ai_prompt(page, attempts_left=16)
            else:
                error = (
                    status[4:]
                    if status.startswith("err:")
                    else "Could not fill the AI input"
                )
                self._finish_caption_send(False, error)

        try:
            page.runJavaScript(fill_script, _after_fill)
        except Exception as exc:
            self._finish_caption_send(False, str(exc))

    def _submit_ai_prompt(self, page: Any, attempts_left: int = 12) -> None:
        submit_script = r"""
(() => {
  try {
    const selectors = [
      'button[data-testid="send-button"]',
      'button[aria-label="Send prompt"]',
      'button[aria-label="Send message"]',
      'button[aria-label*="Send" i]',
      'button[type="submit"]',
      'button.send-button',
      'button[class*="send-button"]',
      'button[aria-label*="Submit" i]'
    ];
    let button = null;
    for (const selector of selectors) {
      const matches = Array.from(document.querySelectorAll(selector));
      button = matches.find((el) => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      });
      if (button) break;
    }
    if (!button) {
      const matches = Array.from(document.querySelectorAll('button'));
      button = matches.find((el) => {
        const label = [
          el.innerText, el.getAttribute('aria-label'), el.title
        ].filter(Boolean).join(' ').toLowerCase();
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0 && /\b(send|submit)\b/.test(label);
      });
    }
    if (!button) return 'notfound';
    const disabled = button.disabled ||
      button.getAttribute('aria-disabled') === 'true';
    if (disabled) return 'disabled';
    const composer = button.closest('form') || document;
    const uploading = Array.from(composer.querySelectorAll(
      '[role="progressbar"], circle[stroke-dasharray], .animate-spin, [class*="spinner" i], [class*="loading" i]'
    )).some((el) => {
      const r = el.getBoundingClientRect();
      return r.width > 0 && r.height > 0;
    });
    if (uploading) return 'disabled';
    button.click();
    return 'clicked';
  } catch (e) {
    return 'err:' + (e && e.message ? e.message : e);
  }
})()
"""

        def _after_submit(result: Any) -> None:
            status = result if isinstance(result, str) else ""
            if status == "clicked":
                self._finish_caption_send(True)
                return
            if status in ("disabled", "notfound") and attempts_left > 0:
                # Wait for the site to enable the button, then retry.
                QTimer.singleShot(
                    300, lambda: self._submit_ai_prompt(page, attempts_left - 1)
                )
                return
            # Last resort: press Enter in the focused input.
            if attempts_left <= 0:
                self._submit_ai_prompt_via_enter(page)
                return
            error = (
                status[4:] if status.startswith("err:") else "Submit button unavailable"
            )
            self._finish_caption_send(False, error)

        try:
            page.runJavaScript(submit_script, _after_submit)
        except Exception as exc:
            self._finish_caption_send(False, str(exc))

    def _submit_ai_prompt_via_enter(self, page: Any) -> None:
        enter_script = r"""
(() => {
  try {
    const selectors = [
      '#prompt-textarea',
      'div.ProseMirror[contenteditable="true"]',
      'textarea',
      '[contenteditable="true"][role="textbox"]',
      '[contenteditable="true"]'
    ];
    let input = null;
    for (const selector of selectors) {
      const matches = Array.from(document.querySelectorAll(selector));
      input = matches.find((el) => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      });
      if (input) break;
    }
    if (!input) return 'notfound';
    input.focus();
    const opts = {
      key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true
    };
    input.dispatchEvent(new KeyboardEvent('keydown', opts));
    input.dispatchEvent(new KeyboardEvent('keypress', opts));
    input.dispatchEvent(new KeyboardEvent('keyup', opts));
    return 'entered';
  } catch (e) {
    return 'err:' + (e && e.message ? e.message : e);
  }
})()
"""

        def _after_enter(result: Any) -> None:
            status = result if isinstance(result, str) else ""
            if status == "entered":
                self._finish_caption_send(True)
            else:
                self._finish_caption_send(
                    False,
                    "Send button stayed disabled and Enter did not submit",
                )

        try:
            page.runJavaScript(enter_script, _after_enter)
        except Exception as exc:
            self._finish_caption_send(False, str(exc))

    def _finish_caption_send(self, success: bool, error: str = "") -> None:
        button = getattr(self, "_btn_copy_captions", None)
        if button is not None:
            button.setEnabled(True)
            button.setText(_("Send notes to AI"))
        if success:
            box = getattr(self, "_caption_history", None)
            had_text = bool(box is not None and box.toPlainText().strip())
            if had_text:
                self._clear_caption_history()
            self._start_ai_response_wait()
            self._kick_ai_response_scan()
            update_status(
                "Submitted to AI — waiting for response…"
                if had_text
                else "Photo submitted to AI — waiting for response…"
            )
        else:
            update_status(
                f"Send failed: {error}. "
                "Content was kept so you can retry."
            )

    def _build_ai_browser_settings_card(self) -> QGroupBox:
        card = QGroupBox(_("AI Browser"))
        form = QVBoxLayout(card)
        form.setSpacing(8)
        tip = QLabel(
            _(
                "Browser sits to the right of the preview (can be hidden). "
                "Log in with Google — no API key. Assistant replies are scanned "
                "and shown on the preview in caption style (bottom 3/4)."
            )
        )
        tip.setObjectName("statusLabel")
        tip.setWordWrap(True)
        form.addWidget(tip)

        form.addWidget(QLabel(_("Default site")))
        self._ai_default_site = QComboBox()
        current = getattr(modules.globals, "ai_browser_url", "https://chatgpt.com")
        for label, url in self._AI_SITES:
            self._ai_default_site.addItem(label, url)
        idx = self._ai_default_site.findData(current)
        if idx < 0:
            idx = 0
        self._ai_default_site.setCurrentIndex(idx)
        self._ai_default_site.currentIndexChanged.connect(
            self._on_ai_default_site_changed
        )
        form.addWidget(self._ai_default_site)
        return card

    def _on_ai_default_site_changed(self, _index: int) -> None:
        url = self._ai_default_site.currentData()
        if url:
            self._open_ai_site(url)

    def _populate_enhancer_choices(self) -> None:
        current = self.cb_enhancer.currentText() if hasattr(self, "cb_enhancer") else None
        self.cb_enhancer.blockSignals(True)
        self.cb_enhancer.clear()
        for label, _fp_key in available_enhancer_choices():
            self.cb_enhancer.addItem(label)
        initial = "None"
        for fp_key in ("face_enhancer", "face_enhancer_gpen512", "face_enhancer_gpen256"):
            if modules.globals.fp_ui.get(fp_key, False):
                label = enhancer_label_for_fp_key(fp_key)
                if label and self.cb_enhancer.findText(label) >= 0:
                    initial = label
                    break
        if current and self.cb_enhancer.findText(current) >= 0:
            initial = current
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.blockSignals(False)

    def _populate_swap_model_choices(self) -> None:
        current = modules.globals.swap_model_choice
        self.cb_swap_model.blockSignals(True)
        self.cb_swap_model.clear()
        selected_index = 0
        for index, (choice_id, label) in enumerate(available_swap_choices()):
            self.cb_swap_model.addItem(label, choice_id)
            if choice_id == current:
                selected_index = index
        if self.cb_swap_model.count() > 0:
            self.cb_swap_model.setCurrentIndex(selected_index)
        self.cb_swap_model.blockSignals(False)

    # ── sliders card ─────────────────────────────────────────────────────

    def _build_sliders_card(self) -> QGroupBox:
        card = QGroupBox(_("Refinement"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        def slider(min_v, max_v, default, denom, on_change):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(min_v * denom), int(max_v * denom))
            s.setValue(int(default * denom))
            s.valueChanged.connect(lambda iv: on_change(iv / denom))
            return s

        # Transparency
        grid.addWidget(QLabel(_("Transparency")), 0, 0)
        self.s_transparency = slider(0.0, 1.0, 1.0, 100, self._on_transparency_change)
        self.s_transparency.setToolTip(
            _("Blend between original and swapped face (0% = original, 100% = fully swapped)")
        )
        grid.addWidget(self.s_transparency, 0, 1)

        # Sharpness
        grid.addWidget(QLabel(_("Sharpness")), 1, 0)
        self.s_sharpness = slider(0.0, 5.0, 0.0, 10, self._on_sharpness_change)
        self.s_sharpness.setToolTip(_("Sharpen the enhanced face output"))
        grid.addWidget(self.s_sharpness, 1, 1)

        # Mouth mask — always starts at 0 (disabled) on launch
        grid.addWidget(QLabel(_("Mouth Mask")), 2, 0)
        self.s_mouth = slider(0.0, 100.0, 0.0, 1,
                              self._on_mouth_mask_change)
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)
        self.s_mouth.setToolTip(
            _("0 = use swapped mouth, 100 = expose original mouth to chin area")
        )
        grid.addWidget(self.s_mouth, 2, 1)

        # Window background / overall opacity
        opacity_row = QHBoxLayout()
        opacity_row.setSpacing(8)
        opacity_row.addWidget(QLabel(_("Window Opacity")))
        self.lbl_window_opacity = QLabel("100%")
        self.lbl_window_opacity.setObjectName("dockLabel")
        self.lbl_window_opacity.setMinimumWidth(40)
        self.lbl_window_opacity.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        opacity_row.addWidget(self.lbl_window_opacity)
        grid.addLayout(opacity_row, 3, 0)

        initial_opacity = float(
            getattr(modules.globals, "window_opacity", 1.0) or 1.0
        )
        self.s_window_opacity = slider(
            0.30, 1.0, initial_opacity, 100, self._on_window_opacity_change
        )
        self.s_window_opacity.setToolTip(
            _(
                "Opacity of the main window background (30%–100%). "
                "Lower values make the window more transparent."
            )
        )
        grid.addWidget(self.s_window_opacity, 3, 1)
        self.lbl_window_opacity.setText(f"{int(round(initial_opacity * 100))}%")
        return card

    def _apply_window_opacity(self, value: float) -> None:
        value = max(0.30, min(1.0, float(value)))
        modules.globals.window_opacity = value
        self.setWindowOpacity(value)

    def _on_window_opacity_change(self, value: float) -> None:
        self._apply_window_opacity(value)
        if getattr(self, "lbl_window_opacity", None) is not None:
            self.lbl_window_opacity.setText(f"{int(round(value * 100))}%")
        save_switch_states()

    def _on_import_face(self) -> None:
        kind = self._current_kind or "source"
        self._add_face(kind)

    def _on_add_face_header(self) -> None:
        kind = self._current_kind or "source"
        self._add_face(kind)

    def _refresh_face_library(
        self,
        selected_id: Optional[str] = None,
        selected_kind: Optional[str] = None,
    ) -> None:
        """Reload entries, restore active selection, and re-render card pages."""
        for kind in ("source", "target"):
            entries = face_library.list_faces(kind)
            self._face_entries[kind] = entries
            entry_ids = {entry["id"] for entry in entries}

            desired_id = None
            if selected_id in entry_ids and selected_kind in (None, kind):
                desired_id = selected_id
            if desired_id is None:
                desired_id = face_library.get_active(kind)

            if desired_id:
                match = next((e for e in entries if e["id"] == desired_id), None)
                if match:
                    # Jump the pager to the page holding the selected face.
                    per = self._cards_per_page(kind)
                    item_idx = entries.index(match) + 1
                    self._page_index[kind] = item_idx // max(1, per)
                    self._select_saved_face(kind, match, announce=False)

            self._render_face_page(kind)

        self._update_active_previews()

    def _highlight_active_face(self, kind: str, entry_id: Optional[str]) -> None:
        for card in self._face_cards.get(kind, []):
            card.set_selected(card.entry.get("id") == entry_id)

    def _select_saved_face(
        self, kind: str, entry: dict, announce: bool = True
    ) -> None:
        path = entry.get("path")
        if not path or not os.path.isfile(path) or not is_image(path):
            update_status("Saved face file is missing")
            self._refresh_face_library()
            return
        face_library.set_active(kind, entry["id"])
        self._highlight_active_face(kind, entry["id"])
        if kind == "source":
            if modules.globals.source_path != path:
                modules.globals.source_path = path
                modules.globals.source_revision += 1
            else:
                modules.globals.source_path = path
            if announce:
                if self.is_live():
                    update_status(
                        f"Switching live source to '{entry['name']}'..."
                    )
                else:
                    update_status(f"Src face: {entry['name']}")
        else:
            modules.globals.target_path = path
            if announce:
                update_status(f"Dst face: {entry['name']}")
        self._update_active_previews()

    def _add_face(self, kind: str) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        recent = (
            _RECENT_SOURCE_DIR if kind == "source" else _RECENT_TARGET_DIR
        )
        path, _filter = QFileDialog.getOpenFileName(
            self,
            _("Add source face") if kind == "source" else _("Add destination face"),
            recent or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp)",
        )
        if not path:
            return
        if not is_image(path):
            update_status("Please choose an image file")
            return
        if kind == "source":
            _RECENT_SOURCE_DIR = os.path.dirname(path)
        else:
            _RECENT_TARGET_DIR = os.path.dirname(path)

        image = imread_unicode(path)
        if image is None or get_one_face(image) is None:
            update_status("No face detected; image was not added")
            return

        default_name = os.path.splitext(os.path.basename(path))[0]
        name, accepted = QInputDialog.getText(
            self,
            _("Add face"),
            _("Name:"),
            text=default_name,
        )
        if not accepted:
            return
        try:
            entry = face_library.add_face(kind, name, path)
        except (OSError, ValueError) as exc:
            update_status(f"Could not add face: {exc}")
            return
        self._refresh_face_library(entry["id"], kind)
        update_status(f"Added {kind} face '{entry['name']}'")

    def _delete_saved_face(self, kind: str, entry: dict) -> None:
        reply = QMessageBox.question(
            self,
            _("Delete face"),
            _("Delete '{name}'?").format(name=entry.get("name", "")),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if not face_library.remove_face(entry["id"]):
            update_status("Could not delete saved face")
            return
        if kind == "source" and modules.globals.source_path == entry.get("path"):
            modules.globals.source_path = None
            modules.globals.source_revision += 1
        elif kind == "target" and modules.globals.target_path == entry.get("path"):
            modules.globals.target_path = None
        self._refresh_face_library()
        update_status(f"Deleted '{entry['name']}'")

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_enhancer_change(self, choice: str) -> None:
        fp_key = enhancer_fp_key_for_label(choice)
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            _update_tumbler(key, False)
        if fp_key:
            _update_tumbler(fp_key, True)
        else:
            save_switch_states()

    def _on_swap_model_change(self, index: int) -> None:
        if index < 0:
            return
        choice_id = self.cb_swap_model.itemData(index)
        if not choice_id:
            return
        modules.globals.swap_model_choice = choice_id
        save_switch_states()
        from modules.processors.frame.face_swapper import reload_face_swapper

        reload_face_swapper()
        if self.is_live():
            update_status(
                f"Changing live swap model to {self.cb_swap_model.currentText()}..."
            )
        else:
            update_status(f"Swap model: {self.cb_swap_model.currentText()}")

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        pct = int(value * 100)
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        update_status(f"Sharpness set to {value:.1f}")

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                "Images (*.png *.jpg *.jpeg *.bmp)",
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                "Videos (*.mp4 *.mkv)",
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()
            apply_tray_taskbar_hiding(_PREVIEW)
            QTimer.singleShot(0, lambda: apply_tray_taskbar_hiding(_PREVIEW))

    def _on_live(self) -> None:
        # Toggle: stop if already running face-swap live.
        if self.is_live():
            self._live_preview.stop()
            self._set_live_button(False)
            self.set_status_pill(_("Ready"), live=False)
            update_status("Live preview stopped")
            return

        if self.is_image_output_running():
            self._live_preview.stop()
            self._set_image_output_button(False)

        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        if not modules.globals.map_faces:
            if modules.globals.source_path is None:
                update_status("Please select a Src face first")
                return
            if modules.globals.opacity <= 0.0:
                update_status("Transparency is 0% — increase it to see the face swap")
                return

            self.btn_live.setEnabled(False)
            try:
                self._live_preview.show_progress(_("Loading source face…"), 5)
                update_status("Loading source face…")
                source_face = _load_source_face_for_live()
                if source_face is None:
                    self._live_preview.hide_progress()
                    return

                self._live_preview.show_progress(_("Loading face detector…"), 30)
                update_status("Loading face detector…")
                from modules.face_analyser import get_face_detector
                from modules.processors.frame.face_swapper import get_face_swapper
                get_face_detector()

                self._live_preview.show_progress(_("Loading swap model…"), 55)
                update_status("Loading swap model…")
                get_face_swapper()

                enhancer_on = any(
                    modules.globals.fp_ui.get(k, False)
                    for k in (
                        "face_enhancer",
                        "face_enhancer_gpen256",
                        "face_enhancer_gpen512",
                    )
                )
                if enhancer_on:
                    self._live_preview.show_progress(_("Loading enhancer model…"), 75)
                    update_status("Loading enhancer model…")
                    get_frame_processors_modules(modules.globals.frame_processors)

                self._live_preview.show_progress(_("Starting camera…"), 90)
                update_status("Starting camera…")
                started = self._live_preview.start(camera_index, source_face)
                if started:
                    self._live_preview.show_progress(_("Streaming…"), 100)
                    self._set_live_button(True)
                    self.set_status_pill(_("Live"), live=True)
                else:
                    self._live_preview.hide_progress()
            finally:
                self.btn_live.setEnabled(True)
        else:
            modules.globals.source_target_map = []
            _open_live_mapper_dialog(camera_index, modules.globals.source_target_map)

    def closeEvent(self, event):
        if (
            not self._force_quit
            and self._tray is not None
            and getattr(modules.globals, "minimize_to_tray", True)
        ):
            event.ignore()
            self.hide_to_tray()
            return
        if not self._force_quit:
            self.quit_app()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _MAIN is not None and _MAIN.is_live():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if getattr(modules.globals, "stealth_mode", False):
            set_window_stealth(self, True)
        apply_tray_taskbar_hiding(self)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(
                get_one_face(imread_unicode(modules.globals.source_path)), temp_frame
            )
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


# Balanced live profile: 20 FPS remains smooth for calls while reducing the
# number of detector/swap/enhancer inferences by ~17% versus 24 FPS.
LIVE_TARGET_FPS = 20.0


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow.

    Frames are enqueued at most at LIVE_TARGET_FPS; extra camera frames are
    dropped so the live pipeline never runs faster than the target rate.
    """

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event

    def run(self) -> None:
        min_interval = 1.0 / LIVE_TARGET_FPS
        next_emit = time.monotonic()
        while not self._stop.is_set():
            # Pace before reading. The old loop continuously decoded and
            # discarded camera frames while waiting for the next emit slot.
            remaining = next_emit - time.monotonic()
            if remaining > 0 and self._stop.wait(remaining):
                break
            ret, frame = self._cap.read()
            if not ret:
                self._stop.set()
                break
            next_emit = max(next_emit + min_interval, time.monotonic())
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass


class _StaticImageWorker(QThread):
    """Enqueue a letterboxed still image at LIVE_TARGET_FPS for virtual cam output."""

    def __init__(
        self,
        frame: np.ndarray,
        processed_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        super().__init__()
        self._frame = frame
        self._queue = processed_queue
        self._stop = stop_event

    def run(self) -> None:
        min_interval = 1.0 / LIVE_TARGET_FPS
        next_emit = time.monotonic()
        while not self._stop.is_set():
            remaining = next_emit - time.monotonic()
            if remaining > 0 and self._stop.wait(remaining):
                break
            next_emit = max(next_emit + min_interval, time.monotonic())
            try:
                self._queue.put_nowait(self._frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(self._frame)
                except queue.Full:
                    pass


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float, source_face=None):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps
        self._source_face = source_face
        self._last_source_path = modules.globals.source_path if source_face else None
        self._source_revision = modules.globals.source_revision
        self._no_face_warned = False

    def run(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = self._source_face
        last_source_path = self._last_source_path
        source_revision = self._source_revision
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        # Detect around 6–7 times/sec and reuse the bbox/keypoints between
        # detections. Swap inference still runs every output frame.
        det_interval = max(1, round(self._fps * 0.15))
        det_count = max(0, det_interval - 1)
        cached_target_face = None
        cached_many_faces = None

        while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            temp_frame = frame
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            if not modules.globals.map_faces:
                selected_path = modules.globals.source_path
                selected_revision = modules.globals.source_revision
                if (
                    selected_revision != source_revision
                    or selected_path != last_source_path
                ):
                    source_revision = selected_revision
                    last_source_path = selected_path
                    if not selected_path:
                        source_image = None
                        self._source_face = None
                    else:
                        candidate = get_one_face(imread_unicode(selected_path))
                        if candidate is None:
                            update_status(
                                "No face found in new source image; keeping current live face"
                            )
                        else:
                            source_image = candidate
                            self._source_face = candidate
                            update_status("Live face changed")

                if source_image is None:
                    if not self._no_face_warned:
                        update_status("Live swap waiting for a valid source face")
                        self._no_face_warned = True
                    continue
                self._no_face_warned = False

                det_count += 1
                if det_count % det_interval == 0:
                    if modules.globals.many_faces:
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                    else:
                        cached_target_face = detect_one_face_fast(temp_frame)
                        cached_many_faces = None

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                # Fast detection skips the 2d106 landmark model, but the mouth
                # mask needs it. Attach landmarks on demand (computed once per
                # detection cycle — the helper no-ops if already present).
                if modules.globals.mouth_mask and cached_faces:
                    ensure_landmarks(temp_frame, cached_faces)

                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                result = fp.swap_face(source_image, t_face, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None:
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)
            else:
                modules.globals.target_path = None
                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame_v2(temp_frame)
                    elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                        fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                        if modules.globals.fp_ui.get(fp_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if modules.globals.show_fps:
                cv2.putText(
                    temp_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            try:
                self._pq.put_nowait(temp_frame)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait(temp_frame)
                except queue.Full:
                    pass


class LivePreviewPanel(QFrame):
    """Embedded live camera preview — sits above the floating dock like a spectrum bar."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("livePreviewPanel")
        self.setMinimumHeight(LIVE_PREVIEW_HEIGHT)
        # Expanding so the preview can grow to ~3/4 of the window height.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._badge = QLabel("\u25cf  " + _("PREVIEW"))
        self._badge.setObjectName("livePreviewBadge")
        header.addWidget(self._badge)
        header.addStretch(1)
        self._hint = QLabel(_("Select a source face, then hit Go Live"))
        self._hint.setObjectName("livePreviewHint")
        header.addWidget(self._hint)
        layout.addLayout(header)

        self._image_label = QLabel(_("Camera idle"))
        self._image_label.setObjectName("livePreviewVideo")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumHeight(LIVE_PREVIEW_HEIGHT - 40)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._image_label, 1)

        # Loading progress (hidden until Go Live triggers model/image loading).
        self._progress_stage = QLabel("")
        self._progress_stage.setObjectName("loadStage")
        self._progress_stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_stage.setVisible(False)
        layout.addWidget(self._progress_stage)

        self._progress = QProgressBar()
        self._progress.setObjectName("loadProgress")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._stop_event: Optional[threading.Event] = None
        self._capture_worker = None
        self._processing_worker = None
        self._timer: Optional[QTimer] = None
        self._cap = None
        self._capture_queue: Optional[queue.Queue] = None
        self._processed_queue: Optional[queue.Queue] = None
        self._static_worker = None
        self._virtual_cam_device: Optional[str] = None
        self._output_mode: Optional[str] = None
        self._running = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep caption overlay stacked above the video when the panel resizes.
        parent = self.window()
        if parent is not None and hasattr(parent, "_reposition_caption_overlay"):
            parent._reposition_caption_overlay()

    def is_running(self) -> bool:
        return self._running

    def is_face_swap_live(self) -> bool:
        return self._running and self._output_mode == "face_swap"

    def is_static_output(self) -> bool:
        return self._running and self._output_mode == "static_image"

    def show_progress(self, stage: str, percent: int) -> None:
        """Show/update the loading progress bar and stage label."""
        self._image_label.setText("")
        self._progress_stage.setVisible(True)
        self._progress.setVisible(True)
        self._progress_stage.setText(stage)
        self._progress.setValue(max(0, min(100, percent)))
        self._progress.setFormat(f"{percent}%")
        if _APP is not None:
            _APP.processEvents()

    def hide_progress(self) -> None:
        self._progress_stage.setVisible(False)
        self._progress.setVisible(False)
        self._progress.setValue(0)

    def start(self, camera_index: int, source_face=None) -> bool:
        if self._running:
            self.stop()
            if _APP is not None:
                for _i in range(10):
                    _APP.processEvents()
            time.sleep(0.35)

        self._stop_event = threading.Event()
        self._cap = VideoCapturer(camera_index)
        if not self._cap.start():
            self._cap.release()
            self._cap = None
            update_status(
                "Camera is in use. Close Google Meet/Zoom/browser tabs using the "
                "webcam, then click Live again."
            )
            return False

        camera_fps = (
            min(self._cap.actual_fps, LIVE_TARGET_FPS)
            if self._cap.actual_fps
            else LIVE_TARGET_FPS
        )
        print(
            f"[webcam] Using camera resolution {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{self._cap.actual_fps:.0f}fps, "
            f"live pipeline capped at {camera_fps:.0f}fps"
        )

        self._capture_queue = queue.Queue(maxsize=2)
        self._processed_queue = queue.Queue(maxsize=2)
        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue,
            self._processed_queue,
            self._stop_event,
            camera_fps,
            source_face,
        )
        self._capture_worker.start()
        self._processing_worker.start()

        try:
            self._virtual_cam_device = virtual_camera.start(
                self._cap.actual_width,
                self._cap.actual_height,
                camera_fps,
            )
            update_status(
                f"Virtual camera on: {self._virtual_cam_device}. "
                f"In Google Meet → Settings → Video, pick '{self._virtual_cam_device}' "
                "(not your physical webcam)."
            )
            print(f"[virtual_camera] publishing to {self._virtual_cam_device}", flush=True)
        except Exception as exc:
            print(f"[virtual_camera] failed to start: {exc}", flush=True)
            update_status(
                "Virtual camera unavailable. Install OBS Studio, then restart Live."
            )

        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

        self._output_mode = "face_swap"
        self._running = True
        self._badge.setText("\u25cf  " + _("LIVE"))
        self._badge.setStyleSheet("color: #6ccb5f;")
        self._hint.setText(_("Streaming to virtual camera"))
        self._image_label.setText(_("Starting camera…"))
        return True

    def start_static_image(self, image_path: str, width: int, height: int) -> bool:
        if self._running:
            self.stop()
            if _APP is not None:
                for _i in range(10):
                    _APP.processEvents()
            time.sleep(0.35)

        image = imread_unicode(image_path)
        if image is None:
            update_status("Could not load image — try another file")
            return False

        frame = letterbox_image_to_size(image, int(width), int(height))
        self._stop_event = threading.Event()
        self._processed_queue = queue.Queue(maxsize=2)
        self._static_worker = _StaticImageWorker(
            frame, self._processed_queue, self._stop_event
        )
        self._static_worker.start()

        fps = LIVE_TARGET_FPS
        try:
            self._virtual_cam_device = virtual_camera.start(int(width), int(height), fps)
            update_status(
                f"Image output on: {self._virtual_cam_device}. "
                f"Pick '{self._virtual_cam_device}' in your meeting app."
            )
            print(f"[virtual_camera] static image → {self._virtual_cam_device}", flush=True)
        except Exception as exc:
            print(f"[virtual_camera] failed to start: {exc}", flush=True)
            update_status(
                "Virtual camera unavailable. Install OBS Studio, then try again."
            )

        poll_ms = max(1, min(16, int(500 / max(fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

        self._output_mode = "static_image"
        self._running = True
        self._badge.setText("\u25cf  " + _("IMAGE OUTPUT"))
        self._badge.setStyleSheet("color: #c9a84c;")
        self._hint.setText(_("Static image → virtual camera"))
        self._image_label.setText("")
        w = max(1, self._image_label.width())
        h = max(1, self._image_label.height())
        preview = fit_image_to_size(frame, w, h)
        self._image_label.setPixmap(_bgr_to_qpixmap(preview))
        return True

    def stop(self) -> None:
        if not self._running and self._stop_event is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
            self._timer = None
        for worker in (self._capture_worker, self._processing_worker, self._static_worker):
            if worker is None:
                continue
            try:
                worker.wait(2000)
            except Exception:
                pass
        self._capture_worker = None
        self._processing_worker = None
        self._static_worker = None
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        try:
            virtual_camera.stop()
        except Exception:
            pass
        self._virtual_cam_device = None
        self._output_mode = None
        self._running = False
        self._stop_event = None
        self._badge.setText("\u25cf  " + _("PREVIEW"))
        self._badge.setStyleSheet("")
        self._hint.setText(_("Select a source face, then hit Go Live"))
        self.hide_progress()
        self._image_label.clear()
        self._image_label.setText(_("Camera idle"))

    def _tick(self) -> None:
        if self._stop_event is None or self._stop_event.is_set():
            if self._running:
                was_static = self._output_mode == "static_image"
                self.stop()
                if _MAIN is not None:
                    _MAIN._set_live_button(False)
                    if was_static and hasattr(_MAIN, "_set_image_output_button"):
                        _MAIN._set_image_output_button(False)
                    _MAIN.set_status_pill(_("Ready"), live=False)
            return
        try:
            bgr_frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return
        # If rendering fell behind, discard stale processed frames and publish
        # the newest one. This lowers latency and avoids needless UI work.
        while True:
            try:
                bgr_frame = self._processed_queue.get_nowait()
            except queue.Empty:
                break
        # First live frame arrived — clear the loading progress UI.
        if self._progress.isVisible():
            self.hide_progress()
        if virtual_camera.is_active():
            virtual_camera.send_bgr(bgr_frame)
        w = max(1, self._image_label.width())
        h = max(1, self._image_label.height())
        bgr_frame = fit_image_to_size(bgr_frame, w, h)
        self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))
        # Camera-only panel — caption/AI overlays live on the center display host.


def _load_source_face_for_live() -> Any:
    """Load and validate the source face before starting live preview."""
    if not modules.globals.source_path:
        update_status("Please select a source image first")
        return None
    source_img = imread_unicode(modules.globals.source_path)
    if source_img is None:
        update_status("Could not load source image — try another file")
        return None
    source_face = get_one_face(source_img)
    if source_face is None:
        update_status("No face found in source image — use a clear front-facing photo")
        return None
    if not hasattr(source_face, "normed_embedding") or source_face.normed_embedding is None:
        update_status("Source face embedding failed — try a different source image")
        return None
    return source_face


def _open_webcam_preview(camera_index: int, source_face=None) -> None:
    """Start embedded live preview on the main window (mapper / legacy entry)."""
    if _MAIN is None:
        return
    if _MAIN._live_preview.is_running():
        _MAIN._live_preview.stop()
        _MAIN._set_live_button(False)
        if hasattr(_MAIN, "_set_image_output_button"):
            _MAIN._set_image_output_button(False)
        if _APP is not None:
            for _i in range(10):
                _APP.processEvents()
        time.sleep(0.35)
    if _MAIN._live_preview.start(camera_index, source_face):
        _MAIN._set_live_button(True)
        _MAIN.set_status_pill(_("Live"), live=True)


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


def _populate_saved_face_combo(combo: QComboBox, kind: str) -> None:
    combo.addItem(
        _("Saved source face") if kind == "source" else _("Saved target face"),
        None,
    )
    for entry in face_library.list_faces(kind):
        combo.addItem(entry["name"], entry)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if getattr(modules.globals, "stealth_mode", False):
            set_window_stealth(self, True)
        apply_tray_taskbar_hiding(self)

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            source_controls = QWidget()
            source_layout = QVBoxLayout(source_controls)
            source_layout.setContentsMargins(0, 0, 0, 0)
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            source_layout.addWidget(btn)
            saved_combo = QComboBox()
            _populate_saved_face_combo(saved_combo, "source")
            saved_combo.currentIndexChanged.connect(
                lambda index, n=row, combo=saved_combo:
                self._select_saved_source(n, combo, index)
            )
            source_layout.addWidget(saved_combo)
            grid.addWidget(source_controls, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        if not path:
            return
        self._set_source_path(row, path)

    def _select_saved_source(
        self, row: int, combo: QComboBox, index: int
    ) -> None:
        entry = combo.itemData(index)
        if entry:
            self._set_source_path(row, entry["path"])

    def _set_source_path(self, row: int, path: str) -> None:
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if getattr(modules.globals, "stealth_mode", False):
            set_window_stealth(self, True)
        apply_tray_taskbar_hiding(self)

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            source_controls = QWidget()
            source_layout = QVBoxLayout(source_controls)
            source_layout.setContentsMargins(0, 0, 0, 0)
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            source_layout.addWidget(btn_s)
            saved_source = QComboBox()
            _populate_saved_face_combo(saved_source, "source")
            saved_source.currentIndexChanged.connect(
                lambda index, n=row, combo=saved_source:
                self._select_saved_face(n, "source", combo, index)
            )
            source_layout.addWidget(saved_source)
            grid.addWidget(source_controls, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            target_controls = QWidget()
            target_layout = QVBoxLayout(target_controls)
            target_layout.setContentsMargins(0, 0, 0, 0)
            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            target_layout.addWidget(btn_t)
            saved_target = QComboBox()
            _populate_saved_face_combo(saved_target, "target")
            saved_target.currentIndexChanged.connect(
                lambda index, n=row, combo=saved_target:
                self._select_saved_face(n, "target", combo, index)
            )
            target_layout.addWidget(saved_target)
            grid.addWidget(target_controls, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        if not path:
            return
        self._set_face_path(row, kind, path)

    def _select_saved_face(
        self,
        row: int,
        kind: str,
        combo: QComboBox,
        index: int,
    ) -> None:
        entry = combo.itemData(index)
        if entry:
            self._set_face_path(row, kind, entry["path"])

    def _set_face_path(self, row: int, kind: str, path: str) -> None:
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()
    apply_tray_taskbar_hiding(_MAPPER)
    QTimer.singleShot(0, lambda: apply_tray_taskbar_hiding(_MAPPER))


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()
    apply_tray_taskbar_hiding(_LIVE_MAPPER)
    QTimer.singleShot(0, lambda: apply_tray_taskbar_hiding(_LIVE_MAPPER))


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._main.raise_()
        self._main.activateWindow()
        apply_tray_taskbar_hiding(self._main)
        QTimer.singleShot(0, lambda: apply_tray_taskbar_hiding(self._main))
        # Ensure stealth is cleared first so a previous WDA_MONITOR session
        # cannot leave the window blank.
        set_window_stealth(self._main, False)
        if getattr(modules.globals, "stealth_mode", False):
            set_window_stealth(self._main, True)
        print(
            f"Window ready ({self._main.width()}x{self._main.height()})",
            flush=True,
        )
        self._app.exec()


def _apply_windows_app_identity(app: QApplication) -> None:
    """Use a custom taskbar icon instead of the default python.exe icon."""
    icon_path = os.path.join(ROOT_DIR, "media", "Download.png")
    if os.path.isfile(icon_path):
        icon = QIcon(icon_path)
        app.setWindowIcon(icon)
    if platform.system() != "Windows":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "SwapfaceLive.SwapfaceLive.1"
        )
        if os.path.isfile(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)

    # Qt WebEngine must be imported and ShareOpenGLContexts set before QApplication.
    if QApplication.instance() is None:
        QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True
        )
        try:
            import PySide6.QtWebEngineWidgets  # noqa: F401
        except ImportError:
            pass
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    if QSystemTrayIcon.isSystemTrayAvailable():
        _APP.setQuitOnLastWindowClosed(False)
    _apply_windows_app_identity(_APP)
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    _MAIN = MainWindow(start, destroy)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)

    return _Window(_APP, _MAIN)
