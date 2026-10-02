"""Global hotkey registration (Windows) with in-app fallback."""

from __future__ import annotations

import sys
from typing import Callable, Optional

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

# Reserve IDs 1..99 for app actions.
_HOTKEY_ID_BASE = 1


def default_hotkeys() -> dict[str, str]:
    return {
        "screenshot": "Ctrl+Meta+R",
        "send_notes_to_ai": "Ctrl+Meta+Enter",
        "toggle_always_on_top": "",
        "focus_window": "Ctrl+Meta+F",
        "focus_session_notes": "",
        # Avoid Ctrl+Meta+Arrow — Windows often refuses Ctrl+Win+Down.
        "ai_chat_scroll_up": "Ctrl+Alt+Up",
        "ai_chat_scroll_down": "Ctrl+Alt+Down",
    }


def _norm_part(part: str) -> str:
    p = part.strip().lower()
    aliases = {
        "control": "ctrl",
        "command": "meta",
        "cmd": "meta",
        "win": "meta",
        "windows": "meta",
        "option": "alt",
    }
    return aliases.get(p, p)


def sequence_to_parts(seq: str) -> tuple[int, int] | None:
    """Parse a shortcut string into (modifiers, virtual_key) for RegisterHotKey."""
    text = (seq or "").strip()
    if not text:
        return None

    parts = [_norm_part(p) for p in text.replace("+", " ").split() if p.strip()]
    if len(parts) < 2:
        return None

    mods = 0
    key_token = parts[-1]
    for part in parts[:-1]:
        if part == "ctrl":
            mods |= MOD_CONTROL
        elif part == "alt":
            mods |= MOD_ALT
        elif part == "shift":
            mods |= MOD_SHIFT
        elif part in {"meta", "win", "super"}:
            mods |= MOD_WIN
        else:
            return None

    if len(key_token) == 1 and key_token.isalnum():
        return mods, ord(key_token.upper())

    if key_token.startswith("f") and key_token[1:].isdigit():
        n = int(key_token[1:])
        if 1 <= n <= 24:
            return mods, 0x6F + n

    named = {
        "space": 0x20,
        "return": 0x0D,
        "enter": 0x0D,
        "escape": 0x1B,
        "esc": 0x1B,
        "tab": 0x09,
        "backspace": 0x08,
        "delete": 0x2E,
        "insert": 0x2D,
        "home": 0x24,
        "end": 0x23,
        "pageup": 0x21,
        "pagedown": 0x22,
        "left": 0x25,
        "up": 0x26,
        "right": 0x27,
        "down": 0x28,
        "print": 0x2A,
        "pause": 0x13,
    }
    vk = named.get(key_token)
    if vk is not None:
        return mods, vk

    # Fall back to Qt's parser for uncommon keys.
    try:
        qseq = QKeySequence(text)
        if qseq.isEmpty():
            return None
        combo = qseq[0]
        key = int(combo.key())
        qt_mods = combo.keyboardModifiers()
        if qt_mods & Qt.KeyboardModifier.AltModifier:
            mods |= MOD_ALT
        if qt_mods & Qt.KeyboardModifier.ControlModifier:
            mods |= MOD_CONTROL
        if qt_mods & Qt.KeyboardModifier.ShiftModifier:
            mods |= MOD_SHIFT
        if qt_mods & Qt.KeyboardModifier.MetaModifier:
            mods |= MOD_WIN
        vk = _qt_key_to_vk(key)
        if vk is None:
            return None
        return mods, vk
    except Exception:
        return None


def _qt_key_to_vk(key: int) -> int | None:
    if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
        return ord(chr(key).upper())
    if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
        return ord(chr(key))
    if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F24:
        return 0x70 + (key - int(Qt.Key.Key_F1))

    special = {
        int(Qt.Key.Key_Space): 0x20,
        int(Qt.Key.Key_Return): 0x0D,
        int(Qt.Key.Key_Enter): 0x0D,
        int(Qt.Key.Key_Escape): 0x1B,
        int(Qt.Key.Key_Tab): 0x09,
        int(Qt.Key.Key_Backspace): 0x08,
        int(Qt.Key.Key_Delete): 0x2E,
        int(Qt.Key.Key_Insert): 0x2D,
        int(Qt.Key.Key_Home): 0x24,
        int(Qt.Key.Key_End): 0x23,
        int(Qt.Key.Key_PageUp): 0x21,
        int(Qt.Key.Key_PageDown): 0x22,
        int(Qt.Key.Key_Left): 0x25,
        int(Qt.Key.Key_Up): 0x26,
        int(Qt.Key.Key_Right): 0x27,
        int(Qt.Key.Key_Down): 0x28,
        int(Qt.Key.Key_Print): 0x2A,
        int(Qt.Key.Key_Pause): 0x13,
    }
    return special.get(key)


class _WinHotkeyFilter(QAbstractNativeEventFilter):
    def __init__(self, on_hotkey: Callable[[int], None]):
        super().__init__()
        self._on_hotkey = on_hotkey

    def nativeEventFilter(self, event_type, message):
        if sys.platform != "win32":
            return False, 0
        if event_type != b"windows_generic_MSG":
            return False, 0
        try:
            import ctypes
            from ctypes import wintypes

            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY:
                self._on_hotkey(int(msg.wParam))
                return True, 0
        except Exception:
            pass
        return False, 0


class HotkeyCaptureButton(QPushButton):
    """Click, then press a key combo to assign a shortcut."""

    hotkeyChanged = Signal(str)

    def __init__(self, sequence: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._sequence = (sequence or "").strip()
        self._recording = False
        self._refresh_label()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._start_recording)

    def sequence(self) -> str:
        return self._sequence

    def set_sequence(self, sequence: str) -> None:
        self._sequence = (sequence or "").strip()
        self._refresh_label()

    def _refresh_label(self) -> None:
        if self._recording:
            self.setText("Press shortcut…")
            return
        self.setText(self._sequence or _("(none)"))

    def _start_recording(self) -> None:
        self._recording = True
        self._refresh_label()
        self.grabKeyboard()
        self.setFocus()

    def keyPressEvent(self, event) -> None:
        if not self._recording:
            super().keyPressEvent(event)
            return

        key = event.key()
        if key in (
            Qt.Key.Key_Control,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
        ):
            return

        if key in (Qt.Key.Key_Escape, Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self._sequence = ""
            self._stop_recording()
            self.hotkeyChanged.emit("")
            return

        mods = event.modifiers()
        parts: list[str] = []
        if mods & Qt.KeyboardModifier.ControlModifier:
            parts.append("Ctrl")
        if mods & Qt.KeyboardModifier.AltModifier:
            parts.append("Alt")
        if mods & Qt.KeyboardModifier.ShiftModifier:
            parts.append("Shift")
        if mods & Qt.KeyboardModifier.MetaModifier:
            parts.append("Meta")

        key_text = QKeySequence(key).toString(QKeySequence.SequenceFormat.PortableText)
        if key_text:
            parts.append(key_text)

        if len(parts) < 2:
            return

        self._sequence = "+".join(parts)
        self._stop_recording()
        self.hotkeyChanged.emit(self._sequence)

    def focusOutEvent(self, event) -> None:
        if self._recording:
            self._stop_recording()
        super().focusOutEvent(event)

    def _stop_recording(self) -> None:
        self._recording = False
        self.releaseKeyboard()
        self._refresh_label()


def _(text: str) -> str:
    try:
        from modules.gettext import LanguageManager

        return LanguageManager().translate(text)
    except Exception:
        return text


class GlobalHotkeyManager(QObject):
    """Register global hotkeys on Windows; in-app QShortcut elsewhere."""

    actionTriggered = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._action_for_id: dict[int, str] = {}
        self._id_for_action: dict[str, int] = {}
        self._next_id = _HOTKEY_ID_BASE
        self._filter: _WinHotkeyFilter | None = None
        self._shortcuts: list[QShortcut] = []
        self._host: QWidget | None = None

    def set_host(self, host: QWidget) -> None:
        self._host = host

    def clear(self) -> None:
        if sys.platform == "win32":
            import ctypes

            user32 = ctypes.windll.user32
            for hotkey_id in list(self._action_for_id):
                user32.UnregisterHotKey(None, hotkey_id)
        self._action_for_id.clear()
        self._id_for_action.clear()
        for sc in self._shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._shortcuts.clear()
        if self._filter is not None:
            app = QApplication.instance()
            if app is not None:
                app.removeNativeEventFilter(self._filter)
            self._filter = None

    def apply_bindings(self, bindings: dict[str, str]) -> list[str]:
        """Apply action→sequence map. Returns human-readable errors."""
        self.clear()
        errors: list[str] = []
        for action, seq in bindings.items():
            seq = (seq or "").strip()
            if not seq:
                continue
            err = self._register_one(action, seq)
            if err:
                errors.append(f"{action}: {err}")
        return errors

    def _register_one(self, action: str, seq: str) -> str | None:
        parsed = sequence_to_parts(seq)
        if parsed is None:
            return f"invalid shortcut '{seq}'"

        if sys.platform == "win32":
            import ctypes

            user32 = ctypes.windll.user32
            hotkey_id = self._next_id
            self._next_id += 1
            mods, vk = parsed
            if not user32.RegisterHotKey(None, hotkey_id, mods, vk):
                return f"could not register '{seq}' (already used?)"
            self._action_for_id[hotkey_id] = action
            self._id_for_action[action] = hotkey_id
            if self._filter is None:
                self._filter = _WinHotkeyFilter(self._on_native_hotkey)
                app = QApplication.instance()
                if app is not None:
                    app.installNativeEventFilter(self._filter)
            return None

        if self._host is None:
            return "no host window for in-app shortcut"
        shortcut = QShortcut(QKeySequence(seq), self._host)
        shortcut.activated.connect(lambda a=action: self.actionTriggered.emit(a))
        self._shortcuts.append(shortcut)
        self._id_for_action[action] = -1
        return None

    def _on_native_hotkey(self, hotkey_id: int) -> None:
        action = self._action_for_id.get(hotkey_id)
        if action:
            self.actionTriggered.emit(action)
