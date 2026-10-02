"""Live speech-to-text captions from system audio (Windows WASAPI loopback).

Uses faster-whisper (OpenAI Whisper `small.en`) for punctuation, casing, and
accuracy closer to Windows Live Captions — streaming via rolling utterance
windows rather than waiting on multi-second cloud round-trips.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal


TARGET_RATE = 16000
# Emit live overlay updates as soon as recognition text changes (~20 Hz cap).
MIN_EMIT_INTERVAL = 0.04
# Read ~30 ms of loopback audio per iteration for snappy capture.
CAPTURE_SECONDS = 0.03
DEVICE_CHECK_INTERVAL = 1.0
# If a device switch is not reflected by PortAudio's live enumeration, the old
# endpoint produces digital silence. Reopen the default endpoint after a short
# silent period, but not repeatedly more often than this interval.
SILENT_RECONNECT_AFTER = 2.0
SILENT_RECONNECT_INTERVAL = 6.0
OVERLAY_MAX_CHARS = 320  # unused legacy; overlay manages visible length via lines

# Whisper small.en — best live-caption balance (accuracy + punctuation + latency).
# Larger than vosk-small; far closer to Windows Live Captions quality.
WHISPER_MODEL_NAME = "small.en"
WHISPER_MODEL_DIR_NAME = "faster-whisper-small.en"
WHISPER_HF_ID = "Systran/faster-whisper-small.en"

# Streaming utterance window.
PARTIAL_INTERVAL = 0.55
MIN_PARTIAL_SECONDS = 0.55
SILENCE_TO_FINAL = 0.70
SPEECH_RMS = 0.010
MAX_UTTERANCE_SECONDS = 26.0
# Drop Whisper hallucinations on near-silence / hold music.
NO_SPEECH_THRESHOLD = 0.65
COMPRESSION_RATIO_THRESHOLD = 2.4


def _models_root() -> Path:
    from modules.paths import MODELS_DIR

    return Path(MODELS_DIR)


def _bundled_model_dir() -> Path:
    return _models_root() / WHISPER_MODEL_DIR_NAME


def _find_local_whisper_dir() -> Optional[Path]:
    """Return a directory containing model.bin if one is already on disk."""
    bundled = _bundled_model_dir()
    if (bundled / "model.bin").is_file():
        return bundled

    root = _models_root()
    for model_bin in root.glob(
        "models--Systran--faster-whisper-small.en/snapshots/*/model.bin"
    ):
        return model_bin.parent
    return None


def captions_available() -> tuple[bool, str]:
    """Return (ok, message) describing whether STT deps are installed."""
    try:
        import pyaudiowpatch  # noqa: F401
    except ImportError:
        return False, "Install PyAudioWPatch (pip install PyAudioWPatch)"
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False, "Install faster-whisper (pip install faster-whisper)"
    return True, "ok"


def ensure_whisper_model(status_cb=None) -> tuple[str, str]:
    """Load path/id for WhisperModel. Prefer bundled files; download only in source.

    Returns (model_path_or_id, download_root).
    """
    import sys

    local = _find_local_whisper_dir()
    if local is not None:
        return str(local), str(_models_root())

    if getattr(sys, "frozen", False):
        raise RuntimeError(
            f"Bundled caption model missing: {_bundled_model_dir()}. "
            f"Redeploy with models/{WHISPER_MODEL_DIR_NAME} included."
        )

    root = _models_root()
    root.mkdir(parents=True, exist_ok=True)
    if status_cb:
        status_cb("Downloading Whisper caption model (one-time)…")
    # WhisperModel will fetch into download_root on first construct.
    return WHISPER_MODEL_NAME, str(root)


def _pick_whisper_device() -> tuple[str, str]:
    """Prefer CUDA float16; fall back to CPU int8."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _load_whisper_model(status_cb=None):
    from faster_whisper import WhisperModel

    model_ref, download_root = ensure_whisper_model(status_cb=status_cb)
    device, compute_type = _pick_whisper_device()
    if status_cb:
        status_cb(f"Loading Whisper {WHISPER_MODEL_NAME} ({device}/{compute_type})…")
    try:
        return WhisperModel(
            model_ref,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
    except Exception:
        if device == "cuda":
            if status_cb:
                status_cb("CUDA Whisper failed — falling back to CPU…")
            return WhisperModel(
                model_ref,
                device="cpu",
                compute_type="int8",
                download_root=download_root,
            )
        raise


def _to_mono_int16(frames: bytes, channels: int) -> np.ndarray:
    """Downmix interleaved int16 PCM to mono."""
    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels <= 1:
        return pcm.copy()
    usable = (len(pcm) // channels) * channels
    pcm = pcm[:usable].reshape(-1, channels)
    return pcm.mean(axis=1).astype(np.int16)


def _resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Cheap linear resample for STT (quality is fine for speech)."""
    if src_rate == dst_rate or len(pcm) == 0:
        return pcm
    duration = len(pcm) / float(src_rate)
    n_out = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm.astype(np.float32)).astype(np.int16)


def _pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def _find_wasapi_host(pa: Any) -> dict:
    for i in range(pa.get_host_api_count()):
        info = pa.get_host_api_info_by_index(i)
        if "WASAPI" in str(info.get("name", "")).upper():
            return info
    raise RuntimeError("WASAPI host API not available")


def _resolve_system_loopback(pa: Any) -> tuple[dict, dict]:
    """Resolve the current Windows default playback endpoint and its loopback."""
    wasapi_info = _find_wasapi_host(pa)
    default_speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    loopback = None
    if not default_speakers.get("isLoopbackDevice"):
        speaker_name = str(default_speakers.get("name", ""))
        for loop in pa.get_loopback_device_info_generator():
            if speaker_name and speaker_name in str(loop.get("name", "")):
                loopback = loop
                break
    else:
        loopback = default_speakers

    if loopback is None:
        loopback = pa.get_default_wasapi_loopback()

    sample_rate = int(loopback["defaultSampleRate"])
    channels = max(1, int(loopback["maxInputChannels"]))
    frames_per_buffer = max(256, int(sample_rate * CAPTURE_SECONDS))

    meta = {
        "name": str(loopback.get("name", "System audio")),
        "output_name": str(default_speakers.get("name", "")),
        "device_index": int(loopback["index"]),
        "sample_rate": sample_rate,
        "channels": channels,
        "frames_per_buffer": frames_per_buffer,
    }
    return loopback, meta


def _open_system_loopback(pa: Any) -> tuple[Any, dict]:
    """Open a WASAPI loopback stream on the current default playback device."""
    loopback, meta = _resolve_system_loopback(pa)
    stream = pa.open(
        format=pa.get_format_from_width(2),  # 16-bit PCM
        channels=meta["channels"],
        rate=meta["sample_rate"],
        input=True,
        input_device_index=meta["device_index"],
        frames_per_buffer=meta["frames_per_buffer"],
    )
    return stream, meta


def _device_key(meta: dict) -> tuple:
    """Stable identity used to detect Windows default-output changes."""
    return (
        meta.get("output_name", ""),
        meta.get("name", ""),
        int(meta.get("device_index", -1)),
        int(meta.get("sample_rate", 0)),
        int(meta.get("channels", 0)),
    )


def _overlay_text(committed: str, partial: str) -> str:
    """Join committed + partial without left-truncation (keeps line breaks stable)."""
    return f"{committed} {partial}".strip()


def _trim_committed(committed: str) -> str:
    """Drop oldest whole words only — never mid-string chops that break prefixes."""
    words = committed.split()
    if len(words) <= 120:
        return committed
    return " ".join(words[-120:])


def _transcribe_window(model: Any, audio: np.ndarray) -> str:
    """Transcribe a mono float32 16 kHz window; return cleaned text."""
    if audio.size < int(TARGET_RATE * 0.25):
        return ""
    # Skip near-silent windows — Whisper invents text on silence.
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms < SPEECH_RMS * 0.45:
        return ""

    segments, _info = model.transcribe(
        audio,
        language="en",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=350,
            speech_pad_ms=120,
        ),
        condition_on_previous_text=False,
        without_timestamps=True,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
    )
    parts: list[str] = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


class LiveCaptionWorker(QThread):
    """Background system-audio → streaming Whisper recognition loop."""

    # Live overlay text (partials + committed). Fired many times per second.
    captionReady = Signal(str)
    # Current unfinalized phrase, emitted immediately for the history box.
    captionPartial = Signal(str)
    # Finalized phrase for the history panel.
    captionFinal = Signal(str)
    # New speech after silence — overlay should accept a fresh shorter string.
    utteranceBegan = Signal()
    # Fired after recognizer state is wiped (e.g. post-submit history clear).
    historyReset = Signal()
    statusChanged = Signal(str)
    errorOccurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False
        self._reset_requested = False

    def stop(self) -> None:
        self._stop = True

    def request_reset(self) -> None:
        """Drop in-flight utterance so history can start fresh immediately."""
        self._reset_requested = True

    def run(self) -> None:
        ok, msg = captions_available()
        if not ok:
            self.errorOccurred.emit(msg)
            return

        import pyaudiowpatch as pyaudio

        try:
            model = _load_whisper_model(
                status_cb=lambda t: self.statusChanged.emit(t)
            )
        except Exception as exc:
            self.errorOccurred.emit(f"Caption model error: {exc}")
            return

        pa = pyaudio.PyAudio()
        stream: Optional[Any] = None
        try:
            stream, meta = _open_system_loopback(pa)
            self.statusChanged.emit(
                f"Listening (Whisper {WHISPER_MODEL_NAME}): {meta['name']}"
            )
        except Exception as exc:
            pa.terminate()
            self.errorOccurred.emit(f"No system audio loopback: {exc}")
            return

        sample_rate = int(meta["sample_rate"])
        channels = int(meta["channels"])
        frames_per_buffer = int(meta["frames_per_buffer"])

        committed = ""
        partial = ""
        last_emit = 0.0
        last_overlay = ""
        pending_text = ""
        last_speech_at = 0.0
        fresh_utterance = True
        last_device_check = time.monotonic()
        last_audio_activity = time.monotonic()
        last_silent_reconnect = time.monotonic()
        active_device_key = _device_key(meta)

        # Rolling utterance buffer (float32 mono @ 16 kHz).
        utterance: deque[np.ndarray] = deque()
        utterance_samples = 0
        in_speech = False
        last_partial_at = 0.0
        max_samples = int(TARGET_RATE * MAX_UTTERANCE_SECONDS)

        def clear_utterance() -> None:
            nonlocal utterance_samples, in_speech, last_partial_at
            utterance.clear()
            utterance_samples = 0
            in_speech = False
            last_partial_at = 0.0

        def append_audio(chunk: np.ndarray) -> None:
            nonlocal utterance_samples
            if chunk.size == 0:
                return
            utterance.append(chunk)
            utterance_samples += int(chunk.size)
            while utterance_samples > max_samples and utterance:
                dropped = utterance.popleft()
                utterance_samples -= int(dropped.size)

        def utterance_audio() -> np.ndarray:
            if not utterance:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(list(utterance))

        def reset_recognizer_state() -> None:
            nonlocal committed, partial, pending_text, last_overlay
            nonlocal fresh_utterance, last_speech_at
            clear_utterance()
            committed = ""
            partial = ""
            pending_text = ""
            last_overlay = ""
            last_speech_at = 0.0
            fresh_utterance = True
            self.captionPartial.emit("")
            self.utteranceBegan.emit()

        def reopen_default_output(reason: str = "") -> None:
            nonlocal pa, stream, meta, sample_rate, channels
            nonlocal frames_per_buffer, active_device_key, last_device_check
            nonlocal last_audio_activity, last_silent_reconnect
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                pa.terminate()
            except Exception:
                pass

            pa = pyaudio.PyAudio()
            stream, meta = _open_system_loopback(pa)
            sample_rate = int(meta["sample_rate"])
            channels = int(meta["channels"])
            frames_per_buffer = int(meta["frames_per_buffer"])
            active_device_key = _device_key(meta)
            last_device_check = time.monotonic()
            last_audio_activity = last_device_check
            last_silent_reconnect = last_device_check
            reset_recognizer_state()
            prefix = "Output changed — " if reason == "device_changed" else ""
            self.statusChanged.emit(
                f"{prefix}Listening (Whisper {WHISPER_MODEL_NAME}): {meta['name']}"
            )

        def emit_overlay(force: bool = False) -> None:
            nonlocal last_emit, last_overlay, pending_text
            text = _overlay_text(committed, partial)
            if last_overlay and text and not fresh_utterance:
                old_w = last_overlay.split()
                new_w = text.split()
                if len(new_w) < len(old_w) and old_w[: len(new_w)] == new_w:
                    return
            if not text and last_overlay:
                return
            pending_text = text
            now = time.monotonic()
            if text == last_overlay and not force:
                return
            if not force and (now - last_emit) < MIN_EMIT_INTERVAL:
                return
            last_emit = now
            last_overlay = text or last_overlay
            if text:
                self.captionReady.emit(text)

        def finalize_utterance() -> None:
            nonlocal committed, partial, last_speech_at, fresh_utterance
            audio = utterance_audio()
            clear_utterance()
            phrase = _transcribe_window(model, audio) if audio.size else ""
            partial = ""
            self.captionPartial.emit("")
            if phrase:
                committed = _trim_committed(f"{committed} {phrase}".strip())
                self.captionFinal.emit(phrase)
                last_speech_at = time.monotonic()
                fresh_utterance = False
            emit_overlay(force=True)

        def refresh_partial(force: bool = False) -> None:
            nonlocal partial, last_partial_at, last_overlay, fresh_utterance
            nonlocal last_speech_at
            now = time.monotonic()
            if not force and (now - last_partial_at) < PARTIAL_INTERVAL:
                return
            if utterance_samples < int(TARGET_RATE * MIN_PARTIAL_SECONDS):
                return
            last_partial_at = now
            text = _transcribe_window(model, utterance_audio())
            if text == partial:
                return
            if text:
                restarted = not partial
                paused = bool(last_speech_at) and (now - last_speech_at) >= 0.35
                if fresh_utterance or restarted or paused:
                    last_overlay = ""
                    fresh_utterance = False
                    self.utteranceBegan.emit()
                last_speech_at = now
            partial = text
            self.captionPartial.emit(partial)
            emit_overlay(force=True)

        try:
            while not self._stop:
                try:
                    if self._reset_requested:
                        self._reset_requested = False
                        reset_recognizer_state()
                        self.historyReset.emit()

                    now = time.monotonic()
                    if (now - last_device_check) >= DEVICE_CHECK_INTERVAL:
                        last_device_check = now
                        _loopback, current_meta = _resolve_system_loopback(pa)
                        if _device_key(current_meta) != active_device_key:
                            reopen_default_output("device_changed")

                    if (
                        pending_text
                        and pending_text != last_overlay
                        and (time.monotonic() - last_emit) >= MIN_EMIT_INTERVAL
                    ):
                        emit_overlay(force=True)

                    data = stream.read(frames_per_buffer, exception_on_overflow=False)
                    pcm = _to_mono_int16(data, channels)
                    now = time.monotonic()
                    peak = (
                        int(np.max(np.abs(pcm.astype(np.int32))))
                        if pcm.size
                        else 0
                    )
                    if peak > 8:
                        last_audio_activity = now
                    elif (
                        (now - last_audio_activity) >= SILENT_RECONNECT_AFTER
                        and (now - last_silent_reconnect)
                        >= SILENT_RECONNECT_INTERVAL
                    ):
                        last_silent_reconnect = now
                        reopen_default_output("silent_endpoint")
                        continue

                    pcm16k = _resample_linear(pcm, sample_rate, TARGET_RATE)
                    audio_f = _pcm16_to_float32(pcm16k)
                    rms = (
                        float(np.sqrt(np.mean(np.square(audio_f))))
                        if audio_f.size
                        else 0.0
                    )
                    speech = rms >= SPEECH_RMS

                    if speech:
                        if not in_speech:
                            # New utterance after silence.
                            if utterance_samples and partial:
                                finalize_utterance()
                            in_speech = True
                            if fresh_utterance or not partial:
                                last_overlay = ""
                                self.utteranceBegan.emit()
                                fresh_utterance = False
                        append_audio(audio_f)
                        last_speech_at = now
                        refresh_partial()
                    else:
                        if in_speech:
                            append_audio(audio_f)
                            if last_speech_at and (
                                now - last_speech_at
                            ) >= SILENCE_TO_FINAL:
                                finalize_utterance()
                        elif utterance_samples:
                            # Trailing silence already finalized elsewhere.
                            clear_utterance()
                except Exception as exc:
                    if self._stop:
                        break
                    self.statusChanged.emit(
                        f"Caption audio reconnecting: {exc}"
                    )
                    time.sleep(0.35)
                    try:
                        reopen_default_output("stream_error")
                    except Exception as reopen_exc:
                        self.errorOccurred.emit(
                            f"Caption audio unavailable: {reopen_exc}"
                        )
                        time.sleep(0.75)
        finally:
            try:
                if utterance_samples:
                    finalize_utterance()
            except Exception:
                pass
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                pa.terminate()
            except Exception:
                pass
            self.statusChanged.emit("Captions stopped")
