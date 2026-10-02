import cv2
import numpy as np
import time
from typing import Optional, Tuple, Callable, List
import platform
import threading

# Only import Windows-specific library if on Windows
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

# Common modes tried (largest first) when DirectShow format list is incomplete.
_CANDIDATE_RESOLUTIONS: List[Tuple[int, int]] = [
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1600, 1200),
    (1280, 960),
    (1280, 720),
    (1024, 768),
    (960, 720),
    (960, 540),
    (800, 600),
    (640, 480),
]


class VideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.frame_callback = None
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.is_running = False
        self.cap = None
        # Actual values reported by the camera after configuration
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0

        # Verify device exists without holding a DirectShow lock.
        if platform.system() == "Windows":
            devices = FilterGraph().get_input_devices()
            if self.device_index >= len(devices):
                raise ValueError(
                    f"Invalid device index {device_index}. Available devices: {len(devices)}"
                )

    def _probe_native_resolution(self) -> Optional[Tuple[int, int, Optional[str]]]:
        """Return (width, height, fourcc) for the largest format this camera offers.

        Important: the DirectShow graph must be fully released before OpenCV
        opens the same device, or every CAP_DSHOW open fails.
        """
        if platform.system() != "Windows":
            return None

        graph = None
        best = None
        best_area = -1
        format_count = 0
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if self.device_index >= len(devices):
                return None
            graph.add_video_input_device(self.device_index)
            formats = graph.get_input_device().get_formats() or []
            format_count = len(formats)

            for fmt in formats:
                try:
                    w = int(fmt.get("width", 0))
                    h = int(fmt.get("height", 0))
                except Exception:
                    continue
                if w <= 0 or h <= 0:
                    continue
                area = w * h
                media = str(fmt.get("media_type_str", "")).upper()
                # Prefer MJPG at the same size (USB bandwidth), else keep current.
                better = area > best_area or (
                    area == best_area
                    and best is not None
                    and best[2] != "MJPG"
                    and media == "MJPG"
                )
                if better:
                    fourcc = "MJPG" if media == "MJPG" else (media if media else None)
                    best = (w, h, fourcc)
                    best_area = area
        except Exception as exc:
            print(f"[VideoCapturer] format probe failed: {exc}", flush=True)
            best = None
        finally:
            if graph is not None:
                try:
                    graph.stop()
                except Exception:
                    pass
                try:
                    graph.remove_filters()
                except Exception:
                    pass
                try:
                    del graph
                except Exception:
                    pass
            # Give Windows a beat to free the DirectShow pin before OpenCV opens.
            time.sleep(0.2)

        if best:
            print(
                f"[VideoCapturer] probed native mode {best[0]}x{best[1]} "
                f"({best[2] or 'default'}) from {format_count} format(s)",
                flush=True,
            )
        return best

    def _open_capture(
        self,
        width: Optional[int],
        height: Optional[int],
        fourcc: Optional[str],
        fps: Optional[int],
    ) -> Optional[cv2.VideoCapture]:
        open_params: List[int] = []
        if fourcc and len(fourcc) == 4:
            open_params.extend([cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)])
        if width is not None:
            open_params.extend([cv2.CAP_PROP_FRAME_WIDTH, int(width)])
        if height is not None:
            open_params.extend([cv2.CAP_PROP_FRAME_HEIGHT, int(height)])
        # Do not force FPS here — many drivers downgrade resolution when FPS
        # is set at open time. Live mode throttles FPS in software.

        # Indices come from DirectShow (pygrabber), so prefer CAP_DSHOW.
        backends = (
            [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            if platform.system() == "Windows"
            else [cv2.CAP_ANY]
        )
        for backend in backends:
            try:
                if open_params:
                    cap = cv2.VideoCapture(self.device_index, backend, open_params)
                else:
                    cap = cv2.VideoCapture(self.device_index, backend)
            except Exception:
                continue
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                continue
            # Reinforce size for backends that honor post-open sets.
            if fourcc and len(fourcc) == 4:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            return cap
        return None

    def _read_size(self, cap: cv2.VideoCapture) -> Optional[Tuple[int, int]]:
        ret, frame = cap.read()
        if ret and frame is not None and frame.size:
            h, w = frame.shape[:2]
            return w, h
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            return w, h
        return None

    def start(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> bool:
        """Initialize and start video capture at the camera's native resolution.

        If width/height are omitted, probe DirectShow formats and open the
        largest available mode. FPS requests are applied in software by the
        live pipeline — not forced on the camera driver.
        """
        try:
            target_w, target_h = width, height
            fourcc: Optional[str] = None

            if target_w is None or target_h is None:
                probed = self._probe_native_resolution()
                if probed:
                    target_w, target_h, fourcc = probed

            # Prefer the probed native mode, then a few safe fallbacks.
            # Avoid dozens of open attempts — each failed DSHOW open is noisy
            # and can leave the device in a bad state.
            attempts: List[Tuple[Optional[int], Optional[int], Optional[str]]] = []
            seen = set()

            def _add(w, h, fcc):
                key = (w, h, fcc)
                if key in seen:
                    return
                seen.add(key)
                attempts.append((w, h, fcc))

            if target_w and target_h:
                if fourcc == "MJPG" or fourcc is None:
                    _add(target_w, target_h, "MJPG")
                _add(target_w, target_h, fourcc)
                _add(target_w, target_h, None)
            for w, h in ((1280, 720), (640, 480)):
                _add(w, h, "MJPG")
                _add(w, h, None)
            _add(None, None, None)

            best_cap = None
            best_size = None
            best_area = -1
            for w, h, fcc in attempts:
                cap = self._open_capture(w, h, fcc, fps)
                if cap is None:
                    continue
                size = self._read_size(cap)
                if not size:
                    cap.release()
                    continue
                got_w, got_h = size
                area = got_w * got_h
                if area > best_area:
                    if best_cap is not None:
                        best_cap.release()
                    best_cap = cap
                    best_size = (got_w, got_h)
                    best_area = area
                    # Requested this size and got it — good enough.
                    if w and h and abs(got_w - w) <= 8 and abs(got_h - h) <= 8:
                        break
                else:
                    cap.release()

            if best_cap is None or best_size is None:
                raise RuntimeError("Failed to open camera")

            self.cap = best_cap
            self.actual_width, self.actual_height = best_size

            reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
            fallback_fps = float(fps) if fps is not None else (reported_fps or 30.0)
            self.actual_fps = self._measure_fps(
                warmup=5, sample=20, fallback=fallback_fps
            )

            print(
                f"[VideoCapturer] {self.actual_width}x{self.actual_height} "
                f"@ {self.actual_fps:.1f}fps (reported={reported_fps:.0f})",
                flush=True,
            )

            self.is_running = True
            return True

        except Exception as e:
            print(f"Failed to start capture: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera"""
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self._current_frame = frame
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None

    def release(self) -> None:
        """Stop capture and release resources"""
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _measure_fps(self, warmup: int = 10, sample: int = 30,
                     fallback: float = 30.0) -> float:
        """Read warmup+sample frames and return measured FPS.

        This is more reliable than CAP_PROP_FPS which often lies on
        DirectShow.  Takes ~0.5-1s at startup but gives a ground-truth
        number for adaptive polling/detection intervals.
        """
        try:
            for _ in range(warmup):
                self.cap.read()
            t0 = time.perf_counter()
            for _ in range(sample):
                ret, _ = self.cap.read()
                if not ret:
                    return fallback
            elapsed = time.perf_counter() - t0
            if elapsed <= 0:
                return fallback
            return sample / elapsed
        except Exception:
            return fallback

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing"""
        self.frame_callback = callback
