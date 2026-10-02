# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
source_revision: int = 0  # Incremented whenever the live source selection changes
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False      # Enable Poisson Blending for smoother face swaps
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False
# Exclude app windows from screen capture / share (Windows SetWindowDisplayAffinity).
# Default OFF — on some NVIDIA + Windows 11 setups exclude-from-capture can make
# the window appear blank/invisible. Enable from Settings → Stealth mode.
stealth_mode: bool = False
# Live speech-to-text captions overlay (system audio → streaming Vosk).
live_captions: bool = True
# Embedded AI browser (ChatGPT / Claude / Gemini web UI — no API key).
ai_browser_url: str = "https://chatgpt.com"
ai_browser_visible: bool = True
# Main window opacity (1.0 = fully opaque, lower = more see-through).
window_opacity: float = 1.0
# Keep the main window above other apps (even when not focused).
always_on_top: bool = False
# Minimal UI for coding: captions + AI browser only.
code_test_mode: bool = False
# Face-swap controls panel collapsed (tabs stay visible).
controls_collapsed: bool = False
# Close button hides to system tray instead of quitting.
minimize_to_tray: bool = True
# Static image output (separate from source/target face library).
image_output_path: str | None = None
image_output_width: int = 1280
image_output_height: int = 720
# Global / in-app hotkeys (action name → QKeySequence string).
hotkeys: Dict[str, str] = {
    "screenshot": "Ctrl+Meta+R",
    "send_notes_to_ai": "Ctrl+Meta+Enter",
    "toggle_always_on_top": "",
    "focus_window": "Ctrl+Meta+F",
    "focus_session_notes": "",
    # Avoid Ctrl+Meta+Arrow — Windows often refuses Ctrl+Win+Down.
    "ai_chat_scroll_up": "Ctrl+Alt+Up",
    "ai_chat_scroll_down": "Ctrl+Alt+Down",
}

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}

# Face Swapper Specific Options
swap_model_choice: str = "auto"  # auto | fp16 | fp32
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()
