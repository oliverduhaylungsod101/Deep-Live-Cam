"""Download URLs and helpers for ONNX models used by the app."""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from modules.paths import MODELS_DIR
from modules.utilities import conditional_download

HF_BASE = "https://huggingface.co/hacksider/deep-live-cam/resolve/main"
GPEN_BASE = "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models"

SWAP_MODELS: Dict[str, Dict[str, str]] = {
    "auto": {"label": "Auto (FP16 on CUDA)", "file": ""},
    "fp16": {
        "label": "InSwapper 128 FP16 (fast)",
        "file": "inswapper_128_fp16.onnx",
        "url": f"{HF_BASE}/inswapper_128_fp16.onnx",
    },
    "fp32": {
        "label": "InSwapper 128 FP32",
        "file": "inswapper_128.onnx",
        "url": f"{HF_BASE}/inswapper_128.onnx",
    },
}

ENHANCER_MODELS: Dict[str, Dict[str, str]] = {
    "gfpgan": {
        "label": "GFPGAN",
        "file": "gfpgan-1024.onnx",
        "url": f"{HF_BASE}/gfpgan-1024.onnx",
        "fp_key": "face_enhancer",
    },
    "gpen512": {
        "label": "GPEN-512",
        "file": "GPEN-BFR-512.onnx",
        "url": f"{GPEN_BASE}/GPEN-BFR-512.onnx",
        "fp_key": "face_enhancer_gpen512",
    },
    "gpen256": {
        "label": "GPEN-256",
        "file": "GPEN-BFR-256.onnx",
        "url": f"{GPEN_BASE}/GPEN-BFR-256.onnx",
        "fp_key": "face_enhancer_gpen256",
    },
}


def model_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def is_model_present(filename: str) -> bool:
    return bool(filename) and os.path.isfile(model_path(filename))


def ensure_app_models(
    update_status: Optional[Callable[[str], None]] = None,
) -> bool:
    """Ensure swap/enhancer ONNX models exist under models/.

    Source builds may download missing files. Frozen deploys must ship them.
    """
    import sys

    os.makedirs(MODELS_DIR, exist_ok=True)
    urls: List[str] = []
    missing: List[str] = []
    for key, info in SWAP_MODELS.items():
        if key == "auto":
            continue
        if not is_model_present(info["file"]):
            urls.append(info["url"])
            missing.append(info["file"])
    for info in ENHANCER_MODELS.values():
        if not is_model_present(info["file"]):
            urls.append(info["url"])
            missing.append(info["file"])
    if not urls:
        return True
    if getattr(sys, "frozen", False):
        msg = (
            "Missing bundled model(s): "
            + ", ".join(missing)
            + f". Place them in {MODELS_DIR}."
        )
        if update_status:
            update_status(msg)
        raise FileNotFoundError(msg)
    if update_status:
        update_status(f"Downloading {len(urls)} missing model(s)...")
    conditional_download(MODELS_DIR, urls)
    return True


def available_swap_choices() -> List[Tuple[str, str]]:
    """Return (choice_id, label) pairs for the swap-model dropdown."""
    choices: List[Tuple[str, str]] = []
    has_fp16 = is_model_present(SWAP_MODELS["fp16"]["file"])
    has_fp32 = is_model_present(SWAP_MODELS["fp32"]["file"])
    if has_fp16 or has_fp32:
        choices.append(("auto", SWAP_MODELS["auto"]["label"]))
    if has_fp16:
        choices.append(("fp16", SWAP_MODELS["fp16"]["label"]))
    if has_fp32:
        choices.append(("fp32", SWAP_MODELS["fp32"]["label"]))
    return choices


def available_enhancer_choices() -> List[Tuple[str, Optional[str]]]:
    """Return (label, fp_key) pairs; fp_key is None for 'None'."""
    choices: List[Tuple[str, Optional[str]]] = [("None", None)]
    for info in ENHANCER_MODELS.values():
        if is_model_present(info["file"]):
            choices.append((info["label"], info["fp_key"]))
    return choices


def enhancer_label_for_fp_key(fp_key: str) -> Optional[str]:
    for info in ENHANCER_MODELS.values():
        if info["fp_key"] == fp_key:
            return info["label"]
    return None


def enhancer_fp_key_for_label(label: str) -> Optional[str]:
    for info in ENHANCER_MODELS.values():
        if info["label"] == label:
            return info["fp_key"]
    return None
