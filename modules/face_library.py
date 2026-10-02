"""Persistent named source/target face image library.

Assets live outside the repository so upgrades, git operations, and virtual
environment rebuilds do not remove a user's saved faces.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


APP_DATA_DIR = Path(
    os.environ.get(
        "LOCALAPPDATA",
        Path.home() / "AppData" / "Local",
    )
) / "DeepLiveCam"
LIBRARY_DIR = APP_DATA_DIR / "face_library"
MANIFEST_PATH = LIBRARY_DIR / "library.json"
VALID_KINDS = {"source", "target"}
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

_LOCK = threading.RLock()


def _empty_manifest() -> Dict[str, Any]:
    return {"version": 1, "faces": [], "active": {"source": None, "target": None}}


def _load_manifest() -> Dict[str, Any]:
    with _LOCK:
        if not MANIFEST_PATH.exists():
            return _empty_manifest()
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _empty_manifest()
        if not isinstance(data, dict) or not isinstance(data.get("faces"), list):
            return _empty_manifest()
        if not isinstance(data.get("active"), dict):
            data["active"] = {"source": None, "target": None}
        return data


def _write_manifest(data: dict) -> None:
    with _LOCK:
        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix="library-", suffix=".json.tmp", dir=LIBRARY_DIR
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, MANIFEST_PATH)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def list_faces(kind: Optional[str] = None) -> List[dict]:
    """Return valid saved entries sorted by display name."""
    if kind is not None and kind not in VALID_KINDS:
        raise ValueError(f"Unsupported face kind: {kind}")
    entries = []
    for entry in _load_manifest()["faces"]:
        if kind is not None and entry.get("kind") != kind:
            continue
        path = entry.get("path")
        if path and Path(path).is_file():
            entries.append(dict(entry))
    return sorted(entries, key=lambda item: item.get("name", "").casefold())


def get_active(kind: str) -> Optional[str]:
    if kind not in VALID_KINDS:
        raise ValueError(f"Unsupported face kind: {kind}")
    value = _load_manifest().get("active", {}).get(kind)
    return value if isinstance(value, str) else None


def set_active(kind: str, entry_id: Optional[str]) -> None:
    if kind not in VALID_KINDS:
        raise ValueError(f"Unsupported face kind: {kind}")
    with _LOCK:
        data = _load_manifest()
        if entry_id is not None and not any(
            item.get("id") == entry_id and item.get("kind") == kind
            for item in data["faces"]
        ):
            raise ValueError("Saved face does not exist")
        data.setdefault("active", {})[kind] = entry_id
        _write_manifest(data)


def add_face(kind: str, name: str, source_path: str) -> dict:
    """Copy an image into the library and return its manifest entry."""
    if kind not in VALID_KINDS:
        raise ValueError(f"Unsupported face kind: {kind}")
    clean_name = " ".join(name.strip().split())
    if not clean_name:
        raise ValueError("A name is required")

    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(source_path)
    extension = source.suffix.lower()
    if extension not in VALID_EXTENSIONS:
        extension = ".png"

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    entry_id = uuid.uuid4().hex
    destination = LIBRARY_DIR / f"{kind}-{entry_id}{extension}"
    shutil.copy2(source, destination)

    entry = {
        "id": entry_id,
        "kind": kind,
        "name": clean_name,
        "path": str(destination.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _LOCK:
        data = _load_manifest()
        # Names are unique within each kind. Replacing a name removes its old
        # managed asset after the new copy is safely in place.
        replaced = [
            item
            for item in data["faces"]
            if item.get("kind") == kind
            and item.get("name", "").casefold() == clean_name.casefold()
        ]
        data["faces"] = [
            item
            for item in data["faces"]
            if item not in replaced
        ]
        data["faces"].append(entry)
        data.setdefault("active", {})[kind] = entry_id
        _write_manifest(data)
        for old in replaced:
            old_path = Path(old.get("path", ""))
            if old_path != destination:
                try:
                    old_path.unlink(missing_ok=True)
                except OSError:
                    pass
    return dict(entry)


def remove_face(entry_id: str) -> bool:
    """Delete one saved entry and its managed image."""
    with _LOCK:
        data = _load_manifest()
        matched = [item for item in data["faces"] if item.get("id") == entry_id]
        if not matched:
            return False
        data["faces"] = [
            item for item in data["faces"] if item.get("id") != entry_id
        ]
        for kind in VALID_KINDS:
            if data.setdefault("active", {}).get(kind) == entry_id:
                data["active"][kind] = None
        _write_manifest(data)
        for item in matched:
            try:
                Path(item.get("path", "")).unlink(missing_ok=True)
            except OSError:
                pass
        return True
