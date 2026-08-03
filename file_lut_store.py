"""Per-file LUT override storage: one JSON sidecar per ND2 file, kept in a
subfolder alongside the data (same pattern as glnd2_rois.json)."""
from __future__ import annotations

import json
import os

LUTS_DIRNAME = "glnd2_luts"


def luts_dir(folder: str) -> str:
    return os.path.join(folder, LUTS_DIRNAME)


def _lut_filename(nd2_filename: str) -> str:
    return os.path.splitext(nd2_filename)[0] + ".json"


def lut_path(folder: str, nd2_filename: str) -> str:
    return os.path.join(luts_dir(folder), _lut_filename(nd2_filename))


def has_override(folder: str, nd2_filename: str) -> bool:
    return os.path.exists(lut_path(folder, nd2_filename))


def load_override(folder: str, nd2_filename: str) -> dict | None:
    path = lut_path(folder, nd2_filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_override(folder: str, nd2_filename: str, data: dict) -> None:
    os.makedirs(luts_dir(folder), exist_ok=True)
    with open(lut_path(folder, nd2_filename), "w") as f:
        json.dump(data, f, indent=2)


def clear_override(folder: str, nd2_filename: str) -> None:
    path = lut_path(folder, nd2_filename)
    if os.path.exists(path):
        os.remove(path)


def list_overridden_stems(folder: str) -> set[str]:
    """Basenames (without extension) of ND2 files that have a saved override."""
    d = luts_dir(folder)
    if not os.path.isdir(d):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".json")}
