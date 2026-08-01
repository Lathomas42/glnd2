"""ROI data model and per-folder JSON persistence."""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass

ROI_FILENAME = "glnd2_rois.json"


@dataclass
class ROI:
    id: str
    name: str
    file: str  # basename of the .nd2 file this ROI belongs to
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def new(cls, name: str, file: str, x: float, y: float, w: float, h: float) -> "ROI":
        return cls(id=str(uuid.uuid4()), name=name, file=file, x=x, y=y, w=w, h=h)


def rois_path(folder: str) -> str:
    return os.path.join(folder, ROI_FILENAME)


def load_rois(folder: str) -> list[ROI]:
    path = rois_path(folder)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [ROI(**d) for d in data]


def save_rois(folder: str, rois: list[ROI]) -> None:
    path = rois_path(folder)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in rois], f, indent=2)


_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|]')


def safe_name(name: str) -> str:
    """Sanitize an ROI name for use in a filename."""
    return _UNSAFE_CHARS.sub("_", name).strip() or "roi"
