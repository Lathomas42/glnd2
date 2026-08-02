"""ND2 file loading and per-channel data helpers."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import nd2

# Default channel tint colors, keyed by a rough wavelength->color mapping.
# Falls back to cycling through this palette (in order) when wavelength
# can't be parsed from the channel name.
_WAVELENGTH_COLOR_STOPS = [
    (400, (0.60, 0.20, 1.00)),   # violet / DAPI-ish
    (450, (0.20, 0.45, 1.00)),   # blue
    (500, (0.20, 1.00, 0.30)),   # green
    (570, (1.00, 0.85, 0.10)),   # yellow
    (600, (1.00, 0.45, 0.05)),   # orange
    (650, (1.00, 0.10, 0.10)),   # red
    (700, (1.00, 0.10, 0.80)),   # magenta / far-red
]
_FALLBACK_PALETTE = [
    (0.20, 0.45, 1.00),
    (0.20, 1.00, 0.30),
    (1.00, 0.85, 0.10),
    (1.00, 0.10, 0.10),
    (1.00, 0.10, 0.80),
    (0.10, 1.00, 1.00),
    (1.00, 1.00, 1.00),
    (0.60, 0.20, 1.00),
]

_WAVELENGTH_RE = re.compile(r"(\d{3,4})\s*nm")

# Standard fluorophores, keyed by a short canonical id. Channels are matched
# to one of these either by excitation wavelength (e.g. "640 nm") or by name
# pattern (e.g. "AF647", "AF647 Prescan", "GFP") -- different microscopes in
# this lab label channels either way.
_FLUOROPHORE_DISPLAY_NAMES: dict[str, str] = {
    "DAPI": "DAPI",
    "AF488": "Alexa Fluor 488",
    "AF555": "Alexa Fluor 555",
    "AF568": "Alexa Fluor 568",
    "AF647": "Alexa Fluor 647",
}
_FLUOROPHORE_WAVELENGTHS: dict[str, int] = {
    "DAPI": 395,
    "AF488": 470,
    "AF555": 555,
    "AF647": 640,
    # AF568 has no wavelength-only acquisitions in this lab yet; add one
    # here if a scope starts labeling it that way.
}
_FLUOROPHORE_TOLERANCE_NM = 10
_FLUOROPHORE_NAME_PATTERNS: dict[str, re.Pattern] = {
    "DAPI": re.compile(r"\bDAPI\b", re.IGNORECASE),
    "AF488": re.compile(r"\b(AF\s*488|Alexa\s*Fluor\s*488|GFP)\b", re.IGNORECASE),
    "AF555": re.compile(r"\b(AF\s*555|Alexa\s*Fluor\s*555)\b", re.IGNORECASE),
    "AF568": re.compile(r"\b(AF\s*568|Alexa\s*Fluor\s*568)\b", re.IGNORECASE),
    "AF647": re.compile(r"\b(AF\s*647|Alexa\s*Fluor\s*647)\b", re.IGNORECASE),
}

# Named, switchable color palettes for the standard panel above.
COLOR_PALETTES: dict[str, dict[str, tuple[float, float, float]]] = {
    "Classic": {
        "DAPI": (0.20, 0.40, 1.00),   # blue
        "AF488": (0.10, 0.95, 0.20),  # green
        "AF555": (1.00, 0.75, 0.00),  # yellow/orange
        "AF568": (1.00, 0.45, 0.05),  # orange
        "AF647": (1.00, 0.10, 0.10),  # red
    },
    "Vivid": {
        "DAPI": (0.10, 0.15, 0.95),   # deep blue
        "AF488": (0.05, 1.00, 0.10),  # bright green
        "AF555": (1.00, 0.05, 0.05),  # red
        "AF568": (1.00, 0.45, 0.00),  # vivid orange
        "AF647": (1.00, 0.15, 0.60),  # bright pink
    },
    "High Contrast": {
        "DAPI": (0.10, 0.85, 1.00),   # cyan
        "AF488": (1.00, 0.90, 0.10),  # yellow
        "AF555": (1.00, 0.50, 0.05),  # orange
        "AF568": (0.60, 0.30, 1.00),  # violet
        "AF647": (0.90, 0.10, 0.90),  # magenta
    },
}
DEFAULT_PALETTE = "Vivid"
_active_palette = DEFAULT_PALETTE

# User-saved palettes live outside version control, under the app folder.
_PALETTES_DIR = os.path.join(os.path.dirname(__file__), "palettes")
_CUSTOM_PALETTES_FILE = os.path.join(_PALETTES_DIR, "custom_palettes.json")


def set_active_palette(name: str):
    global _active_palette
    if name in COLOR_PALETTES:
        _active_palette = name


def get_active_palette() -> str:
    return _active_palette


def _wavelength_to_key(wavelength_str: str) -> str | None:
    """Migrate an old-format palette key (raw excitation wavelength, e.g.
    "395") to the current canonical fluorophore key (e.g. "DAPI")."""
    try:
        nm = int(wavelength_str)
    except ValueError:
        return None
    for key, w in _FLUOROPHORE_WAVELENGTHS.items():
        if w == nm:
            return key
    return None


def load_custom_palettes():
    """Load any user-saved palettes from disk into COLOR_PALETTES.

    Transparently migrates palettes saved before channels were matched by
    name as well as wavelength (when keys were raw wavelengths like "395"
    instead of "DAPI").
    """
    if not os.path.exists(_CUSTOM_PALETTES_FILE):
        return
    try:
        with open(_CUSTOM_PALETTES_FILE) as f:
            data = json.load(f)
    except Exception:
        return

    migrated = False
    for name, entries in data.items():
        palette = {}
        for raw_key, color in entries.items():
            key = raw_key if raw_key in _FLUOROPHORE_DISPLAY_NAMES else _wavelength_to_key(raw_key)
            if key is None:
                continue
            if key != raw_key:
                migrated = True
            palette[key] = tuple(color)
        COLOR_PALETTES[name] = palette

    if migrated:
        for name in data:
            if name in COLOR_PALETTES:
                data[name] = {k: list(c) for k, c in COLOR_PALETTES[name].items()}
        try:
            with open(_CUSTOM_PALETTES_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


def save_custom_palette(name: str, palette: dict[str, tuple[float, float, float]]):
    """Save (or overwrite) a named palette to disk and register it in COLOR_PALETTES."""
    os.makedirs(_PALETTES_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(_CUSTOM_PALETTES_FILE):
        try:
            with open(_CUSTOM_PALETTES_FILE) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing[name] = {k: list(c) for k, c in palette.items()}
    with open(_CUSTOM_PALETTES_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    COLOR_PALETTES[name] = dict(palette)


load_custom_palettes()


def fluorophore_key_for_channel(name: str) -> str | None:
    """Canonical fluorophore key (e.g. "AF647") for a channel name, matched
    either by name pattern ("AF647 Prescan", "GFP") or, failing that, by
    excitation wavelength ("640 nm"). None if neither matches."""
    for key, pattern in _FLUOROPHORE_NAME_PATTERNS.items():
        if pattern.search(name):
            return key
    m = _WAVELENGTH_RE.search(name)
    if m:
        nm = float(m.group(1))
        best = min(_FLUOROPHORE_WAVELENGTHS, key=lambda k: abs(_FLUOROPHORE_WAVELENGTHS[k] - nm))
        if abs(_FLUOROPHORE_WAVELENGTHS[best] - nm) <= _FLUOROPHORE_TOLERANCE_NM:
            return best
    return None


def _color_for_wavelength(nm: float) -> tuple[float, float, float]:
    if nm <= _WAVELENGTH_COLOR_STOPS[0][0]:
        return _WAVELENGTH_COLOR_STOPS[0][1]
    if nm >= _WAVELENGTH_COLOR_STOPS[-1][0]:
        return _WAVELENGTH_COLOR_STOPS[-1][1]
    for (w0, c0), (w1, c1) in zip(_WAVELENGTH_COLOR_STOPS, _WAVELENGTH_COLOR_STOPS[1:]):
        if w0 <= nm <= w1:
            t = (nm - w0) / (w1 - w0)
            return tuple(a + (b - a) * t for a, b in zip(c0, c1))
    return (1.0, 1.0, 1.0)


def default_color_for_channel(name: str, index: int) -> tuple[float, float, float]:
    key = fluorophore_key_for_channel(name)
    if key is not None:
        # A custom or older palette may not have every standard key (e.g. one
        # saved before AF568 was added); fall back to the default palette.
        active = COLOR_PALETTES[_active_palette]
        return active.get(key) or COLOR_PALETTES[DEFAULT_PALETTE][key]
    m = _WAVELENGTH_RE.search(name)
    if m:
        return _color_for_wavelength(float(m.group(1)))
    return _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)]


def display_name_for_channel(name: str) -> str:
    """Human-readable label for UI display (e.g. "Alexa Fluor 647").

    Purely cosmetic: the channel's underlying identity (used as the LUT
    preset/carry-over key) stays the raw ND2 metadata name so existing
    presets keep matching.
    """
    key = fluorophore_key_for_channel(name)
    return _FLUOROPHORE_DISPLAY_NAMES[key] if key is not None else name


@dataclass
class ChannelData:
    name: str
    array: np.ndarray  # 2D uint16
    data_max: int  # actual max pixel value observed (for slider range)
    default_color: tuple[float, float, float]
    hist_counts: np.ndarray = field(repr=False)
    hist_max_bin: int
    p_low: float  # suggested auto-contrast black point
    p_high: float  # suggested auto-contrast white point
    display_name: str = ""  # human-readable label, e.g. "Alexa Fluor 647"


@dataclass
class Nd2Image:
    path: str
    width: int
    height: int
    channels: list[ChannelData]
    pixel_size_um: float | None = None


def _compute_stats(arr: np.ndarray, bins: int = 256):
    """Downsampled histogram + percentile auto-contrast suggestion."""
    step_y = max(1, arr.shape[0] // 512)
    step_x = max(1, arr.shape[1] // 512)
    sample = arr[::step_y, ::step_x]
    data_max = int(arr.max()) if arr.size else 1
    hi = max(data_max, 1)
    counts, _ = np.histogram(sample, bins=bins, range=(0, hi))
    p_low, p_high = np.percentile(sample, [0.5, 99.7])
    if p_high <= p_low:
        p_high = p_low + 1
    return data_max, counts, int(counts.max() if counts.size else 1), float(p_low), float(p_high)


def load_nd2(path: str) -> Nd2Image:
    """Load an ND2 file and split it into per-channel 2D arrays with stats."""
    with nd2.ND2File(path) as f:
        arr = f.asarray()  # dask-backed lazy array materialized here
        sizes = f.sizes
        names = [c.channel.name for c in f.metadata.channels] if f.metadata and f.metadata.channels else None
        try:
            voxel = f.voxel_size()
            pixel_size_um = float(voxel.x) if voxel and voxel.x else None
        except Exception:
            pixel_size_um = None

    arr = np.asarray(arr)

    if "C" in sizes:
        c_axis = list(sizes.keys()).index("C")
        n_channels = sizes["C"]
        arr = np.moveaxis(arr, c_axis, 0)
        # Collapse any remaining leading dims (e.g. multi-position/time) by
        # taking the first plane; ND2s from a single acquisition are (C,Y,X).
        while arr.ndim > 3:
            arr = arr[0]
    else:
        n_channels = 1
        arr = arr[np.newaxis, ...]

    height, width = arr.shape[-2], arr.shape[-1]
    if names is None or len(names) != n_channels:
        names = [f"Ch{i + 1}" for i in range(n_channels)]

    channels = []
    for i in range(n_channels):
        plane = np.ascontiguousarray(arr[i])
        if plane.dtype != np.uint16:
            plane = plane.astype(np.uint16)
        data_max, counts, hist_max_bin, p_low, p_high = _compute_stats(plane)
        channels.append(
            ChannelData(
                name=names[i],
                array=plane,
                data_max=data_max,
                default_color=default_color_for_channel(names[i], i),
                hist_counts=counts,
                hist_max_bin=hist_max_bin,
                p_low=p_low,
                p_high=p_high,
                display_name=display_name_for_channel(names[i]),
            )
        )

    return Nd2Image(path=path, width=width, height=height, channels=channels, pixel_size_um=pixel_size_um)
