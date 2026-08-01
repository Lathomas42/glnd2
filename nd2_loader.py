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

# A standard 4-color immunofluorescence panel: excitation wavelength ->
# fluorophore display name. Matched within a tolerance so small
# wavelength-calibration differences between acquisitions still hit.
_FLUOROPHORE_NAMES: dict[int, str] = {
    395: "DAPI",
    470: "Alexa Fluor 488",
    555: "Alexa Fluor 555",
    640: "Alexa Fluor 647",
}
_FLUOROPHORE_TOLERANCE_NM = 10

# Named, switchable color palettes for the standard panel above.
COLOR_PALETTES: dict[str, dict[int, tuple[float, float, float]]] = {
    "Classic": {
        395: (0.20, 0.40, 1.00),   # blue
        470: (0.10, 0.95, 0.20),   # green
        555: (1.00, 0.75, 0.00),   # yellow/orange
        640: (1.00, 0.10, 0.10),   # red
    },
    "Vivid": {
        395: (0.10, 0.15, 0.95),   # deep blue
        470: (0.05, 1.00, 0.10),   # bright green
        555: (1.00, 0.05, 0.05),   # red
        640: (1.00, 0.15, 0.60),   # bright pink
    },
    "High Contrast": {
        395: (0.10, 0.85, 1.00),   # cyan
        470: (1.00, 0.90, 0.10),   # yellow
        555: (1.00, 0.50, 0.05),   # orange
        640: (0.90, 0.10, 0.90),   # magenta
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


def load_custom_palettes():
    """Load any user-saved palettes from disk into COLOR_PALETTES."""
    if not os.path.exists(_CUSTOM_PALETTES_FILE):
        return
    try:
        with open(_CUSTOM_PALETTES_FILE) as f:
            data = json.load(f)
        for name, wavelengths in data.items():
            COLOR_PALETTES[name] = {int(w): tuple(c) for w, c in wavelengths.items()}
    except Exception:
        pass


def save_custom_palette(name: str, palette: dict[int, tuple[float, float, float]]):
    """Save (or overwrite) a named palette to disk and register it in COLOR_PALETTES."""
    os.makedirs(_PALETTES_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(_CUSTOM_PALETTES_FILE):
        try:
            with open(_CUSTOM_PALETTES_FILE) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing[name] = {str(w): list(c) for w, c in palette.items()}
    with open(_CUSTOM_PALETTES_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    COLOR_PALETTES[name] = dict(palette)


load_custom_palettes()


def wavelength_for_channel(name: str) -> int | None:
    """The matched standard wavelength (395/470/555/640) for a channel name,
    or None if it doesn't correspond to one of the standard fluorophores."""
    m = _WAVELENGTH_RE.search(name)
    if not m:
        return None
    nm = float(m.group(1))
    best = min(_FLUOROPHORE_NAMES, key=lambda w: abs(w - nm))
    return best if abs(best - nm) <= _FLUOROPHORE_TOLERANCE_NM else None


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
    w = wavelength_for_channel(name)
    if w is not None:
        return COLOR_PALETTES[_active_palette][w]
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
    w = wavelength_for_channel(name)
    return _FLUOROPHORE_NAMES[w] if w is not None else name


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
