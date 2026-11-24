"""ND2 file loading and per-channel data helpers."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
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
    "AF568": 561,
    "AF647": 640,
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


@dataclass(frozen=True)
class ChannelStats:
    """Intensity statistics used for auto-contrast and the LUT controls.

    Computed over non-zero pixels only: stitched mosaics pad unscanned tiles
    with exact zeros, which would otherwise dominate the low end.
    """
    background: float  # peak of the histogram: camera offset + slide/tissue background
    noise: float  # robust sigma of that background peak
    p98: float
    display_max: float  # upper end of the useful range, ignoring hot pixels


# Auto-contrast puts black just above the background peak (so background is
# inky black) and white `headroom` times further out than the 98th
# percentile is (so the image is mostly dark, with a few bright pixels and
# almost nothing saturated). `headroom` is adjustable at runtime.
AUTO_BLACK_SIGMAS = 2.0
DEFAULT_AUTO_HEADROOM = 6.0
# Never stretch less than this many noise sigmas, or a near-empty image
# would blow its background noise up into bright speckle.
_MIN_RANGE_SIGMAS = 25.0
_STATS_SAMPLES = 1_000_000
_HIST_BINS = 256


def auto_levels(stats: ChannelStats, headroom: float = DEFAULT_AUTO_HEADROOM) -> tuple[float, float]:
    """(black, white) auto-contrast levels for a channel."""
    black = stats.background + AUTO_BLACK_SIGMAS * stats.noise
    span = max(headroom * (stats.p98 - black), _MIN_RANGE_SIGMAS * stats.noise, 1.0)
    white = min(black + span, 65535.0)
    black = min(black, white - 1.0)
    return float(black), float(white)


def _compute_stats(plane: np.ndarray) -> tuple[ChannelStats, np.ndarray]:
    """Stats plus a `_HIST_BINS`-bin histogram over [0, display_max].

    Everything is derived from one exact per-intensity count of a ~1M pixel
    sample (the data is 16-bit), which is ~10x faster than float
    percentiles/histograms on the same sample.
    """
    stride = max(1, int(np.sqrt(plane.size / _STATS_SAMPLES)))
    counts = np.bincount(plane[::stride, ::stride].ravel(), minlength=65536)
    counts[0] = 0  # stitch padding, not signal
    cdf = np.cumsum(counts)
    total = int(cdf[-1])
    if total == 0:
        return ChannelStats(0.0, 0.0, 1.0, 1.0), np.zeros(_HIST_BINS, dtype=np.int64)

    def pct(q: float) -> int:
        return int(np.searchsorted(cdf, q / 100.0 * total))

    lo, q25, hi = pct(0.1), pct(25), pct(90)

    # Background = the dominant low-intensity peak. Smooth over a window
    # matched to the peak's width before taking the argmax, so noise in
    # individual intensity counts doesn't pick a spurious maximum.
    width = max(5, int(round(5 * max(1.0, (q25 - lo) / 50.0))))
    smooth = np.convolve(counts[lo:hi + 1].astype(np.float64), np.ones(width), mode="same")
    mode = lo + int(np.argmax(smooth))

    # The left flank of the peak is pure noise (signal only adds intensity),
    # so its spread is a clean estimate of background noise: the median
    # distance below the mode, as a robust sigma.
    below = counts[1:mode][::-1]  # distance 1, 2, ... below the mode
    noise = 0.0
    if below.sum() > 0:
        noise = 1.4826 * (int(np.searchsorted(np.cumsum(below), below.sum() / 2.0)) + 1)

    p98, p9999 = pct(98), pct(99.99)
    partial = ChannelStats(float(mode), noise, float(p98), 1.0)
    display_max = float(min(max(p9999, auto_levels(partial)[1], 1.0), 65535.0))
    stats = ChannelStats(float(mode), noise, float(p98), display_max)

    top = int(display_max)
    bins = (np.arange(top + 1) * _HIST_BINS // (top + 1)).astype(np.int64)
    hist = np.bincount(bins, weights=counts[:top + 1], minlength=_HIST_BINS).astype(np.int64)
    return stats, hist


@dataclass
class ChannelData:
    name: str  # raw ND2 channel name: the channel's identity for LUT presets/overrides
    display_name: str  # human-readable label, e.g. "Alexa Fluor 647"
    # 2D uint16 (possibly a strided view into Nd2Image.packed); None once on the GPU.
    array: np.ndarray | None = field(repr=False)
    stats: ChannelStats
    hist_counts: np.ndarray = field(repr=False)  # over [0, stats.display_max]


@dataclass
class Nd2Image:
    path: str
    width: int
    height: int
    channels: list[ChannelData]
    pixel_size_um: float | None = None
    # ND2 stores channels interleaved per pixel. When that buffer is a plain
    # contiguous (Y, X, C) array with C <= 4 it is uploaded to the GPU as-is,
    # as one multi-component texture, instead of channel by channel.
    packed: np.ndarray | None = field(default=None, repr=False)
    # gl_widget.GpuTextures when a loader thread already put the pixels on the
    # GPU; whoever displays (or discards) the image takes ownership.
    gpu: object | None = None


def _to_cyx(arr: np.ndarray, axes: list[str]) -> np.ndarray:
    """Reduce an ND2 array to (C, Y, X).

    Z stacks are max-projected; any other extra axis (time, stage position)
    contributes its first plane. RGB samples are treated as channels when
    the file has no explicit channel axis.
    """
    axes = list(axes)
    if "C" not in axes and "S" in axes:
        axes[axes.index("S")] = "C"
    for ax in [a for a in axes if a not in ("C", "Y", "X")]:
        i = axes.index(ax)
        arr = arr.max(axis=i) if ax == "Z" else np.take(arr, 0, axis=i)
        axes.pop(i)
    if "C" not in axes:
        arr = arr[np.newaxis]
        axes.insert(0, "C")
    return arr.transpose([axes.index(a) for a in ("C", "Y", "X")])


MAX_PACKED_CHANNELS = 4  # an RGBA texture holds at most four


def _prepare_channel(name: str, plane: np.ndarray, copy: bool) -> ChannelData:
    if copy:  # de-interleave into its own contiguous buffer
        plane = np.ascontiguousarray(plane, dtype=np.uint16)
    stats, counts = _compute_stats(plane)
    return ChannelData(
        name=name,
        display_name=display_name_for_channel(name),
        array=plane,
        stats=stats,
        hist_counts=counts,
    )


def load_nd2(path: str) -> Nd2Image:
    """Load an ND2 file into per-channel 2D arrays with stats."""
    with nd2.ND2File(path) as f:
        arr = np.asarray(f.asarray())
        axes = list(f.sizes.keys())
        metadata_channels = f.metadata.channels if f.metadata and f.metadata.channels else []
        names = [c.channel.name for c in metadata_channels]
        try:
            voxel = f.voxel_size()
            pixel_size_um = float(voxel.x) if voxel and voxel.x else None
        except Exception:
            pixel_size_um = None

    arr = _to_cyx(arr, axes)
    n_channels, height, width = arr.shape
    if len(names) != n_channels:
        names = [f"Ch{i + 1}" for i in range(n_channels)]

    interleaved = arr.transpose(1, 2, 0)  # (Y, X, C)
    packed = None
    if (n_channels <= MAX_PACKED_CHANNELS and interleaved.dtype == np.uint16
            and interleaved.flags.c_contiguous):
        packed = interleaved  # upload as-is; channels stay strided views into it

    # Per-channel work (copies, stats) releases the GIL, so threads run it in
    # parallel.
    with ThreadPoolExecutor(max_workers=min(n_channels, os.cpu_count() or 1)) as pool:
        channels = list(pool.map(lambda name, plane: _prepare_channel(name, plane, packed is None), names, arr))

    return Nd2Image(path=path, width=width, height=height, channels=channels,
                    pixel_size_um=pixel_size_um, packed=packed)
