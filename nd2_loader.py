"""ND2 file loading and per-channel data helpers."""
from __future__ import annotations

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
    m = _WAVELENGTH_RE.search(name)
    if m:
        return _color_for_wavelength(float(m.group(1)))
    return _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)]


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


@dataclass
class Nd2Image:
    path: str
    width: int
    height: int
    channels: list[ChannelData]


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
            )
        )

    return Nd2Image(path=path, width=width, height=height, channels=channels)
