"""Per-channel LUT control widget: visibility, color, black/white/gamma, histogram."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_UINT16_MAX = 65535


def _to_qcolor(rgb01: tuple[float, float, float]) -> QColor:
    return QColor(*(int(round(c * 255)) for c in rgb01))


def _from_qcolor(c: QColor) -> tuple[float, float, float]:
    return (c.redF(), c.greenF(), c.blueF())


class HistogramView(QWidget):
    """Log-scaled histogram over [0, range_max] with black/white markers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._log_counts: np.ndarray | None = None
        self._range_max = 1.0
        self._black = 0.0
        self._white = 1.0
        self._color = QColor(200, 200, 200)

    def set_data(self, counts: np.ndarray, range_max: float):
        self._log_counts = np.log1p(counts.astype(np.float64))
        self._range_max = max(float(range_max), 1.0)
        self.update()

    def set_color(self, color: tuple[float, float, float]):
        self._color = _to_qcolor(color)
        self.update()

    def set_markers(self, black: float, white: float):
        self._black = black
        self._white = white
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 24, 28))
        if self._log_counts is not None and self._log_counts.size:
            w, h = self.width(), self.height()
            peak = self._log_counts.max() or 1.0
            bar_w = w / len(self._log_counts)
            p.setPen(QPen(self._color))
            for i, v in enumerate(self._log_counts):
                x = int(i * bar_w)
                p.drawLine(x, h, x, int(h - (v / peak) * (h - 2)))

            for val, color in ((self._black, QColor(255, 255, 255)), (self._white, QColor(255, 220, 80))):
                x = int(min(max(val / self._range_max, 0.0), 1.0) * (w - 1))
                p.setPen(QPen(color, 1))
                p.drawLine(x, 0, x, h)
        p.end()


class ChannelPanel(QFrame):
    """Controls for a single channel.

    `changed` fires only for edits the user makes here; programmatic updates
    via `set_values`/`set_color` are silent. Auto and Reset are delegated to
    the owner (which knows the auto-contrast settings and palette) via
    `autoRequested`/`resetRequested`.
    """

    changed = Signal(int)
    autoRequested = Signal(int)
    resetRequested = Signal(int)

    def __init__(self, index: int, name: str, display_max: float, parent=None):
        super().__init__(parent)
        self.index = index
        self._display_max = max(float(display_max), 1.0)
        self._color = QColor(255, 255, 255)
        self.setFrameShape(QFrame.StyledPanel)
        self._build_ui(name)

    # ------------------------------------------------------------------
    def _build_ui(self, name: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.visible_box = QCheckBox()
        self.visible_box.setChecked(True)
        self.visible_box.stateChanged.connect(self._emit_changed)
        header.addWidget(self.visible_box)

        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(22, 22)
        self.color_btn.clicked.connect(self._pick_color)
        header.addWidget(self.color_btn)

        self.name_label = QLabel(f"<b>{name}</b>")
        header.addWidget(self.name_label)
        header.addStretch(1)

        auto_btn = QPushButton("Auto")
        auto_btn.setFixedWidth(48)
        auto_btn.setToolTip("Auto-contrast black/white for this channel")
        auto_btn.clicked.connect(lambda: self.autoRequested.emit(self.index))
        header.addWidget(auto_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(52)
        reset_btn.setToolTip("Auto-contrast, gamma 1, palette color, visible")
        reset_btn.clicked.connect(lambda: self.resetRequested.emit(self.index))
        header.addWidget(reset_btn)

        layout.addLayout(header)

        self.hist = HistogramView()
        layout.addWidget(self.hist)

        self.black_slider, self.black_spin = self._add_level_row(layout, "Black")
        self.white_slider, self.white_spin = self._add_level_row(layout, "White")

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma"))
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 400)  # -> 0.10 .. 4.00
        gamma_row.addWidget(self.gamma_slider, 1)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 4.0)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setDecimals(2)
        gamma_row.addWidget(self.gamma_spin)
        layout.addLayout(gamma_row)

        self.black_slider.valueChanged.connect(self._black_slider_changed)
        self.black_spin.valueChanged.connect(lambda v: self._spin_changed(self.black_slider, v))
        self.white_slider.valueChanged.connect(self._white_slider_changed)
        self.white_spin.valueChanged.connect(lambda v: self._spin_changed(self.white_slider, v))
        self.gamma_slider.valueChanged.connect(lambda v: self.gamma_spin.setValue(v / 100.0))
        self.gamma_spin.valueChanged.connect(self._gamma_spin_changed)

    @staticmethod
    def _add_level_row(layout: QVBoxLayout, label: str) -> tuple[QSlider, QSpinBox]:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        slider = QSlider(Qt.Horizontal)
        row.addWidget(slider, 1)
        spin = QSpinBox()
        spin.setRange(0, _UINT16_MAX)
        row.addWidget(spin)
        layout.addLayout(row)
        return slider, spin

    # ------------------------------------------------------------------
    def set_values(self, black: float, white: float, gamma: float, enabled: bool,
                   color: tuple[float, float, float]):
        """Show the given settings without emitting `changed`."""
        # The sliders cover the image's useful range, widened if needed so a
        # value from a preset or another file is never clamped.
        slider_max = int(max(self._display_max, white * 1.25, black + 1))
        self._block(True)
        for slider in (self.black_slider, self.white_slider):
            slider.setRange(0, min(slider_max, _UINT16_MAX))
        self.visible_box.setChecked(enabled)
        self.black_slider.setValue(int(round(black)))
        self.black_spin.setValue(int(round(black)))
        self.white_slider.setValue(int(round(white)))
        self.white_spin.setValue(int(round(white)))
        self.gamma_slider.setValue(int(round(gamma * 100)))
        self.gamma_spin.setValue(gamma)
        self._block(False)
        self.set_color(color)
        self.hist.set_markers(black, white)

    def set_color(self, color: tuple[float, float, float]):
        """Show `color` on the swatch and histogram without emitting `changed`."""
        self._color = _to_qcolor(color)
        self.color_btn.setStyleSheet(
            f"background-color: {self._color.name()}; border: 1px solid #888;"
        )
        self.hist.set_color(color)

    def set_histogram(self, counts: np.ndarray, range_max: float):
        self.hist.set_data(counts, range_max)

    def values(self) -> dict:
        return dict(
            black=float(self.black_spin.value()),
            white=float(self.white_spin.value()),
            gamma=float(self.gamma_spin.value()),
            enabled=self.visible_box.isChecked(),
            color=_from_qcolor(self._color),
        )

    def _block(self, on: bool):
        for w in (self.visible_box, self.black_slider, self.black_spin,
                  self.white_slider, self.white_spin, self.gamma_slider, self.gamma_spin):
            w.blockSignals(on)

    # ------------------------------------------------------------------
    def _pick_color(self):
        c = QColorDialog.getColor(self._color, self, f"Color for {self.name_label.text()}")
        if c.isValid():
            self.set_color(_from_qcolor(c))
            self._emit_changed()

    def _spin_changed(self, slider: QSlider, v: int):
        # A typed value past the slider's range widens it rather than being
        # clamped back by the slider.
        if v > slider.maximum():
            new_max = min(int(v * 1.25), _UINT16_MAX)
            for s in (self.black_slider, self.white_slider):
                s.setMaximum(new_max)
        slider.setValue(v)

    def _black_slider_changed(self, v: int):
        if v > self.white_slider.value():
            self.white_slider.setValue(v)
        self.black_spin.blockSignals(True)
        self.black_spin.setValue(v)
        self.black_spin.blockSignals(False)
        self.hist.set_markers(v, self.white_spin.value())
        self._emit_changed()

    def _white_slider_changed(self, v: int):
        if v < self.black_slider.value():
            self.black_slider.setValue(v)
        self.white_spin.blockSignals(True)
        self.white_spin.setValue(v)
        self.white_spin.blockSignals(False)
        self.hist.set_markers(self.black_spin.value(), v)
        self._emit_changed()

    def _gamma_spin_changed(self, v: float):
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(int(round(v * 100)))
        self.gamma_slider.blockSignals(False)
        self._emit_changed()

    def _emit_changed(self, *_):
        self.changed.emit(self.index)
