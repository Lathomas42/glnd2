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


def _to_qcolor(rgb01: tuple[float, float, float]) -> QColor:
    return QColor(*(int(round(c * 255)) for c in rgb01))


def _from_qcolor(c: QColor) -> tuple[float, float, float]:
    return (c.redF(), c.greenF(), c.blueF())


class HistogramView(QWidget):
    """Log-scaled histogram with black/white markers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._counts = None
        self._data_max = 1
        self._black = 0
        self._white = 1
        self._color = QColor(200, 200, 200)

    def set_data(self, counts: np.ndarray, data_max: int, color: tuple[float, float, float]):
        self._counts = counts
        self._data_max = max(data_max, 1)
        self._color = _to_qcolor(color)
        self.update()

    def set_markers(self, black: float, white: float):
        self._black = black
        self._white = white
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 24, 28))
        if self._counts is not None and self._counts.size:
            w, h = self.width(), self.height()
            n = len(self._counts)
            log_counts = np.log1p(self._counts.astype(np.float64))
            peak = log_counts.max() or 1.0
            bar_w = max(w / n, 1.0)
            pen = QPen(self._color)
            p.setPen(pen)
            for i, v in enumerate(log_counts):
                bh = (v / peak) * (h - 2)
                x = i * bar_w
                p.drawLine(int(x), h, int(x), int(h - bh))

            for val, colr in ((self._black, QColor(255, 255, 255)), (self._white, QColor(255, 220, 80))):
                frac = min(max(val / self._data_max, 0.0), 1.0)
                x = int(frac * (w - 1))
                p.setPen(QPen(colr, 1))
                p.drawLine(x, 0, x, h)
        p.end()


class ChannelPanel(QFrame):
    """Controls for a single channel. Emits `changed` whenever a value updates."""

    changed = Signal(int)
    removed = Signal(int)  # not used yet, reserved for future channel removal UI

    def __init__(self, index: int, name: str, data_max: int, color, black: float,
                 white: float, gamma: float, enabled: bool, default_black: float,
                 default_white: float, parent=None):
        super().__init__(parent)
        self.index = index
        self.data_max = max(int(data_max), 1)
        self.default_black = default_black
        self.default_white = default_white
        self.default_color = color

        self.setFrameShape(QFrame.StyledPanel)
        self._build_ui(name, color)
        self.set_values(black, white, gamma, enabled, color)

    # ------------------------------------------------------------------
    def _build_ui(self, name, color):
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

        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setFixedWidth(48)
        self.auto_btn.clicked.connect(self._apply_auto)
        header.addWidget(self.auto_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFixedWidth(52)
        self.reset_btn.clicked.connect(self._apply_reset)
        header.addWidget(self.reset_btn)

        layout.addLayout(header)

        self.hist = HistogramView()
        layout.addWidget(self.hist)

        self.black_slider, self.black_spin, black_row = self._make_row("Black")
        self.white_slider, self.white_spin, white_row = self._make_row("White")
        layout.addLayout(black_row)
        layout.addLayout(white_row)

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma"))
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 400)  # -> 0.10 .. 4.00
        self.gamma_slider.valueChanged.connect(self._gamma_slider_changed)
        gamma_row.addWidget(self.gamma_slider, 1)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 4.0)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.valueChanged.connect(self._gamma_spin_changed)
        gamma_row.addWidget(self.gamma_spin)
        layout.addLayout(gamma_row)

        self.black_slider.valueChanged.connect(self._black_slider_changed)
        self.black_spin.valueChanged.connect(self._black_spin_changed)
        self.white_slider.valueChanged.connect(self._white_slider_changed)
        self.white_spin.valueChanged.connect(self._white_spin_changed)

    def _make_row(self, label):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, self.data_max)
        row.addWidget(slider, 1)
        spin = QSpinBox()
        spin.setRange(0, self.data_max)
        row.addWidget(spin)
        return slider, spin, row

    # ------------------------------------------------------------------
    def set_values(self, black, white, gamma, enabled, color):
        self._block(True)
        self.visible_box.setChecked(enabled)
        self.black_slider.setValue(int(black))
        self.black_spin.setValue(int(black))
        self.white_slider.setValue(int(white))
        self.white_spin.setValue(int(white))
        self.gamma_slider.setValue(int(round(gamma * 100)))
        self.gamma_spin.setValue(gamma)
        self._set_color(color, emit=False)
        self.hist.set_markers(black, white)
        self._block(False)

    def set_histogram(self, counts, data_max, color):
        self.hist.set_data(counts, data_max, color)

    def _block(self, on: bool):
        for w in (self.visible_box, self.black_slider, self.black_spin,
                   self.white_slider, self.white_spin, self.gamma_slider, self.gamma_spin):
            w.blockSignals(on)

    def values(self) -> dict:
        return dict(
            black=float(self.black_spin.value()),
            white=float(self.white_spin.value()),
            gamma=float(self.gamma_spin.value()),
            enabled=self.visible_box.isChecked(),
            color=_from_qcolor(self.color_btn.palette().button().color()),
        )

    # ------------------------------------------------------------------
    def _set_color(self, rgb01, emit=True):
        qc = _to_qcolor(rgb01)
        self.color_btn.setStyleSheet(
            f"background-color: rgb({qc.red()},{qc.green()},{qc.blue()}); border: 1px solid #888;"
        )
        self.color_btn.setProperty("qcolor", qc)
        if emit:
            self._emit_changed()

    def _pick_color(self):
        current = self.color_btn.property("qcolor") or QColor(255, 255, 255)
        c = QColorDialog.getColor(current, self, f"Color for {self.name_label.text()}")
        if c.isValid():
            self._set_color(_from_qcolor(c))

    def _apply_auto(self):
        self.set_values(self.default_black, self.default_white, self.gamma_spin.value(),
                         self.visible_box.isChecked(), self._current_color())
        self._emit_changed()

    def _apply_reset(self):
        self.set_values(self.default_black, self.default_white, 1.0, True, self.default_color)
        self._emit_changed()

    def _current_color(self):
        qc = self.color_btn.property("qcolor") or QColor(255, 255, 255)
        return _from_qcolor(qc)

    # ------------------------------------------------------------------
    def _black_slider_changed(self, v):
        if v > self.white_slider.value():
            self.white_slider.setValue(v)
        self.black_spin.setValue(v)
        self.hist.set_markers(v, self.white_spin.value())
        self._emit_changed()

    def _black_spin_changed(self, v):
        self.black_slider.setValue(v)

    def _white_slider_changed(self, v):
        if v < self.black_slider.value():
            self.black_slider.setValue(v)
        self.white_spin.setValue(v)
        self.hist.set_markers(self.black_spin.value(), v)
        self._emit_changed()

    def _white_spin_changed(self, v):
        self.white_slider.setValue(v)

    def _gamma_slider_changed(self, v):
        self.gamma_spin.setValue(v / 100.0)

    def _gamma_spin_changed(self, v):
        self.gamma_slider.setValue(int(round(v * 100)))
        self._emit_changed()

    def _emit_changed(self, *_):
        self.changed.emit(self.index)
