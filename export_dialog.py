"""Export dialog: scope, LUT preset, downsample factor, format, destination."""
from __future__ import annotations

import os
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_DOWNSAMPLE_PRESETS = [
    ("Full resolution", 1.0),
    ("1/2", 0.5),
    ("1/4", 0.25),
    ("1/8", 0.125),
    ("Custom…", None),
]


@dataclass
class ExportOptions:
    scope_all_files: bool
    preset_path: str | None
    scale: float
    fmt: str  # ".png" or ".tif"
    out_dir: str
    export_rois: bool = False


class ExportDialog(QDialog):
    def __init__(self, default_dir: str, initial_preset_path: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Composite")
        self._preset_path: str | None = initial_preset_path

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Scope
        scope_row = QHBoxLayout()
        self.scope_current = QRadioButton("Current file")
        self.scope_all = QRadioButton("All files in folder")
        self.scope_current.setChecked(True)
        scope_row.addWidget(self.scope_current)
        scope_row.addWidget(self.scope_all)
        form.addRow("Export:", scope_row)

        self.roi_check = QCheckBox("Export ROI crops only (named by ROI, instead of the full composite)")
        form.addRow("", self.roi_check)

        # LUT preset
        preset_row = QHBoxLayout()
        self.preset_label = QLabel("Use current on-screen settings")
        self.preset_label.setStyleSheet("color: palette(mid);")
        preset_btn = QPushButton("Choose Preset…")
        preset_btn.clicked.connect(self._choose_preset)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_preset)
        preset_row.addWidget(self.preset_label, 1)
        preset_row.addWidget(preset_btn)
        preset_row.addWidget(clear_btn)
        form.addRow("LUT preset:", preset_row)

        # Downsample
        ds_row = QHBoxLayout()
        self.downsample_combo = QComboBox()
        for label, _ in _DOWNSAMPLE_PRESETS:
            self.downsample_combo.addItem(label)
        self.downsample_combo.currentIndexChanged.connect(self._on_downsample_changed)
        self.custom_spin = QSpinBox()
        self.custom_spin.setRange(1, 32)
        self.custom_spin.setValue(3)
        self.custom_spin.setPrefix("÷ ")
        self.custom_spin.setEnabled(False)
        ds_row.addWidget(self.downsample_combo, 1)
        ds_row.addWidget(self.custom_spin)
        form.addRow("Downsample:", ds_row)

        # Format
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "TIFF"])
        form.addRow("Format:", self.format_combo)

        # Output folder
        out_row = QHBoxLayout()
        self.out_dir_edit = QLineEdit(default_dir)
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._choose_out_dir)
        out_row.addWidget(self.out_dir_edit, 1)
        out_row.addWidget(out_browse)
        form.addRow("Save to:", out_row)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if self._preset_path:
            self._set_preset_label(self._preset_path)

    def _on_downsample_changed(self, idx: int):
        self.custom_spin.setEnabled(_DOWNSAMPLE_PRESETS[idx][1] is None)

    def _set_preset_label(self, path: str):
        self.preset_label.setText(os.path.basename(path))
        self.preset_label.setStyleSheet("")

    def _choose_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose LUT preset", "", "JSON (*.json)")
        if path:
            self._preset_path = path
            self._set_preset_label(path)

    def _clear_preset(self):
        self._preset_path = None
        self.preset_label.setText("Use current on-screen settings")
        self.preset_label.setStyleSheet("color: palette(mid);")

    def _choose_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", self.out_dir_edit.text())
        if d:
            self.out_dir_edit.setText(d)

    def get_options(self) -> ExportOptions:
        idx = self.downsample_combo.currentIndex()
        scale = _DOWNSAMPLE_PRESETS[idx][1]
        if scale is None:
            scale = 1.0 / self.custom_spin.value()
        fmt = ".png" if self.format_combo.currentText() == "PNG" else ".tif"
        return ExportOptions(
            scope_all_files=self.scope_all.isChecked(),
            preset_path=self._preset_path,
            scale=scale,
            fmt=fmt,
            out_dir=self.out_dir_edit.text() or ".",
            export_rois=self.roi_check.isChecked(),
        )
