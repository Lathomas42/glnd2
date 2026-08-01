from __future__ import annotations

import json
import os
import re
import traceback

import numpy as np
from PySide6.QtCore import QSettings, QThread, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import nd2_loader
from channel_panel import ChannelPanel
from export_dialog import ExportDialog
from gl_widget import ChannelState, CompositeGLWidget

ND2_EXT = ".nd2"


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


class LoaderThread(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            img = nd2_loader.load_nd2(self.path)
            self.loaded.emit(img)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("glnd2")
        self.resize(1500, 950)

        self._folder = None
        self._files: list[str] = []
        self._current_index = -1
        self._carried_states: dict[str, ChannelState] = {}
        self._loader_thread: LoaderThread | None = None
        self._panels: list[ChannelPanel] = []
        self._last_out_dir: str | None = None
        self._last_preset_path: str | None = None

        self._settings = QSettings("glnd2", "glnd2")
        self._default_lut_path: str | None = self._settings.value("default_lut_path", None)
        self._use_default_lut: bool = self._settings.value("use_default_lut", False, type=bool)
        self._default_lut_data: dict | None = None
        if self._default_lut_path and os.path.exists(self._default_lut_path):
            try:
                with open(self._default_lut_path) as f:
                    self._default_lut_data = json.load(f)
            except Exception:
                self._default_lut_data = None
                self._default_lut_path = None

        self._build_ui()
        self._build_actions()

    # ------------------------------------------------------------------
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # left: file browser
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        open_btn = QPushButton("Open Folder…")
        open_btn.clicked.connect(self.open_folder)
        left_layout.addWidget(open_btn)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentRowChanged.connect(self._on_row_selected)
        left_layout.addWidget(self.file_list, 1)
        splitter.addWidget(left)

        # center: GL composite view
        self.gl = CompositeGLWidget()
        self.gl.measured.connect(self._on_measured)
        splitter.addWidget(self.gl)

        # right: channel controls
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(QLabel("<b>Channels</b>"))
        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)
        self.panel_container = QWidget()
        self.panel_layout = QVBoxLayout(self.panel_container)
        self.panel_layout.addStretch(1)
        self.panel_scroll.setWidget(self.panel_container)
        right_layout.addWidget(self.panel_scroll, 1)
        splitter.addWidget(right)

        splitter.setSizes([260, 950, 380])
        self.setCentralWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_label = QLabel("No folder opened")
        self.status.addWidget(self.status_label, 1)

    def _build_actions(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_open = QAction("Open Folder…", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self.open_folder)
        tb.addAction(act_open)
        tb.addSeparator()

        act_prev = QAction("◀ Prev", self)
        act_prev.setShortcut(QKeySequence("Ctrl+Left"))
        act_prev.triggered.connect(self.go_prev)
        tb.addAction(act_prev)

        act_next = QAction("Next ▶", self)
        act_next.setShortcut(QKeySequence("Ctrl+Right"))
        act_next.triggered.connect(self.go_next)
        tb.addAction(act_next)
        tb.addSeparator()

        act_save = QAction("Save Composite", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self.save_composite)
        tb.addAction(act_save)

        act_save_as = QAction("Save As…", self)
        act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self.save_composite_as)
        tb.addAction(act_save_as)

        act_save_next = QAction("Save && Next", self)
        act_save_next.setShortcut(QKeySequence("Ctrl+Return"))
        act_save_next.triggered.connect(self.save_and_next)
        tb.addAction(act_save_next)
        tb.addSeparator()

        act_export = QAction("Export…", self)
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self.run_export_dialog)
        tb.addAction(act_export)
        tb.addSeparator()

        act_reset_view = QAction("Reset View", self)
        act_reset_view.triggered.connect(self.gl.reset_view)
        tb.addAction(act_reset_view)

        act_measure = QAction("Measure", self)
        act_measure.setCheckable(True)
        act_measure.setShortcut(QKeySequence("Ctrl+M"))
        act_measure.toggled.connect(self.gl.set_measure_mode)
        tb.addAction(act_measure)
        tb.addSeparator()

        act_save_preset = QAction("Save Preset…", self)
        act_save_preset.triggered.connect(self.save_preset)
        tb.addAction(act_save_preset)

        act_load_preset = QAction("Load Preset…", self)
        act_load_preset.triggered.connect(self.load_preset)
        tb.addAction(act_load_preset)
        tb.addSeparator()

        act_set_default_lut = QAction("Set Default LUT…", self)
        act_set_default_lut.triggered.connect(self.set_default_lut)
        tb.addAction(act_set_default_lut)

        self.act_use_default_lut = QAction("Use Default LUT", self)
        self.act_use_default_lut.setCheckable(True)
        self.act_use_default_lut.setChecked(self._use_default_lut)
        self.act_use_default_lut.setEnabled(self._default_lut_data is not None)
        self.act_use_default_lut.toggled.connect(self._toggle_use_default_lut)
        tb.addAction(self.act_use_default_lut)

    # ------------------------------------------------------------------
    # Folder / file navigation
    # ------------------------------------------------------------------
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder with .nd2 files")
        if not folder:
            return
        files = sorted(
            (f for f in os.listdir(folder) if f.lower().endswith(ND2_EXT)),
            key=_natural_key,
        )
        if not files:
            QMessageBox.warning(self, "No files", "No .nd2 files found in that folder.")
            return
        self._folder = folder
        self._files = files
        self._carried_states = {}
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for f in files:
            self.file_list.addItem(QListWidgetItem(f))
        self.file_list.blockSignals(False)
        self.file_list.setCurrentRow(0)

    def _on_row_selected(self, row: int):
        if row < 0 or row >= len(self._files):
            return
        self._current_index = row
        path = os.path.join(self._folder, self._files[row])
        self._load_file(path)

    def go_prev(self):
        if self._current_index > 0:
            self.file_list.setCurrentRow(self._current_index - 1)

    def go_next(self):
        if self._current_index < len(self._files) - 1:
            self.file_list.setCurrentRow(self._current_index + 1)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_file(self, path: str):
        if self._loader_thread is not None and self._loader_thread.isRunning():
            return
        self.status_label.setText(f"Loading {os.path.basename(path)}…")
        self.setEnabled(False)
        thread = LoaderThread(path)
        thread.loaded.connect(self._on_loaded)
        thread.failed.connect(self._on_load_failed)
        thread.finished.connect(lambda: self.setEnabled(True))
        self._loader_thread = thread
        thread.start()

    def _on_loaded(self, img: nd2_loader.Nd2Image):
        effective = self._effective_carried_states([c.name for c in img.channels])
        self.gl.set_image(img, carried_states=effective)
        self._rebuild_panels(img)
        # Merge, don't replace: a file with fewer channels than a previous
        # one (e.g. a single-channel acquisition mixed into a folder of
        # 4-channel ones) must not wipe out carried settings for channels
        # it simply doesn't have.
        self._carried_states.update({st.name: st for st in self.gl.get_states()})
        warn = "  [downsampled: exceeds GPU max texture size]" if self.gl.is_downsampled() else ""
        self.status_label.setText(
            f"[{self._current_index + 1}/{len(self._files)}] {os.path.basename(img.path)}  "
            f"({img.width}×{img.height}, {len(img.channels)} ch){warn}"
        )

    def _on_measured(self, result):
        if result is None:
            return
        dist_px, dist_um = result
        if dist_um is not None:
            self.status_label.setText(f"Measured: {dist_um:.2f} µm  ({dist_px:.1f} px)")
        else:
            self.status_label.setText(f"Measured: {dist_px:.1f} px  (no pixel calibration in this file)")

    def _on_load_failed(self, err: str):
        QMessageBox.critical(self, "Failed to load ND2", err)
        self.status_label.setText("Load failed")

    def _rebuild_panels(self, img: nd2_loader.Nd2Image):
        for p in self._panels:
            p.setParent(None)
            p.deleteLater()
        self._panels = []

        states = self.gl.get_states()
        for i, (ch, st) in enumerate(zip(img.channels, states)):
            panel = ChannelPanel(
                index=i, name=ch.name, data_max=ch.data_max, color=st.color,
                black=st.black, white=st.white, gamma=st.gamma, enabled=st.enabled,
                default_black=ch.p_low, default_white=ch.p_high,
            )
            panel.set_histogram(ch.hist_counts, ch.data_max, st.color)
            panel.changed.connect(self._on_panel_changed)
            self.panel_layout.insertWidget(i, panel)
            self._panels.append(panel)

    def _on_panel_changed(self, index: int):
        panel = self._panels[index]
        vals = panel.values()
        self.gl.set_channel_param(index, **vals)
        panel.hist.set_data(panel.hist._counts, panel.hist._data_max, vals["color"])
        panel.hist.set_markers(vals["black"], vals["white"])

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _default_out_path(self, ext=".png") -> str | None:
        if not (0 <= self._current_index < len(self._files)):
            return None
        base = os.path.splitext(self._files[self._current_index])[0]
        return os.path.join(self._folder, f"{base}_composite{ext}")

    def _render_and_save(self, out_path: str, scale: float = 1.0) -> bool:
        arr = self.gl.render_to_array(scale=scale)
        if arr is None:
            QMessageBox.warning(self, "Nothing to save", "No image is loaded.")
            return False
        try:
            if out_path.lower().endswith((".tif", ".tiff")):
                import tifffile
                tifffile.imwrite(out_path, arr, photometric="rgb")
            else:
                from PIL import Image
                Image.fromarray(arr).save(out_path)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return False
        return True

    def save_composite(self):
        out_path = self._default_out_path(".png")
        if not out_path:
            return
        if self._render_and_save(out_path):
            self.status_label.setText(f"Saved {out_path}")

    def save_composite_as(self):
        default = self._default_out_path(".png") or ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save composite as", default, "PNG (*.png);;TIFF (*.tif)"
        )
        if path and self._render_and_save(path):
            self.status_label.setText(f"Saved {path}")

    def save_and_next(self):
        self.save_composite()
        self.go_next()

    # ------------------------------------------------------------------
    # Export dialog (scope, LUT preset, downsample, format, destination)
    # ------------------------------------------------------------------
    def run_export_dialog(self):
        if not self._files:
            QMessageBox.information(self, "No files", "Open a folder first.")
            return
        default_dir = self._last_out_dir or self._folder
        dlg = ExportDialog(default_dir, initial_preset_path=self._last_preset_path, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.get_options()

        try:
            os.makedirs(opts.out_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self._last_out_dir = opts.out_dir

        preset_names = None
        if opts.preset_path:
            try:
                with open(opts.preset_path) as f:
                    data = json.load(f)
            except Exception as e:
                QMessageBox.critical(self, "Failed to load preset", str(e))
                return
            preset_names = set(data.keys())
            self._carried_states.update(self._preset_dict_to_states(data))
            self._apply_preset_data_to_live(data)
            self._last_preset_path = opts.preset_path

        if not opts.scope_all_files:
            fname = self._files[self._current_index]
            if preset_names is not None:
                live_names = {st.name for st in self.gl.get_states()}
                missing = live_names - preset_names
                if missing:
                    QMessageBox.warning(
                        self, "Preset channel mismatch",
                        f"This file has channel(s) not in the preset, using current/default "
                        f"values instead: {', '.join(sorted(missing))}\n\n"
                        f"Preset has: {', '.join(sorted(preset_names))}",
                    )
            out_path = os.path.join(opts.out_dir, f"{os.path.splitext(fname)[0]}_composite{opts.fmt}")
            if self._render_and_save(out_path, scale=opts.scale):
                self.status_label.setText(f"Exported {out_path}")
            return

        self._export_all_files(opts.out_dir, opts.fmt, opts.scale, preset_names=preset_names)

    def _export_all_files(self, out_dir: str, fmt: str, scale: float, preset_names: set[str] | None = None):
        progress = QProgressDialog("Exporting composites…", "Cancel", 0, len(self._files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        unmatched: dict[str, set[str]] = {}
        for i, fname in enumerate(self._files):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(f"Exporting {fname}")
            QApplication.processEvents()

            path = os.path.join(self._folder, fname)
            try:
                img = nd2_loader.load_nd2(path)
            except Exception as e:
                QMessageBox.warning(self, "Skipped", f"Failed to load {fname}: {e}")
                continue

            if preset_names is not None:
                missing = {c.name for c in img.channels} - preset_names
                if missing:
                    unmatched[fname] = missing

            effective = self._effective_carried_states([c.name for c in img.channels])
            self.gl.set_image(img, carried_states=effective)
            self._carried_states.update({st.name: st for st in self.gl.get_states()})
            out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_composite{fmt}")
            self._render_and_save(out_path, scale=scale)

        progress.setValue(len(self._files))

        if unmatched:
            lines = [f"{fname}: {', '.join(sorted(names))}" for fname, names in unmatched.items()]
            QMessageBox.warning(
                self, "Preset channel mismatch",
                "The preset didn't cover every channel in these files (they used "
                "current/default values instead):\n\n" + "\n".join(lines[:15])
                + ("\n…" if len(lines) > 15 else ""),
            )

        self.status_label.setText("Export complete")
        # Reload currently-selected file so the on-screen view matches the list selection.
        if 0 <= self._current_index < len(self._files):
            self._load_file(os.path.join(self._folder, self._files[self._current_index]))

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    @staticmethod
    def _preset_dict_to_states(data: dict) -> dict[str, ChannelState]:
        return {
            name: ChannelState(
                name=name, data_max=65535, color=tuple(vals["color"]),
                black=vals["black"], white=vals["white"], gamma=vals["gamma"],
                enabled=vals["enabled"],
            )
            for name, vals in data.items()
        }

    def _apply_preset_data_to_live(self, data: dict):
        """Push preset values onto the currently loaded channels/panels, if any match by name."""
        for i, st in enumerate(self.gl.get_states()):
            if st.name in data:
                v = data[st.name]
                self.gl.set_channel_param(i, black=v["black"], white=v["white"],
                                           gamma=v["gamma"], enabled=v["enabled"],
                                           color=tuple(v["color"]))
                if i < len(self._panels):
                    self._panels[i].set_values(v["black"], v["white"], v["gamma"],
                                                v["enabled"], tuple(v["color"]))

    def save_preset(self):
        if not self._carried_states:
            QMessageBox.information(self, "No data", "Load a file first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save LUT preset", "lut_preset.json", "JSON (*.json)")
        if not path:
            return
        data = {
            name: dict(black=st.black, white=st.white, gamma=st.gamma,
                       enabled=st.enabled, color=list(st.color))
            for name, st in self._carried_states.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._last_preset_path = path
        self.status_label.setText(f"Preset saved to {path}")

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load LUT preset", "", "JSON (*.json)")
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        self._carried_states.update(self._preset_dict_to_states(data))
        self._apply_preset_data_to_live(data)
        self._last_preset_path = path
        self.status_label.setText(f"Preset loaded from {path}")

    # ------------------------------------------------------------------
    # Default LUT (used instead of auto-contrast for channels with no
    # carried-over value yet, e.g. the first time a channel name is seen)
    # ------------------------------------------------------------------
    def set_default_lut(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose default LUT preset", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load preset", str(e))
            return
        self._default_lut_path = path
        self._default_lut_data = data
        self._use_default_lut = True
        self._settings.setValue("default_lut_path", path)
        self._settings.setValue("use_default_lut", True)
        self.act_use_default_lut.setEnabled(True)
        self.act_use_default_lut.setChecked(True)
        self.status_label.setText(f"Default LUT set to {path}")

    def _toggle_use_default_lut(self, checked: bool):
        self._use_default_lut = checked
        self._settings.setValue("use_default_lut", checked)

    def _effective_carried_states(self, channel_names: list[str]) -> dict[str, ChannelState]:
        """`_carried_states`, plus default-LUT values for any of `channel_names`
        that have no carried value yet. Session-carried values always win over
        the default; auto-contrast is still the fallback for channels absent
        from both."""
        if not self._use_default_lut or not self._default_lut_data:
            return self._carried_states
        missing = [n for n in channel_names if n not in self._carried_states and n in self._default_lut_data]
        if not missing:
            return self._carried_states
        merged = dict(self._carried_states)
        for name in missing:
            merged[name] = self._preset_dict_to_states({name: self._default_lut_data[name]})[name]
        return merged
