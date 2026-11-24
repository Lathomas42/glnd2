from __future__ import annotations

import dataclasses
import json
import os
import re
import traceback

import numpy as np
from PySide6.QtCore import QSettings, QThread, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QOffscreenSurface
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
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

import file_lut_store
import nd2_loader
import roi_store
from channel_panel import ChannelPanel
from export_dialog import ExportDialog
from gl_widget import MAX_CHANNELS, ChannelState, CompositeGLWidget, default_gl_format, upload_in_background

ND2_EXT = ".nd2"


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)


def _rgb255(color) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in color)


def _state_to_dict(st: ChannelState) -> dict:
    return dict(black=st.black, white=st.white, gamma=st.gamma, enabled=st.enabled, color=list(st.color))


def _state_from_dict(name: str, vals: dict, default_color) -> ChannelState:
    """Tolerates presets missing optional keys (hand-edited or older files)."""
    return ChannelState(
        name=name,
        color=tuple(vals.get("color", default_color)),
        black=float(vals["black"]),
        white=float(vals["white"]),
        gamma=float(vals.get("gamma", 1.0)),
        enabled=bool(vals.get("enabled", True)),
    )


class LoaderThread(QThread):
    loaded = Signal(object)  # Nd2Image
    failed = Signal(str, str)  # path, traceback

    def __init__(self, path: str, surface: QOffscreenSurface | None):
        super().__init__()
        self.path = path
        self.surface = surface

    def run(self):
        try:
            img = nd2_loader.load_nd2(self.path)
        except Exception:
            self.failed.emit(self.path, traceback.format_exc())
            return
        # Upload here too, so showing the image never stalls the UI.
        upload_in_background(img, self.surface)
        self.loaded.emit(img)


class MainWindow(QMainWindow):
    """File browser + LUT controls around the GPU compositor.

    Each file's channels are displayed with, in order of precedence: the
    file's saved per-file LUT (Save); otherwise whatever the same channel
    looked like on the previous file (levels, gamma, color, visibility carry
    over as you browse); otherwise the Default LUT, if enabled; otherwise
    auto-contrast. Saved per-file LUTs never carry over to other files.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("glnd2")
        self.resize(1500, 950)

        self._folder: str | None = None
        self._files: list[str] = []
        self._current_index = -1
        # Channels of the displayed file, minus pixel data (that lives on the GPU).
        self._channels: list[nd2_loader.ChannelData] = []
        # Last-seen settings per raw channel name, carried to the next file.
        self._carried: dict[str, ChannelState] = {}
        # Channels of the displayed file whose settings came from its saved LUT.
        self._override_names: set[str] = set()
        self._panels: list[ChannelPanel] = []

        # The single background loader. Cleared only by its own `finished`
        # handler, so a new loader never starts (and never replaces this
        # one's last reference) while it is still running.
        self._loader: LoaderThread | None = None
        self._unreadable: set[str] = set()  # paths that failed; not read ahead again
        self._want_row: int | None = None  # row requested but not yet shown
        self._prefetched: nd2_loader.Nd2Image | None = None
        self._export_view: CompositeGLWidget | None = None  # hidden, for batch export

        self._rois: list[roi_store.ROI] = []
        self._pending_center_roi: roi_store.ROI | None = None
        self._overridden_stems: set[str] = set()
        self._last_out_dir: str | None = None

        # Surface for the loader thread's GL context (must be made on this thread).
        self._upload_surface = QOffscreenSurface()
        self._upload_surface.setFormat(default_gl_format())
        self._upload_surface.create()

        self._settings = QSettings("glnd2", "glnd2")
        self._auto_headroom = float(self._settings.value(
            "auto_headroom", nd2_loader.DEFAULT_AUTO_HEADROOM, type=float))
        self._default_lut_path: str | None = self._settings.value("default_lut_path", None)
        self._use_default_lut: bool = self._settings.value("use_default_lut", False, type=bool)
        self._default_lut_data: dict | None = None
        if self._default_lut_path:
            try:
                self._default_lut_data = _read_json(self._default_lut_path)
            except Exception:
                self._default_lut_path = None

        palette_name = self._settings.value("color_palette", nd2_loader.DEFAULT_PALETTE)
        if palette_name not in nd2_loader.COLOR_PALETTES:
            palette_name = nd2_loader.DEFAULT_PALETTE
        nd2_loader.set_active_palette(palette_name)

        self._build_ui()
        self._build_actions()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

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

        self.gl = CompositeGLWidget()
        self.gl.measured.connect(self._on_measured)
        self.gl.roiDrawn.connect(self._on_roi_drawn)
        splitter.addWidget(self.gl)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        channels_header = QHBoxLayout()
        channels_header.addWidget(QLabel("<b>Channels</b>"))
        channels_header.addStretch(1)
        auto_all_btn = QPushButton("Auto")
        auto_all_btn.setToolTip("Auto-contrast every channel of this file (see Auto headroom).\n"
                                "The result carries to the next files like any other setting.")
        auto_all_btn.clicked.connect(self.auto_all_channels)
        channels_header.addWidget(auto_all_btn)
        right_layout.addLayout(channels_header)
        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_container = QWidget()
        self.panel_layout = QVBoxLayout(panel_container)
        self.panel_layout.addStretch(1)
        panel_scroll.setWidget(panel_container)
        right_layout.addWidget(panel_scroll, 1)
        splitter.addWidget(right)

        splitter.setSizes([260, 950, 380])
        self.setCentralWidget(splitter)

        roi_dock = QDockWidget("ROIs", self)
        roi_dock_widget = QWidget()
        roi_dock_layout = QVBoxLayout(roi_dock_widget)
        roi_dock_layout.setContentsMargins(4, 4, 4, 4)
        self.roi_list = QListWidget()
        self.roi_list.itemClicked.connect(self._on_roi_list_clicked)
        roi_dock_layout.addWidget(self.roi_list, 1)
        roi_btn_row = QHBoxLayout()
        roi_rename_btn = QPushButton("Rename")
        roi_rename_btn.clicked.connect(self._rename_selected_roi)
        roi_delete_btn = QPushButton("Delete")
        roi_delete_btn.clicked.connect(self._delete_selected_roi)
        roi_btn_row.addWidget(roi_rename_btn)
        roi_btn_row.addWidget(roi_delete_btn)
        roi_dock_layout.addLayout(roi_btn_row)
        roi_dock.setWidget(roi_dock_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, roi_dock)

        status = QStatusBar()
        self.setStatusBar(status)
        self.status_label = QLabel("No folder opened")
        status.addWidget(self.status_label, 1)

    def _build_actions(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        def action(text, slot, shortcut=None, tip=None, checkable=False):
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            if tip:
                act.setToolTip(tip)
            act.setCheckable(checkable)
            (act.toggled if checkable else act.triggered).connect(slot)
            tb.addAction(act)
            return act

        action("Open Folder…", self.open_folder, "Ctrl+O")
        tb.addSeparator()
        action("◀ Prev", self.go_prev, "Ctrl+Left")
        action("Next ▶", self.go_next, "Ctrl+Right")
        tb.addSeparator()
        action("Save", self.save_file_lut, "Ctrl+S",
               "Save this file's channel LUTs; it will always open with them (● in the file list)")
        action("Revert", self.revert_file_lut, None,
               "Delete this file's saved LUTs and go back to the carried-over settings")
        tb.addSeparator()
        action("Export…", self.run_export_dialog, "Ctrl+E")
        tb.addSeparator()
        action("Reset View", self.gl.reset_view)
        self.act_measure = action("Measure", self._on_measure_toggled, "Ctrl+M", checkable=True)
        self.act_roi = action("Draw ROI", self._on_roi_mode_toggled, "Ctrl+R", checkable=True)
        tb.addSeparator()

        tb.addWidget(QLabel(" Auto headroom: "))
        self.headroom_spin = QDoubleSpinBox()
        self.headroom_spin.setRange(1.0, 20.0)
        self.headroom_spin.setSingleStep(0.5)
        self.headroom_spin.setDecimals(1)
        self.headroom_spin.setSuffix("×")
        self.headroom_spin.setValue(self._auto_headroom)
        self.headroom_spin.setToolTip(
            "Auto-contrast white point = black + this × (98th percentile − black).\n"
            "Higher = darker image, fewer saturated pixels. Used by the Auto\n"
            "buttons, and applies live to channels currently on auto.")
        self.headroom_spin.valueChanged.connect(self._on_headroom_changed)
        tb.addWidget(self.headroom_spin)
        tb.addSeparator()

        action("Save Preset…", self.save_preset, tip="Save this file's channel LUTs to a preset file")
        action("Load Preset…", self.load_preset, tip="Apply a preset file's LUTs to this file")
        tb.addSeparator()
        action("Set Default LUT…", self.set_default_lut,
               tip="A preset to start from instead of auto-contrast, for channels with nothing to carry over")
        self.act_use_default_lut = action("Use Default LUT", self._toggle_use_default_lut, checkable=True)
        self.act_use_default_lut.setChecked(self._use_default_lut)
        self.act_use_default_lut.setEnabled(self._default_lut_data is not None)
        tb.addSeparator()

        tb.addWidget(QLabel(" Palette: "))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list(nd2_loader.COLOR_PALETTES.keys()))
        self.palette_combo.setCurrentText(nd2_loader.get_active_palette())
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        tb.addWidget(self.palette_combo)
        action("Save Palette…", self.save_new_palette)

    # ------------------------------------------------------------------
    # Folder / file navigation
    # ------------------------------------------------------------------
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder with .nd2 files")
        if not folder:
            return
        files = sorted((f for f in os.listdir(folder) if f.lower().endswith(ND2_EXT)), key=_natural_key)
        if not files:
            QMessageBox.warning(self, "No files", "No .nd2 files found in that folder.")
            return
        try:
            rois = roi_store.load_rois(folder)
        except Exception as e:
            QMessageBox.warning(self, "ROIs not loaded",
                                f"Couldn't read {roi_store.rois_path(folder)}; starting with no ROIs.\n\n{e}")
            rois = []

        self._folder = folder
        self._files = files
        self._current_index = -1
        self._want_row = None
        self._set_prefetched(None)
        self._rois = rois
        self._pending_center_roi = None
        self._overridden_stems = file_lut_store.list_overridden_stems(folder)
        self._refresh_roi_list()
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for f in files:
            self.file_list.addItem(QListWidgetItem(self._file_list_label(f)))
        self.file_list.blockSignals(False)
        self.file_list.setCurrentRow(0)

    def _file_list_label(self, fname: str) -> str:
        return f"● {fname}" if os.path.splitext(fname)[0] in self._overridden_stems else fname

    def _refresh_file_list_labels(self):
        for row, fname in enumerate(self._files):
            item = self.file_list.item(row)
            if item is not None:
                item.setText(self._file_list_label(fname))

    def _current_file(self) -> str | None:
        if 0 <= self._current_index < len(self._files):
            return self._files[self._current_index]
        return None

    def _path(self, row: int) -> str:
        return os.path.join(self._folder, self._files[row])

    def _on_row_selected(self, row: int):
        if 0 <= row < len(self._files):
            self._request_load(row)

    def go_prev(self):
        row = self.file_list.currentRow()
        if row > 0:
            self.file_list.setCurrentRow(row - 1)

    def go_next(self):
        row = self.file_list.currentRow()
        if row < len(self._files) - 1:
            self.file_list.setCurrentRow(row + 1)

    # ------------------------------------------------------------------
    # Loading: one background loader at a time. After a file is shown, the
    # next file is read ahead so Next ▶ only has to upload it to the GPU.
    # ------------------------------------------------------------------
    def _loader_running(self) -> bool:
        return self._loader is not None

    def _request_load(self, row: int):
        self._want_row = row
        path = self._path(row)
        self._unreadable.discard(path)  # an explicit request always retries
        if self._prefetched is not None and self._prefetched.path == path:
            img, self._prefetched = self._prefetched, None
            self._display(row, img)
            return
        self.status_label.setText(f"Loading {self._files[row]}…")
        # If a loader (maybe a read-ahead) is busy, it finishes first and
        # _on_loader_finished then starts the wanted row.
        if not self._loader_running():
            self._start_loader(path)

    def _start_loader(self, path: str):
        thread = LoaderThread(path, self._upload_surface)
        thread.loaded.connect(self._on_loaded)
        thread.failed.connect(self._on_load_failed)
        thread.finished.connect(self._on_loader_finished)
        self._loader = thread
        thread.start()

    def _on_loaded(self, img: nd2_loader.Nd2Image):
        if os.path.dirname(img.path) != self._folder:
            self.gl.discard(img)  # from a folder that has since been closed
        elif self._want_row is not None and self._path(self._want_row) == img.path:
            self._display(self._want_row, img)
        else:
            self._set_prefetched(img)  # read-ahead, or a request since superseded

    def _set_prefetched(self, img: nd2_loader.Nd2Image | None):
        """Hold at most one not-displayed image, freeing the one it replaces."""
        if self._prefetched is not None and self._prefetched is not img:
            self.gl.discard(self._prefetched)
        self._prefetched = img

    def _on_load_failed(self, path: str, err: str):
        self._unreadable.add(path)
        if self._want_row is not None and self._path(self._want_row) == path:
            self._want_row = None
            QMessageBox.critical(self, "Failed to load ND2", err)
            self.status_label.setText("Load failed")

    def _on_loader_finished(self):
        thread, self._loader = self._loader, None
        thread.wait()  # returns at once: run() is over, the thread is just winding down
        if self._folder is None:
            return
        if self._want_row is not None:
            self._start_loader(self._path(self._want_row))
        else:
            self._read_ahead()

    def _read_ahead(self):
        nxt = self._current_index + 1
        if not (0 <= nxt < len(self._files)) or self._loader_running():
            return
        path = self._path(nxt)
        if path in self._unreadable or (self._prefetched is not None and self._prefetched.path == path):
            return
        self._set_prefetched(None)
        self._start_loader(path)

    def closeEvent(self, event):
        if self._loader is not None:
            self._loader.wait()  # a QThread must not be destroyed while running
        self._set_prefetched(None)
        super().closeEvent(event)

    def _display(self, row: int, img: nd2_loader.Nd2Image):
        self._want_row = None
        self._current_index = row
        self._show_image(img)
        self._read_ahead()

    def _show_image(self, img: nd2_loader.Nd2Image):
        fname = os.path.basename(img.path)
        overridden = os.path.splitext(fname)[0] in self._overridden_stems
        self._capture_carry()
        states, self._override_names = self._resolve_states(img.path, img.channels)
        self.gl.set_image(img, states)
        self._channels = [dataclasses.replace(ch, array=None) for ch in img.channels[:MAX_CHANNELS]]
        self._rebuild_panels()

        notes = []
        if overridden:
            notes.append("saved LUT")
        if self.gl.is_downsampled():
            notes.append("downsampled: exceeds GPU max texture size")
        self.status_label.setText(
            f"[{self._current_index + 1}/{len(self._files)}] {fname}  "
            f"({img.width}×{img.height}, {len(img.channels)} ch)"
            + "".join(f"  [{n}]" for n in notes)
        )

        self.gl.set_rois([r for r in self._rois if r.file == fname])
        roi = self._pending_center_roi
        if roi is not None and roi.file == fname:
            self._pending_center_roi = None
            self.gl.center_on_roi(roi.x, roi.y, roi.w, roi.h)

    # ------------------------------------------------------------------
    # LUT resolution: the single place deciding how a file's channels look
    # ------------------------------------------------------------------
    def _resolve_states(self, path: str, channels: list[nd2_loader.ChannelData], preset: dict | None = None,
                        carried: dict[str, ChannelState] | None = None) -> tuple[list[ChannelState], set[str]]:
        """Display settings for each channel of the file at `path`, plus the
        names of the channels that came from its saved LUT. Precedence: the
        file's saved LUT, then `preset` (an export-time choice), then the
        settings carried over from previous files, then the Default LUT (if
        enabled), then auto-contrast."""
        folder, fname = os.path.split(path)
        override = file_lut_store.load_override(folder, fname) or {}
        preset = preset or {}
        carried = self._carried if carried is None else carried
        default = (self._default_lut_data or {}) if self._use_default_lut else {}

        states, override_names = [], set()
        for i, ch in enumerate(channels[:MAX_CHANNELS]):
            palette_color = nd2_loader.default_color_for_channel(ch.name, i)
            if ch.name in override:
                st = _state_from_dict(ch.name, override[ch.name], palette_color)
                override_names.add(ch.name)
            elif ch.name in preset:
                st = _state_from_dict(ch.name, preset[ch.name], palette_color)
            elif ch.name in carried:
                st = dataclasses.replace(carried[ch.name])
            elif ch.name in default:
                st = _state_from_dict(ch.name, default[ch.name], palette_color)
            else:
                black, white = nd2_loader.auto_levels(ch.stats, self._auto_headroom)
                st = ChannelState(ch.name, palette_color, black, white, auto=True)
            states.append(st)
        return states, override_names

    def _capture_carry(self):
        """Remember the displayed file's settings for the next files, except
        channels that came from its saved LUT: those are specific to it."""
        for st in self.gl.get_states():
            if st.name not in self._override_names:
                self._carried[st.name] = dataclasses.replace(st, auto=False)

    def _set_live_states(self, states: list[ChannelState]):
        """Replace the displayed file's channel settings and sync the panels."""
        for i, st in enumerate(states):
            self.gl.set_channel_param(i, **dataclasses.asdict(st))
            if i < len(self._panels):
                self._panels[i].set_values(st.black, st.white, st.gamma, st.enabled, st.color)

    def _rebuild_panels(self):
        for p in self._panels:
            p.setParent(None)
            p.deleteLater()
        self._panels = []
        for i, (ch, st) in enumerate(zip(self._channels, self.gl.get_states())):
            panel = ChannelPanel(i, ch.display_name, ch.stats.display_max)
            panel.set_histogram(ch.hist_counts, ch.stats.display_max)
            panel.set_values(st.black, st.white, st.gamma, st.enabled, st.color)
            panel.changed.connect(self._on_panel_changed)
            panel.autoRequested.connect(self._on_auto_requested)
            panel.resetRequested.connect(self._on_reset_requested)
            self.panel_layout.insertWidget(i, panel)
            self._panels.append(panel)

    def _on_panel_changed(self, index: int):
        st = self.gl.get_states()[index]
        vals = self._panels[index].values()
        # The panel shows whole counts; only take black/white from it when
        # the user actually moved them, so auto levels keep full precision.
        if (vals["black"], vals["white"]) != (round(st.black), round(st.white)):
            self.gl.set_channel_param(index, black=vals["black"], white=vals["white"], auto=False)
        if _rgb255(vals["color"]) != _rgb255(st.color):
            self.gl.set_channel_param(index, color=vals["color"])
        self.gl.set_channel_param(index, gamma=vals["gamma"], enabled=vals["enabled"])

    def _on_auto_requested(self, index: int):
        st = self.gl.get_states()[index]
        black, white = nd2_loader.auto_levels(self._channels[index].stats, self._auto_headroom)
        self.gl.set_channel_param(index, black=black, white=white, auto=True)
        self._panels[index].set_values(black, white, st.gamma, st.enabled, st.color)

    def auto_all_channels(self):
        for i in range(len(self.gl.get_states())):
            self._on_auto_requested(i)
        if self.gl.get_states():
            self.status_label.setText("Auto-contrasted all channels of this file")

    def _on_reset_requested(self, index: int):
        st = self.gl.get_states()[index]
        black, white = nd2_loader.auto_levels(self._channels[index].stats, self._auto_headroom)
        color = nd2_loader.default_color_for_channel(st.name, index)
        self.gl.set_channel_param(index, black=black, white=white, gamma=1.0, enabled=True,
                                  color=color, auto=True)
        self._panels[index].set_values(black, white, 1.0, True, color)

    def _on_headroom_changed(self, value: float):
        self._auto_headroom = value
        self._settings.setValue("auto_headroom", value)
        for i, st in enumerate(self.gl.get_states()):
            if st.auto:
                black, white = nd2_loader.auto_levels(self._channels[i].stats, value)
                self.gl.set_channel_param(i, black=black, white=white)
                self._panels[i].set_values(black, white, st.gamma, st.enabled, st.color)

    def _reresolve_current(self, preset: dict | None = None):
        """Re-derive the displayed file's settings, e.g. after its saved LUT
        was removed. Discards unsaved tweaks."""
        fname = self._current_file()
        if fname is None or not self._channels:
            return
        states, self._override_names = self._resolve_states(self._path(self._current_index), self._channels, preset)
        self._set_live_states(states)

    # ------------------------------------------------------------------
    # Per-file LUTs
    # ------------------------------------------------------------------
    def save_file_lut(self):
        fname = self._current_file()
        if fname is None:
            return
        data = {st.name: _state_to_dict(st) for st in self.gl.get_states()}
        file_lut_store.save_override(self._folder, fname, data)
        # From now on these settings belong to this file alone: they won't
        # carry over to the next file.
        self._override_names = set(data)
        self._overridden_stems.add(os.path.splitext(fname)[0])
        self._refresh_file_list_labels()
        self.status_label.setText(f"Saved LUTs for {fname}")

    def revert_file_lut(self):
        fname = self._current_file()
        if fname is None:
            return
        file_lut_store.clear_override(self._folder, fname)
        self._overridden_stems.discard(os.path.splitext(fname)[0])
        self._refresh_file_list_labels()
        self._reresolve_current()
        self.status_label.setText(f"Reverted {fname} to the carried-over settings")

    # ------------------------------------------------------------------
    # Measurement / ROI tools
    # ------------------------------------------------------------------
    def _on_measured(self, result):
        if result is None:
            return
        dist_px, dist_um = result
        if dist_um is not None:
            self.status_label.setText(f"Measured: {dist_um:.2f} µm  ({dist_px:.1f} px)")
        else:
            self.status_label.setText(f"Measured: {dist_px:.1f} px  (no pixel calibration in this file)")

    def _on_measure_toggled(self, checked: bool):
        self.gl.set_measure_mode(checked)
        if checked:
            self.act_roi.setChecked(False)

    def _on_roi_mode_toggled(self, checked: bool):
        self.gl.set_roi_mode(checked)
        if checked:
            self.act_measure.setChecked(False)

    def _on_roi_drawn(self, x: float, y: float, w: float, h: float):
        fname = self._current_file()
        if fname is None:
            return
        name, ok = QInputDialog.getText(self, "Name ROI", "Name:")
        if not ok or not name.strip():
            return
        self._rois.append(roi_store.ROI.new(name.strip(), fname, x, y, w, h))
        self._rois_changed(fname)

    def _rois_changed(self, fname: str):
        roi_store.save_rois(self._folder, self._rois)
        self._refresh_roi_list()
        if fname == self._current_file():
            self.gl.set_rois([r for r in self._rois if r.file == fname])

    def _refresh_roi_list(self):
        self.roi_list.clear()
        for roi in sorted(self._rois, key=lambda r: (_natural_key(r.file), r.name.lower())):
            item = QListWidgetItem(f"{roi.name}  —  {roi.file}")
            item.setData(Qt.UserRole, roi)
            self.roi_list.addItem(item)

    def _on_roi_list_clicked(self, item: QListWidgetItem):
        roi: roi_store.ROI = item.data(Qt.UserRole)
        if roi.file == self._current_file():
            self.gl.center_on_roi(roi.x, roi.y, roi.w, roi.h)
        elif roi.file in self._files:
            self._pending_center_roi = roi
            self.file_list.setCurrentRow(self._files.index(roi.file))
        else:
            QMessageBox.warning(self, "File not found", f"{roi.file} is not in the current folder.")

    def _rename_selected_roi(self):
        item = self.roi_list.currentItem()
        if item is None:
            return
        roi: roi_store.ROI = item.data(Qt.UserRole)
        name, ok = QInputDialog.getText(self, "Rename ROI", "Name:", text=roi.name)
        if ok and name.strip():
            roi.name = name.strip()
            self._rois_changed(roi.file)

    def _delete_selected_roi(self):
        item = self.roi_list.currentItem()
        if item is None:
            return
        roi: roi_store.ROI = item.data(Qt.UserRole)
        self._rois.remove(roi)
        self._rois_changed(roi.file)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _save_array(self, arr: np.ndarray, out_path: str) -> bool:
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

    def _export_view_of(self, view: CompositeGLWidget, fname: str, out_dir: str, fmt: str,
                        scale: float, rois_only: bool) -> int:
        """Export what `view` shows for `fname`: its full composite, or one
        crop per ROI named `<basename>_<ROIName><ext>`. Returns files written."""
        base = os.path.splitext(fname)[0]
        if not rois_only:
            arr = view.render_to_array(scale=scale)
            return int(arr is not None and self._save_array(arr, os.path.join(out_dir, f"{base}_composite{fmt}")))
        count = 0
        for roi in [r for r in self._rois if r.file == fname]:
            arr = view.render_region_to_array(roi.x, roi.y, roi.w, roi.h, scale=scale)
            out_path = os.path.join(out_dir, f"{base}_{roi_store.safe_name(roi.name)}{fmt}")
            if arr is not None and self._save_array(arr, out_path):
                count += 1
        return count

    def run_export_dialog(self):
        fname = self._current_file()
        if fname is None:
            QMessageBox.information(self, "No files", "Open a folder first.")
            return
        dlg = ExportDialog(self._last_out_dir or self._folder, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.get_options()

        try:
            os.makedirs(opts.out_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self._last_out_dir = opts.out_dir

        preset = None
        if opts.preset_path:
            try:
                preset = _read_json(opts.preset_path)
            except Exception as e:
                QMessageBox.critical(self, "Failed to load preset", str(e))
                return

        if opts.export_rois:
            has_rois = any(opts.scope_all_files or r.file == fname for r in self._rois)
            if not has_rois:
                where = "in this folder" if opts.scope_all_files else "on this file"
                QMessageBox.information(self, "No ROIs", f"No ROIs have been drawn {where} yet.")
                return

        if not opts.scope_all_files:
            if preset is not None:
                self._reresolve_current(preset)
            n = self._export_view_of(self.gl, fname, opts.out_dir, opts.fmt, opts.scale, opts.export_rois)
            self.status_label.setText(f"Exported {n} file(s) to {opts.out_dir}")
            return

        self._export_all_files(opts.out_dir, opts.fmt, opts.scale, preset, opts.export_rois)

    def _hidden_export_view(self) -> CompositeGLWidget:
        """An off-screen compositor, so batch export never disturbs the view."""
        if self._export_view is None:
            self._export_view = CompositeGLWidget()
            self._export_view.setAttribute(Qt.WA_DontShowOnScreen)
            self._export_view.resize(64, 64)
            self._export_view.show()  # creates its GL context without appearing
        return self._export_view

    def _export_all_files(self, out_dir: str, fmt: str, scale: float, preset: dict | None, rois_only: bool):
        """Export every file (or only those with ROIs). Each file gets its
        saved LUT if it has one; every other file gets `preset` if given,
        else what's on screen now (as it would carry over), else the
        Default LUT / auto-contrast."""
        targets = self._files
        if rois_only:
            roi_files = {r.file for r in self._rois}
            targets = [f for f in self._files if f in roi_files]

        self._capture_carry()
        carried = {name: dataclasses.replace(st) for name, st in self._carried.items()}
        view = self._hidden_export_view()

        progress = QProgressDialog("Exporting…", "Cancel", 0, len(targets), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        written, failed = 0, []
        try:
            for i, fname in enumerate(targets):
                if progress.wasCanceled():
                    break
                progress.setValue(i)
                progress.setLabelText(f"Exporting {fname}")
                QApplication.processEvents()
                try:
                    img = nd2_loader.load_nd2(os.path.join(self._folder, fname))
                except Exception as e:
                    failed.append(f"{fname}: {e}")
                    continue
                states, _ = self._resolve_states(img.path, img.channels, preset, carried)
                view.set_image(img, states)
                del img  # the pixels are on the GPU now
                written += self._export_view_of(view, fname, out_dir, fmt, scale, rois_only)
            progress.setValue(len(targets))
        finally:
            view.clear()

        if failed:
            QMessageBox.warning(self, "Some files skipped",
                                "Couldn't load:\n\n" + "\n".join(failed[:15]) + ("\n…" if len(failed) > 15 else ""))
        self.status_label.setText(f"Exported {written} file(s) to {out_dir}")

    # ------------------------------------------------------------------
    # Presets and Default LUT
    # ------------------------------------------------------------------
    def save_preset(self):
        if not self.gl.get_states():
            QMessageBox.information(self, "No data", "Load a file first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save LUT preset", "lut_preset.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w") as f:
            json.dump({st.name: _state_to_dict(st) for st in self.gl.get_states()}, f, indent=2)
        self.status_label.setText(f"Preset saved to {path}")

    def load_preset(self):
        if not self.gl.get_states():
            QMessageBox.information(self, "No data", "Load a file first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load LUT preset", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = _read_json(path)
            matched = self._apply_preset_live(data)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load preset", str(e))
            return
        self.status_label.setText(f"Applied {matched} channel(s) from {os.path.basename(path)}")

    def _apply_preset_live(self, data: dict, skip_saved: bool = False) -> int:
        """Apply preset values to the displayed file's matching channels
        (optionally leaving channels from its saved LUT alone). Returns how
        many channels matched."""
        names = {st.name for st in self.gl.get_states() if st.name in data
                 and not (skip_saved and st.name in self._override_names)}
        self._set_live_states([
            _state_from_dict(st.name, data[st.name], st.color) if st.name in names else st
            for st in self.gl.get_states()
        ])
        return len(names)

    def set_default_lut(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose default LUT preset", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = _read_json(path)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load preset", str(e))
            return
        self._default_lut_path = path
        self._default_lut_data = data
        self._settings.setValue("default_lut_path", path)
        self.act_use_default_lut.setEnabled(True)
        if self.act_use_default_lut.isChecked():
            self._apply_preset_live(data, skip_saved=True)
        else:
            self.act_use_default_lut.setChecked(True)  # applies it via the toggle
        self.status_label.setText(f"Default LUT set to {path}")

    def _toggle_use_default_lut(self, checked: bool):
        self._use_default_lut = checked
        self._settings.setValue("use_default_lut", checked)
        # Turning it on applies it now (and so it carries on from here);
        # turning it off just stops it seeding channels with nothing carried.
        if checked and self._default_lut_data:
            self._apply_preset_live(self._default_lut_data, skip_saved=True)

    # ------------------------------------------------------------------
    # Palettes
    # ------------------------------------------------------------------
    def _on_palette_changed(self, name: str):
        if name not in nd2_loader.COLOR_PALETTES:
            return
        nd2_loader.set_active_palette(name)
        self._settings.setValue("color_palette", name)
        for i, st in enumerate(self.gl.get_states()):
            if nd2_loader.fluorophore_key_for_channel(st.name) is None:
                continue
            color = nd2_loader.default_color_for_channel(st.name, i)
            self.gl.set_channel_param(i, color=color)
            self._panels[i].set_color(color)
        # Recolor carried settings too, for channels not in this file.
        for name, st in self._carried.items():
            if nd2_loader.fluorophore_key_for_channel(name) is not None:
                st.color = nd2_loader.default_color_for_channel(name, 0)
        self.status_label.setText(f"Color palette: {name}")

    def save_new_palette(self):
        name, ok = QInputDialog.getText(self, "Save Palette", "Palette name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        # Start from the active palette so every standard channel has a
        # color, then overlay whatever's on screen right now.
        palette = dict(nd2_loader.COLOR_PALETTES[nd2_loader.get_active_palette()])
        for st in self.gl.get_states():
            key = nd2_loader.fluorophore_key_for_channel(st.name)
            if key is not None:
                palette[key] = tuple(st.color)
        nd2_loader.save_custom_palette(name, palette)
        if self.palette_combo.findText(name) < 0:
            self.palette_combo.addItem(name)
        self.palette_combo.setCurrentText(name)
        self.status_label.setText(f"Saved palette '{name}'")
