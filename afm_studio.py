from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, Signal, QTimer, QPoint, QRect, QSize, QObject, QEvent
from PySide6.QtGui import QAction, QFont, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QAbstractScrollArea, QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QDial, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLayout, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QProgressBar, QSlider, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QToolButton, QVBoxLayout, QWidget, QWidgetItem
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib import colormaps, image as mpl_image
from scipy import ndimage
import tifffile
from openpyxl import Workbook
from openpyxl.styles import Font as XLFont, PatternFill, Alignment

try:
    import pyqtgraph.opengl as gl
except Exception:
    gl = None

try:
    from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR, GL_VERSION
except Exception:
    glGetString = None
    GL_RENDERER = GL_VENDOR = GL_VERSION = None

from nid_reader import read_nid, NIDChannel, NIDFile
from gpu_compute import cuda_status, gaussian_smooth_cuda, median_filter_cuda
from morphology_analysis import analyze_particles, MorphologyResult
from smart_analysis import analyze_smart_qc, smart_morphology_recommendation, SmartQCResult
from batch_analysis import BatchSettings, BatchResult, analyze_file as analyze_batch_file, export_batch_workbook
from workspace_ui import CollapsibleCard, WelcomePanel, WorkspaceHeader, WORKSPACE_QSS
from motion_ui import OrbitTabBar, OrbitalScene, motion_enabled
from space_theme import SPACE_QSS, configure_plots, register_ui_fonts
from analysis_jobs import BatchWorker
from reporting import write_report
from afm_analysis import (
    bearing_curve, fit_plane, gaussian_smooth, line_flatten, line_profile,
    median_filter, radial_psd, remove_row_offsets, roughness_stats,
    xy_unit_scale, z_unit_scale,
    ALGORITHM_VERSION,
)

configure_plots()


APP_NAME = "Afruz AFM Studio"
APP_VERSION = "0.4.0"


class NoWheelValueChangesFilter(QObject):
    """Prevent accidental value changes caused by the mouse wheel.

    Qt spin boxes, combo boxes, sliders and dials normally react to wheel events
    whenever the cursor is over them. In a long, scrollable analysis panel this is
    easy to trigger accidentally. This application-wide filter consumes those wheel
    events and, when possible, applies the same wheel motion to the nearest scroll
    area instead. Keyboard, clicking, dragging and typed value entry are unaffected.
    """

    _BLOCKED_TYPES = (QAbstractSpinBox, QComboBox, QSlider, QDial)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel and isinstance(watched, self._BLOCKED_TYPES):
            self._scroll_nearest_panel(watched, event)
            event.accept()
            return True
        return super().eventFilter(watched, event)

    @staticmethod
    def _scroll_nearest_panel(widget, event):
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                # Do not hijack wheel scrolling from a popup/list's own view.
                if isinstance(parent, QAbstractItemView):
                    parent = parent.parentWidget()
                    continue

                vbar = parent.verticalScrollBar()
                hbar = parent.horizontalScrollBar()
                pixel = event.pixelDelta()
                angle = event.angleDelta()

                if not pixel.isNull():
                    # Trackpads provide pixel deltas; preserve their smooth motion.
                    if pixel.y() and vbar.isVisible():
                        vbar.setValue(vbar.value() - pixel.y())
                        return
                    if pixel.x() and hbar.isVisible():
                        hbar.setValue(hbar.value() - pixel.x())
                        return

                if angle.y() and vbar.isVisible():
                    steps = angle.y() / 120.0
                    amount = int(round(steps * max(24, vbar.singleStep() * 3)))
                    vbar.setValue(vbar.value() - amount)
                    return
                if angle.x() and hbar.isVisible():
                    steps = angle.x() / 120.0
                    amount = int(round(steps * max(24, hbar.singleStep() * 3)))
                    hbar.setValue(hbar.value() - amount)
                    return
                return
            parent = parent.parentWidget()



DARK_QSS = SPACE_QSS


class FlowLayout(QLayout):
    """Simple wrapping layout so toolbars remain usable at narrow widths."""
    def __init__(self, parent=None, margin=0, hspacing=6, vspacing=6):
        super().__init__(parent)
        self._items = []
        self._hspacing = hspacing
        self._vspacing = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        right = rect.right()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._hspacing
            if next_x - self._hspacing > right and line_height > 0:
                x = rect.x()
                y += line_height + self._vspacing
                next_x = x + hint.width() + self._hspacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor="#0e182a")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


if gl is not None:
    class AFMGLView(gl.GLViewWidget):
        '''OpenGL AFM surface view. Rendering runs on the system OpenGL GPU.'''

        gpuInfoChanged = Signal(str)

        def initializeGL(self):
            super().initializeGL()
            renderer = "OpenGL renderer active"
            try:
                if glGetString is not None:
                    r = glGetString(GL_RENDERER)
                    v = glGetString(GL_VENDOR)
                    ver = glGetString(GL_VERSION)
                    def dec(x):
                        return x.decode("utf-8", errors="replace") if isinstance(x, (bytes, bytearray)) else str(x)
                    renderer = f"{dec(r)} | {dec(v)} | OpenGL {dec(ver)}"
            except Exception as exc:
                renderer = f"OpenGL active ({exc})"
            self.gpuInfoChanged.emit(renderer)
else:
    class AFMGLView(QWidget):
        gpuInfoChanged = Signal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            lay = QVBoxLayout(self)
            msg = QLabel("GPU 3D viewer unavailable. Install pyqtgraph and PyOpenGL.")
            msg.setAlignment(Qt.AlignCenter)
            lay.addWidget(msg)

        def setCameraPosition(self, *args, **kwargs):
            pass


class NumericItem(QTableWidgetItem):
    def __init__(self, value):
        self.is_numeric = False
        self.numeric_value = float("nan")
        if isinstance(value, (int, np.integer)):
            self.numeric_value = float(value)
            self.is_numeric = True
            text = str(int(value))
        elif isinstance(value, (float, np.floating)):
            self.numeric_value = float(value)
            self.is_numeric = True
            text = "—" if not np.isfinite(self.numeric_value) else f"{self.numeric_value:.6g}"
        elif value is None:
            text = "—"
        else:
            try:
                self.numeric_value = float(value)
                self.is_numeric = True
                text = "—" if not np.isfinite(self.numeric_value) else f"{self.numeric_value:.6g}"
            except (TypeError, ValueError):
                text = str(value)
        super().__init__(text)

    def __lt__(self, other):
        if isinstance(other, NumericItem) and self.is_numeric and other.is_numeric:
            a, b = self.numeric_value, other.numeric_value
            if np.isnan(a):
                return False
            if np.isnan(b):
                return True
            return a < b
        return super().__lt__(other)


class StatCard(QFrame):
    def __init__(self, name: str):
        super().__init__()
        self.setObjectName("statCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        self.value = QLabel("—")
        self.value.setObjectName("cardValue")
        self.name = QLabel(name)
        self.name.setObjectName("cardName")
        lay.addWidget(self.value)
        lay.addWidget(self.name)


class AFMStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1500, 920)
        self.setMinimumSize(820, 560)

        self.nid: NIDFile | None = None
        self.channel: NIDChannel | None = None
        self.original: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.x_range_m = 0.0
        self.y_range_m = 0.0
        self.crop_origin = (0, 0)
        self.active_map_tool = None
        self.profile_points = []
        self.profile_artists = []
        self.roi_selector = None
        self.pending_roi = None
        self.cuda = cuda_status()
        self.surface_item = None
        self.surface_grid = None
        self._surface_reset_pending = True
        self._surface_cache = None
        self._surface_msaa_samples = None
        self._dirty_tabs = set()
        self.history = []
        self.history_index = -1
        self.morphology_result: MorphologyResult | None = None
        self.smart_qc_result: SmartQCResult | None = None
        self.smart_morph_recommendation = None
        self.smart_channel_correlations = {}
        self.batch_files: list[str] = []
        self.batch_results: list[BatchResult] = []
        self.batch_worker = None
        self.batch_run_settings = None
        self._close_after_batch = False

        self._build_menu()
        self._build_ui()
        self.setAcceptDrops(True)
        self._update_workspace_header()
        self.statusBar().showMessage("Open a .nid file to begin.")

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        a_open = QAction("Open NID…", self)
        a_open.setShortcut("Ctrl+O")
        a_open.triggered.connect(self.open_file)
        file_menu.addAction(a_open)

        edit_menu = self.menuBar().addMenu("Edit")
        self.a_undo = QAction("Undo", self)
        self.a_undo.setShortcut("Ctrl+Z")
        self.a_undo.triggered.connect(self.undo_history)
        edit_menu.addAction(self.a_undo)
        self.a_redo = QAction("Redo", self)
        self.a_redo.setShortcuts(["Ctrl+Shift+Z", "Ctrl+Y"])
        self.a_redo.triggered.connect(self.redo_history)
        edit_menu.addAction(self.a_redo)
        self.a_reset = QAction("Restore raw channel", self)
        self.a_reset.setShortcut("Ctrl+R")
        self.a_reset.triggered.connect(self.reset_data)
        edit_menu.addAction(self.a_reset)
        self.a_cancel_tool = QAction("Cancel map tool", self)
        self.a_cancel_tool.setShortcut("Escape")
        self.a_cancel_tool.triggered.connect(self.cancel_map_tool)
        edit_menu.addAction(self.a_cancel_tool)

        a_png = QAction("Export 600 DPI PNG…", self)
        a_png.triggered.connect(self.export_png)
        file_menu.addAction(a_png)

        a_csv = QAction("Export matrix CSV…", self)
        a_csv.triggered.connect(self.export_csv)
        file_menu.addAction(a_csv)

        file_menu.addSeparator()
        a_exit = QAction("Exit", self)
        a_exit.triggered.connect(self.close)
        file_menu.addAction(a_exit)

        view_menu = self.menuBar().addMenu("View")
        self.a_sidebar = QAction("Show controls sidebar", self)
        self.a_sidebar.setCheckable(True)
        self.a_sidebar.setChecked(True)
        self.a_sidebar.setShortcut("F9")
        self.a_sidebar.triggered.connect(self.toggle_sidebar)
        view_menu.addAction(self.a_sidebar)
        self.a_reduced_motion = QAction("Reduce motion", self)
        self.a_reduced_motion.setCheckable(True)
        self.a_reduced_motion.setChecked(not motion_enabled())
        self.a_reduced_motion.toggled.connect(self.set_reduced_motion)
        view_menu.addAction(self.a_reduced_motion)

        help_menu = self.menuBar().addMenu("Help")
        a_about = QAction("About", self)
        a_about.triggered.connect(lambda: QMessageBox.information(
            self, "About",
            f"{APP_NAME} v{APP_VERSION}\nAFM/NID visualization, smart QC, roughness, GPU 3D, particle morphology and batch analysis."
        ))
        help_menu.addAction(a_about)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.workspace_header = WorkspaceHeader()
        self.workspace_header.open_button.clicked.connect(self.open_file)
        self.workspace_header.undo_button.clicked.connect(self.undo_history)
        self.workspace_header.redo_button.clicked.connect(self.redo_history)
        self.workspace_header.raw_button.clicked.connect(self.reset_data)
        self.workspace_header.controls_button.toggled.connect(self.a_sidebar.setChecked)
        self.workspace_header.controls_button.toggled.connect(self.toggle_sidebar)
        root_layout.addWidget(self.workspace_header)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(self.main_splitter)

        # Scrollable sidebar: controls never overlap or disappear on small windows.
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setObjectName("sidebarScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.setMinimumWidth(245)
        self.sidebar_scroll.setMaximumWidth(390)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(14, 14, 14, 18)
        side.setSpacing(10)

        title = QLabel("ANALYSIS CONTROLS")
        title.setObjectName("sectionTitle")
        subtitle = QLabel("Choose a channel, then prepare your scan.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        side.addWidget(title)
        side.addWidget(subtitle)

        self.btn_open = self.workspace_header.open_button

        # Dataset card
        frame, lay = self._make_side_card("DATASET")
        self.channel_combo = QComboBox()
        self.channel_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.channel_combo.setMinimumContentsLength(18)
        self.channel_combo.currentIndexChanged.connect(self.select_channel)
        lay.addWidget(self.channel_combo)
        self.file_info = QLabel("No file loaded")
        self.file_info.setWordWrap(True)
        self.file_info.setObjectName("hint")
        lay.addWidget(self.file_info)
        side.addWidget(frame)

        # Preprocessing card
        frame, lay = self._make_side_card("PREPROCESSING")
        self.btn_plane = QPushButton("Plane level · 1st order")
        self.btn_plane.clicked.connect(self.apply_plane)
        lay.addWidget(self.btn_plane)
        self.btn_robust_plane = QPushButton("Robust plane · Huber")
        self.btn_robust_plane.setToolTip("Down-weight isolated high residuals when fitting a plane. Inspect the result; broad structures can still influence the fit.")
        self.btn_robust_plane.clicked.connect(self.apply_robust_plane)
        lay.addWidget(self.btn_robust_plane)

        grid = QGridLayout()
        grid.setHorizontalSpacing(7)
        grid.setVerticalSpacing(7)
        self.flatten_degree = QSpinBox()
        self.flatten_degree.setRange(0, 5)
        self.flatten_degree.setValue(1)
        self.flatten_degree.setToolTip("Polynomial degree")
        self.btn_flatten = QPushButton("Line flatten")
        self.btn_flatten.clicked.connect(self.apply_line_flatten)
        grid.addWidget(self.flatten_degree, 0, 0)
        grid.addWidget(self.btn_flatten, 0, 1)

        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 20.0)
        self.sigma.setSingleStep(0.2)
        self.sigma.setValue(1.0)
        self.sigma.setToolTip("Gaussian sigma (pixels)")
        self.btn_gauss = QPushButton("Gaussian")
        self.btn_gauss.clicked.connect(self.apply_gaussian)
        grid.addWidget(self.sigma, 1, 0)
        grid.addWidget(self.btn_gauss, 1, 1)

        self.median_size = QSpinBox()
        self.median_size.setRange(1, 15)
        self.median_size.setSingleStep(2)
        self.median_size.setValue(3)
        self.median_size.setToolTip("Median kernel size (pixels)")
        self.btn_median = QPushButton("Median filter")
        self.btn_median.clicked.connect(self.apply_median)
        grid.addWidget(self.median_size, 2, 0)
        grid.addWidget(self.btn_median, 2, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        self.btn_row = QPushButton("Remove row offsets")
        self.btn_row.clicked.connect(self.apply_row_offsets)
        lay.addWidget(self.btn_row)

        self.chk_cuda = QCheckBox("CUDA compute for filters")
        self.chk_cuda.setChecked(self.cuda.available)
        self.chk_cuda.setEnabled(self.cuda.available)
        self.chk_cuda.setToolTip("Optional CuPy acceleration. GPU 3D rendering works independently through OpenGL.")
        lay.addWidget(self.chk_cuda)
        self.cuda_info_label = QLabel(
            f"CUDA: {self.cuda.device_name}" if self.cuda.available
            else "CUDA compute: optional / not installed"
        )
        self.cuda_info_label.setWordWrap(True)
        self.cuda_info_label.setObjectName("hint")
        lay.addWidget(self.cuda_info_label)

        self.btn_reset = QPushButton("Restore raw channel")
        self.btn_reset.clicked.connect(self.reset_data)
        lay.addWidget(self.btn_reset)
        side.addWidget(frame)

        # Interactive tools
        frame, lay = self._make_side_card("MEASUREMENT TOOLS")
        self.btn_profile = QPushButton("Line profile · click 2 points")
        self.btn_profile.clicked.connect(self.enable_profile)
        lay.addWidget(self.btn_profile)
        self.btn_roi = QPushButton("Select ROI")
        self.btn_roi.clicked.connect(self.enable_roi)
        lay.addWidget(self.btn_roi)
        self.btn_crop = QPushButton("Crop to selected ROI")
        self.btn_crop.clicked.connect(self.crop_to_roi)
        lay.addWidget(self.btn_crop)
        side.addWidget(frame)

        # Smart analysis shortcuts
        frame, lay = self._make_side_card("SMART ANALYSIS")
        self.btn_smart_qc_side = QPushButton("Run Smart QC")
        self.btn_smart_qc_side.setObjectName("primary")
        self.btn_smart_qc_side.clicked.connect(self.run_smart_qc)
        lay.addWidget(self.btn_smart_qc_side)
        self.btn_smart_morph_side = QPushButton("Recommend morphology")
        self.btn_smart_morph_side.clicked.connect(self.recommend_smart_morphology)
        lay.addWidget(self.btn_smart_morph_side)
        smart_hint = QLabel("Non-destructive diagnostics: tilt, scan lines, isolated spikes, bow, clipping, forward/backward consistency, local roughness and texture.")
        smart_hint.setWordWrap(True)
        smart_hint.setObjectName("hint")
        lay.addWidget(smart_hint)
        side.addWidget(frame)

        # Display
        frame, lay = self._make_side_card("DISPLAY")
        self.cmap_combo = QComboBox()
        preferred = ["viridis", "cividis", "magma", "inferno", "afmhot", "gray", "turbo"]
        self.cmap_combo.addItems([c for c in preferred if c in colormaps])
        self.cmap_combo.currentTextChanged.connect(self.refresh_all)
        lay.addWidget(self.cmap_combo)
        self.chk_reverse = QCheckBox("Reverse colormap")
        self.chk_reverse.stateChanged.connect(self.refresh_all)
        lay.addWidget(self.chk_reverse)
        side.addWidget(frame)

        # Non-destructive processing history
        frame, lay = self._make_side_card("ANALYSIS HISTORY")
        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(105)
        self.history_list.setMaximumHeight(175)
        self.history_list.setAlternatingRowColors(True)
        self.history_list.itemDoubleClicked.connect(lambda _item: self.restore_selected_history())
        lay.addWidget(self.history_list)
        hrow = QHBoxLayout()
        self.btn_history_undo = QPushButton("Undo")
        self.btn_history_undo.clicked.connect(self.undo_history)
        self.btn_history_redo = QPushButton("Redo")
        self.btn_history_redo.clicked.connect(self.redo_history)
        self.btn_history_restore = QPushButton("Restore")
        self.btn_history_restore.clicked.connect(self.restore_selected_history)
        hrow.addWidget(self.btn_history_undo)
        hrow.addWidget(self.btn_history_redo)
        hrow.addWidget(self.btn_history_restore)
        lay.addLayout(hrow)
        hint = QLabel("Every processing step is stored. Double-click a state to restore it.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        lay.addWidget(hint)
        side.addWidget(frame)

        # Export
        frame, lay = self._make_side_card("EXPORT")
        self.btn_export_png = QPushButton("Map · 600 DPI PNG")
        self.btn_export_png.clicked.connect(self.export_png)
        lay.addWidget(self.btn_export_png)
        self.btn_export_csv = QPushButton("Matrix · CSV")
        self.btn_export_csv.clicked.connect(self.export_csv)
        lay.addWidget(self.btn_export_csv)
        self.btn_export_tiff = QPushButton("Matrix · float TIFF")
        self.btn_export_tiff.clicked.connect(self.export_tiff)
        lay.addWidget(self.btn_export_tiff)
        self.btn_export_json = QPushButton("Analysis · JSON")
        self.btn_export_json.clicked.connect(self.export_json)
        lay.addWidget(self.btn_export_json)
        side.addWidget(frame)
        side.addStretch(1)

        self.sidebar_scroll.setWidget(self.sidebar)
        self.main_splitter.addWidget(self.sidebar_scroll)

        # Main content tabs
        self.tabs = QTabWidget()
        self.tabs.setTabBar(OrbitTabBar(self.tabs))
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([300, 1200])

        # 2D map
        map_tab = QWidget()
        map_lay = QVBoxLayout(map_tab)
        map_lay.setContentsMargins(8, 8, 8, 8)
        self.map_controls = QWidget()
        self.map_controls.setObjectName("mapControls")
        map_controls_layout = QVBoxLayout(self.map_controls)
        map_controls_layout.setContentsMargins(4, 0, 4, 0)
        map_controls_layout.setSpacing(4)
        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("Compare"))
        self.map_view_combo = QComboBox()
        self.map_view_combo.addItems(["Processed · current state", "Raw · same field", "Processing difference"])
        self.map_view_combo.setAccessibleName("Map comparison view")
        self.map_view_combo.currentIndexChanged.connect(self._map_view_changed)
        compare_row.addWidget(self.map_view_combo, 1)
        map_controls_layout.addLayout(compare_row)
        map_actions = QHBoxLayout()
        self.btn_map_profile = QPushButton("Line profile")
        self.btn_map_profile.setCheckable(True)
        self.btn_map_profile.clicked.connect(self.enable_profile)
        map_actions.addWidget(self.btn_map_profile)
        self.btn_map_roi = QPushButton("Select ROI")
        self.btn_map_roi.setCheckable(True)
        self.btn_map_roi.clicked.connect(self.enable_roi)
        map_actions.addWidget(self.btn_map_roi)
        self.btn_map_crop = QPushButton("Crop ROI")
        self.btn_map_crop.clicked.connect(self.crop_to_roi)
        map_actions.addWidget(self.btn_map_crop)
        self.btn_map_cancel = QPushButton("Cancel · Esc")
        self.btn_map_cancel.clicked.connect(self.cancel_map_tool)
        map_actions.addWidget(self.btn_map_cancel)
        map_actions.addStretch(1)
        map_controls_layout.addLayout(map_actions)
        self.map_tool_hint = QLabel("Measurements and exports use the current analysis state.")
        self.map_tool_hint.setObjectName("hint")
        self.map_tool_hint.setWordWrap(True)
        map_controls_layout.addWidget(self.map_tool_hint)
        map_lay.addWidget(self.map_controls)
        self.map_stack = QStackedWidget()
        self.welcome_panel = WelcomePanel()
        self.welcome_panel.open_button.clicked.connect(self.open_file)
        self.welcome_panel.batch_button.clicked.connect(lambda: self.tabs.setCurrentIndex(self.batch_tab_index))
        self.map_stack.addWidget(self.welcome_panel)
        self.map_canvas = MplCanvas(width=8, height=7)
        self.map_canvas.mpl_connect("button_press_event", self.on_map_click)
        self.map_stack.addWidget(self.map_canvas)
        map_lay.addWidget(self.map_stack, 1)
        self.tabs.addTab(map_tab, "2D Map")

        # Roughness + distributions
        stats_tab = QWidget()
        stats_lay = QVBoxLayout(stats_tab)
        stats_lay.setContentsMargins(8, 8, 8, 8)
        card_host = QWidget()
        cards = FlowLayout(card_host, hspacing=7, vspacing=7)
        self.stat_cards = {}
        for name in ["Sa", "Sq", "Sz", "Sp", "Sv", "Ssk", "Sku", "Mean"]:
            card = StatCard(name)
            card.setMinimumWidth(135)
            self.stat_cards[name] = card
            cards.addWidget(card)
        stats_lay.addWidget(card_host)
        self.stats_canvas = MplCanvas(width=8, height=5)
        stats_lay.addWidget(self.stats_canvas, 1)
        self.tabs.addTab(stats_tab, "Roughness")

        # Morphology / particles
        morph_tab = QWidget()
        morph_root = QVBoxLayout(morph_tab)
        morph_root.setContentsMargins(8, 8, 8, 8)
        morph_root.setSpacing(7)

        summary_host = QWidget()
        summary_flow = FlowLayout(summary_host, hspacing=7, vspacing=7)
        self.morph_cards = {}
        for name in ["Particles", "Size-QC n", "Density", "Coverage", "Mean diameter", "Mean height", "Pixel pitch"]:
            card = StatCard(name)
            card.setMinimumWidth(145)
            self.morph_cards[name] = card
            summary_flow.addWidget(card)
        morph_root.addWidget(summary_host)

        self.morph_vertical_splitter = QSplitter(Qt.Vertical)
        self.morph_vertical_splitter.setChildrenCollapsible(False)
        morph_root.addWidget(self.morph_vertical_splitter, 1)

        morph_top = QSplitter(Qt.Horizontal)
        morph_top.setChildrenCollapsible(False)
        self.morph_vertical_splitter.addWidget(morph_top)

        morph_visual = QSplitter(Qt.Vertical)
        morph_visual.setChildrenCollapsible(False)
        self.morph_canvas = MplCanvas(width=7, height=5)
        self.morph_dist_canvas = MplCanvas(width=7, height=2.5)
        morph_visual.addWidget(self.morph_canvas)
        morph_visual.addWidget(self.morph_dist_canvas)
        morph_visual.setSizes([520, 230])
        morph_top.addWidget(morph_visual)

        morph_controls_scroll = QScrollArea()
        morph_controls_scroll.setWidgetResizable(True)
        morph_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        morph_controls_scroll.setMinimumWidth(270)
        morph_controls_scroll.setMaximumWidth(390)
        morph_controls = QWidget()
        mc = QVBoxLayout(morph_controls)
        mc.setContentsMargins(10, 8, 10, 10)
        mc.setSpacing(8)

        h = QLabel("Particle / grain segmentation")
        h.setObjectName("sectionTitle")
        mc.addWidget(h)
        note = QLabel("Segmentation is non-destructive. Review the overlay and threshold before reporting morphology statistics.")
        note.setWordWrap(True)
        note.setObjectName("hint")
        mc.addWidget(note)

        form = QGridLayout()
        form.setHorizontalSpacing(7)
        form.setVerticalSpacing(7)
        form.addWidget(QLabel("Direction"), 0, 0)
        self.morph_direction = QComboBox()
        self.morph_direction.addItems(["Above threshold", "Below threshold"])
        form.addWidget(self.morph_direction, 0, 1)

        form.addWidget(QLabel("Threshold"), 1, 0)
        self.morph_threshold_method = QComboBox()
        self.morph_threshold_method.addItems(["Otsu", "Li", "Yen", "Triangle", "Mean", "Median + kσ", "Manual"])
        form.addWidget(self.morph_threshold_method, 1, 1)

        form.addWidget(QLabel("Manual value"), 2, 0)
        self.morph_manual_threshold = QDoubleSpinBox()
        self.morph_manual_threshold.setMinimumWidth(80)
        self.morph_manual_threshold.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.morph_manual_threshold.setDecimals(5)
        self.morph_manual_threshold.setRange(-1e12, 1e12)
        self.morph_manual_threshold.setSingleStep(0.1)
        form.addWidget(self.morph_manual_threshold, 2, 1)

        form.addWidget(QLabel("k σ"), 3, 0)
        self.morph_k_sigma = QDoubleSpinBox()
        self.morph_k_sigma.setRange(-10.0, 10.0)
        self.morph_k_sigma.setSingleStep(0.1)
        self.morph_k_sigma.setValue(1.0)
        form.addWidget(self.morph_k_sigma, 3, 1)

        form.addWidget(QLabel("Smooth σ (px)"), 4, 0)
        self.morph_smooth = QDoubleSpinBox()
        self.morph_smooth.setRange(0.0, 10.0)
        self.morph_smooth.setSingleStep(0.2)
        self.morph_smooth.setValue(0.8)
        form.addWidget(self.morph_smooth, 4, 1)

        form.addWidget(QLabel("Min area (px)"), 5, 0)
        self.morph_min_area = QSpinBox()
        self.morph_min_area.setRange(1, 1000000)
        self.morph_min_area.setValue(8)
        form.addWidget(self.morph_min_area, 5, 1)

        form.addWidget(QLabel("Max area (px)"), 6, 0)
        self.morph_max_area = QSpinBox()
        self.morph_max_area.setRange(0, 100000000)
        self.morph_max_area.setValue(0)
        self.morph_max_area.setSpecialValueText("No limit")
        form.addWidget(self.morph_max_area, 6, 1)

        form.addWidget(QLabel("Watershed spacing"), 7, 0)
        self.morph_watershed_distance = QSpinBox()
        self.morph_watershed_distance.setRange(1, 100)
        self.morph_watershed_distance.setValue(4)
        form.addWidget(self.morph_watershed_distance, 7, 1)

        form.addWidget(QLabel("Opening radius"), 8, 0)
        self.morph_opening = QSpinBox()
        self.morph_opening.setRange(0, 20)
        self.morph_opening.setValue(0)
        form.addWidget(self.morph_opening, 8, 1)

        form.addWidget(QLabel("Closing radius"), 9, 0)
        self.morph_closing = QSpinBox()
        self.morph_closing.setRange(0, 20)
        self.morph_closing.setValue(0)
        form.addWidget(self.morph_closing, 9, 1)
        form.setColumnStretch(1, 1)
        mc.addLayout(form)

        size_h = QLabel("Calibrated size metrology")
        size_h.setObjectName("sectionTitle")
        mc.addWidget(size_h)
        self.morph_scale_readout = QLabel("Scale: —")
        self.morph_scale_readout.setObjectName("hint")
        self.morph_scale_readout.setWordWrap(True)
        mc.addWidget(self.morph_scale_readout)

        size_form = QGridLayout()
        size_form.setHorizontalSpacing(7)
        size_form.setVerticalSpacing(7)
        size_form.addWidget(QLabel("Min diameter (nm)"), 0, 0)
        self.morph_min_diameter_nm = QDoubleSpinBox()
        self.morph_min_diameter_nm.setRange(0.0, 1e7)
        self.morph_min_diameter_nm.setDecimals(3)
        self.morph_min_diameter_nm.setValue(0.0)
        self.morph_min_diameter_nm.setSpecialValueText("No limit")
        size_form.addWidget(self.morph_min_diameter_nm, 0, 1)

        size_form.addWidget(QLabel("Max diameter (nm)"), 1, 0)
        self.morph_max_diameter_nm = QDoubleSpinBox()
        self.morph_max_diameter_nm.setRange(0.0, 1e7)
        self.morph_max_diameter_nm.setDecimals(3)
        self.morph_max_diameter_nm.setValue(0.0)
        self.morph_max_diameter_nm.setSpecialValueText("No limit")
        size_form.addWidget(self.morph_max_diameter_nm, 1, 1)

        size_form.addWidget(QLabel("Minimum pixels across"), 2, 0)
        self.morph_min_pixels_across = QDoubleSpinBox()
        self.morph_min_pixels_across.setRange(1.0, 20.0)
        self.morph_min_pixels_across.setSingleStep(0.5)
        self.morph_min_pixels_across.setValue(3.0)
        self.morph_min_pixels_across.setToolTip("Particles below this area-equivalent sampling width are retained in the table but flagged under-resolved.")
        size_form.addWidget(self.morph_min_pixels_across, 2, 1)

        size_form.addWidget(QLabel("Height background"), 3, 0)
        self.morph_background_mode = QComboBox()
        self.morph_background_mode.addItems(["Local plane", "Local median"])
        self.morph_background_mode.setToolTip("Local plane is recommended for particle height/volume because it compensates residual local slope around each object.")
        size_form.addWidget(self.morph_background_mode, 3, 1)

        size_form.addWidget(QLabel("Background ring (px)"), 4, 0)
        self.morph_background_ring = QSpinBox()
        self.morph_background_ring.setRange(1, 50)
        self.morph_background_ring.setValue(4)
        size_form.addWidget(self.morph_background_ring, 4, 1)
        size_form.setColumnStretch(1, 1)
        mc.addLayout(size_form)

        self.morph_exclude_underresolved = QCheckBox("QC-valid size summary only")
        self.morph_exclude_underresolved.setChecked(True)
        self.morph_exclude_underresolved.setToolTip("All detected particles remain visible and exported. Size summaries omit particles below the coarser-axis sampling criterion or touching missing observations.")
        mc.addWidget(self.morph_exclude_underresolved)

        size_plot_form = QGridLayout()
        size_plot_form.setHorizontalSpacing(7)
        size_plot_form.setVerticalSpacing(7)
        size_plot_form.addWidget(QLabel("Population"), 0, 0)
        self.morph_distribution_population = QComboBox()
        self.morph_distribution_population.addItems(["Size-QC valid only", "All detected particles"])
        self.morph_distribution_population.setToolTip("Controls the particles used in the diameter/height/circularity histograms and exported distribution tables. The complete particle table is always preserved.")
        self.morph_distribution_population.currentIndexChanged.connect(lambda *_: self.render_morphology() if self.morphology_result is not None else None)
        size_plot_form.addWidget(self.morph_distribution_population, 0, 1)

        size_plot_form.addWidget(QLabel("Histogram bins"), 1, 0)
        self.morph_hist_bins = QSpinBox()
        self.morph_hist_bins.setRange(0, 100)
        self.morph_hist_bins.setValue(0)
        self.morph_hist_bins.setSpecialValueText("Auto")
        self.morph_hist_bins.setToolTip("0 = automatic Freedman–Diaconis binning with a conservative fallback. Fixed values force the same bin count.")
        self.morph_hist_bins.valueChanged.connect(lambda *_: self.render_morphology() if self.morphology_result is not None else None)
        size_plot_form.addWidget(self.morph_hist_bins, 1, 1)
        size_plot_form.setColumnStretch(1, 1)
        mc.addLayout(size_plot_form)

        self.btn_morph_apply_qc_min = QPushButton("Set minimum diameter to QC limit")
        self.btn_morph_apply_qc_min.setToolTip("Copies the current sampling-resolution limit into Min apparent diameter. This is optional; it will exclude under-resolved objects from the next segmentation result instead of merely flagging them.")
        self.btn_morph_apply_qc_min.clicked.connect(self.set_morph_min_diameter_to_qc_limit)
        mc.addWidget(self.btn_morph_apply_qc_min)

        tip_note = QLabel("Lateral AFM diameters are reported as apparent sizes because probe-tip convolution is not automatically deconvolved. Height is referenced to the selected local background model.")
        tip_note.setObjectName("hint")
        tip_note.setWordWrap(True)
        mc.addWidget(tip_note)

        self.morph_plane = QCheckBox("Robust plane for segmentation & height")
        self.morph_plane.setToolTip("Huber leveling within morphology; heights use the surface before segmentation smoothing.")
        self.morph_plane.setChecked(True)
        self.morph_plane.setToolTip("Huber leveling is applied within morphology to segmentation and height/volume measurement. The current scan matrix stays unchanged.")
        mc.addWidget(self.morph_plane)
        self.morph_fill_holes = QCheckBox("Fill enclosed holes")
        self.morph_fill_holes.setChecked(True)
        mc.addWidget(self.morph_fill_holes)
        self.morph_clear_border = QCheckBox("Exclude border-touching objects")
        self.morph_clear_border.setChecked(True)
        mc.addWidget(self.morph_clear_border)
        self.morph_watershed = QCheckBox("Split touching objects · watershed")
        self.morph_watershed.setChecked(True)
        mc.addWidget(self.morph_watershed)
        self.morph_live = QCheckBox("Live preview")
        self.morph_live.setChecked(True)
        mc.addWidget(self.morph_live)

        self.morph_threshold_readout = QLabel("Threshold: —")
        self.morph_threshold_readout.setObjectName("hint")
        self.morph_threshold_readout.setWordWrap(True)
        mc.addWidget(self.morph_threshold_readout)

        self.btn_morph_analyze = QPushButton("Analyze morphology")
        self.btn_morph_analyze.setObjectName("primary")
        self.btn_morph_analyze.clicked.connect(self.run_morphology_analysis)
        mc.addWidget(self.btn_morph_analyze)

        erow = QVBoxLayout()
        self.btn_morph_csv = QPushButton("Export particles · CSV")
        self.btn_morph_csv.clicked.connect(self.export_morphology_csv)
        self.btn_morph_xlsx = QPushButton("Export data · Excel")
        self.btn_morph_xlsx.clicked.connect(self.export_morphology_excel)
        erow.addWidget(self.btn_morph_csv)
        erow.addWidget(self.btn_morph_xlsx)
        mc.addLayout(erow)
        self.btn_morph_figure = QPushButton("Export morphology figure · 600 DPI")
        self.btn_morph_figure.setToolTip("Exports the morphology map together with diameter, P95-height and circularity distributions as a publication-ready PNG.")
        self.btn_morph_figure.clicked.connect(self.export_morphology_figure)
        mc.addWidget(self.btn_morph_figure)
        self.btn_morph_exclude = QPushButton("Exclude selected particles")
        self.btn_morph_exclude.setToolTip("Select one or more rows in the particle table, then exclude those objects from the current result.")
        self.btn_morph_exclude.clicked.connect(self.exclude_selected_particles)
        mc.addWidget(self.btn_morph_exclude)
        self.btn_morph_reset_mask = QPushButton("Rebuild mask from threshold")
        self.btn_morph_reset_mask.clicked.connect(self.run_morphology_analysis)
        mc.addWidget(self.btn_morph_reset_mask)
        mc.addStretch(1)
        morph_controls_scroll.setWidget(morph_controls)
        morph_top.addWidget(morph_controls_scroll)
        morph_top.setStretchFactor(0, 1)
        morph_top.setStretchFactor(1, 0)
        morph_top.setSizes([900, 330])

        self.morph_table = QTableWidget()
        self.morph_table.setAlternatingRowColors(True)
        self.morph_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.morph_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.morph_table.setSortingEnabled(True)
        self.morph_table.verticalHeader().setVisible(False)
        self.morph_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.morph_table.horizontalHeader().setStretchLastSection(True)
        self.morph_vertical_splitter.addWidget(self.morph_table)
        self.morph_vertical_splitter.setSizes([650, 250])
        self.tabs.addTab(morph_tab, "Morphology")

        # GPU 3D surface
        surf_tab = QWidget()
        surf_lay = QVBoxLayout(surf_tab)
        surf_lay.setContentsMargins(8, 8, 8, 8)
        surf_lay.setSpacing(6)
        surf_controls_host = QWidget()
        surf_controls = FlowLayout(surf_controls_host, hspacing=6, vspacing=6)
        self.gpu_3d_label = QLabel("GPU 3D: initializing OpenGL…" if gl is not None else "GPU 3D: unavailable")
        self.gpu_3d_label.setObjectName("subtitle")
        self.gpu_3d_label.setMinimumWidth(220)
        surf_controls.addWidget(self.gpu_3d_label)

        self.chk_3d_plane = QCheckBox("Visual plane correction")
        self.chk_3d_plane.setChecked(True)
        self.chk_3d_plane.setToolTip("3D-view-only correction; it does not change analysis data.")
        self.chk_3d_plane.stateChanged.connect(self.render_3d)
        surf_controls.addWidget(self.chk_3d_plane)
        self.chk_3d_center = QCheckBox("Relative Z")
        self.chk_3d_center.setChecked(True)
        self.chk_3d_center.stateChanged.connect(self.render_3d)
        surf_controls.addWidget(self.chk_3d_center)
        self.surface_scale_mode = QComboBox()
        self.surface_scale_mode.addItems(["Auto morphology height", "True XYZ scale"])
        self.surface_scale_mode.currentTextChanged.connect(self.render_3d)
        surf_controls.addWidget(self.surface_scale_mode)
        self.surface_z_scale = QDoubleSpinBox()
        self.surface_z_scale.setRange(0.1, 100.0)
        self.surface_z_scale.setDecimals(1)
        self.surface_z_scale.setSingleStep(0.5)
        self.surface_z_scale.setValue(1.0)
        self.surface_z_scale.setSuffix(" × Z")
        self.surface_z_scale.valueChanged.connect(self.render_3d)
        surf_controls.addWidget(self.surface_z_scale)
        self.surface_detail = QComboBox()
        self.surface_detail.addItems([
            "Fast · 128",
            "Native · measured grid",
            "Ultra · 512 interpolated",
            "Extreme · 1024 interpolated",
        ])
        self.surface_detail.setCurrentText("Native · measured grid")
        self.surface_detail.setToolTip(
            "Native renders every measured AFM pixel. Ultra/Extreme create a smoother visual mesh only; "
            "all measurements and statistics remain on the original AFM grid."
        )
        self.surface_detail.currentTextChanged.connect(self.render_3d)
        surf_controls.addWidget(self.surface_detail)

        self.surface_shading = QComboBox()
        self.surface_shading.addItems(["Smooth shaded", "Color only"])
        self.surface_shading.setToolTip("Smooth shaded uses GPU-computed surface normals; Color only disables lighting shading.")
        self.surface_shading.currentTextChanged.connect(self.render_3d)
        surf_controls.addWidget(self.surface_shading)

        self.surface_camera_mode = QComboBox()
        self.surface_camera_mode.addItems(["Perspective", "Orthographic-like"])
        self.surface_camera_mode.setToolTip("Orthographic-like uses a very narrow field of view to minimize perspective distortion.")
        self.surface_camera_mode.currentTextChanged.connect(self._camera_mode_changed)
        surf_controls.addWidget(self.surface_camera_mode)

        self.surface_export_size = QComboBox()
        self.surface_export_size.addItems(["Viewport", "2K", "4K", "8K"])
        self.surface_export_size.setCurrentText("4K")
        self.surface_export_size.setToolTip("High-resolution PNG export is rendered again by OpenGL at the selected output size.")
        surf_controls.addWidget(self.surface_export_size)

        self.surface_export_bg = QComboBox()
        self.surface_export_bg.addItems(["Dark background", "White background", "Transparent background"])
        self.surface_export_bg.setToolTip("Background used only for exported 3D PNG images.")
        surf_controls.addWidget(self.surface_export_bg)

        self.btn_export_3d = QPushButton("Export 3D PNG")
        self.btn_export_3d.setToolTip("Render the current 3D scene to a high-resolution PNG using OpenGL.")
        self.btn_export_3d.clicked.connect(self.export_3d_png)
        surf_controls.addWidget(self.btn_export_3d)

        self.btn_reset_3d = QPushButton("Reset view")
        self.btn_reset_3d.clicked.connect(self.reset_3d_view)
        surf_controls.addWidget(self.btn_reset_3d)
        surf_lay.addWidget(surf_controls_host)

        self.surface_view = AFMGLView()
        if gl is not None:
            try:
                self.surface_view.setBackgroundColor((14, 24, 42))
            except Exception:
                pass
            self.surface_view.gpuInfoChanged.connect(self._set_gl_info)
        surf_lay.addWidget(self.surface_view, 1)
        self.surface_info = QLabel("Open a topography channel to render the GPU surface.")
        self.surface_info.setObjectName("subtitle")
        self.surface_info.setAlignment(Qt.AlignCenter)
        self.surface_info.setWordWrap(True)
        surf_lay.addWidget(self.surface_info)
        self.tabs.addTab(surf_tab, "3D Surface")

        # PSD
        psd_tab = QWidget()
        psd_lay = QVBoxLayout(psd_tab)
        psd_lay.setContentsMargins(8, 8, 8, 8)
        self.psd_canvas = MplCanvas(width=8, height=6)
        psd_lay.addWidget(self.psd_canvas)
        self.tabs.addTab(psd_tab, "PSD")

        # Smart analysis / scan quality
        smart_tab = QWidget()
        smart_root = QVBoxLayout(smart_tab)
        smart_root.setContentsMargins(8, 8, 8, 8)
        smart_root.setSpacing(7)

        smart_summary_host = QWidget()
        smart_summary_flow = FlowLayout(smart_summary_host, hspacing=7, vspacing=7)
        self.smart_cards = {}
        for name in ["Overall QC", "Tilt", "Lines", "Spikes", "F/B match"]:
            card = StatCard(name)
            card.setMinimumWidth(145)
            self.smart_cards[name] = card
            smart_summary_flow.addWidget(card)
        smart_root.addWidget(smart_summary_host)

        smart_split = QSplitter(Qt.Horizontal)
        smart_split.setChildrenCollapsible(False)
        smart_root.addWidget(smart_split, 1)

        smart_visual_host = QWidget()
        smart_visual_lay = QVBoxLayout(smart_visual_host)
        smart_visual_lay.setContentsMargins(0, 0, 0, 0)
        self.smart_canvas = MplCanvas(width=8, height=6)
        smart_visual_lay.addWidget(self.smart_canvas, 1)
        self.smart_recommend_label = QLabel("Run Smart QC to generate a conservative correction recommendation.")
        self.smart_recommend_label.setObjectName("subtitle")
        self.smart_recommend_label.setWordWrap(True)
        smart_visual_lay.addWidget(self.smart_recommend_label)
        smart_split.addWidget(smart_visual_host)

        smart_controls_scroll = QScrollArea()
        smart_controls_scroll.setWidgetResizable(True)
        smart_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        smart_controls_scroll.setMinimumWidth(290)
        smart_controls_scroll.setMaximumWidth(430)
        smart_controls = QWidget()
        sc = QVBoxLayout(smart_controls)
        sc.setContentsMargins(10, 8, 10, 10)
        sc.setSpacing(8)
        sh = QLabel("Smart Analysis & Scan QC")
        sh.setObjectName("sectionTitle")
        sc.addWidget(sh)
        sn = QLabel("The advisor never edits the AFM matrix automatically. Every flag and correction remains reviewable, and quantitative analysis stays on measured data.")
        sn.setWordWrap(True)
        sn.setObjectName("hint")
        sc.addWidget(sn)

        sc.addWidget(QLabel("Diagnostic view"))
        self.smart_view_combo = QComboBox()
        self.smart_view_combo.addItems([
            "Artifact diagnostics",
            "Local Sa map",
            "Local Sq map",
            "Scale-dependent roughness",
            "Autocorrelation",
            "Forward/backward difference",
            "Leveling candidates",
        ])
        self.smart_view_combo.currentTextChanged.connect(self.render_smart_qc)
        sc.addWidget(self.smart_view_combo)

        qgrid = QGridLayout()
        qgrid.setHorizontalSpacing(7)
        qgrid.setVerticalSpacing(7)
        qgrid.addWidget(QLabel("Local window"), 0, 0)
        self.smart_window_px = QSpinBox()
        self.smart_window_px.setRange(5, 151)
        self.smart_window_px.setSingleStep(2)
        self.smart_window_px.setValue(31)
        self.smart_window_px.setSuffix(" px")
        self.smart_window_px.setToolTip("Sliding window used for local Sa/Sq maps. Odd values are used internally.")
        self.smart_window_px.valueChanged.connect(self._smart_window_changed)
        qgrid.addWidget(self.smart_window_px, 0, 1)
        qgrid.setColumnStretch(1, 1)
        sc.addLayout(qgrid)

        self.btn_smart_run = QPushButton("Run Smart QC")
        self.btn_smart_run.setObjectName("primary")
        self.btn_smart_run.clicked.connect(self.run_smart_qc)
        sc.addWidget(self.btn_smart_run)
        self.btn_smart_apply = QPushButton("Apply recommended correction")
        self.btn_smart_apply.clicked.connect(self.apply_recommended_correction)
        sc.addWidget(self.btn_smart_apply)
        self.btn_smart_morph = QPushButton("Recommend morphology settings")
        self.btn_smart_morph.clicked.connect(self.recommend_smart_morphology)
        sc.addWidget(self.btn_smart_morph)
        self.btn_smart_profile = QPushButton("Profile strongest feature")
        self.btn_smart_profile.clicked.connect(self.auto_profile_strongest_feature)
        sc.addWidget(self.btn_smart_profile)
        self.btn_smart_export = QPushButton("Export QC report · JSON")
        self.btn_smart_export.clicked.connect(self.export_smart_qc_json)
        sc.addWidget(self.btn_smart_export)

        rep_label = QLabel("QC report")
        rep_label.setObjectName("sectionTitle")
        sc.addWidget(rep_label)
        self.smart_report = QTextEdit()
        self.smart_report.setReadOnly(True)
        self.smart_report.setMinimumHeight(280)
        self.smart_report.setPlainText("Open a scan and run Smart QC.\n\nThe report will distinguish measured diagnostics from processing recommendations.")
        sc.addWidget(self.smart_report, 1)
        smart_controls_scroll.setWidget(smart_controls)
        smart_split.addWidget(smart_controls_scroll)
        smart_split.setStretchFactor(0, 1)
        smart_split.setStretchFactor(1, 0)
        smart_split.setSizes([920, 360])
        self.smart_tab_index = self.tabs.addTab(smart_tab, "Smart QC")

        # Batch analysis
        batch_tab = QWidget()
        batch_root = QVBoxLayout(batch_tab)
        batch_root.setContentsMargins(8, 8, 8, 8)
        batch_root.setSpacing(7)

        batch_toolbar_host = QWidget()
        batch_toolbar = FlowLayout(batch_toolbar_host, hspacing=6, vspacing=6)
        self.btn_batch_add = QPushButton("Add NID files…")
        self.btn_batch_add.setObjectName("primary")
        self.btn_batch_add.clicked.connect(self.batch_add_files)
        batch_toolbar.addWidget(self.btn_batch_add)
        self.btn_batch_folder = QPushButton("Add folder…")
        self.btn_batch_folder.clicked.connect(self.batch_add_folder)
        batch_toolbar.addWidget(self.btn_batch_folder)
        self.btn_batch_remove = QPushButton("Remove selected")
        self.btn_batch_remove.clicked.connect(self.batch_remove_selected)
        batch_toolbar.addWidget(self.btn_batch_remove)
        self.btn_batch_clear = QPushButton("Clear")
        self.btn_batch_clear.clicked.connect(self.batch_clear)
        batch_toolbar.addWidget(self.btn_batch_clear)
        self.btn_batch_run = QPushButton("Run batch analysis")
        self.btn_batch_run.setObjectName("primary")
        self.btn_batch_run.clicked.connect(self.run_batch_analysis)
        batch_toolbar.addWidget(self.btn_batch_run)
        self.btn_batch_cancel = QPushButton("Cancel batch")
        self.btn_batch_cancel.setEnabled(False)
        self.btn_batch_cancel.setToolTip("Finish the current file, then stop. Completed results remain exportable.")
        self.btn_batch_cancel.clicked.connect(self.cancel_batch_analysis)
        batch_toolbar.addWidget(self.btn_batch_cancel)
        self.btn_batch_export = QPushButton("Export clean Excel…")
        self.btn_batch_export.clicked.connect(self.export_batch_excel)
        batch_toolbar.addWidget(self.btn_batch_export)
        batch_root.addWidget(batch_toolbar_host)

        batch_split = QSplitter(Qt.Horizontal)
        batch_split.setChildrenCollapsible(False)
        batch_root.addWidget(batch_split, 1)

        batch_table_host = QWidget()
        batch_table_lay = QVBoxLayout(batch_table_host)
        batch_table_lay.setContentsMargins(0, 0, 0, 0)
        self.batch_table = QTableWidget()
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.batch_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.batch_table.verticalHeader().setVisible(False)
        self.batch_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.batch_table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.batch_table.setColumnCount(13)
        self.batch_table.setHorizontalHeaderLabels([
            "File", "Status", "Scan X×Y (µm)", "Pixels", "Prep steps", "Particles",
            "Size-QC n", "Mean apparent diameter (nm)", "Mean P95 height (nm)", "Coverage (%)",
            "Sa (nm)", "Sq (nm)", "Message",
        ])
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        batch_table_lay.addWidget(self.batch_table, 1)
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.batch_progress.setTextVisible(True)
        batch_table_lay.addWidget(self.batch_progress)
        self.batch_status = QLabel("Add .nid files, then run one reproducible morphology pipeline across the set.")
        self.batch_status.setObjectName("subtitle")
        self.batch_status.setWordWrap(True)
        batch_table_lay.addWidget(self.batch_status)
        batch_split.addWidget(batch_table_host)

        batch_controls_scroll = QScrollArea()
        batch_controls_scroll.setWidgetResizable(True)
        batch_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        batch_controls_scroll.setMinimumWidth(285)
        batch_controls_scroll.setMaximumWidth(410)
        batch_controls = QWidget()
        self.batch_control_panel = batch_controls
        bc = QVBoxLayout(batch_controls)
        bc.setContentsMargins(10, 8, 10, 10)
        bc.setSpacing(8)

        bh = QLabel("Batch morphology & size")
        bh.setObjectName("sectionTitle")
        bc.addWidget(bh)
        bn = QLabel(
            "Batch analysis uses the settings currently shown in the Morphology tab. "
            "For comparable samples, keep one fixed segmentation protocol. Every Excel table includes the source file name."
        )
        bn.setObjectName("hint")
        bn.setWordWrap(True)
        bc.addWidget(bn)

        channel_label = QLabel("Topography channel")
        channel_label.setObjectName("sectionTitle")
        bc.addWidget(channel_label)
        self.batch_channel_mode = QComboBox()
        self.batch_channel_mode.addItems([
            "Auto · Forward Z-Axis",
            "Backward Z-Axis",
            "Match current channel",
        ])
        self.batch_channel_mode.setToolTip("Auto prefers Scan forward — Z-Axis. Match current channel uses the currently selected channel name/frame when available.")
        bc.addWidget(self.batch_channel_mode)

        prep_label = QLabel("Preprocessing task engine")
        prep_label.setObjectName("sectionTitle")
        bc.addWidget(prep_label)
        prep_note = QLabel(
            "Check the tasks you want. They are executed from top to bottom; drag rows to change the order. "
            "A useful workflow is: optimize one scan in the main viewer → Capture active scan recipe → run the same recipe on the batch."
        )
        prep_note.setObjectName("hint")
        prep_note.setWordWrap(True)
        bc.addWidget(prep_note)

        batch_recipe_buttons = QVBoxLayout()
        self.btn_batch_apply_active = QPushButton("Apply tasks to active scan")
        self.btn_batch_apply_active.setToolTip("Test the checked batch preprocessing recipe on the currently open scan before running all files.")
        self.btn_batch_apply_active.clicked.connect(self.batch_apply_tasks_to_active_scan)
        batch_recipe_buttons.addWidget(self.btn_batch_apply_active)
        self.btn_batch_capture_recipe = QPushButton("Capture active scan recipe")
        self.btn_batch_capture_recipe.clicked.connect(self.batch_capture_active_recipe)
        batch_recipe_buttons.addWidget(self.btn_batch_capture_recipe)
        bc.addLayout(batch_recipe_buttons)
        self.btn_batch_clear_recipe = QPushButton("Clear preprocessing tasks")
        self.btn_batch_clear_recipe.clicked.connect(self.batch_clear_task_engine)
        bc.addWidget(self.btn_batch_clear_recipe)

        self.batch_task_list = QListWidget()
        self.batch_task_list.setMinimumHeight(154)
        self.batch_task_list.setMaximumHeight(210)
        self.batch_task_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.batch_task_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.batch_task_list.setAlternatingRowColors(True)
        self._batch_populate_task_list()
        self.batch_task_list.itemChanged.connect(lambda _item: self._batch_update_task_summary())
        self.batch_task_list.model().rowsMoved.connect(lambda *_args: self._batch_update_task_summary())
        bc.addWidget(self.batch_task_list)

        param_grid = QGridLayout()
        param_grid.setHorizontalSpacing(6)
        param_grid.setVerticalSpacing(5)
        param_grid.addWidget(QLabel("Line degree"), 0, 0)
        self.batch_line_degree = QSpinBox()
        self.batch_line_degree.setRange(0, 5)
        self.batch_line_degree.setValue(1)
        param_grid.addWidget(self.batch_line_degree, 0, 1)
        param_grid.addWidget(QLabel("Gaussian σ (px)"), 1, 0)
        self.batch_gaussian_sigma = QDoubleSpinBox()
        self.batch_gaussian_sigma.setRange(0.05, 20.0)
        self.batch_gaussian_sigma.setDecimals(2)
        self.batch_gaussian_sigma.setSingleStep(0.1)
        self.batch_gaussian_sigma.setValue(1.0)
        param_grid.addWidget(self.batch_gaussian_sigma, 1, 1)
        param_grid.addWidget(QLabel("Median size (px)"), 2, 0)
        self.batch_median_size = QSpinBox()
        self.batch_median_size.setRange(1, 31)
        self.batch_median_size.setSingleStep(2)
        self.batch_median_size.setValue(3)
        param_grid.addWidget(self.batch_median_size, 2, 1)
        self.batch_line_degree.valueChanged.connect(lambda _v: self._batch_update_task_summary())
        self.batch_gaussian_sigma.valueChanged.connect(lambda _v: self._batch_update_task_summary())
        self.batch_median_size.valueChanged.connect(lambda _v: self._batch_update_task_summary())
        bc.addLayout(param_grid)

        self.batch_task_summary = QLabel("")
        self.batch_task_summary.setObjectName("hint")
        self.batch_task_summary.setWordWrap(True)
        bc.addWidget(self.batch_task_summary)
        self._batch_update_task_summary()

        morph_label = QLabel("Morphology protocol")
        morph_label.setObjectName("sectionTitle")
        bc.addWidget(morph_label)
        self.batch_smart_morph = QCheckBox("Auto segmentation per file")
        self.batch_smart_morph.setChecked(False)
        self.batch_smart_morph.setToolTip(
            "OFF is recommended for direct sample-to-sample comparison because every scan uses identical segmentation settings. "
            "Turn ON only when independent automatic segmentation is intentionally desired."
        )
        bc.addWidget(self.batch_smart_morph)
        self.batch_protocol_summary = QLabel("Current Morphology settings will be captured when Run batch analysis is clicked.")
        self.batch_protocol_summary.setObjectName("hint")
        self.batch_protocol_summary.setWordWrap(True)
        bc.addWidget(self.batch_protocol_summary)
        bc.addStretch(1)
        batch_controls_scroll.setWidget(batch_controls)
        batch_split.addWidget(batch_controls_scroll)
        batch_split.setStretchFactor(0, 1)
        batch_split.setStretchFactor(1, 0)
        batch_split.setSizes([1040, 340])
        self.batch_tab_index = self.tabs.addTab(batch_tab, "Batch Analysis")

        # Metadata
        meta_tab = QWidget()
        meta_lay = QVBoxLayout(meta_tab)
        meta_lay.setContentsMargins(8, 8, 8, 8)
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        meta_lay.addWidget(self.meta_text)
        self.tabs.addTab(meta_tab, "Metadata")

        for index, tip in enumerate([
            "Inspect the map, compare raw and processed data, or select a profile and ROI.",
            "Surface roughness, height distribution, and bearing curve.",
            "Segment particles and inspect calibrated size and height measurements.",
            "Interactive surface visualization and high-resolution export.",
            "Radial power spectrum of the current scan.",
            "Inspect artifacts and review processing recommendations.",
            "Apply one processing recipe across multiple NID scans.",
            "Source metadata and calibration details.",
        ]):
            self.tabs.setTabToolTip(index, tip)

        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Debounced live morphology preview.
        self.morph_timer = QTimer(self)
        self.morph_timer.setSingleShot(True)
        self.morph_timer.setInterval(220)
        self.morph_timer.timeout.connect(self.run_morphology_analysis)
        for w in [
            self.morph_direction, self.morph_threshold_method, self.morph_manual_threshold,
            self.morph_k_sigma, self.morph_smooth, self.morph_min_area, self.morph_max_area,
            self.morph_watershed_distance, self.morph_opening, self.morph_closing,
            self.morph_min_diameter_nm, self.morph_max_diameter_nm, self.morph_min_pixels_across,
            self.morph_background_mode, self.morph_background_ring, self.morph_exclude_underresolved,
            self.morph_plane, self.morph_fill_holes, self.morph_clear_border, self.morph_watershed,
        ]:
            if isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._schedule_morphology)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self._schedule_morphology)
            else:
                w.stateChanged.connect(self._schedule_morphology)

        self._set_enabled(False)

    def _make_side_card(self, title: str):
        frame = CollapsibleCard(title, expanded=title in {"DATASET", "PREPROCESSING"})
        return frame, frame.content_layout

    def set_reduced_motion(self, reduced: bool):
        QApplication.instance().setProperty("reducedMotion", bool(reduced))
        for scene in self.findChildren(OrbitalScene):
            scene.set_motion()
        for card in self.findChildren(CollapsibleCard):
            card.settle_motion()
        if hasattr(self, "tabs"):
            self.tabs.tabBar().animation.stop()
            self.tabs.tabBar().update()

    def closeEvent(self, event):
        if self.batch_worker is not None and self.batch_worker.isRunning():
            self._close_after_batch = True
            self.cancel_batch_analysis()
            self.statusBar().showMessage("Completing the current scan before closing…")
            event.ignore()
            return
        super().closeEvent(event)

    def toggle_sidebar(self, visible: bool):
        self.sidebar_scroll.setVisible(bool(visible))
        self.a_sidebar.setChecked(bool(visible))
        self.workspace_header.controls_button.blockSignals(True)
        self.workspace_header.controls_button.setChecked(bool(visible))
        self.workspace_header.controls_button.blockSignals(False)
        if visible:
            self.main_splitter.setSizes([300, max(600, self.width() - 300)])

    def _snapshot(self, description: str):
        return {
            "description": description,
            "data": self.current.copy() if self.current is not None else None,
            "x_range_m": float(self.x_range_m),
            "y_range_m": float(self.y_range_m),
            "crop_origin": tuple(self.crop_origin),
        }

    def _init_history(self, description: str = "Raw channel"):
        self.history = [self._snapshot(description)] if self.current is not None else []
        self.history_index = 0 if self.history else -1
        self._update_history_list()

    def _commit_history(self, description: str):
        if self.current is None:
            return
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(self._snapshot(description))
        self.history_index = len(self.history) - 1
        self._update_history_list()

    def _update_history_list(self):
        if not hasattr(self, "history_list"):
            return
        self.history_list.blockSignals(True)
        self.history_list.clear()
        for i, state in enumerate(self.history):
            item = QListWidgetItem(f"{i + 1:02d}  {state['description']}")
            if i == self.history_index:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.history_list.addItem(item)
        if 0 <= self.history_index < self.history_list.count():
            self.history_list.setCurrentRow(self.history_index)
            self.history_list.scrollToItem(self.history_list.item(self.history_index))
        self.history_list.blockSignals(False)
        if hasattr(self, "btn_history_undo"):
            can_undo = self.current is not None and self.history_index > 0
            can_redo = self.current is not None and self.history_index < len(self.history) - 1
            self.btn_history_undo.setEnabled(can_undo)
            self.btn_history_redo.setEnabled(can_redo)
            self.a_undo.setEnabled(can_undo)
            self.a_redo.setEnabled(can_redo)
            self.workspace_header.undo_button.setEnabled(can_undo)
            self.workspace_header.redo_button.setEnabled(can_redo)
            self.btn_history_restore.setEnabled(self.current is not None and bool(self.history))

    def _restore_history(self, index: int):
        if not (0 <= index < len(self.history)):
            return
        state = self.history[index]
        if state["data"] is None:
            return
        self.current = state["data"].copy()
        self.x_range_m = float(state["x_range_m"])
        self.y_range_m = float(state["y_range_m"])
        self.crop_origin = tuple(state.get("crop_origin", (0, 0)))
        self.history_index = index
        self.pending_roi = None
        self.profile_points = []
        self._surface_reset_pending = True
        self._invalidate_morphology()
        self._update_history_list()
        self.refresh_all()
        self.statusBar().showMessage(f"Restored analysis state: {state['description']}")

    def undo_history(self):
        if self.history_index > 0:
            self._restore_history(self.history_index - 1)

    def redo_history(self):
        if self.history_index < len(self.history) - 1:
            self._restore_history(self.history_index + 1)

    def restore_selected_history(self):
        row = self.history_list.currentRow()
        if row >= 0:
            self._restore_history(row)

    def _invalidate_morphology(self):
        self.cancel_map_tool()
        self.morphology_result = None
        self._invalidate_smart_qc()
        if hasattr(self, "morph_threshold_readout"):
            self.morph_threshold_readout.setText("Threshold: —")
        if hasattr(self, "morph_table"):
            self.morph_table.setRowCount(0)
        if hasattr(self, "morph_cards"):
            for card in self.morph_cards.values():
                card.value.setText("—")

    def _invalidate_smart_qc(self):
        self.smart_qc_result = None
        self.smart_morph_recommendation = None
        self.smart_channel_correlations = {}
        if hasattr(self, "smart_cards"):
            for card in self.smart_cards.values():
                card.value.setText("—")
        if hasattr(self, "smart_recommend_label"):
            self.smart_recommend_label.setText("Run Smart QC to generate a conservative correction recommendation.")
        if hasattr(self, "smart_report"):
            self.smart_report.setPlainText("Smart QC is not current for this analysis state. Run it again to refresh diagnostics.")

    def _set_enabled(self, enabled: bool):
        for w in [
            self.channel_combo, self.btn_plane, self.btn_robust_plane, self.btn_flatten, self.btn_row,
            self.btn_gauss, self.btn_median, self.btn_reset, self.btn_profile,
            self.btn_roi, self.btn_crop, self.cmap_combo, self.chk_reverse,
            self.btn_export_png, self.btn_export_csv, self.btn_export_tiff,
            self.btn_export_json, self.chk_3d_plane, self.chk_3d_center,
            self.surface_scale_mode, self.surface_z_scale, self.surface_detail, self.surface_shading,
            self.surface_camera_mode, self.surface_export_size, self.surface_export_bg,
            self.btn_reset_3d, self.btn_history_undo, self.btn_history_restore,
            self.morph_direction, self.morph_threshold_method, self.morph_manual_threshold,
            self.morph_k_sigma, self.morph_smooth, self.morph_min_area, self.morph_max_area,
            self.morph_watershed_distance, self.morph_opening, self.morph_closing,
            self.morph_min_diameter_nm, self.morph_max_diameter_nm, self.morph_min_pixels_across,
            self.morph_background_mode, self.morph_background_ring, self.morph_exclude_underresolved,
            self.morph_distribution_population, self.morph_hist_bins, self.btn_morph_apply_qc_min,
            self.morph_plane, self.morph_fill_holes, self.morph_clear_border,
            self.morph_watershed, self.morph_live, self.btn_morph_analyze,
            self.btn_morph_csv, self.btn_morph_xlsx, self.btn_morph_figure, self.btn_morph_exclude, self.btn_morph_reset_mask,
            self.btn_smart_qc_side, self.btn_smart_morph_side, self.smart_view_combo, self.smart_window_px,
            self.btn_smart_run, self.btn_smart_apply, self.btn_smart_morph, self.btn_smart_profile, self.btn_smart_export,
        ]:
            w.setEnabled(enabled)
        self.chk_cuda.setEnabled(enabled and self.cuda.available)
        self.map_controls.setVisible(enabled)
        self.map_stack.setCurrentWidget(self.map_canvas if enabled else self.welcome_panel)
        self.workspace_header.raw_button.setEnabled(enabled)
        self.a_reset.setEnabled(enabled)
        for index in range(self.tabs.count()):
            self.tabs.setTabEnabled(index, enabled or index in {0, self.batch_tab_index})
        self._update_history_list()
        if enabled:
            self.morph_manual_threshold.setEnabled(self.morph_threshold_method.currentText() == "Manual")
            self.morph_k_sigma.setEnabled(self.morph_threshold_method.currentText().startswith("Median + k"))

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open AFM NID", "", "Nanosurf NID (*.nid);;All files (*.*)")
        if not path:
            return
        self.load_nid_file(path)

    def load_nid_file(self, path: str | Path) -> bool:
        try:
            nid = read_nid(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not read NID", str(exc))
            return False

        self.nid = nid

        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        for ch in self.nid.channels:
            self.channel_combo.addItem(ch.label)
        self.channel_combo.blockSignals(False)

        self.file_info.setText(f"{Path(path).name}\n{len(self.nid.channels)} image channels")
        self._set_enabled(True)

        # Prefer forward Z-Axis for morphology/topography if present.
        default = 0
        for i, ch in enumerate(self.nid.channels):
            if ch.frame.lower().startswith("scan forward") and ch.name.lower() == "z-axis":
                default = i
                break
        self.channel_combo.blockSignals(True)
        self.channel_combo.setCurrentIndex(default)
        self.channel_combo.blockSignals(False)
        self.tabs.setCurrentIndex(0)
        self.select_channel(default)
        self.statusBar().showMessage(f"Loaded {Path(path).name}")
        return True

    @staticmethod
    def _dropped_nid_paths(mime_data):
        return [url.toLocalFile() for url in mime_data.urls()
                if url.isLocalFile() and Path(url.toLocalFile()).is_file()
                and Path(url.toLocalFile()).suffix.lower() == ".nid"]

    def dragEnterEvent(self, event):
        if self._dropped_nid_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = self._dropped_nid_paths(event.mimeData())
        if not paths:
            event.ignore()
            return
        if len(paths) == 1:
            self.load_nid_file(paths[0])
        else:
            self._add_batch_paths(paths)
            self.tabs.setCurrentIndex(self.batch_tab_index)
            self.statusBar().showMessage(f"{len(self.batch_files)} NID scans queued for batch analysis.")
        event.acceptProposedAction()

    def select_channel(self, index: int):
        if self.nid is None or index < 0 or index >= len(self.nid.channels):
            return
        self.channel = self.nid.channels[index]
        self.original = self.channel.data.copy()
        self.current = self.original.copy()
        self.x_range_m = self.channel.x_range_m
        self.y_range_m = self.channel.y_range_m
        self.crop_origin = (0, 0)
        self.map_view_combo.setCurrentIndex(0)
        self.pending_roi = None
        self.profile_points = []
        self._surface_reset_pending = True
        self._invalidate_morphology()
        self._init_history("Raw channel")
        self._render_metadata()
        self.refresh_all()

    def reset_data(self):
        if self.original is None or self.channel is None:
            return
        if self.history:
            self._restore_history(0)
        else:
            self.current = self.original.copy()
            self.x_range_m = self.channel.x_range_m
            self.y_range_m = self.channel.y_range_m
            self._invalidate_morphology()
            self.refresh_all()
        self.statusBar().showMessage("Restored raw channel. Existing history remains available until the next edit.")

    def apply_plane(self):
        if self.current is None:
            return
        self.current, _ = fit_plane(self.current)
        self._invalidate_morphology()
        self._commit_history("Plane level · 1st order")
        self.refresh_all()
        self.statusBar().showMessage("First-order plane leveling applied.")

    def apply_robust_plane(self):
        if self.current is None:
            return
        try:
            self.current, _ = fit_plane(self.current, robust=True)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot fit robust plane", str(exc))
            return
        self._invalidate_morphology()
        self._commit_history("Robust plane level · Huber")
        self.refresh_all()
        self.statusBar().showMessage("Robust plane leveling applied; inspect raw/current comparison to assess the correction.")

    def apply_line_flatten(self):
        if self.current is None:
            return
        degree = self.flatten_degree.value()
        self.current = line_flatten(self.current, degree)
        self._invalidate_morphology()
        self._commit_history(f"Line flatten · degree {degree}")
        self.refresh_all()
        self.statusBar().showMessage(f"Line flatten applied (degree {degree}).")

    def apply_row_offsets(self):
        if self.current is None:
            return
        self.current = remove_row_offsets(self.current)
        self._invalidate_morphology()
        self._commit_history("Remove row offsets")
        self.refresh_all()
        self.statusBar().showMessage("Row offsets removed.")

    def apply_gaussian(self):
        if self.current is None:
            return
        backend = "CPU"
        sigma = self.sigma.value()
        if self.chk_cuda.isChecked() and self.cuda.available:
            try:
                self.current = gaussian_smooth_cuda(self.current, sigma)
                backend = f"CUDA · {self.cuda.device_name}"
            except Exception as exc:
                self.current = gaussian_smooth(self.current, sigma)
                self.chk_cuda.setChecked(False)
                backend = "CPU fallback"
                QMessageBox.warning(self, "CUDA filter fallback", f"CUDA filtering failed, so the CPU result was used.\n\n{exc}")
        else:
            self.current = gaussian_smooth(self.current, sigma)
        self._invalidate_morphology()
        self._commit_history(f"Gaussian · σ {sigma:g} px · {backend}")
        self.refresh_all()
        self.statusBar().showMessage(f"Gaussian smoothing applied (σ={sigma:g} px) · {backend}.")

    def apply_median(self):
        if self.current is None:
            return
        backend = "CPU"
        size = self.median_size.value()
        if self.chk_cuda.isChecked() and self.cuda.available:
            try:
                self.current = median_filter_cuda(self.current, size)
                backend = f"CUDA · {self.cuda.device_name}"
            except Exception as exc:
                self.current = median_filter(self.current, size)
                self.chk_cuda.setChecked(False)
                backend = "CPU fallback"
                QMessageBox.warning(self, "CUDA filter fallback", f"CUDA filtering failed, so the CPU result was used.\n\n{exc}")
        else:
            self.current = median_filter(self.current, size)
        self._invalidate_morphology()
        self._commit_history(f"Median filter · {size} px · {backend}")
        self.refresh_all()
        self.statusBar().showMessage(f"Median filter applied ({size} px) · {backend}.")

    def _cmap(self):
        name = self.cmap_combo.currentText() or "afmhot"
        return name + "_r" if self.chk_reverse.isChecked() else name

    def _scaled_axes(self):
        sx = xy_unit_scale(max(self.x_range_m, self.y_range_m))
        x_range = self.x_range_m * sx.factor
        y_range = self.y_range_m * sx.factor
        return sx, x_range, y_range

    def _scaled_z(self):
        assert self.current is not None
        unit = z_unit_scale(self.current, self.channel.unit if self.channel else "")
        return unit, self.current * unit.factor

    def refresh_all(self):
        """Invalidate data-dependent views and redraw only the visible tab."""
        if self.current is None:
            return
        self._update_workspace_header()
        self._dirty_tabs = {0, 1, 2, 3, 4, self.smart_tab_index}
        self._render_tab(self.tabs.currentIndex())

    def _on_tab_changed(self, index: int):
        if self.current is not None:
            self._render_tab(index)

    def _render_tab(self, index: int):
        if index not in self._dirty_tabs:
            return
        if index == 0:
            self.render_map()
        elif index == 1:
            self.render_stats()
        elif index == 2:
            if self.morphology_result is None:
                self.run_morphology_analysis()
            else:
                self.render_morphology()
        elif index == 3:
            self.render_3d()
        elif index == 4:
            self.render_psd()
        elif index == self.smart_tab_index:
            if self.smart_qc_result is None:
                self.run_smart_qc(switch_tab=False)
            else:
                self.render_smart_qc()
        self._dirty_tabs.discard(index)

    def _style_axis(self, ax):
        ax.set_facecolor("#0e182a")
        ax.tick_params(colors="#a8beda", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#354e70")
        ax.xaxis.label.set_color("#c5d7ee")
        ax.yaxis.label.set_color("#c5d7ee")
        ax.title.set_color("#ffffff")
        ax.grid(False)

    def _update_workspace_header(self):
        header = self.workspace_header
        if self.channel is None or self.current is None:
            header.set_scan_title(APP_NAME)
            header.badge.setText("No scan loaded")
            header.details.setText("Open a NID scan or drop files into the workspace.")
            return
        header.set_scan_title(self.nid.path.name if self.nid else self.channel.name)
        header.set_quality(getattr(self.channel, "scale_status", "UNVERIFIED"))
        header.badge.setToolTip(getattr(self.channel, "scale_note", ""))
        sx, xr, yr = self._scaled_axes()
        rows, cols = self.current.shape
        description = self.history[self.history_index]["description"] if self.history else "Raw channel"
        header.details.setText(
            f"{self.channel.label} · {cols} × {rows} px · {xr:.4g} × {yr:.4g} {sx.label}\n"
            f"State {self.history_index + 1}/{len(self.history)} · {description}"
        )
        self.setWindowTitle(f"{self.nid.path.name if self.nid else APP_NAME} — {APP_NAME} v{APP_VERSION}")

    def _raw_current_field(self):
        """Return raw pixels aligned with the active crop, without processing."""
        if self.original is None or self.current is None:
            return None
        r0, c0 = self.crop_origin
        rows, cols = self.current.shape
        raw = self.original[r0:r0 + rows, c0:c0 + cols]
        if raw.shape != self.current.shape:
            raise ValueError("The current crop does not match the raw scan geometry.")
        return raw

    def _map_view_changed(self, *_args):
        self.cancel_map_tool()
        if self.current is not None:
            self.render_map()
            self._dirty_tabs.discard(0)

    def cancel_map_tool(self, *_args, keep_profile=False):
        self.active_map_tool = None
        self.pending_roi = None
        self.profile_points = []
        if not keep_profile:
            for artist in self.profile_artists:
                artist.remove()
            self.profile_artists = []
        if self.roi_selector is not None:
            self.roi_selector.set_active(False)
            self.roi_selector.set_visible(False)
            self.roi_selector.disconnect_events()
            self.roi_selector = None
        if hasattr(self, "btn_map_profile"):
            self.btn_map_profile.setChecked(False)
            self.btn_map_roi.setChecked(False)
            self.btn_map_cancel.setEnabled(False)
            self.btn_map_crop.setEnabled(False)
            self.btn_crop.setEnabled(False)
            mode = self.map_view_combo.currentIndex()
            message = ("Measurements and exports use the current analysis state." if mode == 0 else
                       "Comparison preview. Measurements and exports use the current analysis state.")
            self.map_tool_hint.setText(message)
            self.map_canvas.draw_idle()

    def render_map(self):
        self.cancel_map_tool()
        self.map_canvas.figure.clear()
        self.map_canvas.figure.set_layout_engine("compressed")
        ax = self.map_canvas.figure.add_subplot(111)
        self.map_ax = ax
        sx, xr, yr = self._scaled_axes()
        zu = z_unit_scale(self.current, self.channel.unit if self.channel else "")
        view_mode = self.map_view_combo.currentIndex()
        raw = self._raw_current_field()
        map_data = raw if view_mode == 1 else (self.current - raw if view_mode == 2 else self.current)
        z = map_data * zu.factor
        extent = [0, xr, 0, yr]
        cmap = "RdBu_r" if view_mode == 2 else self._cmap()
        color_limits = {}
        finite = z[np.isfinite(z)]
        if view_mode == 2 and finite.size:
            limit = max(float(np.max(np.abs(finite))), np.finfo(float).eps)
            color_limits = {"vmin": -limit, "vmax": limit}
        im = ax.imshow(z, origin="lower", extent=extent, cmap=cmap, aspect="equal", **color_limits)
        ax.set_xlabel(f"X ({sx.label})")
        ax.set_ylabel(f"Y ({sx.label})")
        title = self.channel.label if self.channel else "AFM map"
        title += "\n" + ["Current state", "Raw · same field", "Processed − raw"][view_mode]
        ax.set_title(title, pad=10, fontweight="bold", fontsize=11)
        self._style_axis(ax)
        cb = self.map_canvas.figure.colorbar(im, ax=ax, shrink=0.86, pad=0.025, panchor=False)
        cb.set_label(zu.label, color="#c5d7ee")
        cb.ax.tick_params(colors="#a8beda")
        cb.outline.set_edgecolor("#354e70")
        self.map_canvas.draw_idle()

    def render_stats(self):
        zu, z = self._scaled_z()
        st = roughness_stats(z)
        for key, card in self.stat_cards.items():
            v = st.get(key, float("nan"))
            if key in {"Ssk", "Sku"}:
                card.value.setText(f"{v:.4g}")
            else:
                card.value.setText(f"{v:.4g} {zu.label}")

        self.stats_canvas.figure.clear()
        ax1 = self.stats_canvas.figure.add_subplot(121)
        ax2 = self.stats_canvas.figure.add_subplot(122)
        vals = z[np.isfinite(z)]
        ax1.hist(vals, bins=60, alpha=0.9)
        ax1.set_title("Height distribution")
        ax1.set_xlabel(f"Height ({zu.label})")
        ax1.set_ylabel("Count")
        self._style_axis(ax1)

        ratio, heights = bearing_curve(z)
        ax2.plot(ratio, heights, linewidth=1.6)
        ax2.set_title("Abbott–Firestone / bearing curve")
        ax2.set_xlabel("Material ratio (%)")
        ax2.set_ylabel(f"Height ({zu.label})")
        self._style_axis(ax2)
        self.stats_canvas.figure.tight_layout()
        self.stats_canvas.draw_idle()

    def _schedule_morphology(self, *_args):
        if not hasattr(self, "morph_live"):
            return
        manual = self.morph_threshold_method.currentText() == "Manual"
        self.morph_manual_threshold.setEnabled(self.current is not None and manual)
        self.morph_k_sigma.setEnabled(self.current is not None and self.morph_threshold_method.currentText().startswith("Median + k"))
        if self.current is not None:
            self.morphology_result = None
            self._dirty_tabs.add(2)
        if self.current is not None and self.morph_live.isChecked() and self.tabs.currentIndex() == 2:
            self.morph_timer.start()

    def run_morphology_analysis(self):
        if self.current is None or self.channel is None:
            return
        try:
            sx, xr, yr = self._scaled_axes()
            zu, z = self._scaled_z()
            method = self.morph_threshold_method.currentText()
            manual = self.morph_manual_threshold.value() if method == "Manual" else None
            nm_in_xy_unit = 1e-9 * sx.factor
            self.morphology_result = analyze_particles(
                z,
                x_range=xr,
                y_range=yr,
                threshold_method=method,
                manual_threshold=manual,
                k_sigma=self.morph_k_sigma.value(),
                direction=self.morph_direction.currentText(),
                smoothing_sigma=self.morph_smooth.value(),
                plane_correct=self.morph_plane.isChecked(),
                fill_holes=self.morph_fill_holes.isChecked(),
                opening_radius=self.morph_opening.value(),
                closing_radius=self.morph_closing.value(),
                min_area_px=self.morph_min_area.value(),
                max_area_px=self.morph_max_area.value(),
                clear_border_objects=self.morph_clear_border.isChecked(),
                watershed_split=self.morph_watershed.isChecked(),
                watershed_min_distance=self.morph_watershed_distance.value(),
                min_equivalent_diameter=self.morph_min_diameter_nm.value() * nm_in_xy_unit,
                max_equivalent_diameter=self.morph_max_diameter_nm.value() * nm_in_xy_unit,
                min_pixels_across=self.morph_min_pixels_across.value(),
                background_mode=self.morph_background_mode.currentText(),
                background_ring_px=self.morph_background_ring.value(),
                exclude_underresolved_from_summary=self.morph_exclude_underresolved.isChecked(),
            )
            r = self.morphology_result
            pitch_x_nm = float(r.summary.get("pixel_pitch_x", np.nan)) * (1e9 / sx.factor)
            pitch_y_nm = float(r.summary.get("pixel_pitch_y", np.nan)) * (1e9 / sx.factor)
            min_qc_nm = float(r.summary.get("minimum_qc_lateral_size", np.nan)) * (1e9 / sx.factor)
            scale_status = getattr(self.channel, "scale_status", "UNVERIFIED")
            self.morph_scale_readout.setText(
                f"Scale {scale_status}: {xr:.6g} × {yr:.6g} {sx.label} · "
                f"pixel pitch {pitch_x_nm:.4g} × {pitch_y_nm:.4g} nm · "
                f"size-QC limit ≈ {min_qc_nm:.4g} nm"
            )
            self.morph_threshold_readout.setText(
                f"Threshold: {r.threshold:.6g} {zu.label} · {len(r.records)} objects · "
                f"mask coverage {r.summary['coverage_percent']:.3g}%"
            )
            if method != "Manual":
                self.morph_manual_threshold.blockSignals(True)
                self.morph_manual_threshold.setValue(float(np.clip(r.threshold, -1e12, 1e12)))
                self.morph_manual_threshold.blockSignals(False)
            self.render_morphology()
            self._dirty_tabs.discard(2)
            self.statusBar().showMessage(
                f"Morphology analyzed: {len(r.records)} objects · threshold {r.threshold:.5g} {zu.label}."
            )
        except Exception as exc:
            self.morphology_result = None
            QMessageBox.critical(self, "Morphology analysis failed", str(exc))

    def _morph_unit_scales(self):
        sx, _, _ = self._scaled_axes()
        zu, _ = self._scaled_z()
        xy_to_nm = 1e9 / sx.factor
        if self.channel is not None and self.channel.unit == "m":
            z_to_nm = 1e9 / zu.factor
            height_unit = "nm"
            volume_unit = "nm³"
        else:
            z_to_nm = 1.0
            height_unit = zu.label
            volume_unit = f"{zu.label}·nm²"
        return sx, zu, xy_to_nm, z_to_nm, height_unit, volume_unit

    def _morph_table_rows(self):
        if self.morphology_result is None:
            return [], []
        sx, zu, xy_to_nm, z_to_nm, height_unit, volume_unit = self._morph_unit_scales()
        area_to_nm2 = xy_to_nm ** 2
        volume_factor = area_to_nm2 * z_to_nm
        headers = [
            "ID", f"X ({sx.label})", f"Y ({sx.label})", "Area (nm²)",
            "Apparent Eq. diameter (nm)", "Eq. low · 1px (nm)", "Eq. high · 1px (nm)",
            "Apparent Feret max (nm)", "Apparent Feret min (nm)", "Perimeter (nm)",
            "Pixels across", "Size QC", f"P95 height ({height_unit})",
            f"Max height ({height_unit})", f"Mean height ({height_unit})",
            f"Volume ({volume_unit})", "Circularity", "Aspect ratio", "Orientation (°)", "Pixels",
        ]
        rows = []
        for rec in self.morphology_result.records:
            rows.append([
                int(rec.particle_id), rec.centroid_x, rec.centroid_y,
                rec.area * area_to_nm2,
                rec.equivalent_diameter * xy_to_nm,
                rec.equivalent_diameter_low * xy_to_nm,
                rec.equivalent_diameter_high * xy_to_nm,
                rec.feret_max * xy_to_nm,
                rec.feret_min * xy_to_nm,
                rec.perimeter * xy_to_nm,
                rec.pixels_across,
                rec.size_qc_note,
                rec.p95_height * z_to_nm,
                rec.max_height * z_to_nm,
                rec.mean_height * z_to_nm,
                rec.volume * volume_factor,
                rec.circularity, rec.aspect_ratio, rec.orientation_deg, int(rec.pixel_count),
            ])
        return headers, rows

    def set_morph_min_diameter_to_qc_limit(self):
        """Set the physical minimum-diameter filter to the current sampling QC limit."""
        if self.channel is None:
            QMessageBox.information(self, "Particle size", "Open a calibrated AFM topography channel first.")
            return
        rows = int(self.current.shape[0]) if self.current is not None else int(getattr(self.channel, "rows", 0) or 0)
        cols = int(self.current.shape[1]) if self.current is not None else int(getattr(self.channel, "cols", 0) or 0)
        if rows <= 0 or cols <= 0 or self.x_range_m <= 0 or self.y_range_m <= 0:
            QMessageBox.warning(self, "Particle size", "A valid calibrated X/Y scan field is required.")
            return
        dx_nm = abs(self.x_range_m) / cols * 1e9
        dy_nm = abs(self.y_range_m) / rows * 1e9
        qc_nm = float(self.morph_min_pixels_across.value()) * max(dx_nm, dy_nm)
        self.morph_min_diameter_nm.setValue(qc_nm)
        self.statusBar().showMessage(
            f"Minimum apparent diameter set to the current size-QC limit: {qc_nm:.4g} nm. Re-run morphology to apply it."
        )

    def _morph_distribution_records(self):
        if self.morphology_result is None:
            return []
        records = list(self.morphology_result.records)
        if hasattr(self, "morph_distribution_population") and self.morph_distribution_population.currentIndex() == 0:
            return [rec for rec in records if rec.size_qc_pass]
        return records

    def _morph_histogram(self, values):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        fixed = int(self.morph_hist_bins.value()) if hasattr(self, "morph_hist_bins") else 0
        if fixed > 0:
            bins = min(max(1, fixed), max(1, vals.size))
            counts, edges = np.histogram(vals, bins=bins)
            return counts, edges
        if vals.size < 3 or np.nanmin(vals) == np.nanmax(vals):
            counts, edges = np.histogram(vals, bins=1)
            return counts, edges
        try:
            edges = np.histogram_bin_edges(vals, bins="fd")
            # Keep automatic histograms readable for small AFM populations.
            if edges.size < 4:
                edges = np.histogram_bin_edges(vals, bins="auto")
            if edges.size > 81:
                edges = np.linspace(float(np.nanmin(vals)), float(np.nanmax(vals)), 81)
            counts, edges = np.histogram(vals, bins=edges)
        except Exception:
            counts, edges = np.histogram(vals, bins=min(30, max(5, int(np.sqrt(vals.size)))))
        return counts, edges

    def _morph_size_statistics(self):
        records = self._morph_distribution_records()
        _, _, xy_to_nm, z_to_nm, height_unit, _ = self._morph_unit_scales()
        diam = np.asarray([r.equivalent_diameter * xy_to_nm for r in records], dtype=float)
        p95 = np.asarray([r.p95_height * z_to_nm for r in records], dtype=float)
        feret_max = np.asarray([r.feret_max * xy_to_nm for r in records], dtype=float)
        feret_min = np.asarray([r.feret_min * xy_to_nm for r in records], dtype=float)

        def stats(vals):
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return {k: np.nan for k in ("n", "mean", "median", "sd", "cv_percent", "min", "p10", "q25", "q75", "p90", "max")}
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
            return {
                "n": int(vals.size),
                "mean": mean,
                "median": float(np.median(vals)),
                "sd": sd,
                "cv_percent": float(sd / mean * 100.0) if np.isfinite(sd) and mean != 0 else np.nan,
                "min": float(np.min(vals)),
                "p10": float(np.percentile(vals, 10)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "p90": float(np.percentile(vals, 90)),
                "max": float(np.max(vals)),
            }

        population = self.morph_distribution_population.currentText() if hasattr(self, "morph_distribution_population") else "All detected particles"
        return {
            "population": population,
            "diameter_nm": stats(diam),
            "feret_max_nm": stats(feret_max),
            "feret_min_nm": stats(feret_min),
            "p95_height": stats(p95),
            "height_unit": height_unit,
        }

    def _morph_distribution_export_rows(self):
        records = self._morph_distribution_records()
        _, _, xy_to_nm, z_to_nm, height_unit, _ = self._morph_unit_scales()
        population = self.morph_distribution_population.currentText() if hasattr(self, "morph_distribution_population") else "All detected particles"
        variables = [
            ("Apparent equivalent diameter", "nm", [r.equivalent_diameter * xy_to_nm for r in records]),
            ("P95 height", height_unit, [r.p95_height * z_to_nm for r in records]),
            ("Circularity", "dimensionless", [r.circularity for r in records]),
        ]
        out = []
        for variable, unit, vals in variables:
            counts, edges = self._morph_histogram(vals)
            total = float(np.sum(counts))
            for i, count in enumerate(counts):
                lo = float(edges[i])
                hi = float(edges[i + 1])
                out.append([
                    variable, unit, population, i + 1, lo, hi, (lo + hi) / 2.0,
                    int(count), float(count / total) if total > 0 else np.nan,
                ])
        return out

    def _update_morph_summary(self):
        if self.morphology_result is None:
            return
        r = self.morphology_result
        sx, zu, xy_to_nm, z_to_nm, height_unit, _ = self._morph_unit_scales()
        scan_area_um2 = abs(self.x_range_m * self.y_range_m) * 1e12
        density_um2 = len(r.records) / scan_area_um2 if scan_area_um2 > 0 else np.nan
        mean_d = float(r.summary.get("mean_equivalent_diameter", np.nan)) * xy_to_nm
        mean_h = float(r.summary.get("mean_p95_height", r.summary.get("mean_max_height", np.nan))) * z_to_nm
        qc_n = int(float(r.summary.get("size_qc_count", 0)))
        pitch_nm = float(r.summary.get("sampling_qc_pitch", np.nan)) * xy_to_nm
        self.morph_cards["Particles"].value.setText(str(len(r.records)))
        self.morph_cards["Size-QC n"].value.setText(str(qc_n))
        self.morph_cards["Density"].value.setText(f"{density_um2:.4g} /µm²")
        self.morph_cards["Coverage"].value.setText(f"{r.summary.get('coverage_percent', np.nan):.4g} %")
        self.morph_cards["Mean diameter"].value.setText(f"{mean_d:.4g} nm")
        self.morph_cards["Mean height"].value.setText(f"{mean_h:.4g} {height_unit}")
        self.morph_cards["Pixel pitch"].value.setText(f"{pitch_nm:.4g} nm")

    def render_morphology(self):
        if self.morphology_result is None or self.current is None:
            return
        r = self.morphology_result
        sx, xr, yr = self._scaled_axes()
        zu, _ = self._scaled_z()
        self._update_morph_summary()

        self.morph_canvas.figure.clear()
        ax = self.morph_canvas.figure.add_subplot(111)
        im = ax.imshow(
            r.measurement_data, origin="lower", extent=[0, xr, 0, yr],
            cmap=self._cmap(), aspect="equal",
        )
        labels = np.ma.masked_where(r.labels <= 0, r.labels)
        if np.any(r.labels > 0):
            ax.imshow(labels, origin="lower", extent=[0, xr, 0, yr], cmap="tab20", alpha=0.34,
                      interpolation="nearest", aspect="equal")
            try:
                ax.contour(r.mask.astype(float), levels=[0.5], origin="lower", extent=[0, xr, 0, yr],
                           colors="white", linewidths=0.45, alpha=0.85)
            except Exception:
                pass
        if len(r.records) <= 120:
            for rec in r.records:
                ax.text(rec.centroid_x, rec.centroid_y, str(rec.particle_id), fontsize=6.5,
                        ha="center", va="center", color="white")
        ax.set_xlabel(f"X ({sx.label})")
        ax.set_ylabel(f"Y ({sx.label})")
        ax.set_title(f"Morphology mask · {self.morph_threshold_method.currentText()} threshold")
        self._style_axis(ax)
        cb = self.morph_canvas.figure.colorbar(im, ax=ax, shrink=0.86, pad=0.025)
        cb.set_label(zu.label, color="#c5d7ee")
        cb.ax.tick_params(colors="#a8beda")
        cb.outline.set_edgecolor("#354e70")
        self.morph_canvas.figure.tight_layout()
        self.morph_canvas.draw_idle()

        headers, rows = self._morph_table_rows()
        self.morph_table.setSortingEnabled(False)
        self.morph_table.clear()
        self.morph_table.setColumnCount(len(headers))
        self.morph_table.setHorizontalHeaderLabels(headers)
        self.morph_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                item = NumericItem(value)
                self.morph_table.setItem(i, j, item)
        self.morph_table.setSortingEnabled(True)

        self.morph_dist_canvas.figure.clear()
        ax1 = self.morph_dist_canvas.figure.add_subplot(131)
        ax2 = self.morph_dist_canvas.figure.add_subplot(132)
        ax3 = self.morph_dist_canvas.figure.add_subplot(133)
        plot_records = self._morph_distribution_records()
        _, _, xy_to_nm, z_to_nm, height_unit, _ = self._morph_unit_scales()
        diam = np.asarray([rec.equivalent_diameter * xy_to_nm for rec in plot_records], dtype=float)
        height = np.asarray([rec.p95_height * z_to_nm for rec in plot_records], dtype=float)
        circ = np.asarray([rec.circularity for rec in plot_records], dtype=float)
        for ax, vals in ((ax1, diam), (ax2, height), (ax3, circ)):
            vals = vals[np.isfinite(vals)]
            if vals.size:
                counts, edges = self._morph_histogram(vals)
                if counts.size:
                    ax.hist(vals, bins=edges)
        pop_short = "QC-valid" if self.morph_distribution_population.currentIndex() == 0 else "all detected"
        ax1.set_title(f"Apparent diameter\n{pop_short} · n={len(plot_records)}", fontsize=10)
        ax1.set_xlabel("nm")
        ax2.set_title("P95 height")
        ax2.set_xlabel(height_unit)
        ax3.set_title("Circularity")
        for a in (ax1, ax2, ax3):
            self._style_axis(a)
            a.tick_params(labelsize=8)
        self.morph_dist_canvas.figure.tight_layout()
        self.morph_dist_canvas.draw_idle()

    def _recalculate_morph_summary_after_mask_edit(self):
        if self.morphology_result is None:
            return
        r = self.morphology_result
        mask = r.labels > 0
        r.mask = mask
        sx, xr, yr = self._scaled_axes()
        image_area = abs(xr * yr)
        qc_records = [x for x in r.records if x.size_qc_pass]
        used = qc_records if self.morph_exclude_underresolved.isChecked() else r.records
        eqs = np.array([x.equivalent_diameter for x in used], dtype=float)
        heights = np.array([x.p95_height for x in used], dtype=float)
        max_heights = np.array([x.max_height for x in used], dtype=float)
        areas = np.array([x.area for x in used], dtype=float)
        volumes = np.array([x.volume for x in used], dtype=float)
        old = dict(r.summary)
        old.update({
            "count": float(len(r.records)),
            "size_qc_count": float(len(qc_records)),
            "size_qc_fraction_percent": float(len(qc_records) / len(r.records) * 100.0) if r.records else 0.0,
            "density": float(len(r.records) / image_area) if image_area > 0 else np.nan,
            "coverage_percent": float(mask.sum() / mask.size * 100.0) if mask.size else 0.0,
            "mean_equivalent_diameter": float(np.nanmean(eqs)) if eqs.size else np.nan,
            "median_equivalent_diameter": float(np.nanmedian(eqs)) if eqs.size else np.nan,
            "sd_equivalent_diameter": float(np.nanstd(eqs, ddof=1)) if eqs.size > 1 else np.nan,
            "mean_p95_height": float(np.nanmean(heights)) if heights.size else np.nan,
            "mean_max_height": float(np.nanmean(max_heights)) if max_heights.size else np.nan,
            "mean_area": float(np.nanmean(areas)) if areas.size else np.nan,
            "total_volume": float(np.nansum(volumes)) if volumes.size else 0.0,
            "summary_excludes_underresolved": self.morph_exclude_underresolved.isChecked(),
        })
        r.summary = old

    def exclude_selected_particles(self):
        if self.morphology_result is None:
            return
        rows = sorted({idx.row() for idx in self.morph_table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(self, "Morphology", "Select one or more particle rows first.")
            return
        ids = set()
        for row in rows:
            item = self.morph_table.item(row, 0)
            if item is not None:
                try:
                    ids.add(int(item.numeric_value if isinstance(item, NumericItem) else item.text()))
                except Exception:
                    pass
        if not ids:
            return
        r = self.morphology_result
        for pid in ids:
            r.labels[r.labels == pid] = 0
        r.records = [rec for rec in r.records if rec.particle_id not in ids]
        self._recalculate_morph_summary_after_mask_edit()
        self.render_morphology()
        self.statusBar().showMessage(f"Excluded {len(ids)} selected particle(s) from the current morphology result.")

    def export_morphology_csv(self):
        if self.morphology_result is None:
            QMessageBox.information(self, "Morphology export", "Run morphology analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export particle table", self._default_stem() + "_particles.csv", "CSV (*.csv)"
        )
        if not path:
            return
        headers, rows = self._morph_table_rows()
        source = self.nid.path.name if self.nid is not None else ""
        channel = self.channel.label if self.channel is not None else ""
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Source file", "Channel"] + headers)
            for row in rows:
                writer.writerow([source, channel] + row)
        self.statusBar().showMessage(f"Exported particle table: {path}")

    def export_morphology_excel(self):
        if self.morphology_result is None or self.channel is None:
            QMessageBox.information(self, "Morphology export", "Run morphology analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export morphology workbook", self._default_stem() + "_morphology.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        wb = Workbook()
        ws = wb.active
        ws.title = "Summary"
        zu, _ = self._scaled_z()
        sx, _, _ = self._scaled_axes()
        r = self.morphology_result
        summary_rows = [
            ("Source file", self.nid.path.name if self.nid else ""),
            ("Channel", self.channel.label),
            ("Threshold method", self.morph_threshold_method.currentText()),
            ("Threshold value", r.threshold),
            ("Threshold unit", zu.label),
            ("Direction", self.morph_direction.currentText()),
            ("Particle count", len(r.records)),
            ("Coverage (%)", r.summary.get("coverage_percent", np.nan)),
            ("X range", self.x_range_m * sx.factor),
            ("Y range", self.y_range_m * sx.factor),
            ("XY unit", sx.label),
            ("Plane correction for segmentation", self.morph_plane.isChecked()),
            ("Smoothing sigma (px)", self.morph_smooth.value()),
            ("Minimum area (px)", self.morph_min_area.value()),
            ("Maximum area (px; 0=no limit)", self.morph_max_area.value()),
            ("Minimum apparent diameter (nm; 0=no limit)", self.morph_min_diameter_nm.value()),
            ("Maximum apparent diameter (nm; 0=no limit)", self.morph_max_diameter_nm.value()),
            ("Minimum pixels across for size QC", self.morph_min_pixels_across.value()),
            ("Local height background", self.morph_background_mode.currentText()),
            ("Background ring (px)", self.morph_background_ring.value()),
            ("Exclude under-resolved from size summary", self.morph_exclude_underresolved.isChecked()),
            ("Pixel pitch X (nm)", float(r.summary.get("pixel_pitch_x", np.nan)) * (1e9 / sx.factor)),
            ("Pixel pitch Y (nm)", float(r.summary.get("pixel_pitch_y", np.nan)) * (1e9 / sx.factor)),
            ("Scale status", getattr(self.channel, "scale_status", "UNVERIFIED")),
            ("Lateral size interpretation", "Apparent AFM lateral size; tip convolution not corrected"),
            ("Watershed", self.morph_watershed.isChecked()),
            ("Watershed spacing (px)", self.morph_watershed_distance.value()),
            ("Exclude border objects", self.morph_clear_border.isChecked()),
        ]
        for key, value in summary_rows:
            ws.append([key, value])
        ws["A1"].font = XLFont(bold=True)
        ws.column_dimensions["A"].width = 36
        ws.column_dimensions["B"].width = 28

        pt = wb.create_sheet("Particles")
        headers, rows = self._morph_table_rows()
        pt.append(["Source file", "Channel"] + headers)
        for cell in pt[1]:
            cell.font = XLFont(bold=True)
            cell.alignment = Alignment(horizontal="center")
            cell.fill = PatternFill("solid", fgColor="D9EAF7")
        source = self.nid.path.name if self.nid is not None else ""
        channel = self.channel.label if self.channel is not None else ""
        for row in rows:
            pt.append([source, channel] + row)
        pt.freeze_panes = "A2"
        pt.auto_filter.ref = pt.dimensions
        for col in pt.columns:
            letter = col[0].column_letter
            pt.column_dimensions[letter].width = min(24, max(10, max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2))

        # Dedicated size-statistics sheet. This makes the population used for
        # reported nanoparticle size explicit instead of hiding it in a figure.
        ss = wb.create_sheet("Size Statistics")
        ss.append(["Source file", "Channel", "Population", "Variable", "Statistic", "Value", "Unit"])
        for cell in ss[1]:
            cell.font = XLFont(bold=True)
            cell.fill = PatternFill("solid", fgColor="D9EAF7")
            cell.alignment = Alignment(horizontal="center")
        size_stats = self._morph_size_statistics()
        stat_order = ["n", "mean", "median", "sd", "cv_percent", "min", "p10", "q25", "q75", "p90", "max"]
        var_specs = [
            ("Apparent equivalent diameter", size_stats["diameter_nm"], "nm"),
            ("Apparent Feret maximum", size_stats["feret_max_nm"], "nm"),
            ("Apparent Feret minimum", size_stats["feret_min_nm"], "nm"),
            ("P95 height", size_stats["p95_height"], size_stats["height_unit"]),
        ]
        for variable, stat_dict, unit in var_specs:
            for stat_name in stat_order:
                value = stat_dict.get(stat_name, np.nan)
                out_unit = "%" if stat_name == "cv_percent" else ("count" if stat_name == "n" else unit)
                ss.append([source, channel, size_stats["population"], variable, stat_name, value, out_unit])
        ss.append([source, channel, size_stats["population"], "Size QC", "detected particles", len(r.records), "count"])
        ss.append([source, channel, size_stats["population"], "Size QC", "QC-valid particles", int(float(r.summary.get("size_qc_count", 0))), "count"])
        ss.append([source, channel, size_stats["population"], "Size QC", "QC-valid fraction", float(r.summary.get("size_qc_fraction_percent", np.nan)), "%"])
        ss.append([source, channel, size_stats["population"], "Calibration", "pixel pitch X", float(r.summary.get("pixel_pitch_x", np.nan)) * (1e9 / sx.factor), "nm/pixel"])
        ss.append([source, channel, size_stats["population"], "Calibration", "pixel pitch Y", float(r.summary.get("pixel_pitch_y", np.nan)) * (1e9 / sx.factor), "nm/pixel"])
        ss.append([source, channel, size_stats["population"], "Calibration", "minimum lateral QC size", float(r.summary.get("minimum_qc_lateral_size", np.nan)) * (1e9 / sx.factor), "nm"])
        ss.freeze_panes = "A2"
        ss.auto_filter.ref = ss.dimensions
        for col in ss.columns:
            letter = col[0].column_letter
            ss.column_dimensions[letter].width = min(32, max(12, max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2))

        # Tidy histogram table: one row per bin so it is immediately usable in
        # Origin, Excel, R, Prism or Python without reverse-engineering the GUI.
        ds = wb.create_sheet("Distributions")
        ds.append(["Source file", "Channel", "Variable", "Unit", "Population", "Bin", "Lower", "Upper", "Center", "Count", "Fraction"])
        for cell in ds[1]:
            cell.font = XLFont(bold=True)
            cell.fill = PatternFill("solid", fgColor="D9EAF7")
            cell.alignment = Alignment(horizontal="center")
        for variable, unit, population, bin_id, lower, upper, center, count, fraction in self._morph_distribution_export_rows():
            ds.append([source, channel, variable, unit, population, bin_id, lower, upper, center, count, fraction])
        ds.freeze_panes = "A2"
        ds.auto_filter.ref = ds.dimensions
        for col in ds.columns:
            letter = col[0].column_letter
            ds.column_dimensions[letter].width = min(30, max(11, max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2))

        wb.save(path)
        self.statusBar().showMessage(f"Exported morphology workbook with particle, size-statistics and distribution data: {path}")

    def export_morphology_figure(self):
        if self.morphology_result is None or self.current is None:
            QMessageBox.information(self, "Morphology figure", "Run morphology analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export morphology figure", self._default_stem() + "_morphology_600dpi.png",
            "PNG (*.png);;TIFF (*.tif *.tiff);;PDF (*.pdf)"
        )
        if not path:
            return
        try:
            r = self.morphology_result
            sx, xr, yr = self._scaled_axes()
            zu, _ = self._scaled_z()
            _, _, xy_to_nm, z_to_nm, height_unit, _ = self._morph_unit_scales()
            records = self._morph_distribution_records()
            population = self.morph_distribution_population.currentText()

            fig = Figure(figsize=(11.5, 8.0), dpi=120, facecolor="white")
            gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.0], hspace=0.34, wspace=0.28)
            axm = fig.add_subplot(gs[0, :])
            im = axm.imshow(
                r.measurement_data, origin="lower", extent=[0, xr, 0, yr],
                cmap=self._cmap(), aspect="equal"
            )
            labels = np.ma.masked_where(r.labels <= 0, r.labels)
            if np.any(r.labels > 0):
                axm.imshow(labels, origin="lower", extent=[0, xr, 0, yr], cmap="tab20", alpha=0.30, interpolation="nearest", aspect="equal")
                try:
                    axm.contour(r.mask.astype(float), levels=[0.5], origin="lower", extent=[0, xr, 0, yr], colors="white", linewidths=0.55, alpha=0.95)
                except Exception:
                    pass
            if len(r.records) <= 100:
                for rec in r.records:
                    axm.text(rec.centroid_x, rec.centroid_y, str(rec.particle_id), fontsize=5.5, ha="center", va="center", color="white")
            axm.set_xlabel(f"X ({sx.label})")
            axm.set_ylabel(f"Y ({sx.label})")
            axm.set_title(f"Morphology mask · {self.morph_threshold_method.currentText()} threshold")
            cb = fig.colorbar(im, ax=axm, shrink=0.90, pad=0.02)
            cb.set_label(zu.label)

            axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
            diam = np.asarray([rec.equivalent_diameter * xy_to_nm for rec in records], dtype=float)
            height = np.asarray([rec.p95_height * z_to_nm for rec in records], dtype=float)
            circ = np.asarray([rec.circularity for rec in records], dtype=float)
            specs = [
                (axes[0], diam, "Apparent equivalent diameter", "nm"),
                (axes[1], height, "P95 height", height_unit),
                (axes[2], circ, "Circularity", ""),
            ]
            for ax, vals, title, xlabel in specs:
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    _, edges = self._morph_histogram(vals)
                    ax.hist(vals, bins=edges, edgecolor="black", linewidth=0.35)
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)

            qc_limit_nm = float(r.summary.get("minimum_qc_lateral_size", np.nan)) * xy_to_nm
            if np.isfinite(qc_limit_nm):
                axes[0].axvline(qc_limit_nm, linestyle="--", linewidth=1.0, color="black")
                axes[0].text(qc_limit_nm, axes[0].get_ylim()[1] * 0.95, "QC limit", rotation=90, va="top", ha="right", fontsize=7)

            source = self.nid.path.name if self.nid is not None else ""
            fig.suptitle(
                f"{source} · {self.channel.label if self.channel else ''} · distribution: {population} · n={len(records)}",
                fontsize=11, y=0.995
            )
            fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
            self.statusBar().showMessage(f"Exported morphology figure at 600 DPI: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Morphology figure export failed", str(exc))

    def _smart_window_changed(self, *_args):
        if self.current is None:
            return
        self._invalidate_smart_qc()
        if hasattr(self, "smart_tab_index"):
            self._dirty_tabs.add(self.smart_tab_index)

    def _same_frame_channel_correlations(self):
        if self.nid is None or self.channel is None or self.original is None:
            return {}
        base, _ = fit_plane(np.asarray(self.original, dtype=float))
        a = base.ravel()
        out = {}
        for ch in self.nid.channels:
            if ch is self.channel or ch.frame != self.channel.frame or ch.data.shape != self.original.shape:
                continue
            try:
                other, _ = fit_plane(np.asarray(ch.data, dtype=float))
                b = other.ravel()
                if np.std(a) <= 0 or np.std(b) <= 0:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(a, b)[0, 1])
                out[ch.label] = corr
            except Exception:
                continue
        return out

    def _matching_counterpart_channel(self):
        if self.nid is None or self.channel is None:
            return None
        frame = self.channel.frame.lower()
        if "forward" not in frame and "backward" not in frame:
            return None
        want = "backward" if "forward" in frame else "forward"
        for ch in self.nid.channels:
            if ch is self.channel:
                continue
            if want in ch.frame.lower() and ch.name == self.channel.name and ch.unit == self.channel.unit and ch.data.shape == self.channel.data.shape:
                return ch
        return None

    def run_smart_qc(self, *_args, switch_tab=True):
        if self.current is None or self.channel is None:
            return
        try:
            counterpart = self._matching_counterpart_channel()
            self.smart_qc_result = analyze_smart_qc(
                self.current,
                x_range_m=self.x_range_m,
                y_range_m=self.y_range_m,
                window_px=self.smart_window_px.value(),
                counterpart=counterpart.data if counterpart is not None else None,
                counterpart_label=counterpart.label if counterpart is not None else "",
                fb_reference=self.original if self.original is not None else None,
            )
            r = self.smart_qc_result
            self.smart_channel_correlations = self._same_frame_channel_correlations()
            if self.smart_channel_correlations:
                r.report_lines += ["", "Same-frame channel correlations (raw full scan; independently plane-leveled):"]
                for label, corr in self.smart_channel_correlations.items():
                    r.report_lines.append(f"• {label}: Pearson r={corr:.4f}" if np.isfinite(corr) else f"• {label}: unavailable")
                r.report_lines.append("Correlation indicates spatial co-variation only; it does not establish a physical mechanism.")
            self.smart_cards["Overall QC"].value.setText(f"{r.overall} · {r.score:.0f}/100")
            self.smart_cards["Tilt"].value.setText(r.assessments.get("Tilt", "—"))
            self.smart_cards["Lines"].value.setText(r.assessments.get("Line artifacts", "—"))
            spike_count = int(round(r.metrics.get("spike_count", 0.0)))
            self.smart_cards["Spikes"].value.setText(f"{spike_count} · {r.assessments.get('Spikes', '—')}")
            if r.fb_comparison is not None:
                corr = float(r.fb_comparison.get("correlation", np.nan))
                self.smart_cards["F/B match"].value.setText(f"r={corr:.3f}")
            else:
                self.smart_cards["F/B match"].value.setText("Not available")
            score_text = " · ".join(f"{k}: {v:.3f}" for k, v in sorted(r.candidate_scores.items(), key=lambda kv: kv[1]))
            self.smart_recommend_label.setText(
                f"Recommended: {r.recommended_correction}. The advisor favors the least aggressive correction when candidate scores are close."
            )
            self.smart_recommend_label.setToolTip(score_text)
            self.smart_report.setPlainText("\n".join(r.report_lines))
            self.render_smart_qc()
            self._dirty_tabs.discard(self.smart_tab_index)
            if switch_tab:
                self.tabs.setCurrentIndex(self.smart_tab_index)
            self.statusBar().showMessage(
                f"Smart QC: {r.overall} ({r.score:.0f}/100) · recommended {r.recommended_correction}."
            )
        except Exception as exc:
            self.smart_qc_result = None
            QMessageBox.critical(self, "Smart QC failed", str(exc))

    def render_smart_qc(self, *_args):
        if self.smart_qc_result is None or self.current is None or self.channel is None:
            return
        r = self.smart_qc_result
        view = self.smart_view_combo.currentText()
        self.smart_canvas.figure.clear()
        sx, xr, yr = self._scaled_axes()
        zu, _z = self._scaled_z()

        if view == "Artifact diagnostics":
            ax1 = self.smart_canvas.figure.add_subplot(121)
            shown = r.leveled * zu.factor
            im = ax1.imshow(shown, origin="lower", extent=[0, xr, 0, yr], cmap=self._cmap(), aspect="equal")
            if r.bad_rows.size:
                for rr in r.bad_rows:
                    y = (rr + 0.5) / max(shown.shape[0], 1) * yr
                    ax1.axhline(y, linewidth=0.8, alpha=0.85)
            ys, xs = np.where(r.spike_mask)
            if xs.size:
                ax1.scatter((xs + 0.5) / max(shown.shape[1], 1) * xr,
                            (ys + 0.5) / max(shown.shape[0], 1) * yr,
                            s=12, marker="x", linewidths=0.8)
            ax1.set_title("Leveled diagnostic map\nlines + isolated spike candidates")
            ax1.set_xlabel(f"X ({sx.label})")
            ax1.set_ylabel(f"Y ({sx.label})")
            self._style_axis(ax1)
            cb = self.smart_canvas.figure.colorbar(im, ax=ax1, shrink=0.82, pad=0.025)
            cb.set_label(zu.label, color="#c5d7ee")
            cb.ax.tick_params(colors="#a8beda")

            ax2 = self.smart_canvas.figure.add_subplot(122)
            row_med = np.nanmedian(r.leveled * zu.factor, axis=1)
            ycoord = (np.arange(row_med.size, dtype=float) + 0.5) / max(row_med.size, 1) * yr
            ax2.plot(row_med, ycoord, linewidth=1.2)
            if r.bad_rows.size:
                ax2.scatter(row_med[r.bad_rows], ycoord[r.bad_rows], s=20)
            ax2.set_title("Slow-scan row median diagnostic")
            ax2.set_xlabel(f"Row median ({zu.label})")
            ax2.set_ylabel(f"Y ({sx.label})")
            self._style_axis(ax2)

        elif view in {"Local Sa map", "Local Sq map"}:
            ax = self.smart_canvas.figure.add_subplot(111)
            arr = r.local_sa if view.startswith("Local Sa") else r.local_sq
            arr = arr * zu.factor
            im = ax.imshow(arr, origin="lower", extent=[0, xr, 0, yr], cmap=self._cmap(), aspect="equal")
            ax.set_title(f"{view} · window {self.smart_window_px.value()} px")
            ax.set_xlabel(f"X ({sx.label})")
            ax.set_ylabel(f"Y ({sx.label})")
            self._style_axis(ax)
            cb = self.smart_canvas.figure.colorbar(im, ax=ax, shrink=0.86, pad=0.025)
            cb.set_label(zu.label, color="#c5d7ee")
            cb.ax.tick_params(colors="#a8beda")

        elif view == "Scale-dependent roughness":
            ax = self.smart_canvas.figure.add_subplot(111)
            rows, cols = self.current.shape
            dx = abs(self.x_range_m) / max(cols, 1)
            dy = abs(self.y_range_m) / max(rows, 1)
            pitch = np.sqrt(max(dx * dy, 0.0))
            spatial = r.scale_sizes_px * pitch * sx.factor
            ax.plot(spatial, r.scale_sa * zu.factor, marker="o", linewidth=1.6, label="Sa")
            ax.plot(spatial, r.scale_sq * zu.factor, marker="s", linewidth=1.6, label="Sq")
            ax.set_xlabel(f"Window size ({sx.label})")
            ax.set_ylabel(f"Roughness ({zu.label})")
            ax.set_title("Scale-dependent roughness\nmean of independently plane-leveled non-overlapping windows")
            ax.legend()
            self._style_axis(ax)

        elif view == "Autocorrelation":
            ax = self.smart_canvas.figure.add_subplot(111)
            im = ax.imshow(r.autocorrelation, origin="lower", extent=[-xr/2, xr/2, -yr/2, yr/2],
                           cmap="viridis", aspect="equal", vmin=-1.0, vmax=1.0)
            corr = r.correlation_length_m * sx.factor
            suffix = f" · 1/e length ≈ {corr:.4g} {sx.label}" if np.isfinite(corr) else ""
            ax.set_title("Normalized 2D autocorrelation" + suffix)
            ax.set_xlabel(f"ΔX ({sx.label})")
            ax.set_ylabel(f"ΔY ({sx.label})")
            self._style_axis(ax)
            cb = self.smart_canvas.figure.colorbar(im, ax=ax, shrink=0.86, pad=0.025)
            cb.set_label("Normalized correlation", color="#c5d7ee")
            cb.ax.tick_params(colors="#a8beda")

        elif view == "Forward/backward difference":
            ax = self.smart_canvas.figure.add_subplot(111)
            fb = r.fb_comparison
            if fb is None:
                ax.text(0.5, 0.5, "Matching forward/backward channel not available", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                diff = np.asarray(fb["difference"], dtype=float) * zu.factor
                im = ax.imshow(diff, origin="lower", extent=[0, self.channel.x_range_m * sx.factor, 0, self.channel.y_range_m * sx.factor],
                               cmap="coolwarm", aspect="equal")
                ax.set_title(f"Forward/backward registered difference · r={float(fb['correlation']):.4f}")
                ax.set_xlabel(f"X ({sx.label})")
                ax.set_ylabel(f"Y ({sx.label})")
                self._style_axis(ax)
                cb = self.smart_canvas.figure.colorbar(im, ax=ax, shrink=0.86, pad=0.025)
                cb.set_label(zu.label, color="#c5d7ee")
                cb.ax.tick_params(colors="#a8beda")

        else:  # Leveling candidates
            ax = self.smart_canvas.figure.add_subplot(111)
            ordered = sorted(r.candidate_scores.items(), key=lambda kv: kv[1])
            names = [k for k, _v in ordered]
            vals = [v for _k, v in ordered]
            y = np.arange(len(names))
            ax.barh(y, vals)
            ax.set_yticks(y, names)
            ax.invert_yaxis()
            ax.set_xlabel("Diagnostic score · lower is better")
            ax.set_title("Leveling advisor candidates\ncomplexity penalty favors conservative processing")
            self._style_axis(ax)
            for i, v in enumerate(vals):
                ax.text(v, i, f"  {v:.3f}", va="center", color="#c5d7ee", fontsize=8.5)

        self.smart_canvas.figure.tight_layout()
        self.smart_canvas.draw_idle()

    def apply_recommended_correction(self):
        if self.current is None:
            return
        if self.smart_qc_result is None:
            self.run_smart_qc(switch_tab=False)
        if self.smart_qc_result is None:
            return
        recommendation = self.smart_qc_result.recommended_correction
        if recommendation == "No correction":
            QMessageBox.information(self, "Smart leveling advisor", "The advisor does not recommend a correction for the current analysis state.")
            return
        proposed = self.smart_qc_result.candidate_data.get(recommendation)
        if proposed is None:
            return
        answer = QMessageBox.question(
            self, "Apply recommended correction?",
            f"Recommended processing:\n\n{recommendation}\n\n"
            "This creates a new non-destructive history state. The raw channel remains restorable.\n\nApply it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.current = np.asarray(proposed, dtype=float).copy()
        self._invalidate_morphology()
        self._commit_history(f"Smart advisor · {recommendation}")
        self.refresh_all()
        self.statusBar().showMessage(f"Applied Smart QC recommendation: {recommendation}")

    def recommend_smart_morphology(self):
        if self.current is None or self.channel is None:
            return
        try:
            sx, xr, yr = self._scaled_axes()
            zu, z = self._scaled_z()
            self.statusBar().showMessage("Evaluating transparent morphology candidates…")
            QApplication.processEvents()
            rec = smart_morphology_recommendation(
                z, x_range=xr, y_range=yr, min_area_px=self.morph_min_area.value()
            )
            self.smart_morph_recommendation = rec
            widgets = [self.morph_direction, self.morph_threshold_method, self.morph_k_sigma,
                       self.morph_smooth, self.morph_watershed_distance]
            for w in widgets:
                w.blockSignals(True)
            self.morph_direction.setCurrentText(str(rec["direction"]))
            self.morph_threshold_method.setCurrentText(str(rec["method"]))
            self.morph_k_sigma.setValue(float(rec["k_sigma"]))
            self.morph_smooth.setValue(float(rec["smoothing_sigma"]))
            self.morph_watershed_distance.setValue(int(rec["watershed_distance"]))
            for w in widgets:
                w.blockSignals(False)
            self.tabs.setCurrentIndex(2)
            self.run_morphology_analysis()
            QMessageBox.information(
                self, "Morphology recommendation",
                f"Recommended starting settings:\n\n"
                f"Threshold: {rec['method']}\nDirection: {rec['direction']}\n"
                f"Smoothing σ: {rec['smoothing_sigma']} px\nWatershed spacing: {rec['watershed_distance']} px\n"
                f"Objects in preview: {rec['count']}\nCoverage: {rec['coverage_percent']:.3g}%\n"
                f"Confidence: {rec['confidence']}\n\n"
                f"The program tested {rec['tested_candidates']} deterministic candidates. "
                "This is a starting point only; verify the overlay before reporting particle statistics."
            )
            self.statusBar().showMessage("Morphology recommendation applied to the visible controls for review.")
        except Exception as exc:
            QMessageBox.critical(self, "Morphology recommendation failed", str(exc))

    def auto_profile_strongest_feature(self):
        if self.current is None:
            return
        if self.smart_qc_result is None:
            self.run_smart_qc(switch_tab=False)
        if self.smart_qc_result is None:
            return
        f = self.smart_qc_result.features
        rows, cols = self.current.shape
        sx, xr, yr = self._scaled_axes()
        def point(px, py):
            x = (float(px) + 0.5) / cols * xr
            y = (float(py) + 0.5) / rows * yr
            return (float(px), float(py), x, y)
        self.profile_points = [
            point(f["profile_x0_px"], f["profile_y0_px"]),
            point(f["profile_x1_px"], f["profile_y1_px"]),
        ]
        self.show_profile()
        self.statusBar().showMessage("Created a profile across the strongest local-relief feature detected by Smart QC.")

    def export_smart_qc_json(self):
        if self.current is None:
            return
        if self.smart_qc_result is None:
            self.run_smart_qc(switch_tab=False)
        if self.smart_qc_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Smart QC report", self._default_stem() + "_smart_qc.json", "JSON (*.json)")
        if not path:
            return
        r = self.smart_qc_result
        fb = None
        if r.fb_comparison is not None:
            fb = {k: v for k, v in r.fb_comparison.items() if k not in {"aligned", "difference"}}
        payload = {
            "app": APP_NAME,
            "version": APP_VERSION,
            "source": str(self.nid.path) if self.nid else "",
            "channel": self.channel.label if self.channel else "",
            "overall": r.overall,
            "score": r.score,
            "assessments": r.assessments,
            "metrics": r.metrics,
            "recommended_correction": r.recommended_correction,
            "candidate_scores": r.candidate_scores,
            "correlation_length_m": r.correlation_length_m,
            "texture": r.texture,
            "features": r.features,
            "forward_backward": fb,
            "same_frame_channel_correlations": self.smart_channel_correlations,
            "scale_dependent_roughness": {
                "window_size_px": r.scale_sizes_px.tolist(),
                "Sa_signal_unit": r.scale_sa.tolist(),
                "Sq_signal_unit": r.scale_sq.tolist(),
            },
            "report": r.report_lines,
            "algorithm_version": ALGORITHM_VERSION,
            "acf_method": "Linear overlap-normalized; no periodic wrap",
            "local_sa_method": "Exact absolute deviation from each window mean",
            "note": "Smart QC values are diagnostic heuristics and require visual verification before data exclusion or correction.",
        }
        write_report(path, payload)
        self.statusBar().showMessage(f"Exported Smart QC report: {path}")

    @staticmethod
    def _batch_task_catalog():
        return [
            ("plane_level", "Plane level · 1st order"),
            ("robust_plane", "Robust plane · Huber"),
            ("row_offsets", "Remove row offsets"),
            ("line_flatten", "Line flatten"),
            ("gaussian", "Gaussian smoothing"),
            ("median", "Median filter"),
        ]

    def _batch_populate_task_list(self, order=None, checked_codes=None):
        if not hasattr(self, "batch_task_list"):
            return
        catalog = dict(self._batch_task_catalog())
        default_order = [code for code, _label in self._batch_task_catalog()]
        order = list(order or default_order)
        order.extend(code for code in default_order if code not in order)
        checked = set(checked_codes if checked_codes is not None else {"plane_level"})
        self.batch_task_list.blockSignals(True)
        self.batch_task_list.clear()
        for code in order:
            if code not in catalog:
                continue
            item = QListWidgetItem(catalog[code])
            item.setData(Qt.ItemDataRole.UserRole, code)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled)
            item.setCheckState(Qt.CheckState.Checked if code in checked else Qt.CheckState.Unchecked)
            self.batch_task_list.addItem(item)
        self.batch_task_list.blockSignals(False)

    @staticmethod
    def _batch_task_label(task):
        code = task.get("task", "")
        if code == "robust_plane":
            return "Robust plane level · Huber"
        if code == "plane_level":
            return "Plane level · 1st order"
        if code == "row_offsets":
            return "Remove row offsets"
        if code == "line_flatten":
            return f"Line flatten · degree {int(task.get('degree', 1))}"
        if code == "gaussian":
            return f"Gaussian · σ {float(task.get('sigma', 1.0)):g} px"
        if code == "median":
            return f"Median filter · {int(task.get('size', 3))} px"
        return str(code or "Unknown")

    def _batch_task_recipe(self):
        tasks = []
        if not hasattr(self, "batch_task_list"):
            return tasks
        for row in range(self.batch_task_list.count()):
            item = self.batch_task_list.item(row)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            code = item.data(Qt.ItemDataRole.UserRole)
            task = {"task": code}
            if code == "line_flatten":
                task["degree"] = self.batch_line_degree.value()
            elif code == "gaussian":
                task["sigma"] = self.batch_gaussian_sigma.value()
            elif code == "median":
                size = self.batch_median_size.value()
                if size % 2 == 0:
                    size += 1
                    self.batch_median_size.setValue(size)
                task["size"] = size
            tasks.append(task)
        return tasks

    def _batch_update_task_summary(self):
        if not hasattr(self, "batch_task_summary"):
            return
        tasks = self._batch_task_recipe()
        if not tasks:
            self.batch_task_summary.setText("0/5 preprocessing tasks selected · raw topography will be segmented/analyzed.")
            return
        text = " → ".join(self._batch_task_label(t) for t in tasks)
        self.batch_task_summary.setText(f"{len(tasks)}/5 task(s), top → bottom: {text}")

    def batch_clear_task_engine(self):
        if not hasattr(self, "batch_task_list"):
            return
        self.batch_task_list.blockSignals(True)
        for row in range(self.batch_task_list.count()):
            self.batch_task_list.item(row).setCheckState(Qt.CheckState.Unchecked)
        self.batch_task_list.blockSignals(False)
        self._batch_update_task_summary()

    def batch_apply_tasks_to_active_scan(self):
        if self.current is None:
            QMessageBox.information(self, "Batch engine preview", "Open a scan first, then test the checked preprocessing tasks on it.")
            return
        tasks = self._batch_task_recipe()
        if not tasks:
            QMessageBox.information(self, "Batch engine preview", "No preprocessing tasks are checked. The batch engine would analyze raw topography.")
            return
        for task in tasks:
            code = task.get("task")
            if code == "plane_level":
                self.current, _ = fit_plane(self.current)
            elif code == "robust_plane":
                self.current, _ = fit_plane(self.current, robust=True)
            elif code == "row_offsets":
                self.current = remove_row_offsets(self.current)
            elif code == "line_flatten":
                self.current = line_flatten(self.current, int(task.get("degree", 1)))
            elif code == "gaussian":
                self.current = gaussian_smooth(self.current, float(task.get("sigma", 1.0)))
            elif code == "median":
                self.current = median_filter(self.current, int(task.get("size", 3)))
            else:
                continue
            self._invalidate_morphology()
            self._commit_history(self._batch_task_label(task))
        self.refresh_all()
        self.statusBar().showMessage(f"Batch preprocessing preview applied to active scan · {len(tasks)} task(s).")

    def batch_capture_active_recipe(self):
        if self.current is None or not self.history or self.history_index <= 0:
            QMessageBox.information(
                self, "Capture preprocessing recipe",
                "Preprocess one scan in the main viewer first, then use this button to copy those recognized steps into the batch engine."
            )
            return

        descriptions = [
            str(state.get("description", ""))
            for state in self.history[1:self.history_index + 1]
        ]
        order = []
        checked = set()
        ignored = []
        repeated = set()
        for desc in descriptions:
            low = desc.lower()
            code = None
            if "robust plane" in low:
                code = "robust_plane"
            elif "plane level" in low:
                code = "plane_level"
            elif "row offsets" in low:
                code = "row_offsets"
            elif "line flatten" in low:
                code = "line_flatten"
                m = re.search(r"degree\s+(\d+)", low)
                if m:
                    self.batch_line_degree.setValue(max(0, min(5, int(m.group(1)))))
            elif "gaussian" in low:
                code = "gaussian"
                m = re.search(r"[σs]\s*=?\s*([0-9]+(?:\.[0-9]+)?)", low)
                if m:
                    self.batch_gaussian_sigma.setValue(float(m.group(1)))
            elif "median filter" in low:
                code = "median"
                m = re.search(r"median filter\s*·?\s*(\d+)", low)
                if m:
                    size = int(m.group(1))
                    if size % 2 == 0:
                        size += 1
                    self.batch_median_size.setValue(max(1, min(31, size)))
            elif "raw channel" not in low:
                ignored.append(desc)

            if code:
                checked.add(code)
                if code not in order:
                    order.append(code)
                else:
                    repeated.add(code)

        if not checked:
            QMessageBox.information(
                self, "Capture preprocessing recipe",
                "No supported batch preprocessing steps were found in the active scan history. "
                "Supported steps are plane level, row-offset correction, line flatten, Gaussian smoothing, and median filtering."
            )
            return

        self._batch_populate_task_list(order=order, checked_codes=checked)
        self._batch_update_task_summary()
        notes = []
        if ignored:
            notes.append("Ignored non-batch/history operations: " + "; ".join(ignored))
        if repeated:
            notes.append("Repeated task types were collapsed to one batch step using the most recent parameter value.")
        msg = f"Captured {len(checked)} supported preprocessing task type(s) from the active scan."
        if notes:
            msg += "\n\n" + "\n".join(notes)
        QMessageBox.information(self, "Batch recipe captured", msg)

    def _batch_capture_settings(self) -> BatchSettings:
        method = self.morph_threshold_method.currentText()
        morphology = {
            "threshold_method": method,
            "manual_threshold": self.morph_manual_threshold.value() if method == "Manual" else None,
            "k_sigma": self.morph_k_sigma.value(),
            "direction": self.morph_direction.currentText(),
            "smoothing_sigma": self.morph_smooth.value(),
            "plane_correct": self.morph_plane.isChecked(),
            "fill_holes": self.morph_fill_holes.isChecked(),
            "opening_radius": self.morph_opening.value(),
            "closing_radius": self.morph_closing.value(),
            "min_area_px": self.morph_min_area.value(),
            "max_area_px": self.morph_max_area.value(),
            "clear_border_objects": self.morph_clear_border.isChecked(),
            "watershed_split": self.morph_watershed.isChecked(),
            "watershed_min_distance": self.morph_watershed_distance.value(),
            "min_equivalent_diameter_nm": self.morph_min_diameter_nm.value(),
            "max_equivalent_diameter_nm": self.morph_max_diameter_nm.value(),
            "min_pixels_across": self.morph_min_pixels_across.value(),
            "background_mode": self.morph_background_mode.currentText(),
            "background_ring_px": self.morph_background_ring.value(),
            "exclude_underresolved_from_summary": self.morph_exclude_underresolved.isChecked(),
        }
        current_name = self.channel.name if self.channel is not None else ""
        current_frame = self.channel.frame if self.channel is not None else ""
        return BatchSettings(
            channel_mode=self.batch_channel_mode.currentText(),
            match_channel_name=current_name,
            match_channel_frame=current_frame,
            preprocessing_tasks=self._batch_task_recipe(),
            smart_morphology_per_file=self.batch_smart_morph.isChecked(),
            morphology=morphology,
        )

    @staticmethod
    def _batch_fmt(value, digits=4):
        try:
            value = float(value)
            if not np.isfinite(value):
                return "—"
            return f"{value:.{digits}g}"
        except Exception:
            return "—" if value is None else str(value)

    def _batch_rebuild_table(self):
        if not hasattr(self, "batch_table"):
            return
        result_map = {str(Path(r.source_path)): r for r in self.batch_results}
        self.batch_table.setSortingEnabled(False)
        self.batch_table.setRowCount(len(self.batch_files))
        for row, path in enumerate(self.batch_files):
            norm = str(Path(path))
            result = result_map.get(norm)
            values = [Path(path).name, "Queued", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", ""]
            if result is not None:
                values[1] = result.status
                if result.status == "Done":
                    sc = result.scan
                    mo = result.morphology
                    ro = result.roughness
                    values[2] = f"{self._batch_fmt(sc.get('X size (µm)'))} × {self._batch_fmt(sc.get('Y size (µm)'))}"
                    values[3] = f"{sc.get('Pixels X', '—')} × {sc.get('Pixels Y', '—')}"
                    values[4] = str(sc.get("Preprocessing steps", 0))
                    values[5] = str(mo.get("Particle count", "—"))
                    values[6] = str(mo.get("Size-QC particle count", "—"))
                    values[7] = self._batch_fmt(mo.get("Mean apparent equivalent diameter (nm)"))
                    # Topography channels report nm; keep a tolerant fallback for non-length signals.
                    hkey = next((k for k in mo if k.startswith("Mean P95 height (")), None)
                    if hkey is None:
                        hkey = next((k for k in mo if k.startswith("Mean max height (")), None)
                    values[8] = self._batch_fmt(mo.get(hkey)) if hkey else "—"
                    values[9] = self._batch_fmt(mo.get("Coverage (%)"))
                    sakey = next((k for k in ro if k.startswith("Sa (")), None)
                    sqkey = next((k for k in ro if k.startswith("Sq (")), None)
                    values[10] = self._batch_fmt(ro.get(sakey)) if sakey else "—"
                    values[11] = self._batch_fmt(ro.get(sqkey)) if sqkey else "—"
                    values[12] = result.scan.get("Channel", "")
                else:
                    values[12] = result.error
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 0:
                    item.setToolTip(str(path))
                    item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.batch_table.setItem(row, col, item)
        self.batch_table.setSortingEnabled(True)

    def batch_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add AFM NID files", "", "Nanosurf NID (*.nid);;All files (*.*)"
        )
        if not paths:
            return
        self._add_batch_paths(paths)

    def _add_batch_paths(self, paths):
        if self._batch_is_running():
            self.statusBar().showMessage("Wait for the current batch or cancel it before editing the queue.")
            return
        known = {str(Path(p).resolve()) for p in self.batch_files}
        for path in paths:
            resolved = str(Path(path).resolve())
            if resolved not in known:
                self.batch_files.append(resolved)
                known.add(resolved)
        self._batch_rebuild_table()
        self.batch_status.setText(f"{len(self.batch_files)} file(s) queued. Batch results use one captured morphology protocol per run.")

    def batch_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add folder containing NID files", "")
        if not folder:
            return
        paths = sorted(Path(folder).rglob("*.nid"))
        if not paths:
            QMessageBox.information(self, "Batch analysis", "No .nid files were found in that folder or its subfolders.")
            return
        known = {str(Path(p).resolve()) for p in self.batch_files}
        for path in paths:
            resolved = str(path.resolve())
            if resolved not in known:
                self.batch_files.append(resolved)
                known.add(resolved)
        self._batch_rebuild_table()
        self.batch_status.setText(f"{len(self.batch_files)} file(s) queued from the selected folder.")

    def batch_remove_selected(self):
        if self._batch_is_running():
            self.statusBar().showMessage("Wait for the current batch or cancel it before editing the queue.")
            return
        selected_rows = sorted({idx.row() for idx in self.batch_table.selectionModel().selectedRows()})
        if not selected_rows:
            return
        removed_paths = set()
        for row in selected_rows:
            item = self.batch_table.item(row, 0)
            if item is not None:
                path = item.data(Qt.ItemDataRole.UserRole) or item.toolTip()
                if path:
                    removed_paths.add(str(Path(path)))
        self.batch_files = [p for p in self.batch_files if str(Path(p)) not in removed_paths]
        self.batch_results = [r for r in self.batch_results if str(Path(r.source_path)) not in removed_paths]
        self._batch_rebuild_table()
        self.batch_status.setText(f"{len(self.batch_files)} file(s) remain in the batch.")

    def batch_clear(self):
        if self._batch_is_running():
            self.statusBar().showMessage("Wait for the current batch or cancel it before editing the queue.")
            return
        self.batch_files = []
        self.batch_results = []
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self._batch_rebuild_table()
        self.batch_status.setText("Batch cleared.")

    def _batch_is_running(self):
        return self.batch_worker is not None and self.batch_worker.isRunning()

    def _set_batch_busy(self, busy):
        for widget in (self.btn_batch_add, self.btn_batch_folder, self.btn_batch_remove,
                       self.btn_batch_clear, self.btn_batch_run, self.batch_control_panel):
            widget.setEnabled(not busy)
        self.btn_batch_export.setEnabled(not busy and bool(self.batch_results))
        self.btn_batch_cancel.setEnabled(busy)

    def run_batch_analysis(self):
        if self._batch_is_running():
            return
        if not self.batch_files:
            QMessageBox.information(self, "Batch analysis", "Add one or more .nid files first.")
            return
        settings = self._batch_capture_settings()
        self.batch_run_settings = settings
        m = settings.morphology
        task_labels = [self._batch_task_label(t) for t in settings.preprocessing_tasks]
        self.batch_protocol_summary.setText(
            f"Captured protocol: {len(task_labels)} preprocessing task(s) · "
            f"{m.get('threshold_method')} · {m.get('direction')} · "
            f"size QC ≥{m.get('min_pixels_across')} px · "
            f"smart per-file {'ON' if settings.smart_morphology_per_file else 'OFF'}"
        )
        self.batch_results = []
        self._batch_rebuild_table()
        self.batch_progress.setRange(0, len(self.batch_files))
        self.batch_progress.setValue(0)
        self._set_batch_busy(True)
        worker = BatchWorker(self.batch_files, settings, self)
        self.batch_worker = worker
        worker.file_started.connect(self._batch_file_started)
        worker.file_finished.connect(self._batch_file_finished)
        worker.finished.connect(self._batch_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _batch_file_started(self, index, path):
        self.batch_status.setText(f"Analyzing {index}/{len(self.batch_files)} · {Path(path).name}")

    def _batch_file_finished(self, index, result):
        self.batch_results.append(result)
        self.batch_progress.setValue(index)
        self._batch_rebuild_table()

    def _batch_finished(self):
        cancelled = bool(self.batch_worker and self.batch_worker.cancelled)
        self.batch_worker = None
        self._set_batch_busy(False)
        done = sum(r.status == "Done" for r in self.batch_results)
        failed = len(self.batch_results) - done
        state = "Batch cancelled" if cancelled else "Batch complete"
        self.batch_status.setText(
            f"{state}: {done} successful, {failed} failed, "
            f"{len(self.batch_files) - len(self.batch_results)} pending. "
            "Export uses the protocol captured when this run started."
        )
        self.statusBar().showMessage(f"{state} · {done} successful · {failed} failed")
        if self._close_after_batch:
            QTimer.singleShot(0, self.close)

    def cancel_batch_analysis(self):
        if self._batch_is_running():
            self.batch_worker.requestInterruption()
            self.btn_batch_cancel.setEnabled(False)
            self.batch_status.setText("Cancellation requested · completing the current scan…")

    def export_batch_excel(self):
        if self._batch_is_running():
            return
        if not self.batch_results:
            QMessageBox.information(self, "Batch Excel export", "Run batch analysis first.")
            return
        default_dir = Path(self.batch_files[0]).parent if self.batch_files else Path.cwd()
        default = str(default_dir / "Afruz_AFM_Batch_Results.xlsx")
        path, _ = QFileDialog.getSaveFileName(self, "Export batch AFM workbook", default, "Excel (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            settings = self._batch_capture_settings()
            export_batch_workbook(self.batch_results, self.batch_run_settings or settings, path)
            self.statusBar().showMessage(f"Exported batch workbook: {path}")
            QMessageBox.information(
                self, "Batch export complete",
                "The workbook contains clean tabular sheets for Scan Summary, Roughness, Morphology Summary, "
                "Particle Data, Size Metrology, Segmentation, Processing Pipeline, and Analysis Settings. "
                "Source file names are included in every analytical table."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Batch Excel export failed", str(exc))

    def _set_gl_info(self, text: str):
        self.gpu_3d_label.setText(f"GPU 3D: {text}")

    def _camera_mode_changed(self, *_args):
        if gl is None:
            return
        try:
            self.surface_view.opts["fov"] = 1.5 if self.surface_camera_mode.currentText().startswith("Orthographic") else 45.0
            self.reset_3d_view()
            self.surface_view.update()
        except Exception:
            pass

    @staticmethod
    def _resample_visual_surface(data: np.ndarray, detail_text: str):
        """Return a visualization mesh without changing the quantitative AFM data.

        Native mode returns the measured grid exactly. Upsampled modes use cubic
        interpolation only for rendering and clip any interpolation overshoot back
        to the measured finite-value range.
        """
        z = np.asarray(data, dtype=float)
        rows, cols = z.shape
        max_dim = max(rows, cols)
        if detail_text.startswith("Fast"):
            target = min(128, max_dim)
        elif detail_text.startswith("Ultra"):
            target = max(512, max_dim)
        elif detail_text.startswith("Extreme"):
            target = max(1024, max_dim)
        else:
            return z.copy(), False

        scale = float(target) / float(max_dim)
        nr = max(2, int(round(rows * scale)))
        nc = max(2, int(round(cols * scale)))
        if nr == rows and nc == cols:
            return z.copy(), False

        finite = np.isfinite(z)
        if not finite.any():
            return z.copy(), False
        fill = float(np.nanmedian(z))
        work = np.where(finite, z, fill)
        order = 1 if target <= max_dim else 3
        out = ndimage.zoom(work, (nr / rows, nc / cols), order=order, mode="nearest", prefilter=(order > 1))
        if finite.all():
            lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
            out = np.clip(out, lo, hi)
        return out, target > max_dim

    def _surface_export_dimensions(self):
        text = self.surface_export_size.currentText()
        if text == "Viewport":
            return max(1, self.surface_view.width()), max(1, self.surface_view.height())
        widths = {"2K": 2560, "4K": 3840, "8K": 7680}
        width = widths.get(text, 3840)
        vw = max(1, self.surface_view.width())
        vh = max(1, self.surface_view.height())
        aspect = float(vw) / float(vh)
        height = max(1, int(round(width / max(aspect, 0.1))))
        # Keep pathological narrow/tall window shapes from creating giant framebuffers.
        max_h = {"2K": 2560, "4K": 3840, "8K": 7680}.get(text, 3840)
        if height > max_h:
            height = max_h
            width = max(1, int(round(height * aspect)))
        return width, height

    def export_3d_png(self):
        """Export the active OpenGL scene without creating a hidden GL window.

        pyqtgraph's GLViewWidget.renderToArray() renders the already-initialized
        scene through the active OpenGL context and tiles large outputs internally.
        This avoids the blank/white framebuffer problem that can occur when a new
        off-screen QOpenGLWidget is created solely for export on some Windows/NVIDIA
        driver combinations.
        """
        if gl is None or self.current is None or self._surface_cache is None:
            QMessageBox.information(self, "3D export", "Render a 3D surface first.")
            return
        default = self._default_stem() + "_3D.png"
        path, _ = QFileDialog.getSaveFileName(self, "Export high-resolution 3D PNG", default, "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        width, height = self._surface_export_dimensions()
        bg_text = self.surface_export_bg.currentText()
        if bg_text.startswith("White"):
            export_bg = (255, 255, 255, 255)
        elif bg_text.startswith("Transparent"):
            export_bg = (0, 0, 0, 0)
        else:
            export_bg = (17, 23, 32, 255)

        old_bg = None
        try:
            old_bg = getattr(self.surface_view, "opts", {}).get("bgcolor", None)
        except Exception:
            old_bg = None

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.surface_view.setBackgroundColor(export_bg)
            self.surface_view.makeCurrent()
            self.surface_view.update()
            self.surface_view.repaint()
            QApplication.processEvents()

            used_fallback = False
            array = None
            if hasattr(self.surface_view, "renderToArray"):
                try:
                    # renderToArray uses a tiled FBO, so 4K/8K exports do not require
                    # one enormous framebuffer allocation.
                    array = np.asarray(
                        self.surface_view.renderToArray(
                            (int(width), int(height)), textureSize=2048, padding=128
                        )
                    )
                    if array.ndim != 3 or array.shape[2] < 3:
                        raise RuntimeError(f"Unexpected OpenGL array shape: {array.shape}")
                    # Old pyqtgraph builds used (w,h,4); modern builds use (h,w,4).
                    if array.shape[:2] == (width, height):
                        array = np.transpose(array, (1, 0, 2))
                    if array.shape[0] != height or array.shape[1] != width:
                        raise RuntimeError(
                            f"OpenGL returned {array.shape[1]}×{array.shape[0]} instead of {width}×{height}."
                        )
                    array = np.ascontiguousarray(array)
                    if array.shape[2] >= 4:
                        # renderToArray defaults to BGRA; convert to standard RGBA.
                        rgba = array[..., [2, 1, 0, 3]].copy()
                    else:
                        rgb = array[..., [2, 1, 0]].copy()
                        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])

                    # Detect the failure mode reported by the user instead of silently
                    # saving another blank export. A real AFM scene must have visible
                    # spatial/color variation beyond a uniform background.
                    rgb_std = float(np.std(rgba[..., :3].astype(np.float32)))
                    if not np.isfinite(rgb_std) or rgb_std < 0.35:
                        raise RuntimeError("High-resolution OpenGL render was blank/uniform.")

                    mpl_image.imsave(path, rgba)
                except Exception:
                    array = None

            if array is None:
                # Reliable fallback: grab the framebuffer from the visible, already
                # initialized widget. Scale only as a last resort; this is preferable
                # to producing a white image without warning.
                used_fallback = True
                self.surface_view.repaint()
                QApplication.processEvents()
                image = self.surface_view.grabFramebuffer()
                if image.isNull() or image.width() < 2 or image.height() < 2:
                    raise RuntimeError("The active OpenGL framebuffer could not be captured.")
                if image.width() != width or image.height() != height:
                    image = image.scaled(
                        int(width), int(height),
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                if not image.save(path, "PNG"):
                    raise RuntimeError("Qt could not save the fallback 3D PNG.")

            note = " · viewport fallback" if used_fallback else " · tiled OpenGL render"
            self.statusBar().showMessage(f"Exported GPU 3D PNG: {path} · {width}×{height} px{note}")
        except Exception as exc:
            QMessageBox.critical(
                self, "3D export failed",
                f"Could not export the 3D scene at {width}×{height} pixels.\n\n{exc}\n\n"
                "Try 4K or Viewport if the GPU driver rejects a very large render."
            )
        finally:
            try:
                if old_bg is not None:
                    self.surface_view.setBackgroundColor(old_bg)
                else:
                    self.surface_view.setBackgroundColor((14, 24, 42, 255))
                self.surface_view.update()
            except Exception:
                pass
            QApplication.restoreOverrideCursor()

    def reset_3d_view(self):
        if gl is None:
            return
        sx, xr, yr = self._scaled_axes() if self.current is not None else (None, 5.0, 5.0)
        base_distance = max(float(xr), float(yr), 1.0) * 1.55
        perspective_fov = 45.0
        if hasattr(self, "surface_camera_mode") and self.surface_camera_mode.currentText().startswith("Orthographic"):
            fov = 1.5
            # Narrow-FOV perspective is an orthographic-like projection. Increase
            # distance so the object keeps approximately the same screen size.
            distance = base_distance * np.tan(np.deg2rad(perspective_fov / 2.0)) / np.tan(np.deg2rad(fov / 2.0))
        else:
            fov = perspective_fov
            distance = base_distance
        try:
            self.surface_view.opts["fov"] = fov
            self.surface_view.setCameraPosition(distance=distance, elevation=32, azimuth=-45)
        except Exception:
            pass

    def render_3d(self, *_args):
        if self.current is None or self.channel is None:
            return
        if gl is None:
            self.surface_info.setText("Install pyqtgraph + PyOpenGL to enable the GPU 3D surface.")
            return

        # The optional visual plane correction is deliberately non-destructive.
        # Analysis/export data remain exactly as stored in self.current.
        view_data = np.asarray(self.current, dtype=float)
        if self.chk_3d_plane.isChecked():
            view_data, _ = fit_plane(view_data)

        sx, xr, yr = self._scaled_axes()
        zu = z_unit_scale(view_data, self.channel.unit)
        z_phys = view_data * zu.factor
        if self.chk_3d_center.isChecked():
            z_phys = z_phys - np.nanmedian(z_phys)

        rows, cols = z_phys.shape
        zz, visually_interpolated = self._resample_visual_surface(z_phys, self.surface_detail.currentText())
        vr, vc = zz.shape
        xs = np.linspace(-xr / 2.0, xr / 2.0, vc).astype(np.float32)
        ys = np.linspace(-yr / 2.0, yr / 2.0, vr).astype(np.float32)

        finite = zz[np.isfinite(zz)]
        if finite.size == 0:
            return
        p1, p99 = np.nanpercentile(finite, [1.0, 99.0])
        robust_span = float(p99 - p1)
        if not np.isfinite(robust_span) or robust_span <= 0:
            robust_span = float(np.nanmax(finite) - np.nanmin(finite))
        if not np.isfinite(robust_span) or robust_span <= 0:
            robust_span = 1.0

        z_mult = float(self.surface_z_scale.value())
        mode = self.surface_scale_mode.currentText()
        if mode == "True XYZ scale" and self.channel.unit == "m":
            # Convert the raw Z length to the same displayed spatial unit as X/Y.
            true_z_native = view_data * sx.factor
            if self.chk_3d_center.isChecked():
                true_z_native = true_z_native - np.nanmedian(true_z_native)
            true_z, _ = self._resample_visual_surface(true_z_native, self.surface_detail.currentText())
            z_geom = true_z * z_mult
            scale_note = f"true XYZ · Z ×{z_mult:g}"
        else:
            # Normalize only the visual geometry so nanoscale relief is visible on a micrometre scan.
            # Colors and the numeric Z readout still use calibrated physical heights.
            center = float(np.nanmedian(zz))
            auto_height = 0.34 * max(min(float(xr), float(yr)), 1e-12)
            z_geom = (zz - center) / robust_span * auto_height * z_mult
            if mode == "True XYZ scale" and self.channel.unit != "m":
                scale_note = f"auto height (signal channel) · Z ×{z_mult:g}"
            else:
                scale_note = f"auto morphology height · Z ×{z_mult:g}"

        # Robust color limits prevent a few spikes from washing out the morphology.
        denom = max(robust_span, np.finfo(float).eps)
        norm = np.clip((zz - p1) / denom, 0.0, 1.0)
        colors = colormaps[self._cmap()](norm).astype(np.float32)

        # pyqtgraph GLSurfacePlotItem expects z as (len(x), len(y)).
        z_for_gl = np.ascontiguousarray(z_geom.T.astype(np.float32))
        c_for_gl = np.ascontiguousarray(np.transpose(colors, (1, 0, 2)))

        try:
            if self.surface_item is not None:
                self.surface_view.removeItem(self.surface_item)
            if self.surface_grid is not None:
                self.surface_view.removeItem(self.surface_grid)
        except Exception:
            pass

        try:
            shaded = self.surface_shading.currentText().startswith("Smooth")
            shader_name = "shaded" if shaded else None
            self.surface_item = gl.GLSurfacePlotItem(
                x=xs, y=ys, z=z_for_gl, colors=c_for_gl,
                shader=shader_name, smooth=True, computeNormals=shaded,
            )
            self.surface_item.setGLOptions("opaque")
            self.surface_view.addItem(self.surface_item)

            self.surface_grid = gl.GLGridItem()
            self.surface_grid.setSize(x=float(xr), y=float(yr), z=0.0)
            self.surface_grid.setSpacing(
                x=max(float(xr) / 5.0, 1e-12),
                y=max(float(yr) / 5.0, 1e-12),
                z=1.0,
            )
            base_z = float(np.nanmin(z_geom)) - 0.04 * max(float(np.ptp(z_geom)), 1e-12)
            self.surface_grid.translate(0.0, 0.0, base_z)
            self.surface_view.addItem(self.surface_grid)
            try:
                self._surface_msaa_samples = int(self.surface_view.format().samples())
            except Exception:
                self._surface_msaa_samples = None
            self._surface_cache = {
                "x": xs.copy(), "y": ys.copy(), "z": z_for_gl.copy(), "colors": c_for_gl.copy(),
                "shader": shader_name, "compute_normals": shaded,
                "xr": float(xr), "yr": float(yr), "base_z": float(base_z),
                "camera_distance": max(float(xr), float(yr), 1.0) * 1.55,
            }
        except Exception as exc:
            self.surface_info.setText(f"OpenGL surface error: {exc}")
            return

        if self._surface_reset_pending:
            self.reset_3d_view()
            self._surface_reset_pending = False

        actual_min = float(np.nanmin(z_phys))
        actual_max = float(np.nanmax(z_phys))
        actual_span = actual_max - actual_min
        correction = "visual plane corrected" if self.chk_3d_plane.isChecked() else "raw tilt shown"
        mesh_note = (
            f"visual mesh {z_for_gl.shape[0]}×{z_for_gl.shape[1]} · interpolated display only"
            if visually_interpolated else
            f"native mesh {z_for_gl.shape[0]}×{z_for_gl.shape[1]}"
        )
        msaa_note = f"MSAA {self._surface_msaa_samples}×" if self._surface_msaa_samples else "MSAA requested"
        self.surface_info.setText(
            f"X {xr:.4g} {sx.label}  ·  Y {yr:.4g} {sx.label}  ·  "
            f"Z span {actual_span:.4g} {zu.label}  ·  {scale_note}  ·  {correction}  ·  "
            f"measured grid {rows}×{cols}  ·  {mesh_note}  ·  {msaa_note}"
        )

    def render_psd(self):
        self.psd_canvas.figure.clear()
        ax = self.psd_canvas.figure.add_subplot(111)
        sx = xy_unit_scale(max(self.x_range_m, self.y_range_m))
        # Use the same displayed spatial unit for a readable reciprocal axis.
        x_disp = self.x_range_m * sx.factor
        y_disp = self.y_range_m * sx.factor
        zu, z = self._scaled_z()
        try:
            f, p = radial_psd(z, x_disp, y_disp)
            good = (p > 0) & np.isfinite(p)
            if good.any():
                ax.loglog(f[good], p[good], linewidth=1.7)
            else:
                ax.text(0.5, 0.5, "No resolved spectral power", transform=ax.transAxes, ha="center", color="#9eb4d0")
        except ValueError as exc:
            ax.text(0.5, 0.5, str(exc), transform=ax.transAxes, ha="center", va="center", wrap=True, color="#9eb4d0")
        ax.set_xlabel(f"Spatial frequency (1/{sx.label})")
        ax.set_ylabel(f"2D PSD ({zu.label}² · {sx.label}²)")
        ax.set_title("Surface spectrum · Hann window / plane detrended")
        self._style_axis(ax)
        self.psd_canvas.figure.tight_layout()
        self.psd_canvas.draw_idle()

    def _render_metadata(self):
        if self.nid is None or self.channel is None:
            return
        lines = [f"FILE: {self.nid.path}", "", "SELECTED CHANNEL:"]
        ch = self.channel
        lines += [
            f"  Section: {ch.section}",
            f"  Frame: {ch.frame}",
            f"  Channel: {ch.name}",
            f"  Unit: {ch.unit}",
            f"  Image: {ch.points} × {ch.lines}",
            f"  X range: {ch.x_range_m:.9g} m",
            f"  Y range: {ch.y_range_m:.9g} m",
            f"  Pixel pitch X: {ch.x_range_m / max(ch.points, 1) * 1e9:.9g} nm",
            f"  Pixel pitch Y: {ch.y_range_m / max(ch.lines, 1) * 1e9:.9g} nm",
            f"  Scale status: {getattr(ch, 'scale_status', 'UNVERIFIED')}",
            f"  Scale note: {getattr(ch, 'scale_note', '')}",
            f"  Z calibration min: {ch.z_min:.9g} {ch.unit}",
            f"  Z calibration range: {ch.z_range:.9g} {ch.unit}",
            "",
            "NID METADATA:",
        ]
        for k, v in self.nid.metadata.items():
            lines.append(f"  {k}: {v}")
        self.meta_text.setPlainText("\n".join(lines))

    def enable_profile(self):
        if self.current is None:
            return
        self.map_view_combo.setCurrentIndex(0)
        self.tabs.setCurrentIndex(0)
        self.render_map()
        self.active_map_tool = "profile"
        self.btn_map_profile.setChecked(True)
        self.btn_map_cancel.setEnabled(True)
        self.profile_points = []
        self.map_tool_hint.setText("Line profile: click the first point, then the second point. Press Esc to cancel.")
        self.statusBar().showMessage("Profile mode: click the first point, then the second point on the 2D map.")

    def on_map_click(self, event):
        if self.current is None or self.active_map_tool != "profile":
            return
        if event.inaxes is not getattr(self, "map_ax", None) or event.xdata is None or event.ydata is None:
            return
        sx, xr, yr = self._scaled_axes()
        cols = self.current.shape[1]
        rows = self.current.shape[0]
        px = np.clip(event.xdata / xr * cols - 0.5, 0, cols - 1)
        py = np.clip(event.ydata / yr * rows - 0.5, 0, rows - 1)
        self.profile_points.append((px, py, event.xdata, event.ydata))
        if len(self.profile_points) == 1:
            self.map_tool_hint.setText("First point selected. Click the second point to create the profile, or press Esc.")
            self.profile_artists += self.map_ax.plot(event.xdata, event.ydata, "o", color="#77b9ff", markersize=6)
            self.map_canvas.draw_idle()
        if len(self.profile_points) == 2:
            self.profile_artists += self.map_ax.plot([p[2] for p in self.profile_points], [p[3] for p in self.profile_points], color="#77b9ff", linewidth=1.5)
            self.map_canvas.draw_idle()
            self.show_profile()
            self.cancel_map_tool(keep_profile=True)
            self.statusBar().showMessage("Line profile created. Click 'Line profile' to select another.")

    def show_profile(self):
        p0, p1 = self.profile_points[:2]
        xs, ys, vals = line_profile(self.current, (p0[0], p0[1]), (p1[0], p1[1]))
        sx, xr, yr = self._scaled_axes()
        zu = z_unit_scale(self.current, self.channel.unit if self.channel else "")
        vals = vals * zu.factor
        dx = (p1[2] - p0[2])
        dy = (p1[3] - p0[3])
        distance = np.linspace(0, np.hypot(dx, dy), vals.size)

        dlg = QMainWindow(self)
        dlg.setWindowTitle("AFM line profile")
        dlg.resize(850, 520)
        c = MplCanvas(width=8, height=4.5)
        ax = c.figure.add_subplot(111)
        ax.plot(distance, vals, linewidth=1.7)
        ax.set_xlabel(f"Distance ({sx.label})")
        ax.set_ylabel(f"Height / signal ({zu.label})")
        ax.set_title("Line profile")
        self._style_axis(ax)
        c.figure.tight_layout()
        dlg.setCentralWidget(c)
        dlg.show()
        # keep reference so Qt does not garbage collect the window
        if not hasattr(self, "_profile_windows"):
            self._profile_windows = []
        self._profile_windows.append(dlg)

    def enable_roi(self):
        if self.current is None:
            return
        self.map_view_combo.setCurrentIndex(0)
        self.tabs.setCurrentIndex(0)
        self.cancel_map_tool()
        self.active_map_tool = "roi"
        self.btn_map_roi.setChecked(True)
        self.btn_map_cancel.setEnabled(True)
        self.roi_selector = RectangleSelector(
            self.map_ax, self._roi_selected,
            useblit=True, button=[1], minspanx=2, minspany=2,
            spancoords="pixels", interactive=True,
        )
        self.map_tool_hint.setText("Drag a region on the map, then choose Crop ROI. Press Esc to cancel.")
        self.btn_crop.setEnabled(False)
        self.statusBar().showMessage("Drag a rectangle on the 2D map, then click 'Crop to selected ROI'.")

    def _roi_selected(self, eclick, erelease):
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            return
        self.pending_roi = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
        self.btn_crop.setEnabled(True)
        self.btn_map_crop.setEnabled(True)
        self.map_tool_hint.setText("Region selected. Choose Crop ROI to apply it, or press Esc to cancel.")
        self.statusBar().showMessage("ROI selected. Click 'Crop to selected ROI' to apply.")

    def crop_to_roi(self):
        if self.current is None or self.pending_roi is None:
            QMessageBox.information(self, "ROI", "Select an ROI first.")
            return
        sx, xr, yr = self._scaled_axes()
        x0, y0, x1, y1 = self.pending_roi
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        rows, cols = self.current.shape
        c0 = int(np.clip(np.floor(xmin / xr * cols), 0, cols - 1))
        c1 = int(np.clip(np.ceil(xmax / xr * cols), 1, cols))
        r0 = int(np.clip(np.floor(ymin / yr * rows), 0, rows - 1))
        r1 = int(np.clip(np.ceil(ymax / yr * rows), 1, rows))
        if c1 - c0 < 2 or r1 - r0 < 2:
            QMessageBox.warning(self, "ROI", "The selected ROI is too small.")
            return
        old_cols, old_rows = cols, rows
        self.current = self.current[r0:r1, c0:c1].copy()
        self.crop_origin = (self.crop_origin[0] + r0, self.crop_origin[1] + c0)
        self.x_range_m *= (c1 - c0) / max(old_cols, 1)
        self.y_range_m *= (r1 - r0) / max(old_rows, 1)
        self.pending_roi = None
        self._surface_reset_pending = True
        if self.roi_selector is not None:
            self.roi_selector.set_active(False)
        self._invalidate_morphology()
        self._commit_history(f"Crop ROI · {self.current.shape[1]}×{self.current.shape[0]} px")
        self.refresh_all()
        self.statusBar().showMessage(f"Cropped to ROI: {self.current.shape[1]} × {self.current.shape[0]} pixels.")

    def _default_stem(self):
        if self.nid is None or self.channel is None:
            return "afm_export"
        safe_channel = self.channel.name.replace(" ", "_").replace("/", "-")
        return f"{self.nid.path.stem}_{self.channel.frame.replace(' ', '_')}_{safe_channel}"

    def export_png(self):
        if self.current is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export 600 DPI PNG", self._default_stem() + ".png", "PNG (*.png)")
        if not path:
            return
        fig = Figure(figsize=(6.8, 5.8), dpi=100, facecolor="white")
        ax = fig.add_subplot(111)
        sx, xr, yr = self._scaled_axes()
        zu, z = self._scaled_z()
        im = ax.imshow(z, origin="lower", extent=[0, xr, 0, yr], cmap=self._cmap(), aspect="equal")
        ax.set_xlabel(f"X ({sx.label})", fontfamily="Times New Roman", fontsize=11)
        ax.set_ylabel(f"Y ({sx.label})", fontfamily="Times New Roman", fontsize=11)
        ax.set_title(self.channel.label if self.channel else "AFM map", fontfamily="Times New Roman", fontsize=12)
        cb = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.025)
        cb.set_label(zu.label, fontfamily="Times New Roman", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=600, bbox_inches="tight")
        self.statusBar().showMessage(f"Exported 600 DPI PNG: {path}")

    def export_csv(self):
        if self.current is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export data matrix", self._default_stem() + ".csv", "CSV (*.csv)")
        if not path:
            return
        np.savetxt(path, self.current, delimiter=",", fmt="%.12g")
        self.statusBar().showMessage(f"Exported CSV: {path}")

    def export_tiff(self):
        if self.current is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export float TIFF", self._default_stem() + ".tif", "TIFF (*.tif *.tiff)")
        if not path:
            return
        tifffile.imwrite(path, self.current.astype(np.float32), photometric="minisblack")
        self.statusBar().showMessage(f"Exported float TIFF: {path}")

    def export_json(self):
        if self.current is None or self.channel is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export analysis JSON", self._default_stem() + "_analysis.json", "JSON (*.json)")
        if not path:
            return
        zu = z_unit_scale(self.current, self.channel.unit)
        st_si = roughness_stats(self.current)
        data = {
            "app": APP_NAME,
            "version": APP_VERSION,
            "source": str(self.nid.path) if self.nid else "",
            "algorithm_version": ALGORITHM_VERSION,
            "source_sha256": self.nid.source_sha256 if self.nid else "",
            "calibration_status": getattr(self.channel, "scale_status", "UNVERIFIED"),
            "finite_pixel_fraction": float(np.isfinite(self.current).mean()),
            "channel": self.channel.label,
            "shape": [int(self.current.shape[0]), int(self.current.shape[1])],
            "x_range_m": self.x_range_m,
            "y_range_m": self.y_range_m,
            "crop_origin_px": list(self.crop_origin),
            "signal_unit": self.channel.unit,
            "roughness_in_signal_unit": st_si,
            "display_z_unit": zu.label,
            "history": [state["description"] for state in self.history],
            "active_history_index": self.history_index,
            "metadata": self.nid.metadata if self.nid else {},
        }
        if self.morphology_result is not None:
            headers, rows = self._morph_table_rows()
            data["morphology"] = {
                "threshold_method": self.morph_threshold_method.currentText(),
                "threshold_value_display_unit": self.morphology_result.threshold,
                "direction": self.morph_direction.currentText(),
                "summary": self.morphology_result.summary,
                "columns": headers,
                "particles": rows,
            }
        if self.smart_qc_result is not None:
            r = self.smart_qc_result
            fb = None
            if r.fb_comparison is not None:
                fb = {k: v for k, v in r.fb_comparison.items() if k not in {"aligned", "difference"}}
            data["smart_qc"] = {
                "overall": r.overall,
                "score": r.score,
                "assessments": r.assessments,
                "metrics": r.metrics,
                "recommended_correction": r.recommended_correction,
                "candidate_scores": r.candidate_scores,
                "correlation_length_m": r.correlation_length_m,
                "texture": r.texture,
                "forward_backward": fb,
                "same_frame_channel_correlations": self.smart_channel_correlations,
                "scale_dependent_roughness": {
                    "window_size_px": r.scale_sizes_px.tolist(),
                    "Sa_signal_unit": r.scale_sa.tolist(),
                    "Sq_signal_unit": r.scale_sq.tolist(),
                },
            }
        write_report(path, data)
        self.statusBar().showMessage(f"Exported analysis JSON: {path}")


def main():
    # Request multisampling and vsync before QApplication creates any OpenGL context.
    # Windows/NVIDIA will normally route the pyqtgraph surface through the RTX GPU.
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(32)
    fmt.setSamples(8)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    register_ui_fonts()
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS + WORKSPACE_QSS)
    app.setFont(QFont("Segoe UI", 10))

    # Global UX guard: wheel motion scrolls panels instead of silently changing
    # parameters under the cursor. Keep a Python reference for the app lifetime.
    wheel_value_guard = NoWheelValueChangesFilter(app)
    app.installEventFilter(wheel_value_guard)
    app._wheel_value_guard = wheel_value_guard

    w = AFMStudio()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
