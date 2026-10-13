"""Orbit presentation widgets; scientific content remains in the application."""
from __future__ import annotations
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)
from motion_ui import OrbitalScene, motion_enabled

WORKSPACE_QSS = ""  # The complete palette is centralized in space_theme.py.

class CollapsibleCard(QFrame):
    def __init__(self, title: str, expanded: bool = True):
        super().__init__()
        self.setObjectName("sideCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(4)
        self.toggle = QToolButton()
        self.toggle.setObjectName("sectionToggle")
        self.toggle.setText(title)
        self.toggle.setAccessibleName(title + " controls")
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle.setCheckable(True)
        layout.addWidget(self.toggle)
        self.content = QWidget()
        self.content.setObjectName("cardContent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 4, 0, 3)
        self.content_layout.setSpacing(8)
        layout.addWidget(self.content)
        self.animation = QPropertyAnimation(self.content, b"maximumHeight", self)
        self.animation.setDuration(180)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.finished.connect(self._animation_finished)
        self.toggle.toggled.connect(self.set_expanded)
        self.toggle.setChecked(expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool):
        self.animation.stop()
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        if not self.isVisible() or not motion_enabled():
            self.content.setMaximumHeight(16777215 if expanded else 0)
            self.content.setVisible(expanded)
            return
        start = self.content.height() if self.content.isVisible() else 0
        self.content.setVisible(True)
        self.animation.setStartValue(start)
        self.animation.setEndValue(max(1, self.content.sizeHint().height()) if expanded else 0)
        self.animation.start()

    def _animation_finished(self):
        expanded = self.toggle.isChecked()
        self.content.setVisible(expanded)
        self.content.setMaximumHeight(16777215 if expanded else 0)

    def settle_motion(self):
        self.animation.stop()
        self._animation_finished()

class WorkspaceHeader(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("workspaceHeader")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(7)
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        self.orbit = OrbitalScene(compact=True)
        title_row.addWidget(self.orbit)
        title_host = QWidget()
        title_host.setStyleSheet("background: transparent;")
        heading = QVBoxLayout(title_host)
        heading.setContentsMargins(0, 0, 0, 0)
        heading.setSpacing(2)
        eyebrow = QLabel("AFRUZ  /  AFM STUDIO")
        eyebrow.setObjectName("workspaceEyebrow")
        heading.addWidget(eyebrow)
        self.title = QLabel()
        self.title.setObjectName("scanTitle")
        self.title.setMinimumWidth(0)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        heading.addWidget(self.title)
        title_row.addWidget(title_host, 1)
        status_host = QWidget()
        status_host.setStyleSheet("background: transparent;")
        status = QVBoxLayout(status_host)
        status.setContentsMargins(0, 0, 0, 0)
        status.setSpacing(4)
        edition = QLabel("ORBIT WORKSPACE  ·  0.4")
        edition.setObjectName("hint")
        status.addWidget(edition, 0, Qt.AlignmentFlag.AlignRight)
        self.badge = QLabel("No scan loaded")
        self.badge.setObjectName("scanBadge")
        self.badge.setTextFormat(Qt.TextFormat.PlainText)
        status.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignRight)
        title_row.addWidget(status_host)
        layout.addLayout(title_row)
        self.details = QLabel("Open a NID scan or drop files into the workspace.")
        self.details.setObjectName("scanDetails")
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        actions.setSpacing(7)
        self.open_button = QPushButton("Open scan")
        self.open_button.setObjectName("primary")
        self.open_button.setToolTip("Open a NID scan (Ctrl+O). Drop multiple scans to queue a batch.")
        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Previous analysis state (Ctrl+Z)")
        self.redo_button = QPushButton("Redo")
        self.redo_button.setToolTip("Next analysis state (Ctrl+Shift+Z / Ctrl+Y)")
        self.raw_button = QPushButton("Restore raw")
        self.raw_button.setToolTip("Restore the full raw channel (Ctrl+R)")
        self.controls_button = QPushButton("Controls · F9")
        self.controls_button.setCheckable(True)
        self.controls_button.setChecked(True)
        for button in (self.open_button, self.undo_button, self.redo_button, self.raw_button):
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(self.controls_button)
        layout.addLayout(actions)
        self._full_title = "Afruz AFM Studio"
        self.set_scan_title(self._full_title)

    def set_scan_title(self, title: str):
        self._full_title = title
        self.title.setToolTip(title)
        self.title.setAccessibleName(title)
        self._elide_title()

    def set_quality(self, status: str):
        self.badge.setText(status)
        quality = "verified" if status == "VERIFIED" else "invalid" if status == "INVALID" else "review" if status == "CHECK" else "neutral"
        self.badge.setProperty("quality", quality)
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)

    def _elide_title(self):
        self.title.setText(self.title.fontMetrics().elidedText(
            self._full_title, Qt.TextElideMode.ElideMiddle, max(1, self.title.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide_title()

class WelcomePanel(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self.setWidget(content)
        outer = QVBoxLayout(content)
        outer.setContentsMargins(28, 22, 28, 22)
        outer.addStretch(1)
        host = QWidget()
        self.host = host
        host.setMaximumWidth(900)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(13)
        hero = QHBoxLayout()
        words = QVBoxLayout()
        eyebrow = QLabel("A NEW PERSPECTIVE ON YOUR SURFACE")
        eyebrow.setObjectName("workspaceEyebrow")
        eyebrow.setWordWrap(True)
        words.addWidget(eyebrow)
        title = QLabel("Small surfaces.\nDeeper insight.")
        self.hero_title = title
        title.setObjectName("welcomeTitle")
        title.setWordWrap(True)
        words.addWidget(title)
        description = QLabel("A calibrated workspace for topography, particle metrology, and reproducible AFM analysis.")
        description.setObjectName("welcomeBody")
        description.setWordWrap(True)
        words.addWidget(description)
        words.addStretch()
        hero.addLayout(words, 3)
        self.orbit = OrbitalScene()
        self.orbit.setMinimumWidth(130)
        hero.addWidget(self.orbit, 2)
        layout.addLayout(hero)
        for number, name, body in (
            ("01", "Open & inspect", "Choose a channel. Confirm the scan field, units, and calibration."),
            ("02", "Prepare & compare", "Level or filter the surface, then compare raw and processed maps."),
            ("03", "Measure & publish", "Inspect particles, roughness, and QC. Export data and scientific figures."),
        ):
            card = QFrame()
            card.setObjectName("welcomeStep")
            row = QHBoxLayout(card)
            row.setContentsMargins(18, 13, 18, 13)
            row.setSpacing(18)
            label = QLabel(number)
            label.setObjectName("stepNumber")
            row.addWidget(label)
            text = QVBoxLayout()
            heading = QLabel(name)
            heading.setObjectName("stepTitle")
            text.addWidget(heading)
            description = QLabel(body)
            description.setObjectName("welcomeBody")
            description.setWordWrap(True)
            text.addWidget(description)
            row.addLayout(text, 1)
            layout.addWidget(card)
        buttons = QHBoxLayout()
        self.open_button = QPushButton("Open NID scan · Ctrl+O")
        self.open_button.setObjectName("primary")
        self.batch_button = QPushButton("Batch workspace")
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.batch_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        hint = QLabel("Drop a scan to open it. Drop several to build a batch.\nMotion can be reduced in the View menu.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        outer.addWidget(host, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "host"):
            width = min(900, max(280, self.viewport().width() - 56))
            self.host.setFixedWidth(width)
            self.hero_title.setStyleSheet(f"font-size: {31 if width >= 650 else 22}pt; font-weight: 700;")
