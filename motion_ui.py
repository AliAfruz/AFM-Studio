"""Small, paint-only space visuals and interruptible Qt motion primitives."""
from __future__ import annotations

import math
import os
import random
from PySide6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QApplication, QTabBar, QWidget


def motion_enabled() -> bool:
    app = QApplication.instance()
    return not os.environ.get("AFRUZ_REDUCED_MOTION") == "1" and not bool(
        app and app.property("reducedMotion"))


class OrbitalScene(QWidget):
    """Procedural orbital illustration; timer runs only while the panel is shown."""
    def __init__(self, parent=None, compact=False):
        super().__init__(parent)
        self.compact = compact
        self.phase = 0.0
        self.setMinimumHeight(32 if compact else 200)
        if compact:
            self.setFixedSize(36, 36)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAccessibleName("Orbital surface illustration")
        rng = random.Random(1701)
        self.stars = [(rng.random(), rng.random(), rng.uniform(0.6, 1.8)) for _ in range(55)]
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)

    def _tick(self):
        if not motion_enabled():
            self.timer.stop()
            return
        self.phase += 0.004
        self.update()

    def set_motion(self):
        if self.isVisible() and motion_enabled():
            self.timer.start()
        else:
            self.timer.stop()
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self.set_motion()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        center = QPointF(w * 0.5, h * 0.5)
        radius = min(w, h) * (0.37 if self.compact else 0.36)
        if not self.compact:
            glow = QRadialGradient(center, radius * 1.8)
            glow.setColorAt(0, QColor(49, 72, 142, 100))
            glow.setColorAt(1, QColor(8, 14, 27, 0))
            p.fillRect(self.rect(), glow)
            for x, y, size in self.stars:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(175, 202, 229, 85))
                p.drawEllipse(QPointF(x * w, y * h), size / 2, size / 2)
        for index, angle in enumerate((22, -32, 80)):
            p.save()
            p.translate(center)
            p.rotate(angle)
            p.setPen(QPen(QColor(76, 126, 161, 125), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(-radius, -radius * 0.32, 2 * radius, radius * 0.64))
            theta = self.phase * (index + 1) + index * 2.1
            point = QPointF(radius * math.cos(theta), radius * 0.32 * math.sin(theta))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#7be7df" if index != 1 else "#afa0ff"))
            p.drawEllipse(point, 2.5 if self.compact else 4.0, 2.5 if self.compact else 4.0)
            p.restore()
        globe = QRadialGradient(center - QPointF(radius * 0.2, radius * 0.2), radius * 0.58)
        globe.setColorAt(0, QColor("#426b94"))
        globe.setColorAt(0.7, QColor("#162e51"))
        globe.setColorAt(1, QColor("#0c1a32"))
        p.setPen(QPen(QColor("#41668f"), 1))
        p.setBrush(globe)
        p.drawEllipse(center, radius * 0.44, radius * 0.44)
        p.end()


class OrbitTabBar(QTabBar):
    """Native keyboard-accessible tabs with a moving accent indicator."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setExpanding(False)
        self.setUsesScrollButtons(True)
        self.setElideMode(Qt.TextElideMode.ElideRight)
        self.indicator = QRectF()
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(190)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.valueChanged.connect(self._move_indicator)
        self.currentChanged.connect(self._selection_changed)

    def _move_indicator(self, value):
        self.indicator = value
        self.update()

    def _selection_changed(self, index):
        self.animation.stop()
        if index < 0:
            return
        target = QRectF(self.tabRect(index))
        target.setTop(target.bottom() - 2)
        if motion_enabled() and self.isVisible() and not self.indicator.isNull():
            self.animation.setStartValue(self.indicator)
            self.animation.setEndValue(target)
            self.animation.start()
        else:
            self._move_indicator(target)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.animation.stop()
        if self.currentIndex() >= 0:
            self.indicator = QRectF(self.tabRect(self.currentIndex()))
            self.indicator.setTop(self.indicator.bottom() - 2)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.currentIndex() < 0:
            return
        rect = self.indicator if self.animation.state() == self.animation.State.Running else QRectF(self.tabRect(self.currentIndex()))
        rect.setTop(rect.bottom() - 2)
        p = QPainter(self)
        gradient = QLinearGradient(rect.topLeft(), rect.topRight())
        gradient.setColorAt(0, QColor("#53d8ce"))
        gradient.setColorAt(1, QColor("#a491ff"))
        p.fillRect(rect.adjusted(8, 0, -8, 0), gradient)
        p.end()
