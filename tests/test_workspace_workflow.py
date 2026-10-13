"""Regression checks for navigation, map comparison, and measurement tools."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtCore import QMimeData, Qt, QUrl
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import afm_studio
from workspace_ui import CollapsibleCard, WORKSPACE_QSS
from sample_fixtures import get_sample_files


SAMPLES = get_sample_files()
SAMPLE_DIR = SAMPLES[0].parent


class WorkspaceWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if Path("C:/Windows/Fonts/segoeui.ttf").is_file():
            QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        cls.app.setFont(QFont("Segoe UI", 10))
        cls.app.setStyle("Fusion")
        cls.app.setStyleSheet(afm_studio.DARK_QSS + WORKSPACE_QSS)

    def setUp(self):
        # Offscreen Qt does not support an OpenGL context. Test the documented
        # fallback while keeping all numeric analysis and Qt controls real.
        self.gl_patch = patch.object(afm_studio, "gl", None)
        self.gl_patch.start()
        self.window = afm_studio.AFMStudio()

    def tearDown(self):
        self.window.morph_timer.stop()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.gl_patch.stop()

    def load_sample(self):
        self.assertTrue(self.window.load_nid_file(SAMPLES[0]))
        self.app.processEvents()

    def test_empty_workspace_and_collapsible_controls(self):
        w = self.window
        self.assertIs(w.map_stack.currentWidget(), w.welcome_panel)
        self.assertFalse(w.a_undo.isEnabled())
        self.assertFalse(w.a_redo.isEnabled())
        self.assertFalse(w.a_reset.isEnabled())
        self.assertTrue(w.tabs.isTabEnabled(w.batch_tab_index))
        self.assertFalse(w.tabs.isTabEnabled(1))
        sections = {card.toggle.text(): card for card in w.findChildren(CollapsibleCard)}
        self.assertTrue(sections["PREPROCESSING"].toggle.isChecked())
        card = sections["EXPORT"]
        self.assertTrue(card.content.isHidden())
        card.toggle.click()
        self.assertFalse(card.content.isHidden())

    def test_loading_comparison_and_history_branching(self):
        self.load_sample()
        w = self.window
        raw = w.original.copy()
        self.assertEqual(w.workspace_header.badge.text(), "VERIFIED")
        self.assertIs(w.map_stack.currentWidget(), w.map_canvas)
        w.apply_plane()
        processed = w.current.copy()
        self.assertTrue(w.a_undo.isEnabled())
        w.map_view_combo.setCurrentIndex(1)
        np.testing.assert_array_equal(w.current, processed)
        image = w.map_ax.images[0]
        factor = afm_studio.z_unit_scale(processed, w.channel.unit).factor
        np.testing.assert_allclose(np.asarray(image.get_array()), raw * factor)
        w.map_view_combo.setCurrentIndex(2)
        np.testing.assert_array_equal(w.current, processed)
        np.testing.assert_allclose(np.asarray(w.map_ax.images[0].get_array()), (processed - raw) * factor)
        w.undo_history()
        np.testing.assert_array_equal(w.current, raw)
        self.assertTrue(w.a_redo.isEnabled())
        w.redo_history()
        np.testing.assert_array_equal(w.current, processed)
        w.undo_history()
        w.apply_row_offsets()
        self.assertEqual(len(w.history), 2)
        self.assertFalse(w.a_redo.isEnabled())

    def test_nested_crops_compare_correct_raw_pixels_after_undo_redo(self):
        self.load_sample()
        w = self.window
        original = w.original.copy()
        pitch = (w.y_range_m / original.shape[0], w.x_range_m / original.shape[1])
        for _ in range(2):
            _, xr, yr = w._scaled_axes()
            w.pending_roi = (xr * 0.25, yr * 0.25, xr * 0.75, yr * 0.75)
            w.crop_to_roi()
            rows, cols = w.current.shape
            r0, c0 = w.crop_origin
            np.testing.assert_array_equal(w._raw_current_field(), original[r0:r0 + rows, c0:c0 + cols])
            self.assertAlmostEqual(w.y_range_m / rows, pitch[0])
            self.assertAlmostEqual(w.x_range_m / cols, pitch[1])
        final_origin = w.crop_origin
        final_shape = w.current.shape
        w.undo_history()
        w.redo_history()
        self.assertEqual(w.crop_origin, final_origin)
        self.assertEqual(w.current.shape, final_shape)
        w.apply_plane()
        w.map_view_combo.setCurrentIndex(1)
        factor = afm_studio.z_unit_scale(w.current, w.channel.unit).factor
        np.testing.assert_allclose(np.asarray(w.map_ax.images[0].get_array()), w._raw_current_field() * factor)
        w.reset_data()
        self.assertEqual(w.crop_origin, (0, 0))
        np.testing.assert_array_equal(w.current, original)

    def test_profile_survives_status_messages_and_stops_after_two_points(self):
        self.load_sample()
        w = self.window
        w.enable_profile()
        w.statusBar().showMessage("Another operation completed.")
        _, xr, yr = w._scaled_axes()
        def event(fraction):
            return SimpleNamespace(inaxes=w.map_ax, xdata=xr * fraction, ydata=yr * fraction)
        with patch.object(w, "show_profile") as show:
            w.on_map_click(event(0.25))
            self.assertEqual(len(w.profile_points), 1)
            w.on_map_click(event(0.75))
            show.assert_called_once()
            self.assertIsNone(w.active_map_tool)
            w.on_map_click(event(0.5))
            show.assert_called_once()
        self.assertFalse(w.btn_map_cancel.isEnabled())

    def test_roi_and_profile_are_exclusive_and_escape_cancels(self):
        self.load_sample()
        w = self.window
        w.show()
        self.app.processEvents()
        w.enable_roi()
        selector = w.roi_selector
        self.assertEqual(w.active_map_tool, "roi")
        self.assertFalse(w.btn_map_crop.isEnabled())
        w.enable_profile()
        self.assertFalse(selector.active)
        self.assertIsNone(w.roi_selector)
        QTest.keyClick(w, Qt.Key.Key_Escape)
        self.app.processEvents()
        self.assertIsNone(w.active_map_tool)
        self.assertFalse(w.btn_map_profile.isChecked())

    def test_selected_roi_enables_crop_and_comparison_cancels_it(self):
        self.load_sample()
        w = self.window
        w.enable_roi()
        _, xr, yr = w._scaled_axes()
        w._roi_selected(SimpleNamespace(xdata=xr * 0.25, ydata=yr * 0.25),
                        SimpleNamespace(xdata=xr * 0.75, ydata=yr * 0.75))
        self.assertTrue(w.btn_map_crop.isEnabled())
        w.map_view_combo.setCurrentIndex(1)
        self.assertIsNone(w.pending_roi)
        self.assertIsNone(w.active_map_tool)
        self.assertFalse(w.btn_map_crop.isEnabled())

    def test_drop_routes_multiple_scans_to_batch_and_single_to_map(self):
        w = self.window
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path)) for path in SAMPLES[:2]])
        event = SimpleNamespace(mimeData=lambda: mime, acceptProposedAction=lambda: None,
                                ignore=lambda: self.fail("NID drop was rejected"))
        w.dropEvent(event)
        self.assertEqual(len(w.batch_files), 2)
        self.assertEqual(w.tabs.currentIndex(), w.batch_tab_index)
        w.dropEvent(event)
        self.assertEqual(len(w.batch_files), 2)
        mime.setUrls([QUrl.fromLocalFile(str(SAMPLES[0]))])
        w.dropEvent(event)
        self.assertEqual(w.tabs.currentIndex(), 0)
        self.assertIsNotNone(w.current)

    def test_failed_load_preserves_active_scan_and_controls_toggle(self):
        self.load_sample()
        w = self.window
        nid = w.nid
        current = w.current.copy()
        with patch.object(afm_studio.QMessageBox, "critical"):
            self.assertFalse(w.load_nid_file(SAMPLE_DIR / "missing.nid"))
        self.assertIs(w.nid, nid)
        np.testing.assert_array_equal(w.current, current)
        w.workspace_header.controls_button.click()
        self.assertTrue(w.sidebar_scroll.isHidden())
        self.assertFalse(w.a_sidebar.isChecked())
        w.a_sidebar.trigger()
        self.assertFalse(w.sidebar_scroll.isHidden())
        self.assertTrue(w.workspace_header.controls_button.isChecked())

    def test_map_layout_and_controls_fit_after_resizing(self):
        self.load_sample()
        w = self.window
        w.show()
        for width, height in ((1500, 920), (820, 560), (1500, 920)):
            w.resize(width, height)
            self.app.processEvents()
            w.map_canvas.draw()
            self.assertEqual((w.width(), w.height()), (width, height))
            self.assertLessEqual(w.map_controls.minimumSizeHint().width(), w.map_controls.width())
            title_box = w.map_ax.title.get_window_extent(w.map_canvas.get_renderer())
            canvas_box = w.map_canvas.figure.bbox
            self.assertGreaterEqual(title_box.x0, canvas_box.x0)
            self.assertLessEqual(title_box.x1, canvas_box.x1)
            self.assertIsNotNone(w.map_canvas.figure.get_layout_engine())
            position = w.map_ax.get_position()
            self.assertGreater(position.x0 + position.width / 2, 0.35)
            self.assertLess(position.x0 + position.width / 2, 0.65)


if __name__ == "__main__":
    unittest.main(verbosity=2)
