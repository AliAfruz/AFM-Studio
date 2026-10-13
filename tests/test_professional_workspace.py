"""GUI motion and background-job regressions without a physical OpenGL driver."""
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import afm_studio
from batch_analysis import BatchResult
from motion_ui import OrbitalScene
from workspace_ui import CollapsibleCard
from space_theme import register_ui_fonts
from sample_fixtures import get_sample_files

class ProfessionalWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        register_ui_fonts()
        cls.app.setStyle("Fusion")
        cls.app.setStyleSheet(afm_studio.DARK_QSS)
        cls.samples = get_sample_files()

    def setUp(self):
        self.gl_patch = patch.object(afm_studio, "gl", None)
        self.gl_patch.start()
        self.window = afm_studio.AFMStudio()

    def tearDown(self):
        if self.window.batch_worker is not None:
            self.window.batch_worker.requestInterruption()
            self.assertTrue(self.window.batch_worker.wait(5000))
            self.app.processEvents()
        self.app.setProperty("reducedMotion", False)
        self.window.morph_timer.stop()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.gl_patch.stop()

    def wait_batch(self):
        deadline = time.monotonic() + 5
        while self.window.batch_worker is not None and time.monotonic() < deadline:
            QTest.qWait(15)
        self.assertIsNone(self.window.batch_worker)

    def test_reduced_motion_stops_timers_and_settles_controls(self):
        w = self.window
        w.show()
        self.app.processEvents()
        w.set_reduced_motion(True)
        self.assertTrue(self.app.property("reducedMotion"))
        for scene in w.findChildren(OrbitalScene):
            self.assertFalse(scene.timer.isActive())
        card = next(c for c in w.findChildren(CollapsibleCard) if c.toggle.text() == "EXPORT")
        card.toggle.click()
        self.assertFalse(card.content.isHidden())
        card.toggle.click()
        self.assertTrue(card.content.isHidden())

    def test_welcome_heading_fits_at_large_and_compact_sizes(self):
        w = self.window
        w.show()
        for width, height in ((1500, 920), (820, 560)):
            w.resize(width, height)
            self.app.processEvents()
            title = w.welcome_panel.hero_title
            self.assertGreaterEqual(title.height(), title.heightForWidth(title.width()))

    def test_robust_button_is_undoable(self):
        w = self.window
        self.assertFalse(w.btn_robust_plane.isEnabled())
        self.assertTrue(w.load_nid_file(self.samples[0]))
        raw = w.current.copy()
        w.apply_robust_plane()
        self.assertIn("Huber", w.history[-1]["description"])
        w.undo_history()
        import numpy as np
        np.testing.assert_array_equal(w.current, raw)

    def test_batch_keeps_event_loop_running_and_captures_settings(self):
        w = self.window
        w._add_batch_paths(self.samples[:2])
        ticks = []
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: ticks.append(1))
        captured = []
        def analyze(path, settings):
            captured.append(settings.morphology["min_pixels_across"])
            time.sleep(0.1)
            return BatchResult(str(path), Path(path).name, "Done")
        timer.start()
        try:
            with patch("analysis_jobs.analyze_file", side_effect=analyze):
                start = time.monotonic()
                w.run_batch_analysis()
                self.assertLess(time.monotonic() - start, 0.1)
                original_setting = w.batch_run_settings.morphology["min_pixels_across"]
                w.morph_min_pixels_across.setValue(original_setting + 2)
                self.assertFalse(w.btn_batch_add.isEnabled())
                w.batch_clear()
                self.assertEqual(len(w.batch_files), 2)
                self.wait_batch()
        finally:
            timer.stop()
        self.assertEqual(captured, [original_setting, original_setting])
        self.assertGreater(len(ticks), 3)
        self.assertEqual(len(w.batch_results), 2)
        self.assertTrue(w.btn_batch_export.isEnabled())
        self.assertEqual(w.batch_run_settings.morphology["min_pixels_across"], original_setting)

    def test_batch_cancels_after_current_scan(self):
        w = self.window
        w._add_batch_paths(self.samples[:2])
        def analyze(path, settings):
            time.sleep(0.15)
            return BatchResult(str(path), Path(path).name, "Done")
        with patch("analysis_jobs.analyze_file", side_effect=analyze):
            w.run_batch_analysis()
            deadline = time.monotonic() + 1
            while "Analyzing" not in w.batch_status.text() and time.monotonic() < deadline:
                QTest.qWait(5)
            w.cancel_batch_analysis()
            self.wait_batch()
        self.assertLess(len(w.batch_results), 2)
        self.assertIn("cancelled", w.batch_status.text())
        self.assertTrue(w.btn_batch_run.isEnabled())

    def test_scans_finish_through_native_qt_event_loop(self):
        w = self.window
        w._add_batch_paths(self.samples[:2])
        w._batch_populate_task_list(order=["robust_plane"], checked_codes={"robust_plane"})
        w.run_batch_analysis()
        loop = QEventLoop()
        w.batch_worker.finished.connect(loop.quit)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(60000)
        loop.exec()
        timer.stop()
        self.app.processEvents()
        self.assertIsNone(w.batch_worker)
        self.assertEqual(len(w.batch_results), 2)
        self.assertTrue(all(r.status == "Done" for r in w.batch_results))
        self.assertTrue(all(len(r.scan["Source SHA256"]) == 64 for r in w.batch_results))

    def test_closing_waits_for_scan_and_then_closes_safely(self):
        w = self.window
        w.show()
        w._add_batch_paths(self.samples[:2])
        def analyze(path, settings):
            time.sleep(0.12)
            return BatchResult(str(path), Path(path).name, "Done")
        with patch("analysis_jobs.analyze_file", side_effect=analyze):
            w.run_batch_analysis()
            loop = QEventLoop()
            w.batch_worker.finished.connect(loop.quit)
            QTimer.singleShot(30, w.close)
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            QTest.qWait(20)
        self.assertIsNone(w.batch_worker)
        self.assertFalse(w.isVisible())

if __name__ == "__main__":
    unittest.main()
