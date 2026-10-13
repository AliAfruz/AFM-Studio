"""Known-answer metrology tests; independent of GUI and physical hardware."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afm_analysis import (autocorrelation_2d, fit_plane, gaussian_smooth, median_filter,
                          power_spectral_density_2d, radial_psd, roughness_stats)
from morphology_analysis import analyze_particles
from smart_analysis import _local_roughness, _autocorrelation, analyze_smart_qc
from batch_analysis import BatchSettings, apply_preprocessing_recipe, analyze_file, select_channel
from nid_reader import NIDChannel, NIDFile

class ScientificAlgorithmTests(unittest.TestCase):
    def test_plane_recovers_known_tilt_at_nanometre_scale(self):
        y, x = np.mgrid[:40, :60]
        plane = (3 + 0.07 * x - 0.04 * y) * 1e-9
        leveled, fitted = fit_plane(plane)
        np.testing.assert_allclose(fitted, plane, atol=1e-22)
        self.assertLess(np.max(np.abs(leveled)), 1e-22)

    def test_robust_plane_preserves_sparse_features(self):
        y, x = np.mgrid[:80, :80]
        baseline = (2 + 0.025 * x + 0.035 * y) * 1e-9
        z = baseline.copy()
        z[10:22, 8:20] += 50e-9
        _, ordinary = fit_plane(z)
        leveled, robust = fit_plane(z, robust=True)
        ordinary_error = np.mean(np.abs(ordinary - baseline))
        robust_error = np.mean(np.abs(robust - baseline))
        self.assertLess(robust_error, ordinary_error * 0.02)
        self.assertAlmostEqual(float(np.median(leveled[10:22, 8:20])) / 1e-9, 50, places=3)

    def test_plane_mask_and_degenerate_input(self):
        y, x = np.mgrid[:8, :9]
        baseline = 2 + 3 * x - 4 * y
        z = baseline.astype(float)
        z[2:4, 2:4] = np.nan
        mask = np.ones_like(z, dtype=bool)
        mask[:, 0] = False
        out, plane = fit_plane(z, mask=mask)
        np.testing.assert_allclose(plane, baseline, atol=1e-12)
        self.assertTrue(np.isnan(out[2:4, 2:4]).all())
        with self.assertRaises(ValueError):
            fit_plane(np.full((4, 4), np.nan))
        line_mask = np.zeros((8, 9), bool)
        line_mask[0] = True
        with self.assertRaises(ValueError):
            fit_plane(z, mask=line_mask)

    def test_filters_preserve_missing_data_and_constant_surface(self):
        z = np.full((19, 21), 7.5)
        z[5:8, 6:9] = np.nan
        for filtered in (gaussian_smooth(z, 1.2), median_filter(z, 4)):
            np.testing.assert_allclose(filtered[np.isfinite(z)], 7.5)
            self.assertTrue(np.isnan(filtered[~np.isfinite(z)]).all())

    def test_psd_parseval_and_physical_scaling(self):
        z = np.random.default_rng(704).normal(size=(48, 64)) * 2e-9
        fx, fy, density = power_spectral_density_2d(z, 3e-6, 4e-6)
        residual, _ = fit_plane(z)
        window = np.hanning(48)[:, None] * np.hanning(64)[None, :]
        expected = np.sum((residual * window) ** 2) / np.sum(window ** 2)
        integrated = density.sum() * (fx[1] - fx[0]) * (fy[1] - fy[0])
        self.assertAlmostEqual(integrated / expected, 1, places=12)
        fx2, fy2, density2 = power_spectral_density_2d(z * 10, 6e-6, 8e-6)
        np.testing.assert_allclose(density2, density * 400, rtol=1e-12, atol=0)
        np.testing.assert_allclose(fx2, fx / 2)
        np.testing.assert_allclose(fy2, fy / 2)

    def test_psd_resolves_known_sinusoidal_wavelength(self):
        n = 128
        z = np.tile(np.sin(2 * np.pi * 12 * np.arange(n) / n), (n, 1))
        fx, fy, density = power_spectral_density_2d(z, 8.0, 8.0)
        row, col = np.unravel_index(np.argmax(density), density.shape)
        self.assertAlmostEqual(abs(fx[col]), 12 / 8)
        self.assertAlmostEqual(fy[row], 0)
        frequency, radial = radial_psd(z, 8.0, 8.0)
        self.assertLess(abs(frequency[np.argmax(radial)] - 12 / 8), 0.13)

    def test_psd_rejects_missing_data_and_invalid_calibration(self):
        z = np.zeros((8, 8))
        z[0, 0] = np.nan
        with self.assertRaises(ValueError):
            power_spectral_density_2d(z, 1, 1)
        with self.assertRaises(ValueError):
            power_spectral_density_2d(np.zeros((8, 8)), 0, 1)

    def test_acf_matches_direct_linear_overlap_calculation(self):
        z = np.random.default_rng(512).normal(size=(20, 24))
        residual, _ = fit_plane(z)
        acf = autocorrelation_2d(z)
        variance = np.mean(residual ** 2)
        self.assertAlmostEqual(acf[10, 12], 1, places=12)
        for dy, dx in ((0, 1), (1, 0), (5, 7), (-4, 3)):
            a = residual[max(0, dy):min(20, 20 + dy), max(0, dx):min(24, 24 + dx)]
            b = residual[max(0, -dy):min(20, 20 - dy), max(0, -dx):min(24, 24 - dx)]
            self.assertAlmostEqual(acf[10 + dy, 12 + dx], float(np.mean(a * b)) / variance, places=12)
        _, length = _autocorrelation(z, 20e-6, 24e-6)
        self.assertTrue(np.isfinite(length))

    def test_local_roughness_matches_each_window_reference(self):
        z = np.random.default_rng(231).normal(size=(9, 11))
        sa, sq = _local_roughness(z, 5)
        padded = np.pad(z, 2, mode="symmetric")
        for r, c in ((0, 0), (4, 5), (8, 10)):
            window = padded[r:r + 5, c:c + 5]
            self.assertAlmostEqual(sa[r, c], float(np.mean(np.abs(window - window.mean()))), places=12)
            self.assertAlmostEqual(sq[r, c], float(np.std(window)), places=12)

    def test_roughness_uses_finite_observations(self):
        stats = roughness_stats(np.array([[-1, 1], [np.nan, np.inf]]))
        self.assertEqual(stats["Sa"], 1)
        self.assertEqual(stats["Sq"], 1)
        self.assertEqual(stats["Sz"], 2)

    def test_particle_area_height_volume_and_anisotropic_sampling(self):
        z = np.zeros((40, 50))
        z[10:20, 15:25] = 6
        result = analyze_particles(z, x_range=100, y_range=40, threshold_method="Manual",
            manual_threshold=3, smoothing_sigma=0, plane_correct=False, watershed_split=False,
            background_mode="Local median", min_area_px=1)
        self.assertEqual(len(result.records), 1)
        p = result.records[0]
        self.assertEqual(p.pixel_count, 100)
        self.assertEqual(p.area, 200)
        self.assertEqual(p.p95_height, 6)
        self.assertEqual(p.volume, 1200)
        self.assertAlmostEqual(p.equivalent_diameter, np.sqrt(800 / np.pi))
        self.assertAlmostEqual(p.pixels_across, p.equivalent_diameter / 2)
        self.assertEqual(result.summary["minimum_qc_lateral_size"], 6)

    def test_missing_data_are_not_fabricated_particles_or_heights(self):
        z = np.zeros((30, 30))
        z[10:20, 10:20] = 5
        z[12, 12] = np.nan
        result = analyze_particles(z, x_range=30, y_range=30, threshold_method="Manual",
            manual_threshold=2, smoothing_sigma=0, plane_correct=False, watershed_split=False,
            background_mode="Local median", min_area_px=1)
        self.assertFalse(result.mask[12, 12])
        self.assertEqual(result.records[0].pixel_count, 99)
        self.assertEqual(result.records[0].volume, 495)
        self.assertTrue(result.records[0].touches_missing_data)
        self.assertFalse(result.records[0].size_qc_pass)

    def test_robust_recipe_is_supported(self):
        y, x = np.mgrid[:20, :30]
        z = (2 + x + y).astype(float)
        out, labels, tasks = apply_preprocessing_recipe(z, BatchSettings(preprocessing_tasks=[{"task": "robust_plane"}]))
        self.assertLess(np.max(np.abs(out)), 1e-12)
        self.assertEqual(tasks[0]["task"], "robust_plane")
        self.assertIn("Huber", labels[0])

    def test_batch_rejects_voltage_channel_instead_of_labelling_it_height(self):
        ch = NIDChannel("section", "Scan forward", 0, "Deflection", "V", 8, 8,
            1e-6, 1e-6, 0, 1, np.zeros((8, 8)), scale_status="VERIFIED")
        nid = NIDFile(Path("voltage.nid"), [ch], {}, {})
        with patch("batch_analysis.read_nid", return_value=nid):
            result = analyze_file("voltage.nid", BatchSettings())
        self.assertEqual(result.status, "Error")
        self.assertIn("length channel", result.error)

    def test_smart_qc_reports_missing_observations(self):
        z = np.random.default_rng(7).normal(size=(32, 32))
        z[3:7, 3:7] = np.nan
        result = analyze_smart_qc(z, x_range_m=1e-6, y_range_m=1e-6, window_px=5)
        self.assertAlmostEqual(result.metrics["missing_data_percent"], 100 * 16 / 1024)
        self.assertEqual(result.assessments["Missing data"], "Moderate")

    def test_missing_requested_channel_does_not_silently_fall_back(self):
        ch = NIDChannel("section", "Scan forward", 0, "Z-Axis", "m", 8, 8,
            1e-6, 1e-6, 0, 1, np.zeros((8, 8)), scale_status="VERIFIED")
        nid = NIDFile(Path("forward.nid"), [ch], {}, {})
        for settings in (BatchSettings(channel_mode="Backward Z-Axis"),
                         BatchSettings(channel_mode="Match current channel", match_channel_name="Z-Axis", match_channel_frame="Scan backward")):
            with self.assertRaises(ValueError):
                select_channel(nid, settings)

if __name__ == "__main__":
    unittest.main()
