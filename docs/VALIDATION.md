# v0.4.0 validation

Validation date: 13 October 2026.

The installed application passed 34 tests with zero failures and zero errors.
The checks were run on Windows using:

| Component | Version |
|---|---|
| Python | 3.12.14 |
| PySide6 | 6.11.2 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| scikit-image | 0.26.0 |

## Coverage

- Known plane recovery, fit masks, rank-deficient inputs, and Huber leveling
  that preserves a sparse synthetic feature.
- Missing-data Gaussian/median filters.
- Physical PSD scaling, discrete Parseval energy conservation, and recovery
  of a known sinusoidal wavelength.
- Linear autocorrelation against direct overlap calculations.
- Local Sa/Sq against independently evaluated windows.
- Known particle area, height, volume, and anisotropic sampling.
- Missing-data boundary QC and finite-observation roughness.
- Strict channel matching and rejection of voltage as height metrology.
- Standards-compliant JSON and exact source fingerprints.
- Undo/redo, raw comparison, nested crops, measurement tools, drag-and-drop,
  failed-load preservation, and compact layout.
- Reduced motion, native-event-loop batch processing on both synthetic fixtures,
  protocol capture, cancellation, and safe closing.

The public test suite creates two analytic 64 × 64 NID fixtures, each containing
six calibrated forward/backward image channels. The visual demos use a 256 × 256
version of the same mathematical model. The NID reader's VERIFIED status denotes
agreement between lateral header calibration fields.

CSV, Excel, JSON and 600 DPI morphology exports were exercised. Excel output
was reopened and checked. The Morphology and Batch control panels were checked
to fit their viewport widths.

## Reproduce

From an environment with requirements.txt installed:

    python -m unittest discover -s tests -v

No experimental data is needed or published. Set AFRUZ_AFM_SAMPLE_DIR to a
private folder containing at least two NID scans for an optional local-data run.

The cover is reproducible with `python examples/render_github_preview.py`.
Its heights and contours come from analytic Gaussian particles; the 3D display
uses a visually amplified vertical aspect. It is an illustration of synthetic
topography, not a measured specimen or validation of instrument accuracy.

## Limits

GUI tests use offscreen Qt and the documented 3D fallback. Physical OpenGL and
CUDA drivers were not exercised. These tests check implementation behavior;
they do not establish experimental ground truth or formal metrology uncertainty.

The numerical corrections can change PSD amplitudes, local Sa, ACF lengths,
and some height/volume results relative to v0.3.8. Record the algorithm version
when comparing results. Lateral sizes remain apparent AFM sizes. Smart QC and
segmentation recommendations are deterministic heuristics.

The v0.4.0 release is archived at [DOI 10.5281/zenodo.23348914](https://doi.org/10.5281/zenodo.23348914).
The archive links to the v0.4.0 Git tag. Citation documentation was added
to the main branch after archiving; the release tag is preserved.
