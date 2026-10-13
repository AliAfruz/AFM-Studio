# Afruz AFM Studio v0.4.0

## Orbit workspace and responsive processing

- Centralized the space theme; added orbital visuals, animated controls and tab
  indicators, calibration-state colors, focus styles, and reduced motion.
- Added cancellable background batch processing with an immutable protocol and
  export provenance. Queue mutation is prevented during a run.

## Scientific corrections

- Added Huber robust plane leveling to interactive/batch preprocessing and QC candidates.
- Corrected 2D PSD normalization and display units; incomplete rasters are rejected.
- Corrected local Sa and replaced circular ACF with linear overlap-normalized ACF.
- Preserved missing data in CPU filters and particle measurements.
- Used robust local/background planes, coarse-axis size QC, and calibrated watershed
  spacing; flagged particle boundaries touching missing data.
- Rejected non-length batch height channels and missing explicitly requested channels.
- Added known-answer scientific tests and GUI/background-job regressions.

These changes can alter results compared with v0.3.8. Algorithm version is now
recorded with source hashes in analysis JSON and batch scan summaries.

---

# Afruz AFM Studio v0.3.8

## Workspace and workflow

- Added a welcome screen and a persistent scan summary with calibration and history state.
- Made sidebar control sections collapsible and shortened the 3D tab label.
- Added raw/current/difference map comparison without changing analysis data.
- Made map layout respond to resizing, with centered geometry and a two-line title that fits compact windows.
- Tracked cumulative crop offsets so raw comparisons remain aligned after nested ROI crops, undo, and redo. Included crop offsets in analysis JSON.
- Added redo, Edit-menu actions, and keyboard shortcuts for undo, redo, raw restoration, and cancelling map tools.
- Added direct map controls with active tool buttons, selection instructions, first-point feedback, and a Crop ROI action.
- Replaced status-message-dependent profile activation with explicit tool state; profile and ROI tools are mutually exclusive.
- Added local NID drag and drop: one scan opens, multiple scans queue a batch.
- Corrected automatic profile endpoint display coordinates to use calibrated pixel centers.
- Added offscreen workflow regression checks covering history, comparisons, crop geometry, selection tools, file loading, and batch drop routing.

## v0.3.7

## Morphology export and particle-size management

- Added a dedicated morphology distribution population selector:
  - **Size-QC valid only (recommended)**
  - **All detected particles**
- Morphology histograms now default to the size-QC-valid population so under-resolved segmented objects do not silently bias the reported nanoparticle-size distribution.
- Added configurable histogram binning:
  - automatic Freedman–Diaconis binning with conservative fallbacks;
  - fixed user-selected bin count.
- Added **Set minimum diameter to QC limit**. This optionally converts the current sampling-resolution criterion (minimum pixels across × calibrated pixel pitch) into a physical minimum apparent-diameter filter for the next segmentation run.
- Expanded morphology Excel export with two new tidy sheets:
  - **Size Statistics** — n, mean, median, SD, CV, min, P10, Q25, Q75, P90 and max for apparent equivalent diameter, Feret max/min and P95 height;
  - **Distributions** — bin-by-bin diameter, height and circularity data with source file, channel, population, bin limits, count and fraction.
- The complete **Particles** sheet remains unchanged in principle and retains all particles, including under-resolved particles, with per-particle Size QC flags.
- Added **Export morphology figure · 600 DPI**. It exports the segmentation map plus diameter, P95-height and circularity distributions using the selected distribution population.
- Added the lateral size-QC cutoff marker to exported diameter plots.
- Version bumped to 0.3.7.

## Scientific interpretation

Lateral AFM particle dimensions remain labelled **apparent** because automatic tip-radius deconvolution is not performed. The size-QC threshold is a sampling-resolution criterion, not a correction for probe-tip convolution or a formal uncertainty estimate.
