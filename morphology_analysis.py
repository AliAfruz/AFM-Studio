from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, distance_matrix
from skimage import filters, measure, morphology, segmentation
from afm_analysis import fit_plane, gaussian_smooth, robust_lstsq


@dataclass
class ParticleRecord:
    particle_id: int
    centroid_x: float
    centroid_y: float
    area: float
    equivalent_diameter: float
    equivalent_diameter_low: float
    equivalent_diameter_high: float
    feret_max: float
    feret_min: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    orientation_deg: float
    max_height: float
    p95_height: float
    mean_height: float
    volume: float
    pixel_count: int
    pixels_across: float
    size_qc_pass: bool
    size_qc_note: str
    touches_border: bool
    touches_missing_data: bool = False

    def as_dict(self) -> Dict[str, float | int | bool | str]:
        return asdict(self)


@dataclass
class MorphologyResult:
    analysis_data: np.ndarray
    measurement_data: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    threshold: float
    records: List[ParticleRecord]
    summary: Dict[str, float | str | bool]
    direction: str


def _finite_fill(data: np.ndarray) -> np.ndarray:
    z = np.asarray(data, dtype=float).copy()
    finite = np.isfinite(z)
    if finite.all():
        return z
    if finite.any():
        z[~finite] = float(np.nanmedian(z[finite]))
    else:
        z[:] = 0.0
    return z


def threshold_value(
    data: np.ndarray,
    method: str,
    manual_value: float | None = None,
    k_sigma: float = 1.0,
) -> float:
    z = np.asarray(data, dtype=float)
    vals = z[np.isfinite(z)]
    if vals.size == 0:
        raise ValueError("A finite surface is required to determine a threshold.")

    method = (method or "Otsu").strip().lower()
    if method == "manual":
        return float(np.nanmedian(vals) if manual_value is None else manual_value)
    if method == "mean":
        return float(np.nanmean(vals))
    if method.startswith("median + k"):
        return float(np.nanmedian(vals) + float(k_sigma) * np.nanstd(vals))
    if method == "li":
        return float(filters.threshold_li(vals))
    if method == "yen":
        return float(filters.threshold_yen(vals))
    if method == "triangle":
        return float(filters.threshold_triangle(vals))
    return float(filters.threshold_otsu(vals))


def _watershed_split(mask: np.ndarray, min_distance: int = 4, sampling=None) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    distance = ndimage.distance_transform_edt(mask, sampling=sampling)
    md = max(1, int(min_distance))
    footprint = 2 * md + 1
    local_max = distance == ndimage.maximum_filter(distance, size=footprint, mode="nearest")
    local_max &= mask & (distance > 0)
    markers, n = ndimage.label(local_max)
    if n <= 1:
        return measure.label(mask, connectivity=2).astype(np.int32)
    return segmentation.watershed(-distance, markers, mask=mask).astype(np.int32)


def _physical_axes_from_coords(coords_rc: np.ndarray, dx: float, dy: float) -> Tuple[float, float, float]:
    if coords_rc.shape[0] < 2:
        return 0.0, 0.0, 0.0
    # Pixel-center coordinates. The +0.5 offset does not affect covariance but
    # keeps the convention explicit and consistent with the calibrated raster.
    xy = np.column_stack(((coords_rc[:, 1] + 0.5) * dx, (coords_rc[:, 0] + 0.5) * dy))
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False, bias=True)
    try:
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        evecs = evecs[:, order]
        major = 4.0 * float(np.sqrt(evals[0])) if evals.size else 0.0
        minor = 4.0 * float(np.sqrt(evals[1])) if evals.size > 1 else 0.0
        vec = evecs[:, 0]
        orientation = float(np.degrees(np.arctan2(vec[1], vec[0])))
        return major, minor, orientation
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0


def _physical_contours(mask: np.ndarray, dx: float, dy: float) -> List[np.ndarray]:
    """Return sub-pixel object boundaries in calibrated physical coordinates.

    The image field is interpreted as `cols` pixels spanning x_range and `rows`
    pixels spanning y_range. Pixel centers therefore sit at (j+0.5)*dx and
    (i+0.5)*dy. Padding lets contours close at image edges without inventing an
    extra half-pixel of object size.
    """
    padded = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
    contours = measure.find_contours(padded, 0.5)
    out: List[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        # padded contour coordinate q -> original-array index q-1. Physical
        # center mapping adds 0.5, so boundary mapping is (q - 0.5)*pitch.
        x = (contour[:, 1] - 0.5) * dx
        y = (contour[:, 0] - 0.5) * dy
        out.append(np.column_stack((x, y)))
    return out


def _contour_perimeter(mask: np.ndarray, dx: float, dy: float) -> float:
    contours = _physical_contours(mask, dx, dy)
    if not contours:
        return 0.0
    total = 0.0
    for pts in contours:
        if pts.shape[0] < 2:
            continue
        closed = np.vstack([pts, pts[0]])
        total += float(np.sum(np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))))
    return total


def _feret_diameters_from_mask(mask: np.ndarray, dx: float, dy: float) -> Tuple[float, float]:
    contours = _physical_contours(mask, dx, dy)
    if not contours:
        return 0.0, 0.0
    # Outer contour is the one with the greatest number of boundary samples.
    pts = max(contours, key=lambda p: p.shape[0])
    if pts.shape[0] < 2:
        return 0.0, 0.0
    try:
        hull = ConvexHull(pts)
        hp = pts[hull.vertices]
    except Exception:
        hp = pts

    if hp.shape[0] < 2:
        return 0.0, 0.0
    if hp.shape[0] <= 700:
        feret_max = float(np.max(distance_matrix(hp, hp)))
    else:
        idx = np.linspace(0, hp.shape[0] - 1, 700).astype(int)
        sampled = hp[idx]
        feret_max = float(np.max(distance_matrix(sampled, sampled)))

    feret_min = np.inf
    for i in range(hp.shape[0]):
        p0 = hp[i]
        p1 = hp[(i + 1) % hp.shape[0]]
        edge = p1 - p0
        norm = float(np.linalg.norm(edge))
        if norm <= 0:
            continue
        normal = np.array([-edge[1], edge[0]]) / norm
        proj = hp @ normal
        width = float(np.max(proj) - np.min(proj))
        feret_min = min(feret_min, width)
    if not np.isfinite(feret_min):
        feret_min = 0.0
    return feret_max, float(feret_min)


def _equivalent_diameter_from_area(area: float) -> float:
    return float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0


def _pixel_boundary_size_band(object_mask: np.ndarray, pixel_area: float) -> Tuple[float, float]:
    """One-pixel segmentation-sensitivity band for equivalent diameter.

    This is *not* a formal instrument uncertainty. It quantifies how much the
    reported area-equivalent diameter would move if the binary boundary shifted
    inward or outward by one pixel.
    """
    eroded = ndimage.binary_erosion(object_mask, iterations=1, border_value=0)
    dilated = ndimage.binary_dilation(object_mask, iterations=1, border_value=0)
    low = _equivalent_diameter_from_area(float(eroded.sum()) * pixel_area)
    high = _equivalent_diameter_from_area(float(dilated.sum()) * pixel_area)
    return low, high


def _local_background_surface(
    measurement: np.ndarray,
    object_mask: np.ndarray,
    full_mask: np.ndarray,
    *,
    sign: float,
    dx: float,
    dy: float,
    iterations: int = 4,
    mode: str = "Local plane",
) -> np.ndarray:
    """Evaluate a local background beneath one particle.

    A plane fitted to a clean ring around the particle is preferred because it
    removes residual local slope without changing the particle itself. Median
    mode is retained as a conservative fallback for sparse/irregular rings.
    """
    ring = ndimage.binary_dilation(object_mask, iterations=max(1, int(iterations)))
    ring &= ~object_mask
    ring &= ~full_mask
    ring &= np.isfinite(measurement)

    obj_rc = np.argwhere(object_mask)
    if obj_rc.size == 0:
        return np.array([], dtype=float)

    vals = measurement[ring]
    mode_norm = (mode or "Local plane").strip().lower()
    if vals.size >= 6 and mode_norm.startswith("local plane"):
        rc = np.argwhere(ring)
        x = (rc[:, 1] + 0.5) * dx
        y = (rc[:, 0] + 0.5) * dy
        x_center, y_center = float(np.mean(x)), float(np.mean(y))
        x_scale, y_scale = max(float(np.ptp(x)), dx), max(float(np.ptp(y)), dy)
        A = np.column_stack(((x - x_center) / x_scale, (y - y_center) / y_scale, np.ones_like(x)))
        try:
            coef = robust_lstsq(A, vals)
            ox = (obj_rc[:, 1] + 0.5) * dx
            oy = (obj_rc[:, 0] + 0.5) * dy
            return coef[0] * (ox - x_center) / x_scale + coef[1] * (oy - y_center) / y_scale + coef[2]
        except (np.linalg.LinAlgError, ValueError):
            pass

    if vals.size:
        baseline = float(np.median(vals))
    else:
        background = measurement[(~full_mask) & np.isfinite(measurement)]
        if background.size:
            baseline = float(np.median(background))
        else:
            finite = measurement[np.isfinite(measurement)]
            baseline = float(np.median(finite)) if finite.size else 0.0
    return np.full(obj_rc.shape[0], baseline, dtype=float)


def analyze_particles(
    data: np.ndarray,
    *,
    x_range: float,
    y_range: float,
    threshold_method: str = "Otsu",
    manual_threshold: float | None = None,
    k_sigma: float = 1.0,
    direction: str = "Above threshold",
    smoothing_sigma: float = 0.8,
    plane_correct: bool = True,
    fill_holes: bool = True,
    opening_radius: int = 0,
    closing_radius: int = 0,
    min_area_px: int = 5,
    max_area_px: int = 0,
    clear_border_objects: bool = True,
    watershed_split: bool = True,
    watershed_min_distance: int = 4,
    min_equivalent_diameter: float = 0.0,
    max_equivalent_diameter: float = 0.0,
    min_pixels_across: float = 3.0,
    background_mode: str = "Local plane",
    background_ring_px: int = 4,
    exclude_underresolved_from_summary: bool = True,
) -> MorphologyResult:
    """Segment and quantify AFM morphology with calibrated pixel geometry.

    `x_range` and `y_range` are the *full raster field widths* in a consistent
    physical unit. Pixel pitch is therefore x_range/columns and y_range/rows.
    Lateral dimensions are apparent AFM dimensions and remain subject to tip
    convolution; the function does not invent a tip deconvolution.
    """
    raw = np.asarray(data, dtype=float).copy()
    if raw.ndim != 2:
        raise ValueError("Particle analysis requires a two-dimensional raster.")
    valid = np.isfinite(raw)
    if not valid.any():
        raise ValueError("Particle analysis requires finite height observations.")
    raw[~valid] = np.nan
    measurement = raw.copy()
    rows, cols = raw.shape
    if rows <= 0 or cols <= 0:
        raise ValueError("Empty AFM image.")
    if not np.isfinite(x_range) or not np.isfinite(y_range) or x_range <= 0 or y_range <= 0:
        raise ValueError("Valid positive X/Y scan calibration is required for particle sizing.")

    # NID Dim0Range/Dim1Range describe the raster field width. Matplotlib's
    # extent=[0, range] convention also treats those as image edges, so /N is the
    # internally consistent calibrated pixel pitch (not /(N-1)).
    dx = abs(float(x_range)) / float(cols)
    dy = abs(float(y_range)) / float(rows)
    pixel_area = dx * dy
    pitch_geom = float(np.sqrt(pixel_area))
    pitch_sampling = max(dx, dy)
    if not np.isfinite(min_pixels_across) or min_pixels_across <= 0:
        raise ValueError("Size-QC pixels across must be positive and finite.")
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
        raise ValueError("Segmentation smoothing must be finite and non-negative.")

    if plane_correct:
        measurement, _ = fit_plane(raw, robust=True)

    analysis = measurement.copy()
    if smoothing_sigma and float(smoothing_sigma) > 0:
        analysis = gaussian_smooth(analysis, sigma=float(smoothing_sigma))

    thr = threshold_value(analysis, threshold_method, manual_threshold, k_sigma)
    direction_norm = (direction or "Above threshold").lower()
    if direction_norm.startswith("below"):
        mask = analysis < thr
        sign = -1.0
    else:
        mask = analysis > thr
        sign = 1.0

    mask &= np.isfinite(analysis)
    if opening_radius > 0:
        mask = morphology.binary_opening(mask, morphology.disk(int(opening_radius)))
    if closing_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(int(closing_radius)))
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    mask &= valid  # Hole filling must never fabricate observed surface pixels.

    effective_min_px = max(1, int(min_area_px))
    if min_equivalent_diameter and float(min_equivalent_diameter) > 0:
        min_area_phys = np.pi * (float(min_equivalent_diameter) ** 2) / 4.0
        effective_min_px = max(effective_min_px, int(np.ceil(min_area_phys / pixel_area)))
    if effective_min_px > 1:
        try:
            mask = morphology.remove_small_objects(mask.astype(bool), max_size=effective_min_px - 1)
        except TypeError:
            mask = morphology.remove_small_objects(mask.astype(bool), min_size=effective_min_px)

    if clear_border_objects:
        mask = segmentation.clear_border(mask)

    if watershed_split:
        labels = _watershed_split(mask, min_distance=watershed_min_distance, sampling=(dy, dx))
    else:
        labels = measure.label(mask, connectivity=2).astype(np.int32)

    # Apply final-object size limits after watershed as well. This is important
    # because a merged island can pass a pre-watershed area threshold and then
    # split into sub-threshold particles. Preserve watershed label boundaries
    # instead of binarizing/relabeling adjacent basins back together.
    effective_max_px = int(max_area_px) if max_area_px and int(max_area_px) > 0 else 0
    if max_equivalent_diameter and float(max_equivalent_diameter) > 0:
        max_area_phys = np.pi * (float(max_equivalent_diameter) ** 2) / 4.0
        max_px_from_d = max(1, int(np.floor(max_area_phys / pixel_area)))
        effective_max_px = min(effective_max_px, max_px_from_d) if effective_max_px else max_px_from_d
    for region in measure.regionprops(labels):
        too_small = region.area < effective_min_px
        too_large = effective_max_px > 0 and region.area > effective_max_px
        if too_small or too_large:
            labels[labels == region.label] = 0
    labels, _, _ = segmentation.relabel_sequential(labels)
    labels = labels.astype(np.int32, copy=False)

    mask = labels > 0

    records: List[ParticleRecord] = []
    for new_id, region in enumerate(measure.regionprops(labels), start=1):
        coords = region.coords
        obj = labels == region.label
        area = float(region.area) * pixel_area
        eq = _equivalent_diameter_from_area(area)
        eq_low, eq_high = _pixel_boundary_size_band(obj, pixel_area)
        feret_max, feret_min = _feret_diameters_from_mask(obj, dx, dy)
        perimeter = _contour_perimeter(obj, dx, dy)
        circularity = float(4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else np.nan
        major, minor, orientation = _physical_axes_from_coords(coords, dx, dy)
        # Prefer directly measured caliper ratio when possible; covariance axes
        # are only a fallback for very small/degenerate masks.
        aspect = float(feret_max / feret_min) if feret_min > 0 else (float(major / minor) if minor > 0 else np.nan)

        cy_px, cx_px = region.centroid
        cx = float((cx_px + 0.5) * dx)
        cy = float((cy_px + 0.5) * dy)
        touches = bool(
            np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == rows - 1)
            or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == cols - 1)
        )

        baseline_surface = _local_background_surface(
            measurement,
            obj,
            mask,
            sign=sign,
            dx=dx,
            dy=dy,
            iterations=max(1, int(background_ring_px)),
            mode=background_mode,
        )
        obj_values = measurement[obj]
        if baseline_surface.size == obj_values.size:
            delta = sign * (obj_values - baseline_surface)
        else:
            delta = sign * obj_values
        finite_delta = delta[np.isfinite(delta)]
        if finite_delta.size:
            positive = np.clip(finite_delta, 0.0, None)
            max_h = float(np.max(positive))
            p95_h = float(np.percentile(positive, 95.0))
            mean_h = float(np.mean(positive))
            volume = float(np.sum(positive) * pixel_area)
        else:
            max_h = p95_h = mean_h = volume = np.nan

        touches_missing = bool(np.any(ndimage.binary_dilation(obj) & ~valid))
        pixels_across = float(eq / pitch_sampling) if pitch_sampling > 0 else np.nan
        size_qc_pass = bool(np.isfinite(pixels_across) and pixels_across >= float(min_pixels_across) and not touches_missing)
        if size_qc_pass:
            size_qc_note = "PASS"
        elif touches_missing:
            size_qc_note = "MISSING-DATA BOUNDARY"
        else:
            size_qc_note = f"UNDER-RESOLVED (<{float(min_pixels_across):g} px across)"

        records.append(
            ParticleRecord(
                particle_id=new_id,
                centroid_x=cx,
                centroid_y=cy,
                area=area,
                equivalent_diameter=eq,
                equivalent_diameter_low=eq_low,
                equivalent_diameter_high=eq_high,
                feret_max=feret_max,
                feret_min=feret_min,
                perimeter=perimeter,
                circularity=circularity,
                aspect_ratio=aspect,
                orientation_deg=orientation,
                max_height=max_h,
                p95_height=p95_h,
                mean_height=mean_h,
                volume=volume,
                pixel_count=int(region.area),
                pixels_across=pixels_across,
                size_qc_pass=size_qc_pass,
                size_qc_note=size_qc_note,
                touches_border=touches,
                touches_missing_data=touches_missing,
            )
        )

    image_area = abs(float(x_range) * float(y_range))
    coverage = float(mask.sum() * pixel_area / image_area * 100.0) if image_area > 0 else 0.0
    n = len(records)
    qc_records = [r for r in records if r.size_qc_pass]
    summary_records = qc_records if exclude_underresolved_from_summary else records

    eqs = np.array([r.equivalent_diameter for r in summary_records], dtype=float)
    heights = np.array([r.p95_height for r in summary_records], dtype=float)
    max_heights = np.array([r.max_height for r in summary_records], dtype=float)
    areas = np.array([r.area for r in summary_records], dtype=float)
    volumes = np.array([r.volume for r in summary_records], dtype=float)

    summary: Dict[str, float | str | bool] = {
        "count": float(n),
        "size_qc_count": float(len(qc_records)),
        "size_qc_fraction_percent": float(len(qc_records) / n * 100.0) if n else 0.0,
        "density": float(n / image_area) if image_area > 0 else np.nan,
        "coverage_percent": coverage,
        "mean_equivalent_diameter": float(np.nanmean(eqs)) if eqs.size else np.nan,
        "median_equivalent_diameter": float(np.nanmedian(eqs)) if eqs.size else np.nan,
        "sd_equivalent_diameter": float(np.nanstd(eqs, ddof=1)) if eqs.size > 1 else np.nan,
        "q25_equivalent_diameter": float(np.nanpercentile(eqs, 25.0)) if eqs.size else np.nan,
        "q75_equivalent_diameter": float(np.nanpercentile(eqs, 75.0)) if eqs.size else np.nan,
        "mean_p95_height": float(np.nanmean(heights)) if heights.size else np.nan,
        "mean_max_height": float(np.nanmean(max_heights)) if max_heights.size else np.nan,
        "mean_area": float(np.nanmean(areas)) if areas.size else np.nan,
        "total_volume": float(np.nansum(volumes)) if volumes.size else 0.0,
        "pixel_pitch_x": dx,
        "pixel_pitch_y": dy,
        "pixel_pitch_geometric": pitch_geom,
        "minimum_qc_lateral_size": float(min_pixels_across) * pitch_sampling,
        "sampling_qc_pitch": pitch_sampling,
        "finite_pixel_fraction": float(valid.mean()),
        "plane_method": "Huber robust plane" if plane_correct else "None",
        "local_background_method": "Huber robust plane / median fallback" if str(background_mode).lower().startswith("local plane") else "Local median / global median fallback",
        "min_pixels_across": float(min_pixels_across),
        "summary_excludes_underresolved": bool(exclude_underresolved_from_summary),
        "background_mode": str(background_mode),
        "lateral_size_note": "Apparent AFM lateral dimensions; tip-convolution correction not applied.",
        "pixel_boundary_band_note": "Low/high diameter is a one-pixel boundary-sensitivity band, not formal instrument uncertainty.",
    }

    return MorphologyResult(
        analysis_data=analysis,
        measurement_data=measurement,
        mask=mask,
        labels=labels,
        threshold=thr,
        records=records,
        summary=summary,
        direction=direction,
    )
