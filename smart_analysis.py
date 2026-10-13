from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from afm_analysis import autocorrelation_2d, fit_plane, line_flatten, remove_row_offsets, radial_psd, roughness_stats
from morphology_analysis import threshold_value, _watershed_split
from skimage import morphology, segmentation


@dataclass
class SmartQCResult:
    overall: str
    score: float
    assessments: Dict[str, str]
    metrics: Dict[str, float]
    recommended_correction: str
    candidate_scores: Dict[str, float]
    candidate_data: Dict[str, np.ndarray] = field(repr=False)
    leveled: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    spike_mask: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0), dtype=bool))
    bad_rows: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, dtype=int))
    local_sa: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    local_sq: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    scale_sizes_px: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    scale_sa: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    scale_sq: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    autocorrelation: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    correlation_length_m: float = float("nan")
    texture: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    fb_comparison: Optional[Dict[str, object]] = None
    report_lines: List[str] = field(default_factory=list)


def _finite_fill(data: np.ndarray) -> np.ndarray:
    z = np.asarray(data, dtype=float)
    if z.ndim != 2 or min(z.shape) < 2 or not np.isfinite(z).any():
        raise ValueError("Diagnostics require a 2D raster with finite observations.")
    if np.all(np.isfinite(z)):
        return z.copy()
    finite = z[np.isfinite(z)]
    fill = float(np.median(finite)) if finite.size else 0.0
    return np.nan_to_num(z, nan=fill, posinf=fill, neginf=fill)


def _mad(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _robust_sigma(values: np.ndarray) -> float:
    return 1.4826 * _mad(values)


def _robust_span(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    q5, q95 = np.percentile(x, [5.0, 95.0])
    return float(q95 - q5)


def _severity(value: float, moderate: float, high: float) -> str:
    if not np.isfinite(value):
        return "Unknown"
    if value >= high:
        return "High"
    if value >= moderate:
        return "Moderate"
    return "Low"


def _plane_span(data: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    leveled, plane = fit_plane(data)
    return float(np.nanmax(plane) - np.nanmin(plane)), leveled, plane


def _quadratic_bow_span(data: np.ndarray) -> float:
    z, _ = fit_plane(data)
    rows, cols = z.shape
    yy, xx = np.mgrid[:rows, :cols]
    good = np.isfinite(z)
    if good.sum() < 12:
        return 0.0
    # Only quadratic terms are used as a curvature/bow proxy after plane removal.
    xn = (xx - (cols - 1) / 2.0) / max(cols - 1, 1)
    yn = (yy - (rows - 1) / 2.0) / max(rows - 1, 1)
    A = np.c_[xn[good] ** 2, yn[good] ** 2, xn[good] * yn[good], np.ones(good.sum())]
    coef, *_ = np.linalg.lstsq(A, z[good], rcond=None)
    surface = coef[0] * xn ** 2 + coef[1] * yn ** 2 + coef[2] * xn * yn + coef[3]
    return float(np.nanmax(surface) - np.nanmin(surface))


def _row_artifacts(leveled: np.ndarray) -> Tuple[np.ndarray, float, float]:
    row_median = np.nanmedian(leveled, axis=1)
    jumps = np.diff(row_median)
    center = float(np.median(jumps)) if jumps.size else 0.0
    sig = _robust_sigma(jumps)
    if sig <= 0:
        bad = np.empty(0, dtype=int)
    else:
        boundaries = np.where(np.abs(jumps - center) > 6.0 * sig)[0]
        bad = np.unique(np.r_[boundaries, boundaries + 1]).astype(int)
    sq = roughness_stats(leveled)["Sq"]
    jump_sigma_ratio = float(sig / sq) if sq and np.isfinite(sq) else 0.0
    row_offset_ratio = float(_robust_sigma(row_median) / sq) if sq and np.isfinite(sq) else 0.0
    return bad, jump_sigma_ratio, row_offset_ratio


def _isolated_spikes(leveled: np.ndarray) -> Tuple[np.ndarray, float]:
    med = ndimage.median_filter(leveled, size=3, mode="nearest")
    residual = leveled - med
    sig = _robust_sigma(residual)
    span = _robust_span(leveled)
    threshold = max(15.0 * sig, 0.12 * span, np.finfo(float).eps)
    mask = np.abs(residual) > threshold
    # Keep isolated/small connected defects; broad real topographic features are not called spikes.
    labels, n = ndimage.label(mask)
    if n:
        sizes = np.bincount(labels.ravel())
        keep = np.zeros_like(sizes, dtype=bool)
        keep[1:] = sizes[1:] <= 2
        mask = keep[labels]
    density = float(mask.mean() * 100.0)
    return mask, density


def _clipping_fraction(data: np.ndarray) -> float:
    z = np.asarray(data, dtype=float)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return 0.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        return 0.0  # A constant signal alone is not evidence of ADC saturation.
    return float(np.count_nonzero((finite == lo) | (finite == hi)) / finite.size * 100.0)


def _local_roughness(data: np.ndarray, window_px: int = 31) -> Tuple[np.ndarray, np.ndarray]:
    z = _finite_fill(data)
    z -= np.median(z)  # Avoid cancellation in E[z²] - E[z]² for large offsets.
    w = max(3, int(window_px))
    if w % 2 == 0:
        w += 1
    mean = ndimage.uniform_filter(z, size=w, mode="reflect")
    mean2 = ndimage.uniform_filter(z * z, size=w, mode="reflect")
    var = np.maximum(mean2 - mean * mean, 0.0)
    sq = np.sqrt(var)
    # Every sample is compared with the mean of THAT window. Smoothing a
    # residual from a spatially varying mean would systematically bias Sa.
    padded = np.pad(z, w // 2, mode="symmetric")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (w, w))
    sa = np.empty_like(z)
    chunk = max(1, min(16, 16_000_000 // max(z.shape[1] * w * w * 8, 1)))
    for r0 in range(0, z.shape[0], chunk):
        r1 = min(r0 + chunk, z.shape[0])
        sa[r0:r1] = np.mean(np.abs(windows[r0:r1] - mean[r0:r1, :, None, None]), axis=(-2, -1))
    return sa, sq


def _scale_dependent_roughness(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = _finite_fill(data)
    rows, cols = z.shape
    max_size = min(rows, cols)
    requested = [16, 32, 64, 128, 256, 512]
    sizes = [v for v in requested if v <= max_size]
    if max_size not in sizes and max_size >= 8:
        sizes.append(max_size)
    sizes = sorted(set(sizes))
    out_sa = []
    out_sq = []
    for size in sizes:
        sa_vals = []
        sq_vals = []
        for r0 in range(0, rows - size + 1, size):
            for c0 in range(0, cols - size + 1, size):
                patch = z[r0:r0 + size, c0:c0 + size]
                if patch.size < 16:
                    continue
                leveled, _ = fit_plane(patch)
                st = roughness_stats(leveled)
                sa_vals.append(st["Sa"])
                sq_vals.append(st["Sq"])
        out_sa.append(float(np.mean(sa_vals)) if sa_vals else float("nan"))
        out_sq.append(float(np.mean(sq_vals)) if sq_vals else float("nan"))
    return np.asarray(sizes, dtype=float), np.asarray(out_sa, dtype=float), np.asarray(out_sq, dtype=float)


def _peak_valley_metrics(data: np.ndarray) -> Dict[str, float]:
    z, _ = fit_plane(_finite_fill(data))
    z = ndimage.gaussian_filter(z, sigma=0.8)
    sq = roughness_stats(z)["Sq"]
    if not np.isfinite(sq) or sq <= 0:
        return {"peak_count": 0.0, "valley_count": 0.0, "peak_density_px": 0.0, "valley_density_px": 0.0}
    local_max = z == ndimage.maximum_filter(z, size=5, mode="nearest")
    local_min = z == ndimage.minimum_filter(z, size=5, mode="nearest")
    peaks = local_max & (z > 1.5 * sq)
    valleys = local_min & (z < -1.5 * sq)
    # Exclude a two-pixel edge band where extrema are less reliable.
    edge = np.zeros_like(peaks, dtype=bool)
    edge[:2, :] = edge[-2:, :] = True
    edge[:, :2] = edge[:, -2:] = True
    peaks &= ~edge
    valleys &= ~edge
    return {
        "peak_count": float(np.count_nonzero(peaks)),
        "valley_count": float(np.count_nonzero(valleys)),
        "peak_density_px": float(np.mean(peaks)),
        "valley_density_px": float(np.mean(valleys)),
    }


def _autocorrelation(data: np.ndarray, x_range_m: float, y_range_m: float) -> Tuple[np.ndarray, float]:
    acf = autocorrelation_2d(data)
    rows, cols = acf.shape
    dx, dy = abs(float(x_range_m)) / cols, abs(float(y_range_m)) / rows
    if not np.isfinite(dx + dy) or min(dx, dy) <= 0:
        raise ValueError("Autocorrelation length requires finite positive calibration.")
    yy, xx = np.mgrid[:rows, :cols]
    rr = np.hypot((xx - cols // 2) * dx, (yy - rows // 2) * dy)
    rmax = min(abs(x_range_m), abs(y_range_m)) / 2
    bins = max(2, int(rmax / max(dx, dy)))
    edges = np.linspace(0, rmax, bins + 1)
    good = np.isfinite(acf)
    counts, _ = np.histogram(rr[good], bins=edges)
    totals, _ = np.histogram(rr[good], bins=edges, weights=acf[good])
    radial = np.divide(totals, counts, out=np.full(bins, np.nan), where=counts > 0)
    centers = (edges[:-1] + edges[1:]) / 2
    keep = np.isfinite(radial) & (centers > max(dx, dy) / 2)
    # Anchor zero lag explicitly; interpolate the first 1/e crossing.
    radii = np.r_[0.0, centers[keep]]
    values = np.r_[float(acf[rows // 2, cols // 2]), radial[keep]]
    below = np.flatnonzero(values <= 1 / np.e)
    length = float("nan")
    if values[0] > 0 and below.size and below[0] > 0:
        i = int(below[0])
        if values[i - 1] > values[i]:
            t = (values[i - 1] - 1 / np.e) / (values[i - 1] - values[i])
            length = float(radii[i - 1] + t * (radii[i] - radii[i - 1]))
    return acf, length


def _texture_metrics(data: np.ndarray, x_range_m: float, y_range_m: float) -> Dict[str, float]:
    z, _ = fit_plane(_finite_fill(data))
    rows, cols = z.shape
    dx = abs(float(x_range_m)) / max(cols, 1)
    dy = abs(float(y_range_m)) / max(rows, 1)
    gy, gx = np.gradient(z, max(dy, 1e-30), max(dx, 1e-30))
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    trace = jxx + jyy
    disc = float(np.sqrt(max((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0)))
    l1 = 0.5 * (trace + disc)
    l2 = 0.5 * (trace - disc)
    anisotropy = float((l1 - l2) / (l1 + l2)) if (l1 + l2) > 0 else 0.0
    grad_angle = 0.5 * np.degrees(np.arctan2(2.0 * jxy, jxx - jyy))
    texture_direction = float((grad_angle + 90.0) % 180.0)

    dominant_wavelength = float("nan")
    try:
        freq, psd = radial_psd(z, abs(float(x_range_m)), abs(float(y_range_m)))
        if freq.size > 2:
            # Ignore the first low-frequency bin, which is often dominated by residual form/waviness.
            start = 1
            i = start + int(np.nanargmax(psd[start:]))
            if freq[i] > 0:
                dominant_wavelength = float(1.0 / freq[i])
    except Exception:
        pass
    return {
        "anisotropy_index": anisotropy,
        "texture_direction_deg": texture_direction,
        "dominant_wavelength_m": dominant_wavelength,
    }


def _feature_metrics(data: np.ndarray, texture_direction_deg: float) -> Dict[str, float]:
    z, _ = fit_plane(_finite_fill(data))
    rows, cols = z.shape
    # Use local relief rather than the global highest point so broad tilt/form does not dominate.
    background = ndimage.gaussian_filter(z, sigma=max(2.0, min(rows, cols) / 64.0))
    relief = z - background
    iy, ix = np.unravel_index(int(np.nanargmax(np.abs(relief))), relief.shape)
    peak_y, peak_x = np.unravel_index(int(np.nanargmax(z)), z.shape)
    valley_y, valley_x = np.unravel_index(int(np.nanargmin(z)), z.shape)

    angle = np.deg2rad((texture_direction_deg + 90.0) % 180.0)
    half = max(6.0, min(rows, cols) * 0.14)
    dx = np.cos(angle) * half
    dy = np.sin(angle) * half
    x0 = float(np.clip(ix - dx, 0, cols - 1))
    x1 = float(np.clip(ix + dx, 0, cols - 1))
    y0 = float(np.clip(iy - dy, 0, rows - 1))
    y1 = float(np.clip(iy + dy, 0, rows - 1))
    return {
        "strongest_x_px": float(ix),
        "strongest_y_px": float(iy),
        "strongest_relief": float(relief[iy, ix]),
        "highest_x_px": float(peak_x),
        "highest_y_px": float(peak_y),
        "highest_value": float(z[peak_y, peak_x]),
        "deepest_x_px": float(valley_x),
        "deepest_y_px": float(valley_y),
        "deepest_value": float(z[valley_y, valley_x]),
        "profile_x0_px": x0,
        "profile_y0_px": y0,
        "profile_x1_px": x1,
        "profile_y1_px": y1,
    }


def _candidate_quality(data: np.ndarray, raw_sq: float, complexity_penalty: float) -> float:
    z = _finite_fill(data)
    span, leveled, _ = _plane_span(z)
    robust = max(_robust_span(z), np.finfo(float).eps)
    tilt_ratio = span / robust
    _bad, row_jump, row_offset = _row_artifacts(leveled)
    bow = _quadratic_bow_span(z)
    sq = roughness_stats(z)["Sq"]
    bow_ratio = bow / max(roughness_stats(leveled)["Sq"], np.finfo(float).eps)
    if raw_sq > 0 and sq > 0:
        rough_change = abs(np.log(max(sq, 1e-30) / max(raw_sq, 1e-30)))
    else:
        rough_change = 0.0
    return float(
        2.2 * min(tilt_ratio, 5.0)
        + 1.7 * min(row_jump, 5.0)
        + 0.8 * min(row_offset, 5.0)
        + 0.45 * min(bow_ratio, 5.0)
        + 0.20 * min(rough_change, 4.0)
        + complexity_penalty
    )


def leveling_advisor(data: np.ndarray) -> Tuple[str, Dict[str, float], Dict[str, np.ndarray]]:
    raw = _finite_fill(data)
    plane, _ = fit_plane(raw)
    robust_plane, _ = fit_plane(raw, robust=True)
    candidates = {
        "No correction": raw.copy(),
        "Plane level · 1st order": plane,
        "Robust plane level · Huber": robust_plane,
        "Plane + row offsets": remove_row_offsets(plane),
        "Plane + line flatten · degree 1": line_flatten(plane, 1),
        "Plane + line flatten · degree 2": line_flatten(plane, 2),
    }
    penalties = {
        "No correction": 0.00,
        "Plane level · 1st order": 0.05,
        "Robust plane level · Huber": 0.08,
        "Plane + row offsets": 0.12,
        "Plane + line flatten · degree 1": 0.18,
        "Plane + line flatten · degree 2": 0.34,
    }
    raw_sq = roughness_stats(raw)["Sq"]
    scores = {k: _candidate_quality(v, raw_sq, penalties[k]) for k, v in candidates.items()}
    best_score = min(scores.values())
    # Prefer the least aggressive correction when its diagnostic score is close to
    # the numerical optimum. This keeps the advisor conservative and avoids
    # recommending line-by-line processing for small score improvements.
    order = [
        "No correction",
        "Plane level · 1st order",
        "Robust plane level · Huber",
        "Plane + row offsets",
        "Plane + line flatten · degree 1",
        "Plane + line flatten · degree 2",
    ]
    tolerance = 0.65
    best = next((name for name in order if scores[name] <= best_score + tolerance), min(scores, key=scores.get))
    return best, scores, candidates


def compare_forward_backward(reference: np.ndarray, moving: np.ndarray) -> Dict[str, object]:
    a, _ = fit_plane(_finite_fill(reference))
    b0, _ = fit_plane(_finite_fill(moving))
    if a.shape != b0.shape:
        raise ValueError("Forward/backward images do not have the same pixel dimensions.")

    def one_candidate(b: np.ndarray, flipped: bool) -> Dict[str, object]:
        aa = a - np.mean(a)
        bb = b - np.mean(b)
        sa = float(np.std(aa))
        sb = float(np.std(bb))
        if sa > 0:
            aa = aa / sa
        if sb > 0:
            bb = bb / sb
        shift, error, _ = phase_cross_correlation(aa, bb, upsample_factor=10)
        aligned = ndimage.shift(b, shift=shift, order=1, mode="nearest", prefilter=False)
        va = a.ravel()
        vb = aligned.ravel()
        corr = float(np.corrcoef(va, vb)[0, 1]) if np.std(va) > 0 and np.std(vb) > 0 else float("nan")
        diff = a - aligned
        rms = float(np.sqrt(np.mean(diff ** 2)))
        ref_span = max(_robust_span(a), np.finfo(float).eps)
        return {
            "flipped_x": bool(flipped),
            "shift_y_px": float(shift[0]),
            "shift_x_px": float(shift[1]),
            "correlation": corr,
            "rms_difference": rms,
            "rms_percent_robust_span": float(rms / ref_span * 100.0),
            "registration_error": float(error),
            "aligned": aligned,
            "difference": diff,
        }

    normal = one_candidate(b0, False)
    flipped = one_candidate(np.fliplr(b0), True)
    c0 = normal["correlation"] if np.isfinite(normal["correlation"]) else -np.inf
    c1 = flipped["correlation"] if np.isfinite(flipped["correlation"]) else -np.inf
    return flipped if c1 > c0 else normal


def _score_overall(assessments: Dict[str, str], metrics: Dict[str, float]) -> Tuple[str, float]:
    penalties = {"Low": 0.0, "Moderate": 11.0, "High": 24.0, "Unknown": 4.0, "None": 0.0}
    score = 100.0
    for key in ("Tilt", "Line artifacts", "Spikes", "Bow", "Clipping", "Missing data"):
        score -= penalties.get(assessments.get(key, "Unknown"), 4.0)
    if "Forward/backward" in assessments:
        score -= penalties.get(assessments["Forward/backward"], 4.0) * 0.8
    score = float(np.clip(score, 0.0, 100.0))
    high = sum(v == "High" for v in assessments.values())
    moderate = sum(v == "Moderate" for v in assessments.values())
    if high >= 2 or score < 55:
        label = "POOR"
    elif high >= 1 or moderate >= 3 or score < 78:
        label = "REVIEW"
    else:
        label = "GOOD"
    return label, score


def analyze_smart_qc(
    data: np.ndarray,
    *,
    x_range_m: float,
    y_range_m: float,
    window_px: int = 31,
    counterpart: Optional[np.ndarray] = None,
    counterpart_label: str = "",
    fb_reference: Optional[np.ndarray] = None,
) -> SmartQCResult:
    raw = _finite_fill(data)
    missing_pct = float((~np.isfinite(np.asarray(data, dtype=float))).mean() * 100.0)
    plane_span, leveled, _ = _plane_span(raw)
    robust_span = max(_robust_span(raw), np.finfo(float).eps)
    sq_lev = max(roughness_stats(leveled)["Sq"], np.finfo(float).eps)
    tilt_ratio = float(plane_span / robust_span)
    bad_rows, row_jump_ratio, row_offset_ratio = _row_artifacts(leveled)
    spike_mask, spike_density_pct = _isolated_spikes(leveled)
    clipping_pct = _clipping_fraction(raw)
    bow_span = _quadratic_bow_span(raw)
    bow_ratio = float(bow_span / sq_lev)
    hf = leveled - ndimage.gaussian_filter(leveled, sigma=1.0)
    noise_rms = float(np.sqrt(np.mean(hf ** 2)))
    usable_pct = float(max(0.0, 100.0 - bad_rows.size / max(raw.shape[0], 1) * 100.0 - spike_density_pct - missing_pct))

    local_sa, local_sq = _local_roughness(leveled, window_px=window_px)
    scale_sizes_px, scale_sa, scale_sq = _scale_dependent_roughness(raw)
    acf, corr_len = _autocorrelation(np.asarray(data, dtype=float), x_range_m, y_range_m)
    texture = _texture_metrics(raw, x_range_m, y_range_m)
    extrema = _peak_valley_metrics(raw)
    features = _feature_metrics(raw, texture.get("texture_direction_deg", 0.0))
    recommendation, candidate_scores, candidate_data = leveling_advisor(raw)

    assessments: Dict[str, str] = {
        "Tilt": _severity(tilt_ratio, 0.18, 0.55),
        "Line artifacts": _severity(max(row_jump_ratio, bad_rows.size / max(raw.shape[0], 1)), 0.12, 0.32),
        "Spikes": _severity(spike_density_pct, 0.02, 0.15),
        "Bow": _severity(bow_ratio, 0.45, 1.25),
        "Clipping": _severity(clipping_pct, 0.05, 0.5),
        "Missing data": _severity(missing_pct, 0.01, 5.0),
    }
    metrics: Dict[str, float] = {
        "tilt_span": plane_span,
        "tilt_ratio": tilt_ratio,
        "row_jump_ratio": row_jump_ratio,
        "row_offset_ratio": row_offset_ratio,
        "bad_row_count": float(bad_rows.size),
        "spike_count": float(np.count_nonzero(spike_mask)),
        "spike_density_percent": spike_density_pct,
        "clipping_percent": clipping_pct,
        "bow_span": bow_span,
        "bow_ratio": bow_ratio,
        "noise_rms": noise_rms,
        "usable_area_percent": usable_pct,
        "missing_data_percent": missing_pct,
        "peak_count": extrema["peak_count"],
        "valley_count": extrema["valley_count"],
    }

    fb = None
    if counterpart is not None:
        try:
            fb = compare_forward_backward(_finite_fill(fb_reference) if fb_reference is not None else raw, counterpart)
            corr = float(fb["correlation"])
            mismatch = 1.0 - corr if np.isfinite(corr) else 1.0
            assessments["Forward/backward"] = _severity(mismatch, 0.12, 0.30)
            metrics["forward_backward_correlation"] = corr
            metrics["forward_backward_rms_percent_span"] = float(fb["rms_percent_robust_span"])
        except Exception:
            fb = None

    overall, score = _score_overall(assessments, metrics)

    report = [
        f"Overall QC: {overall} ({score:.0f}/100)",
        f"Recommended correction: {recommendation}",
        "",
        "Diagnostics",
        f"• Tilt: {assessments['Tilt']} — fitted plane span is {tilt_ratio:.3g} × the robust image height span.",
        f"• Line artifacts: {assessments['Line artifacts']} — {bad_rows.size} suspicious scan rows; row-jump ratio {row_jump_ratio:.3g}.",
        f"• Isolated spikes: {assessments['Spikes']} — {np.count_nonzero(spike_mask)} pixels ({spike_density_pct:.4g}% of image).",
        f"• Bow/curvature proxy: {assessments['Bow']} — quadratic form span {bow_ratio:.3g} × leveled Sq.",
        f"• Clipping: {assessments['Clipping']} — extrema occupy {clipping_pct:.4g}% of finite pixels.",
        f"• Missing observations: {missing_pct:.4g}% — filled for diagnostic previews only; quantitative PSD requires a complete raster.",
        f"• High-frequency residual RMS: {noise_rms:.6g} in the current signal unit.",
        f"• Estimated usable area: {usable_pct:.3g}% (diagnostic heuristic, not a replacement for visual inspection).",
        "",
        "Spatial / texture",
        f"• Texture direction: {texture.get('texture_direction_deg', float('nan')):.2f}° (structure-tensor estimate).",
        f"• Anisotropy index: {texture.get('anisotropy_index', float('nan')):.3f} (0 isotropic → 1 strongly directional).",
        f"• Peak / valley candidates: {int(extrema['peak_count'])} / {int(extrema['valley_count'])} (5-pixel neighborhood; ±1.5 Sq threshold).",
        f"• Autocorrelation 1/e length: {corr_len * 1e9:.4g} nm" if np.isfinite(corr_len) else "• Autocorrelation 1/e length: not resolved in this scan.",
    ]
    dw = texture.get("dominant_wavelength_m", float("nan"))
    if np.isfinite(dw):
        report.append(f"• Dominant radial PSD wavelength: {dw * 1e9:.4g} nm (spectral maximum after the first non-zero bin).")
    if fb is not None:
        report += [
            "",
            f"Forward/backward comparison ({counterpart_label or 'matching channel'}; independently plane-leveled):",
            f"• Correlation: {float(fb['correlation']):.4f}",
            f"• Registration shift: Δx={float(fb['shift_x_px']):.2f} px, Δy={float(fb['shift_y_px']):.2f} px",
            f"• Fast-axis flip selected: {'yes' if fb['flipped_x'] else 'no'}",
            f"• RMS difference: {float(fb['rms_percent_robust_span']):.3g}% of robust height span",
        ]
    report += [
        "",
        "Interpretation guardrail",
        "These are diagnostic heuristics. A flagged line, spike, periodicity, or mismatch should be confirmed visually before modifying or excluding data.",
        "Morphology recommendations rank mask plausibility using a 14% coverage preference; confidence is not a calibrated probability.",
    ]

    return SmartQCResult(
        overall=overall,
        score=score,
        assessments=assessments,
        metrics=metrics,
        recommended_correction=recommendation,
        candidate_scores=candidate_scores,
        candidate_data=candidate_data,
        leveled=leveled,
        spike_mask=spike_mask,
        bad_rows=bad_rows,
        local_sa=local_sa,
        local_sq=local_sq,
        scale_sizes_px=scale_sizes_px,
        scale_sa=scale_sa,
        scale_sq=scale_sq,
        autocorrelation=acf,
        correlation_length_m=corr_len,
        texture=texture,
        features=features,
        fb_comparison=fb,
        report_lines=report,
    )


def smart_morphology_recommendation(
    data: np.ndarray,
    *,
    x_range: float,
    y_range: float,
    min_area_px: int = 8,
) -> Dict[str, object]:
    """Rank transparent deterministic segmentation candidates quickly.

    The ranking uses mask plausibility only (coverage, object count, tiny-object
    fraction). Full particle metrics are calculated only after the chosen settings
    are transferred to the Morphology workspace, keeping this advisor responsive.
    """
    raw = _finite_fill(data)
    rows, cols = raw.shape
    yy, xx = np.mgrid[:rows, :cols]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(raw.size)]
    coef, *_ = np.linalg.lstsq(A, raw.ravel(), rcond=None)
    leveled = raw - (coef[0] * xx + coef[1] * yy + coef[2])

    candidates: List[Dict[str, object]] = []
    methods = ["Otsu", "Li", "Yen", "Triangle", "Median + kσ"]
    directions = ["Above threshold", "Below threshold"]
    smooth_values = [0.6, 1.0]
    watersheds = [3, 5]
    min_area = max(2, int(min_area_px))

    for sigma in smooth_values:
        analysis = ndimage.gaussian_filter(leveled, sigma=float(sigma))
        for method in methods:
            k_values = [0.8, 1.2] if method == "Median + kσ" else [1.0]
            for k in k_values:
                try:
                    thr = threshold_value(analysis, method, None, k)
                except Exception:
                    continue
                for direction in directions:
                    base = analysis < thr if direction.startswith("Below") else analysis > thr
                    try:
                        base = morphology.remove_small_objects(base.astype(bool), max_size=min_area - 1)
                    except TypeError:
                        base = morphology.remove_small_objects(base.astype(bool), min_size=min_area)
                    base = segmentation.clear_border(base)
                    for wd in watersheds:
                        try:
                            labels = _watershed_split(base, min_distance=wd)
                        except Exception:
                            labels, _ = ndimage.label(base)
                        counts = np.bincount(labels.ravel())
                        areas = counts[1:].astype(float) if counts.size > 1 else np.empty(0)
                        n = int(areas.size)
                        cov = float(np.count_nonzero(labels) / labels.size * 100.0)
                        if n == 0:
                            score = 1e6
                            tiny_frac = 1.0
                        else:
                            tiny_frac = float(np.mean(areas <= min_area + 2))
                            coverage_penalty = abs(cov - 14.0) / 14.0
                            if cov < 0.25 or cov > 60.0:
                                coverage_penalty += 5.0
                            count_penalty = 0.0 if 2 <= n <= 1500 else min(abs(n - 200) / 200.0, 6.0)
                            score = coverage_penalty + 2.2 * tiny_frac + count_penalty
                        candidates.append({
                            "score": float(score),
                            "method": method,
                            "direction": direction,
                            "k_sigma": float(k),
                            "smoothing_sigma": float(sigma),
                            "watershed_distance": int(wd),
                            "threshold": float(thr),
                            "count": int(n),
                            "coverage_percent": float(cov),
                        })

    if not candidates:
        raise RuntimeError("No morphology candidate could be evaluated.")
    candidates.sort(key=lambda item: item["score"])
    best = dict(candidates[0])
    second = candidates[1]["score"] if len(candidates) > 1 else best["score"] + 1.0
    margin = max(0.0, float(second) - float(best["score"]))
    if best["score"] < 1.0 and margin > 0.15:
        confidence = "High"
    elif best["score"] < 2.5:
        confidence = "Moderate"
    else:
        confidence = "Low"
    best["confidence"] = confidence
    best["tested_candidates"] = len(candidates)
    best["top_candidates"] = candidates[:5]
    return best
