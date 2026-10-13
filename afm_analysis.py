from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


ALGORITHM_VERSION = "0.4.0"


def _surface(data: np.ndarray) -> np.ndarray:
    z = np.asarray(data, dtype=np.float64)
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError("A surface must be a two-dimensional raster of at least 2 × 2 pixels.")
    return z


def robust_lstsq(a: np.ndarray, values: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Huber IRLS with MAD scale; the last design column must be constant 1."""
    coef, _, rank, _ = np.linalg.lstsq(a, values, rcond=None)
    if rank < a.shape[1]:
        raise ValueError("Insufficient independent samples to fit a plane.")
    for _ in range(iterations):
        residual = values - a @ coef
        coef[-1] += np.median(residual)
        residual = values - a @ coef
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        tolerance = np.finfo(float).eps * max(np.max(np.abs(values)), np.finfo(float).tiny) * 32
        if np.max(np.abs(residual)) <= tolerance:
            break
        scale = max(scale, tolerance)
        weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), np.finfo(float).tiny))
        root = np.sqrt(weights)
        updated, _, rank, _ = np.linalg.lstsq(a * root[:, None], values * root, rcond=None)
        if rank < a.shape[1]:
            break
        delta = np.linalg.norm(updated - coef)
        coef = updated
        if delta <= max(tolerance, 1e-8 * np.linalg.norm(coef)):
            break
    return coef


@dataclass
class UnitScale:
    factor: float
    label: str


def xy_unit_scale(range_m: float) -> UnitScale:
    a = abs(range_m)
    if a >= 1e-3:
        return UnitScale(1e3, "mm")
    if a >= 1e-6:
        return UnitScale(1e6, "µm")
    if a >= 1e-9:
        return UnitScale(1e9, "nm")
    return UnitScale(1.0, "m")


def z_unit_scale(data: np.ndarray, unit: str) -> UnitScale:
    if unit != "m":
        return UnitScale(1.0, unit or "a.u.")
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return UnitScale(1.0, "m")
    span = float(np.nanmax(finite) - np.nanmin(finite))
    if span < 2e-6:
        return UnitScale(1e9, "nm")
    if span < 2e-3:
        return UnitScale(1e6, "µm")
    return UnitScale(1e3, "mm")


def fit_plane(data: np.ndarray, *, robust: bool = False, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract a first-order plane, optionally using Huber IRLS or a fit mask.

    A True mask pixel participates in the fit. Missing observations stay missing.
    Standard least squares remains the default for backwards compatibility.
    """
    z = _surface(data)
    rows, cols = z.shape
    yy, xx = np.mgrid[:rows, :cols]
    valid = np.isfinite(z)
    if mask is not None:
        fit_mask = np.asarray(mask, dtype=bool)
        if fit_mask.shape != z.shape:
            raise ValueError("Plane fit mask must match the surface dimensions.")
        valid &= fit_mask
    if valid.sum() < 3:
        raise ValueError("At least three finite, non-collinear pixels are needed to fit a plane.")
    xn = (xx - (cols - 1) / 2.0) / max(cols - 1, 1)
    yn = (yy - (rows - 1) / 2.0) / max(rows - 1, 1)
    a = np.c_[xn[valid], yn[valid], np.ones(valid.sum())]
    if robust:
        coef = robust_lstsq(a, z[valid])
    else:
        coef, _, rank, _ = np.linalg.lstsq(a, z[valid], rcond=None)
        if rank < 3:
            raise ValueError("Plane fit pixels must span both raster axes.")
    plane = coef[0] * xn + coef[1] * yn + coef[2]
    leveled = np.where(np.isfinite(z), z - plane, np.nan)
    return leveled, plane


def line_flatten(data: np.ndarray, degree: int = 1) -> np.ndarray:
    """Subtract each fast-scan row's polynomial baseline and center on zero."""
    z = _surface(data).copy()
    degree = int(degree)
    if not 0 <= degree <= 5:
        raise ValueError("Line flatten degree must be between 0 and 5.")
    x = np.linspace(-1.0, 1.0, z.shape[1])
    out = np.empty_like(z)
    for i, row in enumerate(z):
        mask = np.isfinite(row)
        if mask.sum() <= degree:
            out[i] = row
            continue
        coef = np.polyfit(x[mask], row[mask], degree)
        baseline = np.polyval(coef, x)
        out[i] = row - baseline
    finite = np.isfinite(out)
    if finite.any():
        out -= np.median(out[finite])
    out[~finite] = np.nan
    return out


def remove_row_offsets(data: np.ndarray) -> np.ndarray:
    z = _surface(data).copy()
    valid = np.isfinite(z)
    z[~valid] = np.nan
    present = valid.any(axis=1)
    medians = np.zeros((z.shape[0], 1))
    medians[present, 0] = np.nanmedian(z[present], axis=1)
    z -= medians
    if valid.any():
        z -= np.median(z[valid])
    return z


def gaussian_smooth(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Normalize the Gaussian kernel by valid coverage; retain missing pixels."""
    z = _surface(data)
    valid = np.isfinite(z)
    sigma = max(0.01, float(sigma))
    if valid.all():
        return ndimage.gaussian_filter(z, sigma=sigma)
    weight = ndimage.gaussian_filter(valid.astype(float), sigma=sigma)
    filtered = ndimage.gaussian_filter(np.where(valid, z, 0.0), sigma=sigma)
    out = np.full_like(z, np.nan)
    np.divide(filtered, weight, out=out, where=valid & (weight > 0))
    return out


def median_filter(data: np.ndarray, size: int = 3) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    z = _surface(data)
    valid = np.isfinite(z)
    if valid.all():
        return ndimage.median_filter(z, size=size)
    def finite_median(values):
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else np.nan
    out = ndimage.generic_filter(z, finite_median, size=size, mode="reflect")
    out[~valid] = np.nan
    return out


def roughness_stats(data: np.ndarray) -> Dict[str, float]:
    z = np.asarray(data, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {k: float("nan") for k in ("Mean", "Sa", "Sq", "Sp", "Sv", "Sz", "Ssk", "Sku", "Min", "Max")}

    mean = float(np.mean(z))
    c = z - mean
    sq = float(np.sqrt(np.mean(c ** 2)))
    sa = float(np.mean(np.abs(c)))
    sp = float(np.max(c))
    sv = float(-np.min(c))
    ssk = float(np.mean(c ** 3) / sq ** 3) if sq > 0 else float("nan")
    sku = float(np.mean(c ** 4) / sq ** 4) if sq > 0 else float("nan")

    return {
        "Mean": mean,
        "Sa": sa,
        "Sq": sq,
        "Sp": sp,
        "Sv": sv,
        "Sz": sp + sv,
        "Ssk": ssk,
        "Sku": sku,
        "Min": float(np.min(z)),
        "Max": float(np.max(z)),
    }


def line_profile(data: np.ndarray, p0: Tuple[float, float], p1: Tuple[float, float], samples: int | None = None):
    """Bilinear line profile; points are supplied in image pixel coordinates (x, y)."""
    x0, y0 = p0
    x1, y1 = p1
    length_px = float(np.hypot(x1 - x0, y1 - y0))
    if not np.isfinite(length_px):
        raise ValueError("Profile endpoints must be finite pixel coordinates.")
    if samples is None:
        samples = max(2, int(np.ceil(length_px)) + 1)
    xs = np.linspace(x0, x1, samples)
    ys = np.linspace(y0, y1, samples)
    vals = ndimage.map_coordinates(np.asarray(data, dtype=float), [ys, xs], order=1, mode="nearest")
    return xs, ys, vals


def power_spectral_density_2d(data: np.ndarray, x_range: float, y_range: float):
    """Two-sided 2D periodogram in signal² × spatial-unit².

    Uses cycles/spatial-unit, plane detrending and a separable Hann window.
    The squared-window normalization satisfies the discrete Parseval relation.
    Missing data are rejected because zero/median filling changes the spectrum.
    """
    z = _surface(data)
    if not np.isfinite(z).all():
        raise ValueError("PSD requires a complete finite raster; crop away missing data first.")
    if not np.isfinite(x_range) or not np.isfinite(y_range) or min(x_range, y_range) <= 0:
        raise ValueError("PSD requires positive finite X/Y calibration.")
    z, _ = fit_plane(z)
    rows, cols = z.shape
    wy = (np.hanning(rows) if rows > 2 else np.ones(rows))[:, None]
    wx = (np.hanning(cols) if cols > 2 else np.ones(cols))[None, :]
    w = wy * wx
    dx, dy = float(x_range) / cols, float(y_range) / rows
    fft = np.fft.fftshift(np.fft.fft2(z * w))
    psd2 = np.abs(fft) ** 2 * dx * dy / np.sum(w ** 2)
    fx = np.fft.fftshift(np.fft.fftfreq(cols, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(rows, d=dy))
    return fx, fy, psd2


def radial_psd(data: np.ndarray, x_range: float, y_range: float):
    """Annular mean of the 2D density, not a one-dimensional radial density."""
    fx, fy, psd2 = power_spectral_density_2d(data, x_range, y_range)
    rows, cols = psd2.shape
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    fr = np.sqrt(FX ** 2 + FY ** 2)

    fmax = min(np.max(np.abs(fx)), np.max(np.abs(fy)))
    nbins = max(2, min(rows, cols) // 2)
    edges = np.linspace(0, fmax, nbins + 1)
    counts, _ = np.histogram(fr, bins=edges)
    total, _ = np.histogram(fr, bins=edges, weights=psd2)
    radial = np.divide(total, counts, out=np.full(nbins, np.nan), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    good = np.isfinite(radial) & (centers > 0)
    return centers[good], radial[good]


def autocorrelation_2d(data: np.ndarray) -> np.ndarray:
    """Linear (non-periodic), overlap-normalized ACF, cropped to raster size."""
    z = _surface(data)
    valid = np.isfinite(z)
    leveled, _ = fit_plane(z)
    leveled = np.where(valid, leveled, 0.0)
    numerator = fftconvolve(leveled, leveled[::-1, ::-1], mode="full")
    overlap = fftconvolve(valid.astype(float), valid[::-1, ::-1].astype(float), mode="full")
    acf = np.divide(numerator, overlap, out=np.full_like(numerator, np.nan),
                    where=overlap >= max(3.0, 0.25 * valid.sum()))
    rows, cols = z.shape
    variance = float(acf[rows - 1, cols - 1])
    if variance > 0:
        acf /= variance
    else:
        acf[:] = 0.0
    r0, c0 = rows - 1 - rows // 2, cols - 1 - cols // 2
    return acf[r0:r0 + rows, c0:c0 + cols]


def bearing_curve(data: np.ndarray):
    z = np.asarray(data, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.array([]), np.array([])
    heights = np.sort(z)[::-1]
    material_ratio = np.linspace(0.0, 100.0, heights.size)
    return material_ratio, heights
