from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CUDAStatus:
    available: bool
    device_name: str = ""
    reason: str = ""


try:
    import cupy as cp  # type: ignore
    from cupyx.scipy import ndimage as cndimage  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    cp = None
    cndimage = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""


def cuda_status() -> CUDAStatus:
    if cp is None:
        return CUDAStatus(False, reason=_IMPORT_ERROR or "CuPy is not installed")
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            return CUDAStatus(False, reason="No CUDA device detected")
        props = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = props.get("name", b"CUDA GPU") if isinstance(props, dict) else b"CUDA GPU"
        if isinstance(raw_name, bytes):
            name = raw_name.decode("utf-8", errors="replace")
        else:
            name = str(raw_name)
        return CUDAStatus(True, device_name=name)
    except Exception as exc:  # pragma: no cover - hardware dependent
        return CUDAStatus(False, reason=str(exc))


def gaussian_smooth_cuda(data: np.ndarray, sigma: float) -> np.ndarray:
    if not np.isfinite(data).all():
        raise ValueError("Use the coverage-normalized CPU filter for missing observations.")
    if cp is None or cndimage is None:
        raise RuntimeError("CUDA acceleration is not available")
    arr = cp.asarray(np.asarray(data, dtype=np.float64))
    out = cndimage.gaussian_filter(arr, sigma=max(0.01, float(sigma)))
    return cp.asnumpy(out)


def median_filter_cuda(data: np.ndarray, size: int) -> np.ndarray:
    if not np.isfinite(data).all():
        raise ValueError("Use the finite-sample CPU median for missing observations.")
    if cp is None or cndimage is None:
        raise RuntimeError("CUDA acceleration is not available")
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    arr = cp.asarray(np.asarray(data, dtype=np.float64))
    out = cndimage.median_filter(arr, size=size)
    return cp.asnumpy(out)
