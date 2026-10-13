"""Deterministic AFM-like demonstration surfaces; no experimental data is used.

Run ``python examples/synthetic_afm.py --output demo`` to create two local NID
files. These fixtures exercise the reader, not the complete Nanosurf format.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def synthetic_surface(points: int = 256, variant: int = 0) -> np.ndarray:
    """Return a simulated surface in metres on a square, normalized field."""
    axis = (np.arange(points) + 0.5) / points
    x, y = np.meshgrid(axis, axis)
    # Analytic Gaussian particles on a slightly tilted, rippled substrate.
    z_nm = 3.5 * x - 2.0 * y + 0.8 * np.sin(2 * np.pi * (4 * x + y))
    particles = (
        (0.18, 0.22, 0.046, 36), (0.43, 0.19, 0.037, 25),
        (0.72, 0.24, 0.061, 48), (0.30, 0.49, 0.068, 55),
        (0.60, 0.51, 0.048, 40), (0.85, 0.58, 0.033, 23),
        (0.15, 0.77, 0.041, 29), (0.46, 0.81, 0.055, 43),
        (0.76, 0.83, 0.059, 52),
    )
    for cx, cy, width, height in particles:
        cx += 0.018 * variant
        cy -= 0.013 * variant
        z_nm += height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width ** 2))
    z_nm += 0.3 * np.cos(2 * np.pi * (17 * x - 11 * y))
    return z_nm * 1e-9


def write_synthetic_nid(path: str | Path, *, points: int = 64,
                        variant: int = 0) -> Path:
    """Encode six calibrated forward/backward channels from analytic functions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    span = (4 + 2 * variant) * 1e-6
    height = synthetic_surface(points, variant)
    deflection = np.gradient(height, axis=1) * 1e7
    header = [
        "[DataSet-Info]", "Description=SYNTHETIC demonstration; not an experimental scan",
        "Generator=Afruz AFM Studio analytic Gaussian surface v1",
        "[DataSet\\Parameters\\Imaging]", f"ScanRange=V[{span:.12g},{span:.12g}]*[m,m]",
    ]
    blocks = []
    info = np.iinfo(np.int32)
    for frame in range(2):
        for channel, (name, unit, data) in enumerate((
            ("Deflection", "V", deflection),
            ("Z-Axis", "m", height),
            ("Z-Axis Sensor", "m", height + 0.1e-9),
        )):
            z_min, z_range = (-1.0, 2.0) if unit == "V" else (-100e-9, 200e-9)
            header.extend([
                f"[DataSet-{frame}:{channel}]", "SaveMode=Binary", "SaveBits=32",
                "SaveSign=Signed", "SaveOrder=Intel", f"Points={points}", f"Lines={points}",
                "Dim0Unit=m", "Dim1Unit=m", f"Dim0Range={span:.12g}", f"Dim1Range={span:.12g}",
                f"Dim2Name={name}", f"Dim2Unit={unit}", f"Dim2Min={z_min:.12g}",
                f"Dim2Range={z_range:.12g}", f"Frame=Scan {'forward' if frame == 0 else 'backward'}",
            ])
            stored = np.rint((data - z_min) / z_range * (info.max - info.min) + info.min)
            if np.any((stored < info.min) | (stored > info.max)):
                raise ValueError("Synthetic surface exceeds its encoded range.")
            blocks.append(stored.astype("<i4").tobytes())
    path.write_bytes(("\r\n".join(header) + "\r\n").encode("ascii") + b"#!" + b"".join(blocks))
    return path


def create_demo_files(directory: str | Path, *, points: int = 64) -> list[Path]:
    directory = Path(directory)
    return [write_synthetic_nid(directory / f"synthetic-orbit-{i + 1}.nid",
                                points=points, variant=i) for i in range(2)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo"))
    parser.add_argument("--points", type=int, default=256)
    args = parser.parse_args()
    if args.points < 8:
        parser.error("--points must be at least 8")
    for result in create_demo_files(args.output, points=args.points):
        print(result)
