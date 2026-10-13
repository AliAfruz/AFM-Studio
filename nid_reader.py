from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import hashlib
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class NIDChannel:
    section: str
    frame: str
    channel_id: int
    name: str
    unit: str
    points: int
    lines: int
    x_range_m: float
    y_range_m: float
    z_min: float
    z_range: float
    data: np.ndarray
    x_unit: str = "m"
    y_unit: str = "m"
    x_raw_range: float = 0.0
    y_raw_range: float = 0.0
    scale_status: str = "UNVERIFIED"
    scale_note: str = ""

    @property
    def label(self) -> str:
        unit = f" [{self.unit}]" if self.unit else ""
        return f"{self.frame} — {self.name}{unit}"


@dataclass
class NIDFile:
    path: Path
    channels: List[NIDChannel]
    metadata: Dict[str, str]
    sections: Dict[str, Dict[str, str]]
    source_sha256: str = ""


def _parse_sections(text: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """Parse the INI-like text header while preserving section order."""
    sections: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip("\x00\r\n")
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            if current not in sections:
                sections[current] = {}
                order.append(current)
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        sections[current][key.strip()] = value.strip()
    return order, sections


def _float(d: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def _int(d: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(d.get(key, default)))
    except (TypeError, ValueError):
        return default




def _length_factor_to_m(unit: str) -> float | None:
    u = (unit or "").strip().lower().replace("μ", "µ")
    factors = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
    }
    return factors.get(u)


def _parse_scan_range_m(sections: Dict[str, Dict[str, str]]) -> Tuple[float | None, float | None]:
    """Read the imaging ScanRange vector when present, in metres."""
    raw = sections.get("DataSet\\Parameters\\Imaging", {}).get("ScanRange", "")
    if not raw:
        return None, None
    m = re.search(r"V\[([^\]]+)\]\*\[([^\]]+)\]", raw)
    if not m:
        return None, None
    try:
        values = [float(v.strip()) for v in m.group(1).split(",")]
        units = [u.strip() for u in m.group(2).split(",")]
        if len(values) < 2 or len(units) < 2:
            return None, None
        fx = _length_factor_to_m(units[0])
        fy = _length_factor_to_m(units[1])
        if fx is None or fy is None:
            return None, None
        return values[0] * fx, values[1] * fy
    except Exception:
        return None, None


def _decode_block(raw_bytes: bytes, cfg: Dict[str, str]) -> np.ndarray:
    bits = _int(cfg, "SaveBits", 32)
    signed = cfg.get("SaveSign", "Signed").strip().lower() == "signed"
    order = cfg.get("SaveOrder", "Intel").strip().lower()
    endian = "<" if order in {"intel", "little", "little-endian"} else ">"

    if bits not in (8, 16, 32, 64):
        raise ValueError(f"Unsupported NID SaveBits={bits}")

    kind = "i" if signed else "u"
    dtype = np.dtype(f"{endian}{kind}{bits // 8}")
    raw = np.frombuffer(raw_bytes, dtype=dtype)

    z_min = _float(cfg, "Dim2Min", 0.0)
    z_range = _float(cfg, "Dim2Range", 1.0)

    if signed:
        iinfo = np.iinfo(dtype)
        # Map the complete stored integer range onto [Dim2Min, Dim2Min+Dim2Range].
        scaled = z_min + (raw.astype(np.float64) - iinfo.min) * z_range / (iinfo.max - iinfo.min)
    else:
        iinfo = np.iinfo(dtype)
        scaled = z_min + raw.astype(np.float64) * z_range / iinfo.max
    return scaled


def read_nid(path: str | Path) -> NIDFile:
    """
    Read a Nanosurf-style .nid file containing an INI-like header followed by
    sequential binary image blocks. The parser is intentionally conservative:
    it uses only calibration fields present in the file and raises on truncated
    payloads instead of guessing.
    """
    path = Path(path)
    blob = path.read_bytes()

    marker = blob.find(b"#!")
    if marker < 0:
        raise ValueError("NID binary marker '#!' was not found.")
    payload_offset = marker + 2

    header = blob[:marker].decode("latin-1", errors="replace")
    order, sections = _parse_sections(header)

    data_sections = [
        name for name in order
        if re.fullmatch(r"DataSet-\d+:\d+", name) and sections[name].get("SaveMode", "").lower() == "binary"
    ]
    if not data_sections:
        raise ValueError("No binary DataSet-* image sections were found in the NID header.")

    channels: List[NIDChannel] = []
    cursor = payload_offset
    scan_range_x_m, scan_range_y_m = _parse_scan_range_m(sections)

    for section_name in data_sections:
        cfg = sections[section_name]
        points = _int(cfg, "Points")
        lines = _int(cfg, "Lines")
        bits = _int(cfg, "SaveBits", 32)
        if points <= 0 or lines <= 0:
            raise ValueError(f"Invalid image dimensions in [{section_name}].")

        nbytes = points * lines * (bits // 8)
        if cursor + nbytes > len(blob):
            raise ValueError(
                f"Truncated NID payload while reading [{section_name}]: "
                f"need {nbytes} bytes, only {len(blob) - cursor} remain."
            )

        block = _decode_block(blob[cursor:cursor + nbytes], cfg)
        cursor += nbytes
        data = block.reshape((lines, points))

        m = re.fullmatch(r"DataSet-(\d+):(\d+)", section_name)
        frame_index, channel_id = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
        frame = cfg.get("Frame", f"Frame {frame_index}")
        name = cfg.get("Dim2Name", f"Channel {channel_id}")
        unit = cfg.get("Dim2Unit", "")

        x_unit = cfg.get("Dim0Unit", "m")
        y_unit = cfg.get("Dim1Unit", "m")
        x_raw = _float(cfg, "Dim0Range", 0.0)
        y_raw = _float(cfg, "Dim1Range", 0.0)
        fx = _length_factor_to_m(x_unit)
        fy = _length_factor_to_m(y_unit)
        x_range_m = x_raw * fx if fx is not None else 0.0
        y_range_m = y_raw * fy if fy is not None else 0.0

        if fx is None or fy is None:
            scale_status = "INVALID"
            scale_note = f"Unsupported lateral units: X={x_unit!r}, Y={y_unit!r}."
        elif x_range_m <= 0 or y_range_m <= 0:
            scale_status = "INVALID"
            scale_note = "Non-positive Dim0Range/Dim1Range lateral calibration."
        else:
            scale_status = "CALIBRATED"
            scale_note = "Dim0Range/Dim1Range converted to SI."
            if scan_range_x_m and scan_range_y_m:
                rx = abs(x_range_m - scan_range_x_m) / max(abs(scan_range_x_m), 1e-30)
                ry = abs(y_range_m - scan_range_y_m) / max(abs(scan_range_y_m), 1e-30)
                if rx <= 1e-4 and ry <= 1e-4:
                    scale_status = "VERIFIED"
                    scale_note = "Dim0/Dim1 calibration agrees with Imaging ScanRange."
                else:
                    scale_status = "CHECK"
                    scale_note = (
                        f"Dim0/Dim1 range differs from Imaging ScanRange "
                        f"(X {rx*100:.3g}%, Y {ry*100:.3g}%)."
                    )

        channels.append(
            NIDChannel(
                section=section_name,
                frame=frame,
                channel_id=channel_id,
                name=name,
                unit=unit,
                points=points,
                lines=lines,
                x_range_m=x_range_m,
                y_range_m=y_range_m,
                z_min=_float(cfg, "Dim2Min", 0.0),
                z_range=_float(cfg, "Dim2Range", 1.0),
                data=data,
                x_unit=x_unit,
                y_unit=y_unit,
                x_raw_range=x_raw,
                y_raw_range=y_raw,
                scale_status=scale_status,
                scale_note=scale_note,
            )
        )

    meta: Dict[str, str] = {}
    preferred = [
        "DataSet-Info",
        "DataSet\\DataSetInfos\\Scan",
        "DataSet\\DataSetInfos\\Approach",
        "DataSet\\DataSetInfos\\Global",
    ]
    for sec in preferred:
        for key, value in sections.get(sec, {}).items():
            if not key.startswith("SubSection"):
                meta.setdefault(key, value)

    meta["Source file"] = path.name
    meta["Binary channels"] = str(len(channels))
    meta["Binary payload offset"] = str(payload_offset)

    return NIDFile(path=path, channels=channels, metadata=meta, sections=sections,
                   source_sha256=hashlib.sha256(blob).hexdigest())
