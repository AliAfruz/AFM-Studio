from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from afm_analysis import ALGORITHM_VERSION, fit_plane, gaussian_smooth, line_flatten, median_filter, remove_row_offsets, roughness_stats
from morphology_analysis import analyze_particles
from nid_reader import NIDChannel, NIDFile, read_nid
from smart_analysis import smart_morphology_recommendation


@dataclass
class BatchSettings:
    channel_mode: str = "Auto · Forward Z-Axis"
    match_channel_name: str = ""
    match_channel_frame: str = ""
    # Ordered preprocessing engine. Each task is a dictionary such as
    # {"task": "plane_level"} or {"task": "gaussian", "sigma": 1.0}.
    # The list order is the execution order.
    preprocessing_tasks: List[Dict[str, Any]] | None = None
    # Legacy fields are retained so older saved calls/scripts still work. They
    # are used only when preprocessing_tasks is empty.
    plane_level: bool = True
    row_offsets: bool = False
    line_flatten: bool = False
    line_flatten_degree: int = 1
    smart_morphology_per_file: bool = False
    morphology: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    source_path: str
    file_name: str
    status: str
    error: str = ""
    scan: Dict[str, Any] = field(default_factory=dict)
    roughness: Dict[str, Any] = field(default_factory=dict)
    morphology: Dict[str, Any] = field(default_factory=dict)
    particles: List[Dict[str, Any]] = field(default_factory=list)
    morphology_settings: Dict[str, Any] = field(default_factory=dict)


def select_channel(nid: NIDFile, settings: BatchSettings) -> NIDChannel:
    channels = list(nid.channels)
    if not channels:
        raise ValueError("No image channels were found in the NID file.")

    mode = (settings.channel_mode or "").lower()
    if mode.startswith("match current"):
        if not settings.match_channel_name:
            raise ValueError("Open a scan before matching its channel, or select Auto channel mode.")
        exact = [
            ch for ch in channels
            if ch.name == settings.match_channel_name and ch.frame == settings.match_channel_frame
        ]
        if exact:
            return exact[0]
        same_name = [ch for ch in channels if ch.name == settings.match_channel_name]
        if same_name and not settings.match_channel_frame:
            return same_name[0]
        raise ValueError(f"Requested channel/frame is missing: {settings.match_channel_frame} — {settings.match_channel_name}")

    if "backward" in mode:
        for ch in channels:
            if "backward" in ch.frame.lower() and ch.name.lower() == "z-axis":
                return ch
        raise ValueError("Requested backward Z-Axis channel is missing.")

    # Default/auto: prefer the calibrated forward Z-Axis topography channel.
    for ch in channels:
        if "forward" in ch.frame.lower() and ch.name.lower() == "z-axis":
            return ch
    for ch in channels:
        if ch.name.lower() == "z-axis":
            return ch
    for ch in channels:
        if "z-axis" in ch.name.lower() and "sensor" not in ch.name.lower():
            return ch
    return channels[0]



def normalized_preprocessing_tasks(settings: BatchSettings) -> List[Dict[str, Any]]:
    """Return an ordered, validated preprocessing recipe.

    v0.3.5 introduces an explicit task engine. If no task list is supplied,
    legacy v0.3.4 boolean settings are converted to the equivalent recipe.
    """
    if settings.preprocessing_tasks is not None:
        tasks = [dict(t) for t in settings.preprocessing_tasks if t and t.get("task")]
    else:
        tasks = []
        if settings.plane_level:
            tasks.append({"task": "plane_level"})
        if settings.row_offsets:
            tasks.append({"task": "row_offsets"})
        if settings.line_flatten:
            tasks.append({"task": "line_flatten", "degree": int(settings.line_flatten_degree)})

    out: List[Dict[str, Any]] = []
    for task in tasks:
        code = str(task.get("task", "")).strip().lower()
        if code in {"plane_level", "robust_plane"}:
            out.append({"task": code})
        elif code == "row_offsets":
            out.append({"task": code})
        elif code == "line_flatten":
            out.append({"task": code, "degree": max(0, min(5, int(task.get("degree", 1))))})
        elif code == "gaussian":
            out.append({"task": code, "sigma": max(0.01, float(task.get("sigma", 1.0)))})
        elif code == "median":
            size = max(1, int(task.get("size", 3)))
            if size % 2 == 0:
                size += 1
            out.append({"task": code, "size": size})
    return out


def preprocessing_task_label(task: Dict[str, Any]) -> str:
    code = task.get("task")
    if code == "plane_level":
        return "Plane level · 1st order"
    if code == "robust_plane":
        return "Robust plane level · Huber"
    if code == "row_offsets":
        return "Remove row offsets"
    if code == "line_flatten":
        return f"Line flatten · degree {int(task.get('degree', 1))}"
    if code == "gaussian":
        return f"Gaussian · σ {float(task.get('sigma', 1.0)):g} px"
    if code == "median":
        return f"Median filter · {int(task.get('size', 3))} px"
    return str(code or "Unknown")


def apply_preprocessing_recipe(data: np.ndarray, settings: BatchSettings):
    """Apply the ordered recipe and return (processed_data, labels, tasks)."""
    z = np.asarray(data, dtype=float).copy()
    tasks = normalized_preprocessing_tasks(settings)
    labels: List[str] = []
    for task in tasks:
        code = task["task"]
        if code == "plane_level":
            z, _ = fit_plane(z)
        elif code == "robust_plane":
            z, _ = fit_plane(z, robust=True)
        elif code == "row_offsets":
            z = remove_row_offsets(z)
        elif code == "line_flatten":
            z = line_flatten(z, int(task["degree"]))
        elif code == "gaussian":
            z = gaussian_smooth(z, float(task["sigma"]))
        elif code == "median":
            z = median_filter(z, int(task["size"]))
        labels.append(preprocessing_task_label(task))
    return z, labels, tasks

def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _summary(values: Iterable[float], prefix: str) -> Dict[str, float]:
    arr = _finite(values)
    if arr.size == 0:
        return {
            f"Mean {prefix}": np.nan,
            f"Median {prefix}": np.nan,
            f"SD {prefix}": np.nan,
            f"Min {prefix}": np.nan,
            f"Max {prefix}": np.nan,
        }
    return {
        f"Mean {prefix}": float(np.mean(arr)),
        f"Median {prefix}": float(np.median(arr)),
        f"SD {prefix}": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
        f"Min {prefix}": float(np.min(arr)),
        f"Max {prefix}": float(np.max(arr)),
    }


def _summary_compact(values: Iterable[float], prefix: str) -> Dict[str, float]:
    arr = _finite(values)
    if arr.size == 0:
        return {
            f"Mean {prefix}": np.nan,
            f"Median {prefix}": np.nan,
            f"SD {prefix}": np.nan,
        }
    return {
        f"Mean {prefix}": float(np.mean(arr)),
        f"Median {prefix}": float(np.median(arr)),
        f"SD {prefix}": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
    }


def analyze_file(path: str | Path, settings: BatchSettings) -> BatchResult:
    path = Path(path)
    result = BatchResult(source_path=str(path), file_name=path.name, status="Error")
    try:
        nid = read_nid(path)
        ch = select_channel(nid, settings)
        if ch.unit != "m":
            raise ValueError("Batch particle-height metrology requires a length channel in metres; select Z-Axis topography.")
        if getattr(ch, "scale_status", "INVALID") == "INVALID":
            raise ValueError("Particle sizing requires valid lateral calibration.")
        data = np.asarray(ch.data, dtype=float).copy()
        data, processing, applied_tasks = apply_preprocessing_recipe(data, settings)

        rows, cols = data.shape
        x_um = float(ch.x_range_m) * 1e6
        y_um = float(ch.y_range_m) * 1e6
        dx_nm = abs(float(ch.x_range_m)) * 1e9 / max(cols, 1)
        dy_nm = abs(float(ch.y_range_m)) * 1e9 / max(rows, 1)
        z_factor = 1e9 if ch.unit == "m" else 1.0
        z_unit = "nm" if ch.unit == "m" else (ch.unit or "a.u.")
        finite = data[np.isfinite(data)]
        z_span = float(np.ptp(finite) * z_factor) if finite.size else np.nan

        result.scan = {
            "Algorithm version": ALGORITHM_VERSION,
            "Source SHA256": nid.source_sha256,
            "Source file": path.name,
            "Full path": str(path),
            "Channel": ch.label,
            "Frame": ch.frame,
            "Signal": ch.name,
            "X size (µm)": x_um,
            "Y size (µm)": y_um,
            "Pixels X": int(cols),
            "Pixels Y": int(rows),
            "Pixel size X (nm)": dx_nm,
            "Pixel size Y (nm)": dy_nm,
            "Scale status": getattr(ch, "scale_status", "UNVERIFIED"),
            "Scale source": "NID Dim0Range / Dim1Range",
            "Scale note": getattr(ch, "scale_note", ""),
            f"Z span ({z_unit})": z_span,
            "Preprocessing steps": len(applied_tasks),
            "Processing": " → ".join(processing) if processing else "Raw",
        }

        rs = roughness_stats(data)
        result.roughness = {
            "Source file": path.name,
            "Channel": ch.label,
            "Z unit": z_unit,
            f"Mean ({z_unit})": rs["Mean"] * z_factor,
            f"Sa ({z_unit})": rs["Sa"] * z_factor,
            f"Sq ({z_unit})": rs["Sq"] * z_factor,
            f"Sp ({z_unit})": rs["Sp"] * z_factor,
            f"Sv ({z_unit})": rs["Sv"] * z_factor,
            f"Sz ({z_unit})": rs["Sz"] * z_factor,
            "Ssk": rs["Ssk"],
            "Sku": rs["Sku"],
        }

        morph = dict(settings.morphology or {})
        morph.setdefault("threshold_method", "Otsu")
        morph.setdefault("manual_threshold", None)
        morph.setdefault("k_sigma", 1.0)
        morph.setdefault("direction", "Above threshold")
        morph.setdefault("smoothing_sigma", 0.8)
        morph.setdefault("plane_correct", True)
        morph.setdefault("fill_holes", True)
        morph.setdefault("opening_radius", 0)
        morph.setdefault("closing_radius", 0)
        morph.setdefault("min_area_px", 8)
        morph.setdefault("max_area_px", 0)
        morph.setdefault("clear_border_objects", True)
        morph.setdefault("watershed_split", True)
        morph.setdefault("watershed_min_distance", 4)
        # v0.3.6 calibrated size-metrology settings. Lateral filter values are
        # stored in nm in the GUI/batch protocol and converted to SI here.
        morph.setdefault("min_equivalent_diameter_nm", 0.0)
        morph.setdefault("max_equivalent_diameter_nm", 0.0)
        morph.setdefault("min_pixels_across", 3.0)
        morph.setdefault("background_mode", "Local plane")
        morph.setdefault("background_ring_px", 4)
        morph.setdefault("exclude_underresolved_from_summary", True)

        if settings.smart_morphology_per_file:
            rec = smart_morphology_recommendation(
                data,
                x_range=ch.x_range_m,
                y_range=ch.y_range_m,
                min_area_px=int(morph.get("min_area_px", 8)),
            )
            morph["threshold_method"] = rec["method"]
            morph["direction"] = rec["direction"]
            morph["k_sigma"] = rec["k_sigma"]
            morph["smoothing_sigma"] = rec["smoothing_sigma"]
            morph["watershed_min_distance"] = rec["watershed_distance"]
            morph["manual_threshold"] = None
            morph["smart_confidence"] = rec.get("confidence", "")
            morph["smart_score"] = rec.get("score", np.nan)

        analysis_kwargs = {k: v for k, v in morph.items() if k in {
            "threshold_method", "manual_threshold", "k_sigma", "direction", "smoothing_sigma",
            "plane_correct", "fill_holes", "opening_radius", "closing_radius", "min_area_px",
            "max_area_px", "clear_border_objects", "watershed_split", "watershed_min_distance",
            "min_pixels_across", "background_mode", "background_ring_px",
            "exclude_underresolved_from_summary",
        }}
        analysis_kwargs["min_equivalent_diameter"] = max(0.0, float(morph.get("min_equivalent_diameter_nm", 0.0))) * 1e-9
        analysis_kwargs["max_equivalent_diameter"] = max(0.0, float(morph.get("max_equivalent_diameter_nm", 0.0))) * 1e-9
        mr = analyze_particles(
            data,
            x_range=float(ch.x_range_m),
            y_range=float(ch.y_range_m),
            **analysis_kwargs,
        )
        result.morphology_settings = dict(morph)

        scan_area_um2 = abs(float(ch.x_range_m) * float(ch.y_range_m)) * 1e12
        density = len(mr.records) / scan_area_um2 if scan_area_um2 > 0 else np.nan

        volume_unit = "nm³" if z_unit == "nm" else f"{z_unit}·nm²"
        particles: List[Dict[str, Any]] = []
        for rec in mr.records:
            particles.append({
                "Source file": path.name,
                "Channel": ch.label,
                "Particle ID": int(rec.particle_id),
                "X (µm)": float(rec.centroid_x) * 1e6,
                "Y (µm)": float(rec.centroid_y) * 1e6,
                "Area (nm²)": float(rec.area) * 1e18,
                "Apparent equivalent diameter (nm)": float(rec.equivalent_diameter) * 1e9,
                "Eq diameter low · 1 px band (nm)": float(rec.equivalent_diameter_low) * 1e9,
                "Eq diameter high · 1 px band (nm)": float(rec.equivalent_diameter_high) * 1e9,
                "Apparent Feret max (nm)": float(rec.feret_max) * 1e9,
                "Apparent Feret min (nm)": float(rec.feret_min) * 1e9,
                "Perimeter (nm)": float(rec.perimeter) * 1e9,
                "Pixels across eq diameter": float(rec.pixels_across),
                "Size QC": rec.size_qc_note,
                "Touches missing data": rec.touches_missing_data,
                "Size QC note": rec.size_qc_note,
                f"P95 height ({z_unit})": float(rec.p95_height) * z_factor,
                f"Max height ({z_unit})": float(rec.max_height) * z_factor,
                f"Mean height ({z_unit})": float(rec.mean_height) * z_factor,
                f"Volume ({volume_unit})": float(rec.volume) * 1e18 * z_factor,
                "Circularity": float(rec.circularity),
                "Aspect ratio": float(rec.aspect_ratio),
                "Orientation (°)": float(rec.orientation_deg),
                "Pixels": int(rec.pixel_count),
            })
        result.particles = particles

        use_qc_only = bool(morph.get("exclude_underresolved_from_summary", True))
        sized_particles = [p for p in particles if (p["Size QC"] == "PASS" or not use_qc_only)]
        eq = [p["Apparent equivalent diameter (nm)"] for p in sized_particles]
        feret_max = [p["Apparent Feret max (nm)"] for p in sized_particles]
        feret_min = [p["Apparent Feret min (nm)"] for p in sized_particles]
        areas = [p["Area (nm²)"] for p in sized_particles]
        heights = [p[f"P95 height ({z_unit})"] for p in sized_particles]
        max_heights = [p[f"Max height ({z_unit})"] for p in sized_particles]
        volumes = [p[f"Volume ({volume_unit})"] for p in sized_particles]
        circularity = [p["Circularity"] for p in sized_particles]
        aspect = [p["Aspect ratio"] for p in sized_particles]

        ms: Dict[str, Any] = {
            "Source file": path.name,
            "Channel": ch.label,
            "Particle count": len(particles),
            "Size-QC particle count": len([p for p in particles if p["Size QC"] == "PASS"]),
            "Size-QC used for summary": use_qc_only,
            "Particle density (/µm²)": density,
            "Coverage (%)": float(mr.summary.get("coverage_percent", np.nan)),
            "Threshold method": morph.get("threshold_method", ""),
            "Threshold value (raw Z unit)": float(mr.threshold),
            "Direction": morph.get("direction", ""),
            "Z unit": z_unit,
            "Pixel pitch X (nm)": dx_nm,
            "Pixel pitch Y (nm)": dy_nm,
            "Minimum lateral size for QC (nm)": float(mr.summary.get("minimum_qc_lateral_size", np.nan)) * 1e9,
            "Scale status": getattr(ch, "scale_status", "UNVERIFIED"),
            "Lateral-size interpretation": "Apparent AFM lateral size; tip convolution not corrected",
        }
        ms.update(_summary_compact(eq, "apparent equivalent diameter (nm)"))
        ms["Mean apparent Feret max (nm)"] = float(np.nanmean(_finite(feret_max))) if _finite(feret_max).size else np.nan
        ms["Mean apparent Feret min (nm)"] = float(np.nanmean(_finite(feret_min))) if _finite(feret_min).size else np.nan
        ms.update(_summary_compact(areas, "area (nm²)"))
        ms.update(_summary_compact(heights, f"P95 height ({z_unit})"))
        ms["Mean max height (" + z_unit + ")"] = float(np.nanmean(_finite(max_heights))) if _finite(max_heights).size else np.nan
        ms.update(_summary_compact(volumes, f"volume ({volume_unit})"))
        ms["Mean circularity"] = float(np.nanmean(_finite(circularity))) if _finite(circularity).size else np.nan
        ms["Mean aspect ratio"] = float(np.nanmean(_finite(aspect))) if _finite(aspect).size else np.nan
        result.morphology = ms
        result.status = "Done"
        return result
    except Exception as exc:
        result.error = str(exc)
        result.status = "Error"
        return result


def _safe_excel_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (list, tuple, dict)):
        return str(value)
    return value


def _style_table_sheet(ws, headers: List[str], row_count: int, table_name: str) -> None:
    navy = "1F4E78"
    pale = "D9EAF7"
    border_color = "D7DEE8"
    thin = Side(style="thin", color=border_color)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=navy)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=thin)
    ws.row_dimensions[1].height = 32

    if row_count > 0 and headers:
        end_col = get_column_letter(len(headers))
        ref = f"A1:{end_col}{row_count + 1}"
        tab = Table(displayName=table_name, ref=ref)
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
            showRowStripes=True, showColumnStripes=False,
        )
        ws.add_table(tab)

    for idx, header in enumerate(headers, start=1):
        values = [str(ws.cell(r, idx).value or "") for r in range(1, min(ws.max_row, 300) + 1)]
        longest = max([len(header)] + [len(v) for v in values])
        width = min(34, max(10, longest + 2))
        if "file" in header.lower() or "channel" in header.lower() or "processing" in header.lower():
            width = min(44, max(width, 22))
        ws.column_dimensions[get_column_letter(idx)].width = width

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="center")
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"
    ws.sheet_view.showGridLines = False


def _write_records_sheet(wb: Workbook, title: str, records: List[Dict[str, Any]], table_name: str) -> None:
    ws = wb.create_sheet(title)
    if not records:
        ws.append(["No data"])
        ws["A1"].font = Font(bold=True)
        return
    headers: List[str] = []
    seen = set()
    for rec in records:
        for key in rec.keys():
            if key not in seen:
                headers.append(key)
                seen.add(key)
    ws.append(headers)
    for rec in records:
        ws.append([_safe_excel_value(rec.get(h)) for h in headers])
    _style_table_sheet(ws, headers, len(records), table_name)


def export_batch_workbook(results: List[BatchResult], settings: BatchSettings, path: str | Path) -> None:
    path = Path(path)
    wb = Workbook()
    wb.remove(wb.active)

    successful = [r for r in results if r.status == "Done"]
    scan_rows = [r.scan for r in successful]
    rough_rows = [r.roughness for r in successful]

    # Keep the main morphology table deliberately compact: one clean row per file
    # with the size/morphology outputs most often transferred into manuscripts.
    morph_keys = [
        "Source file", "Channel", "Scale status", "Pixel pitch X (nm)", "Pixel pitch Y (nm)",
        "Particle count", "Size-QC particle count", "Size-QC used for summary",
        "Particle density (/µm²)", "Coverage (%)", "Minimum lateral size for QC (nm)",
        "Mean apparent equivalent diameter (nm)", "Median apparent equivalent diameter (nm)",
        "SD apparent equivalent diameter (nm)", "Mean apparent Feret max (nm)",
        "Mean apparent Feret min (nm)", "Mean area (nm²)",
        "Mean P95 height (nm)", "Median P95 height (nm)", "Mean max height (nm)",
        "Mean volume (nm³)", "Mean circularity", "Mean aspect ratio", "Lateral-size interpretation",
    ]
    morph_rows = [{k: r.morphology.get(k) for k in morph_keys} for r in successful]

    segmentation_rows = []
    for r in successful:
        z_unit = r.morphology.get("Z unit", "")
        ms = r.morphology_settings
        segmentation_rows.append({
            "Source file": r.file_name,
            "Channel": r.morphology.get("Channel", ""),
            "Threshold method": r.morphology.get("Threshold method", ""),
            "Threshold value (raw Z unit)": r.morphology.get("Threshold value (raw Z unit)"),
            "Z unit": z_unit,
            "Direction": r.morphology.get("Direction", ""),
            "Smoothing sigma (px)": ms.get("smoothing_sigma"),
            "Plane correction": ms.get("plane_correct"),
            "Minimum area (px)": ms.get("min_area_px"),
            "Maximum area (px; 0=no limit)": ms.get("max_area_px"),
            "Fill holes": ms.get("fill_holes"),
            "Opening radius (px)": ms.get("opening_radius"),
            "Closing radius (px)": ms.get("closing_radius"),
            "Watershed": ms.get("watershed_split"),
            "Watershed distance (px)": ms.get("watershed_min_distance"),
            "Exclude border objects": ms.get("clear_border_objects"),
            "Min apparent diameter (nm)": ms.get("min_equivalent_diameter_nm"),
            "Max apparent diameter (nm; 0=no limit)": ms.get("max_equivalent_diameter_nm"),
            "Minimum pixels across for size QC": ms.get("min_pixels_across"),
            "Local height background": ms.get("background_mode"),
            "Background ring (px)": ms.get("background_ring_px"),
            "Exclude under-resolved from size summary": ms.get("exclude_underresolved_from_summary"),
            "Smart confidence": ms.get("smart_confidence", ""),
            "Smart score": ms.get("smart_score"),
        })

    particle_rows = [p for r in successful for p in r.particles]
    error_rows = [
        {"Source file": r.file_name, "Full path": r.source_path, "Status": r.status, "Error": r.error}
        for r in results if r.status != "Done"
    ]

    _write_records_sheet(wb, "Scan Summary", scan_rows, "ScanSummaryTable")
    _write_records_sheet(wb, "Roughness", rough_rows, "RoughnessTable")
    _write_records_sheet(wb, "Morphology Summary", morph_rows, "MorphologySummaryTable")
    _write_records_sheet(wb, "Particle Data", particle_rows, "ParticleDataTable")
    _write_records_sheet(wb, "Segmentation", segmentation_rows, "SegmentationTable")

    size_rows = []
    for r in successful:
        size_rows.append({
            "Source file": r.file_name,
            "Channel": r.morphology.get("Channel", ""),
            "Scale status": r.morphology.get("Scale status", ""),
            "Pixel pitch X (nm)": r.morphology.get("Pixel pitch X (nm)"),
            "Pixel pitch Y (nm)": r.morphology.get("Pixel pitch Y (nm)"),
            "Minimum lateral size for QC (nm)": r.morphology.get("Minimum lateral size for QC (nm)"),
            "Detected particles": r.morphology.get("Particle count"),
            "Size-QC particles": r.morphology.get("Size-QC particle count"),
            "Mean apparent equivalent diameter (nm)": r.morphology.get("Mean apparent equivalent diameter (nm)"),
            "Median apparent equivalent diameter (nm)": r.morphology.get("Median apparent equivalent diameter (nm)"),
            "SD apparent equivalent diameter (nm)": r.morphology.get("SD apparent equivalent diameter (nm)"),
            "Mean P95 height (nm)": r.morphology.get("Mean P95 height (nm)"),
            "Interpretation": "Lateral AFM sizes are apparent and may be broadened by the probe tip. Height uses a local background model.",
            "Boundary-band meaning": "Low/high diameter in Particle Data reflects ±1-pixel boundary sensitivity; it is not formal calibration uncertainty.",
        })
    _write_records_sheet(wb, "Size Metrology", size_rows, "SizeMetrologyTable")

    # Explicit per-file processing audit trail. The source filename is repeated
    # intentionally so every exported analytical table remains self-identifying.
    recipe = normalized_preprocessing_tasks(settings)
    pipeline_rows = []
    for r in successful:
        if recipe:
            for i, task in enumerate(recipe, start=1):
                pipeline_rows.append({
                    "Source file": r.file_name,
                    "Step": i,
                    "Task": preprocessing_task_label(task),
                })
        else:
            pipeline_rows.append({"Source file": r.file_name, "Step": 0, "Task": "Raw / no preprocessing"})
    _write_records_sheet(wb, "Processing Pipeline", pipeline_rows, "ProcessingPipelineTable")

    settings_ws = wb.create_sheet("Analysis Settings")
    settings_ws.append(["Setting", "Value"])
    recipe = normalized_preprocessing_tasks(settings)
    settings_rows = [
        ("Channel mode", settings.channel_mode),
        ("Preprocessing task count", len(recipe)),
        ("Preprocessing pipeline", " → ".join(preprocessing_task_label(t) for t in recipe) if recipe else "Raw / none"),
        ("Smart morphology per file", settings.smart_morphology_per_file),
    ]
    for i, task in enumerate(recipe, start=1):
        settings_rows.append((f"Preprocessing step {i}", preprocessing_task_label(task)))
    settings_rows.extend((f"Morphology · {k}", v) for k, v in settings.morphology.items())
    for key, value in settings_rows:
        settings_ws.append([key, _safe_excel_value(value)])
    _style_table_sheet(settings_ws, ["Setting", "Value"], len(settings_rows), "AnalysisSettingsTable")

    if error_rows:
        _write_records_sheet(wb, "Errors", error_rows, "BatchErrorsTable")

    # Make file identity immediately visible on all analytical sheets.
    for ws in wb.worksheets:
        if ws.max_column >= 1:
            ws.column_dimensions["A"].width = max(ws.column_dimensions["A"].width or 10, 24)

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
