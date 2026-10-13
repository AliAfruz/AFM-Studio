"""Bounded browser adapter for the unchanged Afruz AFM Studio v0.4.0 core."""
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import numpy as np
import scipy
import skimage
from afm_analysis import (fit_plane, line_flatten, remove_row_offsets, gaussian_smooth,
    median_filter, roughness_stats, radial_psd, autocorrelation_2d, line_profile)
from morphology_analysis import analyze_particles
from nid_reader import read_nid, NIDChannel, NIDFile, _parse_sections
from reporting import json_safe
from examples.synthetic_afm import write_synthetic_nid

MAX_BYTES = 32 * 1024 * 1024
MAX_SIDE = 1024
STATE = {}
ENGINE = {"name": "Afruz AFM Studio", "version": "0.4.0", "adapter": "1.0.0",
    "commit": "08454897c122f60b617d6909029fa94e9cb7ce2a", "doi": "10.5281/zenodo.23348914",
    "numpy": np.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__}


def number(value, low, high, label):
    result = float(value)
    if not np.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return result


def integer(value, low, high, label):
    result = number(value, low, high, label)
    if result != int(result):
        raise ValueError(label + " must be an integer.")
    return int(result)


def preflight_nid(path):
    blob = path.read_bytes()
    if not 0 < len(blob) <= MAX_BYTES:
        raise ValueError("Choose a nonempty NID scan up to 32 MB.")
    marker = blob.find(b"#!")
    if not 0 <= marker <= 262144:
        raise ValueError("The NID binary header is missing or too large.")
    _, sections = _parse_sections(blob[:marker].decode("latin-1"))
    channels = [s for name, s in sections.items() if re.fullmatch(r"DataSet-\d+:\d+", name)
                and s.get("SaveMode", "").lower() == "binary"]
    if not 1 <= len(channels) <= 32:
        raise ValueError("The browser supports one to 32 image channels.")
    total = 0
    for channel in channels:
        rows = integer(channel.get("Lines", 0), 2, MAX_SIDE, "Raster rows")
        cols = integer(channel.get("Points", 0), 2, MAX_SIDE, "Raster columns")
        total += rows * cols
        if total > 8388608:
            raise ValueError("The combined raster size exceeds the browser limit.")
        bits = integer(channel.get("SaveBits", 32), 8, 64, "Stored bits")
        if bits not in (8, 16, 32, 64):
            raise ValueError("Unsupported NID integer format.")
        if not np.isfinite(float(channel.get("Dim2Min", 0))) or not np.isfinite(float(channel.get("Dim2Range", 1))):
            raise ValueError("The NID signal calibration must be finite.")


def activate(nid, index, name=None):
    channel = nid.channels[index]
    if not np.isfinite([channel.x_range_m, channel.y_range_m]).all() or min(channel.x_range_m, channel.y_range_m) <= 0:
        raise ValueError("This channel needs valid positive lateral calibration.")
    if not np.isfinite(channel.data).any():
        raise ValueError("The scan contains no finite observations.")
    STATE.clear()
    STATE.update(nid=nid, index=index, name=name or nid.path.name, raw=channel.data.copy(),
        z=channel.data.copy(), x=channel.x_range_m, y=channel.y_range_m, origin=(0, 0),
        history=[], steps=["Raw channel"], morphology=None)


def push(label):
    STATE["history"].append((STATE["z"].copy(), STATE["x"], STATE["y"], STATE["origin"], STATE["steps"].copy()))
    STATE["history"] = STATE["history"][-8:]
    STATE["steps"].append(label)
    STATE["morphology"] = None


def raw_field():
    r, c = STATE["origin"]
    rows, cols = STATE["z"].shape
    return STATE["raw"][r:r + rows, c:c + cols]


def preview(z, limit=192):
    rr = np.linspace(0, z.shape[0] - 1, min(limit, z.shape[0])).round().astype(int)
    cc = np.linspace(0, z.shape[1] - 1, min(limit, z.shape[1])).round().astype(int)
    return z[np.ix_(rr, cc)]


def snapshot(view="processed"):
    if not STATE:
        raise ValueError("Load a scan or the synthetic demonstration first.")
    channel = STATE["nid"].channels[STATE["index"]]
    unit = "nm" if channel.unit == "m" else channel.unit or "a.u."
    factor = 1e9 if channel.unit == "m" else 1.0
    z = STATE["z"]
    display = z if view == "processed" else raw_field() if view == "raw" else z - raw_field()
    stats = roughness_stats(z)
    stats = {key: value if key in ("Ssk", "Sku") else value * factor for key, value in stats.items()}
    result = {"engine": ENGINE, "source": STATE["name"], "source_sha256": STATE["nid"].source_sha256,
        "channels": [{"label": c.label, "unit": c.unit} for c in STATE["nid"].channels],
        "channel_index": STATE["index"], "channel": channel.label, "height_unit": unit,
        "calibration": channel.scale_status, "calibration_note": channel.scale_note,
        "rows": z.shape[0], "cols": z.shape[1], "x_um": STATE["x"] * 1e6,
        "y_um": STATE["y"] * 1e6, "pitch_x_nm": STATE["x"] / z.shape[1] * 1e9,
        "pitch_y_nm": STATE["y"] / z.shape[0] * 1e9, "finite_fraction": float(np.isfinite(z).mean()),
        "grid": preview(display * factor), "roughness": stats, "steps": STATE["steps"],
        "can_undo": bool(STATE["history"]), "view": view, "height_metrology": channel.unit == "m"}
    return result


def import_file(request):
    path = Path("/tmp/afm-import.data")
    name = str(request.get("fileName", "scan.nid"))[:180]
    if name.lower().endswith(".nid"):
        preflight_nid(path)
        nid = read_nid(path)
        index = next((i for i, c in enumerate(nid.channels) if c.name == "Z-Axis" and "forward" in c.frame.lower()), 0)
    else:
        blob = path.read_bytes()
        if not 0 < len(blob) <= MAX_BYTES:
            raise ValueError("Choose a nonempty height matrix up to 32 MB.")
        text = blob.decode("utf-8-sig")
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not 2 <= len(lines) <= MAX_SIDE:
            raise ValueError("A matrix must contain two to 1024 numerical rows, without column headers.")
        delimiter = next((d for d in (";", "\t", ",") if d in lines[0]), None)
        rows = [next(csv.reader([line], delimiter=delimiter)) if delimiter else line.split() for line in lines]
        if not 2 <= len(rows[0]) <= MAX_SIDE or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Supply a rectangular matrix with two to 1024 columns.")
        units = {"nm": 1e-9, "um": 1e-6, "m": 1.0}
        unit = request.get("matrixUnit", "nm")
        if unit not in units:
            raise ValueError("Unknown matrix height unit.")
        z = np.asarray([[float(v) if v.strip() else np.nan for v in row] for row in rows]) * units[unit]
        if np.isinf(z).any():
            raise ValueError("Use NaN or blank cells for missing observations, not infinity.")
        x = number(request.get("xUm", 4), 1e-6, 1e6, "X field in µm") * 1e-6
        y = number(request.get("yUm", 4), 1e-6, 1e6, "Y field in µm") * 1e-6
        channel = NIDChannel("matrix", "Imported matrix", 0, "Height", "m", z.shape[1], z.shape[0], x, y,
            float(np.nanmin(z)), float(np.nanmax(z) - np.nanmin(z)), z,
            scale_status="USER SUPPLIED", scale_note="X/Y fields and signal units were provided by the user.")
        nid = NIDFile(path, [channel], {}, {}, hashlib.sha256(blob).hexdigest())
        index = 0
    activate(nid, index, name)
    return snapshot()


def dispatch(request):
    action = request.get("action")
    if action == "catalog":
        return {"engine": ENGINE, "max_bytes": MAX_BYTES, "max_side": MAX_SIDE}
    if action == "demo":
        path = write_synthetic_nid("/tmp/afm-synthetic.nid", points=192)
        activate(read_nid(path), 1, "Synthetic · Gaussian particles")
        return snapshot()
    if action == "import":
        return import_file(request)
    if not STATE:
        raise ValueError("Load a scan first.")
    if action == "channel":
        nid, name = STATE["nid"], STATE["name"]
        activate(nid, integer(request.get("index"), 0, len(nid.channels) - 1, "Channel"), name)
    elif action == "view":
        if request.get("view") not in ("processed", "raw", "difference"):
            raise ValueError("Unknown map view.")
        return snapshot(request["view"])
    elif action == "process":
        op = request.get("operation")
        z = STATE["z"]
        if op in ("plane", "robust"):
            updated, _ = fit_plane(z, robust=op == "robust")
        elif op == "rows":
            updated = remove_row_offsets(z)
        elif op == "line":
            updated = line_flatten(z, integer(request.get("degree", 1), 0, 5, "Polynomial degree"))
        elif op == "gaussian":
            updated = gaussian_smooth(z, number(request.get("sigma", 1), .01, 10, "Gaussian sigma"))
        elif op == "median":
            updated = median_filter(z, integer(request.get("size", 3), 1, 9, "Median window"))
        else:
            raise ValueError("Unknown preprocessing operation.")
        push({"robust": "Robust plane · Huber", "plane": "Least-squares plane", "rows": "Row offsets", "line": "Line flatten", "gaussian": "Gaussian filter", "median": "Median filter"}[op])
        STATE["z"] = updated
    elif action == "undo":
        if STATE["history"]:
            STATE["z"], STATE["x"], STATE["y"], STATE["origin"], STATE["steps"] = STATE["history"].pop()
            STATE["morphology"] = None
    elif action == "reset":
        nid, index, name = STATE["nid"], STATE["index"], STATE["name"]
        activate(nid, index, name)
    elif action == "crop":
        box = request.get("box", [])
        if len(box) != 4:
            raise ValueError("Select two corners of the crop.")
        x0, y0, x1, y1 = [number(v, 0, 1, "Crop coordinate") for v in box]
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        rows, cols = STATE["z"].shape
        c0, c1 = sorted((int(np.floor(x0 * cols)), int(np.ceil(x1 * cols))))
        r0, r1 = sorted((int(np.floor(y0 * rows)), int(np.ceil(y1 * rows))))
        c1, r1 = min(c1, cols), min(r1, rows)
        if c1 - c0 < 2 or r1 - r0 < 2:
            raise ValueError("The crop must retain at least two rows and columns.")
        push("ROI crop")
        STATE["z"] = STATE["z"][r0:r1, c0:c1].copy()
        STATE["x"] *= (c1 - c0) / cols
        STATE["y"] *= (r1 - r0) / rows
        STATE["origin"] = (STATE["origin"][0] + r0, STATE["origin"][1] + c0)
    elif action == "profile":
        points = request.get("points", [])
        if len(points) != 2 or any(len(point) != 2 for point in points):
            raise ValueError("Select two profile endpoints.")
        rows, cols = STATE["z"].shape
        p = [(np.clip(number(x, 0, 1, "Profile X") * cols - .5, 0, cols - 1),
              np.clip(number(y, 0, 1, "Profile Y") * rows - .5, 0, rows - 1)) for x, y in points]
        xs, ys, values = line_profile(STATE["z"], *p, samples=min(1024, max(2, int(np.hypot(*(np.asarray(p[1]) - p[0]))) + 1)))
        distance = np.hypot((xs - xs[0]) * STATE["x"] / cols, (ys - ys[0]) * STATE["y"] / rows) * 1e6
        factor = 1e9 if STATE["nid"].channels[STATE["index"]].unit == "m" else 1
        return {"distance_um": distance, "height": values * factor}
    elif action == "spectrum":
        channel = STATE["nid"].channels[STATE["index"]]
        z = STATE["z"] * (1e9 if channel.unit == "m" else 1)
        frequency, psd = radial_psd(z, STATE["x"] * 1e6, STATE["y"] * 1e6)
        acf = autocorrelation_2d(z)
        row = acf[acf.shape[0] // 2, acf.shape[1] // 2:]
        return {"frequency": frequency, "psd": psd, "acf_distance_um": np.arange(row.size) * STATE["x"] / z.shape[1] * 1e6,
            "acf": row, "psd_unit": f"{'nm' if channel.unit == 'm' else channel.unit or 'a.u.'}² · µm²"}
    elif action == "morphology":
        if STATE["nid"].channels[STATE["index"]].unit != "m":
            raise ValueError("Particle height metrology requires a length channel. Choose Z-Axis or a height matrix.")
        method = request.get("method", "Otsu")
        if method not in ("Otsu", "Li", "Yen", "Triangle", "Mean", "Median + kσ", "Manual"):
            raise ValueError("Unknown threshold method.")
        result = analyze_particles(STATE["z"] * 1e9, x_range=STATE["x"] * 1e9, y_range=STATE["y"] * 1e9,
            threshold_method=method, manual_threshold=number(request.get("threshold", 10), -1e12, 1e12, "Threshold in nm"),
            k_sigma=number(request.get("kSigma", 1), .01, 10, "Threshold k"),
            direction="Below threshold" if request.get("direction") == "below" else "Above threshold",
            smoothing_sigma=number(request.get("smooth", .8), 0, 5, "Segmentation sigma"),
            plane_correct=bool(request.get("plane", True)), min_area_px=integer(request.get("minArea", 8), 1, 100000, "Minimum area"),
            clear_border_objects=bool(request.get("clearBorder", True)), watershed_split=bool(request.get("watershed", True)),
            min_pixels_across=number(request.get("pixelsAcross", 3), 1, 20, "Size QC pixels across"))
        STATE["morphology"] = result
        STATE["morphology_settings"] = {k: v for k, v in request.items() if k != "action"}
        records = [r.as_dict() for r in result.records]
        return {"records": records, "summary": result.summary, "threshold_nm": result.threshold,
            "labels": preview(result.labels), "measurement_grid": preview(result.measurement_data),
            "settings": {k: v for k, v in request.items() if k != "action"}}
    elif action == "export":
        payload = snapshot()
        payload.pop("grid")
        payload["crop_origin"] = STATE["origin"]
        if STATE["morphology"] is not None:
            payload["particles"] = [r.as_dict() for r in STATE["morphology"].records]
            payload["morphology_summary"] = STATE["morphology"].summary
            payload["morphology_settings"] = STATE["morphology_settings"]
            payload["morphology_threshold_nm"] = STATE["morphology"].threshold
        if request.get("matrix"):
            payload["matrix"] = STATE["z"] * (1e9 if STATE["nid"].channels[STATE["index"]].unit == "m" else 1)
        return payload
    else:
        raise ValueError("Unknown browser action.")
    return snapshot()


def web_request(encoded):
    request = json.loads(encoded)
    if not isinstance(request, dict) or len(encoded) > 8192:
        raise ValueError("The browser request is invalid or too large.")
    return json.dumps(json_safe(dispatch(request)), allow_nan=False, separators=(",", ":"))
