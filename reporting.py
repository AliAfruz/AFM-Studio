"""Standards-compliant scientific JSON: non-finite observations become null."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_report(path, payload):
    Path(path).write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
