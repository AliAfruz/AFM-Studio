"""Private opt-in scans or generated synthetic fixtures for portable tests."""
import os
from pathlib import Path
import tempfile

from examples.synthetic_afm import create_demo_files

_temporary = None
_samples = None


def get_sample_files():
    global _temporary, _samples
    if _samples is None:
        private_directory = os.environ.get("AFRUZ_AFM_SAMPLE_DIR")
        if private_directory:
            _samples = sorted(Path(private_directory).glob("*.nid"))[:2]
            if len(_samples) != 2:
                raise ValueError("AFRUZ_AFM_SAMPLE_DIR must contain at least two NID scans.")
        else:
            _temporary = tempfile.TemporaryDirectory(prefix="afruz-synthetic-tests-")
            _samples = create_demo_files(_temporary.name)
    return _samples
