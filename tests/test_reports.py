"""Interoperable export and source provenance checks."""
from pathlib import Path
import hashlib
import json
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reporting import json_safe
from nid_reader import read_nid
from sample_fixtures import get_sample_files


class ScientificReportTests(unittest.TestCase):
    def test_missing_metrics_are_standard_json_null(self):
        payload = json_safe({"metrics": [float("nan"), float("inf"), -float("inf"), 3.0],
                             "flag": np.bool_(True), "count": np.int64(7)})
        encoded = json.dumps(payload, allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["metrics"], [None, None, None, 3.0])
        self.assertIs(decoded["flag"], True)
        self.assertEqual(decoded["count"], 7)

    def test_source_hash_matches_exact_decoded_file(self):
        samples = get_sample_files()
        self.assertEqual(len(samples), 2)
        for sample in samples:
            nid = read_nid(sample)
            self.assertEqual(nid.source_sha256, hashlib.sha256(sample.read_bytes()).hexdigest())
            self.assertEqual(len(nid.channels), 6)


if __name__ == "__main__":
    unittest.main()
