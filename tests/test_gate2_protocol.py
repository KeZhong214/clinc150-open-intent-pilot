"""Protocol checks for the frozen Gate-2 hard-OOS split and results."""

from __future__ import annotations

import csv
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest"
RESULTS = ROOT / "results"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Gate2ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.split = read_json(MANIFEST / "gate2_hard_oos_split.json")
        cls.rows = read_csv(MANIFEST / "gate2_hard_oos_manifest.csv")
        cls.freeze = read_json(RESULTS / "gate2_freeze.json")

    def test_split_is_65_65_and_disjoint(self) -> None:
        dev = set(self.split["dev_classes"])
        final = set(self.split["final_classes"])
        known = set(self.split["known_classes"])
        self.assertEqual(len(known), 20)
        self.assertEqual(len(dev), 65)
        self.assertEqual(len(final), 65)
        self.assertFalse(dev & final)
        self.assertFalse((dev | final) & known)
        self.assertEqual(len(dev | final), 130)

    def test_hard_oos_counts(self) -> None:
        roles = Counter(row["role"] for row in self.rows)
        self.assertEqual(roles["hard_oos_val"], 1300)
        self.assertEqual(roles["hard_oos_test"], 1950)
        val_classes = {row["intent"] for row in self.rows if row["role"] == "hard_oos_val"}
        test_classes = {row["intent"] for row in self.rows if row["role"] == "hard_oos_test"}
        self.assertEqual(val_classes, set(self.split["dev_classes"]))
        self.assertEqual(test_classes, set(self.split["final_classes"]))
        self.assertEqual(len({row["sample_id"] for row in self.rows}), len(self.rows))

    def test_no_text_overlap_against_known_or_each_other(self) -> None:
        overlaps = self.split["checks"]["text_overlap"]
        self.assertTrue(all(value == 0 for value in overlaps.values()), overlaps)

    def test_b2v2_thresholds_never_lower_than_global(self) -> None:
        table = read_csv(RESULTS / "gate2_b2v2_class_thresholds.csv")
        tau = float(self.freeze["thresholds"]["B2_v2"]["tau"])
        self.assertEqual(len(table), 20)
        for row in table:
            self.assertGreaterEqual(float(row["tau_c"]), tau - 1e-9)
            self.assertLessEqual(
                float(row["tau_c"]),
                max(tau, float(row["q_c"])) + 1e-9,
            )

    def test_frozen_test_was_evaluated_once(self) -> None:
        self.assertTrue(self.freeze["frozen"])
        self.assertTrue(self.freeze["hard_oos_test_evaluated"])
        metrics = read_csv(RESULTS / "gate2_test_metrics.csv")
        self.assertEqual({row["method"] for row in metrics}, {"B0", "B2", "B2_v2"})
        for row in metrics:
            self.assertEqual(int(row["n_known"]), 600)
            self.assertEqual(int(row["n_hard_oos"]), 1950)


if __name__ == "__main__":
    unittest.main(verbosity=2)
