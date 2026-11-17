"""Protocol checks for the Gate-3 frozen datasets and 72-cell matrix."""

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


class Gate3ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.all = read_json(MANIFEST / "gate3_all_datasets.json")
        cls.metrics = read_csv(RESULTS / "gate3_test_metrics.csv")
        cls.freeze = read_json(RESULTS / "gate3_freeze.json")

    def test_dataset_class_split_counts(self) -> None:
        expected = {
            "clinc150": (20, 0, 0),
            "banking77": (40, 18, 19),
            "massive": (35, 12, 13),
        }
        for dataset, (known, dev, final) in expected.items():
            info = read_json(MANIFEST / f"gate3_{dataset}_dataset.json")
            self.assertEqual(len(info["known_classes"]), known)
            self.assertEqual(len(info["dev_unknown_classes"]), dev)
            self.assertEqual(len(info["final_unknown_classes"]), final)
            self.assertEqual(info["counts"]["support"], known * 10)
            self.assertEqual(info["support_5_rows"], known * 5)

    def test_role_counts_and_text_isolation(self) -> None:
        for dataset in ("clinc150", "banking77", "massive"):
            rows = read_csv(MANIFEST / f"gate3_{dataset}_manifest.csv")
            roles = Counter(row["role"] for row in rows)
            self.assertEqual(roles["support"], len({row["intent"] for row in rows if row["role"] == "support"}) * 10)
            info = read_json(MANIFEST / f"gate3_{dataset}_dataset.json")
            for role, count in info["counts"].items():
                self.assertEqual(roles[role], count)
            self.assertTrue(all(value == 0 for value in info["text_overlap"].values()), info["text_overlap"])
            self.assertEqual(len({row["sample_id"] for row in rows}), len(rows))

    def test_full_matrix_is_72_and_frozen(self) -> None:
        self.assertEqual(len(self.metrics), 72)
        keys = {(row["dataset"], row["model"], row["shot"], row["method"]) for row in self.metrics}
        self.assertEqual(
            len(keys),
            3 * 4 * 2 * 3,
        )
        self.assertTrue(self.freeze["frozen"])
        self.assertTrue(self.freeze["test_evaluated"])
        self.assertEqual(self.freeze["threshold_configs"], 72)

    def test_training_files_exist(self) -> None:
        for name in (
            "gate3_setfit_metrics.csv",
            "gate3_setfit_training.json",
            "gate3_adapter_metrics.csv",
            "gate3_adapter_training.csv",
        ):
            self.assertTrue((RESULTS / name).exists(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
