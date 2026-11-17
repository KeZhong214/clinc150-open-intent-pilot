"""Protocol checks for the frozen Gate-0 manifests.

Run from the project root:
    python -m unittest discover -s tests -v

These tests do not need a model or network access.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_data import (  # noqa: E402
    EXPECTED_COUNTS,
    load_and_validate_data,
    validate_domains,
    select_intents,
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Gate0ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = [
            MANIFEST_DIR / "selected_intents.json",
            MANIFEST_DIR / "source_manifest.json",
            MANIFEST_DIR / "support_manifest.csv",
            MANIFEST_DIR / "eval_manifest.csv",
            MANIFEST_DIR / "split_counts.csv",
            MANIFEST_DIR / "leakage_report.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest(f"Gate-0 manifests missing: {missing}")

        cls.selected = read_json(MANIFEST_DIR / "selected_intents.json")
        cls.source = read_json(MANIFEST_DIR / "source_manifest.json")
        cls.leakage = read_json(MANIFEST_DIR / "leakage_report.json")
        cls.support = read_csv(MANIFEST_DIR / "support_manifest.csv")
        cls.eval_rows = read_csv(MANIFEST_DIR / "eval_manifest.csv")
        cls.split_counts = read_csv(MANIFEST_DIR / "split_counts.csv")
        cls.samples = load_and_validate_data(RAW_DIR / "data_full.json")

    def test_source_counts_and_hash(self) -> None:
        self.assertEqual(self.source["counts"], EXPECTED_COUNTS)
        data_path = RAW_DIR / "data_full.json"
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        self.assertEqual(digest, self.source["dataset"]["data_file"]["sha256"])
        self.assertEqual(digest, self.selected["dataset"]["data_sha256"])

    def test_20_intents_two_per_domain(self) -> None:
        intents = self.selected["intents"]
        self.assertEqual(len(intents), 20)
        self.assertEqual(len({row["intent"] for row in intents}), 20)
        domains = Counter(row["domain"] for row in intents)
        self.assertEqual(len(domains), 10)
        self.assertEqual(set(domains.values()), {2})

    def test_selection_is_reproducible_without_eval_labels(self) -> None:
        domains = validate_domains(
            RAW_DIR / "domains.json",
            {sample.label for sample in self.samples["train"]},
        )
        selected, _, by_domain = select_intents(
            domains,
            seed=42,
            per_domain=2,
            expected_total=20,
        )
        expected = {domain: sorted(values) for domain, values in by_domain.items()}
        self.assertEqual(self.selected["domains"], expected)
        self.assertEqual(selected, [row["intent"] for row in self.selected["intents"]])
        # select_intents() receives only official domains + seed; no val/test row
        # or label is passed in, so the frozen list cannot have been test-tuned.

    def test_support_counts_are_exact(self) -> None:
        grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
        for row in self.support:
            grouped[(int(row["support_seed"]), int(row["shot"]))].append(row)

        self.assertEqual(
            set(grouped),
            {(seed, shot) for seed in (42, 43, 44) for shot in (5, 10)},
        )
        for (seed, shot), rows in grouped.items():
            self.assertEqual(len(rows), 20 * shot, f"seed={seed}, shot={shot}")
            per_intent = Counter(row["intent"] for row in rows)
            self.assertEqual(set(per_intent), {row["intent"] for row in self.selected["intents"]})
            self.assertEqual(set(per_intent.values()), {shot})
            self.assertEqual(len({row["sample_id"] for row in rows}), len(rows))
            self.assertEqual({row["source_split"] for row in rows}, {"train"})

    def test_k5_is_nested_in_k10(self) -> None:
        for seed in (42, 43, 44):
            k5 = {
                row["sample_id"]
                for row in self.support
                if int(row["support_seed"]) == seed and int(row["shot"]) == 5
            }
            k10 = {
                row["sample_id"]
                for row in self.support
                if int(row["support_seed"]) == seed and int(row["shot"]) == 10
            }
            self.assertTrue(k5.issubset(k10), f"K=5 not nested in K=10 for seed {seed}")

    def test_support_has_no_oos_or_eval_ids(self) -> None:
        eval_ids = {row["sample_id"] for row in self.eval_rows}
        support_ids = {row["sample_id"] for row in self.support}
        self.assertFalse(support_ids & eval_ids)
        for row in self.support:
            self.assertTrue(row["sample_id"].startswith("clinc150:train:"))
            self.assertNotEqual(row["intent"], "oos")

    def test_support_eval_text_overlap_is_zero(self) -> None:
        eval_hashes = {row["text_norm_sha256"] for row in self.eval_rows}
        support_hashes = {row["text_norm_sha256"] for row in self.support}
        self.assertEqual(support_hashes & eval_hashes, set())

    def test_eval_counts_and_roles(self) -> None:
        roles = Counter(row["role"] for row in self.eval_rows)
        self.assertEqual(
            roles,
            {
                "calibration_known": 400,
                "final_known_test": 600,
                "calibration_oos": 100,
                "final_oos_test": 1000,
            },
        )
        self.assertEqual(len({row["sample_id"] for row in self.eval_rows}), len(self.eval_rows))
        self.assertEqual(len({row["open_set_label"] for row in self.eval_rows}), 21)
        self.assertEqual(
            sum(1 for row in self.eval_rows if row["is_oos"] == "1"),
            1100,
        )
        for row in self.eval_rows:
            if row["is_oos"] == "1":
                self.assertEqual(row["open_set_label"], "OOS")
            else:
                self.assertNotEqual(row["open_set_label"], "OOS")

    def test_split_counts_contract(self) -> None:
        rows = {row["source_split"]: row for row in self.split_counts}
        self.assertEqual(set(rows), set(EXPECTED_COUNTS))
        for split, expected in EXPECTED_COUNTS.items():
            self.assertEqual(int(rows[split]["total"]), expected)
        self.assertEqual(int(rows["train"]["selected_total"]), 2000)
        self.assertEqual(int(rows["val"]["selected_total"]), 400)
        self.assertEqual(int(rows["test"]["selected_total"]), 600)
        self.assertEqual(int(rows["oos_val"]["selected_total"]), 100)
        self.assertEqual(int(rows["oos_test"]["selected_total"]), 1000)

    def test_official_cross_split_collisions_are_reported(self) -> None:
        reported = self.leakage["official_data"]["cross_split_normalized_text"]
        self.assertEqual(reported["n_shared_hashes_total"], 5)
        pairs = {(pair["split_a"], pair["split_b"]) for pair in reported["pairs"]}
        self.assertEqual(pairs, {("train", "val"), ("train", "test")})


if __name__ == "__main__":
    unittest.main(verbosity=2)
