"""Check that the human-readable pilot config matches frozen Gate-0 artefacts."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"

import sys

sys.path.insert(0, str(PROJECT_ROOT))


class Gate0ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (PROJECT_ROOT / "configs" / "pilot.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            cls.cfg = yaml.safe_load(handle)
        import json

        with (MANIFEST_DIR / "selected_intents.json").open(
            "r", encoding="utf-8"
        ) as handle:
            cls.selected = json.load(handle)
        with (MANIFEST_DIR / "source_manifest.json").open(
            "r", encoding="utf-8"
        ) as handle:
            cls.source = json.load(handle)

    def test_selection_matches_manifest(self) -> None:
        cfg = self.cfg["selection"]
        self.assertEqual(cfg["seed"], self.selected["selection"]["seed"])
        self.assertEqual(cfg["known_classes"], self.selected["selection"]["known_classes"])
        self.assertEqual(cfg["per_domain"], self.selected["selection"]["per_domain"])

    def test_data_revision_and_counts_match(self) -> None:
        cfg = self.cfg["data"]
        self.assertEqual(cfg["revision"], self.selected["dataset"]["revision"])
        self.assertEqual(cfg["data_full_sha256"], self.source["dataset"]["data_file"]["sha256"])
        self.assertEqual(cfg["expected_counts"], self.source["counts"])

    def test_support_and_eval_protocol_match(self) -> None:
        cfg_support = self.cfg["support"]
        self.assertEqual(cfg_support["seeds"], self.selected["support"]["seeds"])
        self.assertEqual(cfg_support["shots"], self.selected["support"]["shots"])
        cfg_eval = self.cfg["evaluation"]
        self.assertEqual(cfg_eval["known_val"], self.selected["evaluation"]["known_val"])
        self.assertEqual(cfg_eval["known_test"], self.selected["evaluation"]["known_test"])
        self.assertEqual(cfg_eval["oos_val"], self.selected["evaluation"]["oos_val"])
        self.assertEqual(cfg_eval["oos_test"], self.selected["evaluation"]["oos_test"])

    def test_qwen_prompt_hash_is_recorded(self) -> None:
        prompt = self.cfg["models"]["qwen"]["prompt"]
        expected = self.cfg["models"]["qwen"]["prompt_sha256"]
        self.assertEqual(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), expected)
        self.assertTrue(prompt.startswith("Instruct: "))
        self.assertIn("\nQuery: ", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
