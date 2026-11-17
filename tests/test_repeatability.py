"""Repeatability check: regenerate manifests and compare frozen artefacts."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_data import main as prepare_main  # noqa: E402


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def without_timestamps(payload):
    if isinstance(payload, dict):
        return {
            key: without_timestamps(value)
            for key, value in payload.items()
            if key not in {"generated_at_utc", "frozen_at_utc", "retrieved_at_utc"}
        }
    if isinstance(payload, list):
        return [without_timestamps(value) for value in payload]
    return payload


class Gate0RepeatabilityTest(unittest.TestCase):
    def test_regeneration_matches_support_eval_and_selection(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="gate0_repeat_", dir=PROJECT_ROOT / "data"
        ) as tmp:
            tmp_path = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = prepare_main(
                    [
                        "--data-file",
                        str(RAW_DIR / "data_full.json"),
                        "--domains-file",
                        str(RAW_DIR / "domains.json"),
                        "--out-dir",
                        str(tmp_path),
                        "--seed",
                        "42",
                        "--known-classes",
                        "20",
                        "--per-domain",
                        "2",
                    ]
                )
            self.assertEqual(exit_code, 0)

            for name in ("support_manifest.csv", "eval_manifest.csv", "split_counts.csv"):
                self.assertEqual(
                    (tmp_path / name).read_bytes(),
                    (MANIFEST_DIR / name).read_bytes(),
                    f"{name} is not byte-for-byte reproducible",
                )

            for name in ("selected_intents.json", "source_manifest.json", "leakage_report.json"):
                self.assertEqual(
                    without_timestamps(read_json(tmp_path / name)),
                    without_timestamps(read_json(MANIFEST_DIR / name)),
                    f"{name} differs after removing timestamps",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
