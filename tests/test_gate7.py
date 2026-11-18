"""Gate-7 development split, mechanism and multi-seed stability tests."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest"
RESULTS = ROOT / "results"


class Gate7Test(unittest.TestCase):
    def test_dev_split_disjoint_and_frozen(self) -> None:
        thr = pd.read_csv(MANIFEST / "gate7_dev_threshold.csv")
        sel = pd.read_csv(MANIFEST / "gate7_dev_select.csv")
        info = json.loads((RESULTS / "gate7_dev_split.json").read_text(encoding="utf-8"))
        self.assertEqual(len(thr), 760)
        self.assertEqual(len(sel), 760)
        self.assertEqual(int((thr.role == "calibration_known").sum()), 400)
        self.assertEqual(int((sel.role == "dev_unknown").sum()), 360)
        self.assertEqual(set(thr.sample_id).isdisjoint(set(sel.sample_id)), True)
        thr_classes = {x for x in thr[thr.role == "dev_unknown"].intent.unique()}
        sel_classes = {x for x in sel[sel.role == "dev_unknown"].intent.unique()}
        self.assertEqual(len(thr_classes), 9)
        self.assertEqual(len(sel_classes), 9)
        self.assertTrue(thr_classes.isdisjoint(sel_classes))
        for name in ("dev_threshold", "dev_select"):
            p = ROOT / info[name]["path"]
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), info[name]["sha256"])
        self.assertEqual(info["checks"]["sample_id_overlap"], 0)
        self.assertEqual(info["checks"]["text_sha256_overlap"], 0)
        self.assertNotIn("known_test", set(thr.role) | set(sel.role))
        self.assertNotIn("final_unknown", set(thr.role) | set(sel.role))

    def test_diagnosis_metrics_exist(self) -> None:
        d = pd.read_csv(RESULTS / "gate7_dev_diagnosis.csv")
        self.assertEqual(len(d), 8)
        plain = d[(d.encoder == "qwen_plain") & (d.method == "B3-H")].iloc[0]
        frozen = d[(d.encoder == "qwen_frozen") & (d.method == "B3-H")].iloc[0]
        self.assertGreater(float(plain.select_open_set_macro_f1), float(frozen.select_open_set_macro_f1))
        self.assertGreater(float(plain.select_oos_recall), float(frozen.select_oos_recall))

    def test_toy_batch_is_6x6_not_48x48(self) -> None:
        t = json.loads((RESULTS / "gate7_toy_batch.json").read_text(encoding="utf-8"))
        self.assertEqual(t["microbatch_similarity_shape"], [6, 6])
        self.assertEqual(t["grad_accum_8_hypothetical_shape"], [48, 48])
        self.assertEqual(t["leftover_microbatches"], 0)
        self.assertIn("never 48x48", t["actual_current_code"])

    def test_hard_negative_risk(self) -> None:
        h = json.loads((RESULTS / "gate7_hard_negative_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(h["n"], 400)
        self.assertGreater(h["margin_negative_count"], 200)
        self.assertGreater(h["duplicate_hard_negative_count"], 200)
        self.assertLess(h["margin_mean"], 0.0)

    def test_seed_control_files(self) -> None:
        # Filled after B; ensures each trained seed has an independent adapter.
        if (RESULTS / "gate7_seed_runs.csv").exists():
            d = pd.read_csv(RESULTS / "gate7_seed_runs.csv")
            self.assertTrue({"seed", "tag", "adapter_sha256", "epochs"}.issubset(d.columns))
            self.assertGreaterEqual(d.seed.nunique(), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
