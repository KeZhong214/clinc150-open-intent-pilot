"""Gate 8.2 A: v1 offline rescore and FAQ grounding audit tests."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate8_1"


class Gate82Test(unittest.TestCase):
    def test_v1_source_and_diff(self):
        rows = [json.loads(x) for x in (DATA / "benchmark_cases_v1.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(rows), 120)
        self.assertEqual(rows[0]["case_id"], "G81-C001")
        f009 = next(x for x in rows if x["case_id"] == "G81-F009")
        self.assertEqual(f009["turns"][0], "What happens if a top-up fails?")
        v2 = [json.loads(x) for x in (DATA / "benchmark_cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        g2 = next(x for x in v2 if x["case_id"] == "G81-F009")
        self.assertNotEqual(f009["turns"][0], g2["turns"][0])
        diff = pd.read_csv(RESULTS / "gate8_2_v1_v2_diff.csv")
        self.assertEqual(len(diff), 13)
        self.assertTrue(((diff.case_id == "G81-F009") & (diff.field == "turns")).any())

    def test_v1_freeze_discrepancy_documented(self):
        m = json.loads((RESULTS / "gate8_2_v1_source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["v1_commit"], "7824bb70d8b41e395cefee0a108b7060557180ef")
        self.assertNotEqual(m["v1_case_sha256"], m["freeze_declared_case_sha256"])
        self.assertIn("discrepancy", m)

    def test_rescore_per_case_and_summary(self):
        d = pd.read_csv(RESULTS / "gate8_2_v1_rescore_per_case.csv")
        self.assertEqual(len(d), 360)
        self.assertEqual(d.groupby("model").size().nunique(), 1)
        s = pd.read_csv(RESULTS / "gate8_2_v1_rescore_summary.csv").set_index("model")
        self.assertAlmostEqual(float(s.loc["GTE-Frozen-RejectOnly", "v1_original_tsr"]), 0.575, places=3)
        self.assertAlmostEqual(float(s.loc["GTE-Frozen-RejectOnly", "v1_corrected_tsr"]), 0.600, places=3)
        self.assertAlmostEqual(float(s.loc["Qwen-Plain42-RejectOnly", "v1_corrected_tsr"]), 0.708333, places=6)
        # No v2 result may be used to fill a v1 score.
        self.assertTrue((d.v1_corrected_success == d.v1_original_success).sum() >= 300)

    def test_faq_audit_and_f009(self):
        d = pd.read_csv(RESULTS / "gate8_2_faq_grounding_audit.csv")
        self.assertEqual(len(d), 36)
        f9 = d[d.case_id == "G81-F009"]
        self.assertEqual(len(f9), 3)
        self.assertFalse(f9.grounding_ok.any())
        self.assertFalse(f9.v1_corrected_success.any())
        # F001/F008/F011 are the only scoring-bug corrections, and only for GTE/Qwen Frozen.
        corr = d[d.v1_corrected_success & ~d.v1_original_success]
        self.assertEqual(set(corr.case_id), {"G81-F001", "G81-F008", "G81-F011"})

    def test_rubric_has_12_real_cases_and_docs(self):
        r = json.loads((ROOT / "configs" / "gate8_2_faq_rubric.json").read_text(encoding="utf-8"))
        self.assertEqual(len(r["cases"]), 12)
        kb = [json.loads(x) for x in (ROOT / "knowledge_base" / "fictional_bank" / "policies.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        ids = {x["doc_id"] for x in kb}
        allowed = {d for c in r["cases"].values() for d in c["allowed_doc_ids"]}
        self.assertTrue(allowed <= ids)
        self.assertEqual(r["cases"]["G81-F009"]["allowed_doc_ids"], ["KB-016"])

    def test_old_v1_files_unchanged(self):
        for name in ["gate8_1_agent_gte_frozen_rejectonly_v1_results.csv", "gate8_1_agent_qwen_frozen_rejectonly_v1_results.csv", "gate8_1_agent_qwen_plain42_rejectonly_v1_results.csv"]:
            p = RESULTS / name
            self.assertTrue(p.exists())
            self.assertTrue(hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
