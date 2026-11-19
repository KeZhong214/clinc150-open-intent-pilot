"""Gate 8.1: paired model registry, OOS safety, LLM evidence and new cases."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate8_1"

from src.gate4.agent import StubRouter  # noqa: E402
from src.gate5.agent_v2 import has_policy_terms  # noqa: E402
from src.gate5.kb import KBRetriever  # noqa: E402
from src.gate8.agent import Gate8Agent  # noqa: E402
from src.gate8_1.registry import validate  # noqa: E402
from src.gate8_1.routers import ROUTER_NAMES, metadata  # noqa: E402


class Gate81Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.kb = KBRetriever()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_registry_is_paired(self):
        for name in ROUTER_NAMES:
            e = validate(name)
            self.assertTrue(e["paired"])
            self.assertTrue(e["checkpoint_sha256"])
            self.assertTrue(e["threshold_file"])
        gte = metadata("GTE-Frozen-RejectOnly")
        recal = json.loads((RESULTS / "gate8_1_gte_recalibrated_threshold.json").read_text(encoding="utf-8"))
        self.assertEqual(gte["checkpoint_sha256"], recal["checkpoint_sha256"])
        self.assertAlmostEqual(gte["threshold"], recal["tau"], places=12)
        self.assertAlmostEqual(metadata("Qwen-Plain42-RejectOnly")["threshold"], 0.4168315291404724, places=12)

    def test_oos_word_boundary(self):
        self.assertFalse(has_policy_terms("Please open a cryptocurrency trading account."))
        self.assertTrue(has_policy_terms("What is the transfer fee policy?"))

    def test_oos_crypto_does_not_use_kb(self):
        a = Gate8Agent(StubRouter("OOS", accepted=False), session_id="T1", db_dir=Path(self.tmp.name), kb=self.kb)
        try:
            t = a.handle("Please open a cryptocurrency trading account.")
            self.assertEqual([x["tool"] for x in t.tools], ["handoff_to_human"])
            self.assertEqual(t.citations, [])
        finally:
            a.close()

    def test_llm_hallucinated_id_rejected(self):
        class Fake:
            available = True

            def parse_slots(self, *a, **k):
                return {"card_id": "CARD-9999"}

        a = Gate8Agent(StubRouter("card_arrival"), session_id="T2", db_dir=Path(self.tmp.name), kb=self.kb, llm_assistant=Fake())
        try:
            a.handle("Hi, I still have not received my card.")
            t = a.handle("CARD-1001")
            self.assertIn("CARD-1001", t.response)
            self.assertEqual(t.tools[0]["tool"], "get_card_delivery_status")
            self.assertEqual(t.tools[0]["args"]["card_id"], "CARD-1001")
        finally:
            a.close()

    def test_new_cases_quality(self):
        rows = [json.loads(x) for x in (DATA / "benchmark_cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertGreaterEqual(len(rows), 100)
        self.assertGreaterEqual(len({r["template_id"] for r in rows}), 60)
        self.assertEqual(len(rows), len({r["case_id"] for r in rows}))
        audit = json.loads((RESULTS / "gate8_1_case_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["exact_duplicates_against_gate5_or_gate8"], [])
        self.assertEqual(audit["exact_duplicates_within_new"], [])
        self.assertEqual(audit["n_cases"], len(rows))

    def test_freeze_and_old_results_preserved(self):
        f = json.loads((RESULTS / "gate8_1_benchmark_freeze.json").read_text(encoding="utf-8"))
        self.assertTrue(f["frozen"])
        self.assertEqual(f["sha256"], hashlib.sha256((DATA / "benchmark_cases.jsonl").read_bytes()).hexdigest())
        self.assertTrue((RESULTS / "gate8_agent_summary.csv").exists())
        self.assertTrue((RESULTS / "gate8_benchmark_freeze.json").exists())

    def test_old_gate8_artifact_untouched(self):
        old = json.loads((RESULTS / "gate8_benchmark_freeze.json").read_text(encoding="utf-8"))
        self.assertEqual(old["sha256"], "7ca2a276fc9b0be0e9c19113c4181359d31c013982d9615793df28c2079166e1")

    def test_app_switch_release_code(self):
        text = (ROOT / "app_gate8.py").read_text(encoding="utf-8")
        self.assertIn("release_sessions", text)
        self.assertIn("memory_allocated", text)
        self.assertIn("Gate 8.1", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
