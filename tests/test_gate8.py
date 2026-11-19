"""Gate-8 Agent safety/context, frozen cases, routers, LLM guard and benchmark."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate8"

from src.gate4.agent import StubRouter  # noqa: E402
from src.gate4.router import RouteResult  # noqa: E402
from src.gate5.kb import KBRetriever  # noqa: E402
from src.gate8.agent import Gate8Agent  # noqa: E402
from src.gate8.llm_assist import LLMAssistant  # noqa: E402
from src.gate8.routers import ROUTER_NAMES, metadata  # noqa: E402


class SeqRouter:
    def __init__(self):
        self.n = 0

    def route(self, text: str) -> RouteResult:
        c = text.lower()
        if "policy" in c or "fee" in c:
            intent = "transfer_fee_charged"
        elif "dispute" in c or "recognise" in c:
            intent = "card_payment_not_recognised"
        elif "activate" in c:
            intent = "activate_my_card"
        elif "cancel" in c:
            intent = "cancel_transfer"
        elif "card" in c:
            intent = "card_arrival"
        elif "flight" in c:
            intent = "OOS"
        else:
            intent = "card_arrival"
        self.n += 1
        accepted = intent != "OOS"
        return RouteResult("STUB", intent, 0.9 if accepted else 0.2, 0.5, accepted, [(intent, 0.9), ("x", 0.1), ("y", 0.05)], margin=0.2)


class Gate8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.kb = KBRetriever()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def agent(self, router=None):
        import uuid

        return Gate8Agent(router or SeqRouter(), session_id=f"T-{uuid.uuid4().hex[:8]}", db_dir=Path(self.tmp.name), kb=self.kb)

    def test_cases_frozen_and_disjoint(self):
        cases = [json.loads(x) for x in (DATA / "benchmark_cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertGreaterEqual(len(cases), 100)
        self.assertEqual(len({c["case_id"] for c in cases}), len(cases))
        self.assertTrue(all(c["case_id"].startswith("G8-") for c in cases))
        froze = json.loads((RESULTS / "gate8_benchmark_freeze.json").read_text(encoding="utf-8"))
        self.assertTrue(froze["frozen"])
        self.assertEqual(froze["sha256"], hashlib.sha256((DATA / "benchmark_cases.jsonl").read_bytes()).hexdigest())
        self.assertEqual(froze["n_cases"], len(cases))
        self.assertIn("not_external_blind", froze["label"])
        old = {json.loads(x)["case_id"] for x in (ROOT / "data" / "gate5" / "benchmark_cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
        self.assertFalse(old & {c["case_id"] for c in cases})

    def test_three_router_thresholds(self):
        self.assertEqual(len(ROUTER_NAMES), 3)
        for name in ROUTER_NAMES:
            m = metadata(name)
            self.assertGreater(float(m["threshold"]), 0)
            self.assertTrue(m["threshold_source"])
            self.assertTrue(m["checkpoint_sha256"])
        for tag, name in [("frozen", "Qwen-Frozen-RejectOnly"), ("plain42", "Qwen-Plain42-RejectOnly")]:
            d = pd.read_csv(RESULTS / f"gate7_dev_{tag}.csv")
            expected = float(d[d.method == "B4-RejectOnly"].tau.iloc[0])
            self.assertAlmostEqual(metadata(name)["threshold"], expected, places=12)

    def test_context_reset(self):
        a = self.agent()
        try:
            a.handle("Hi, I still have not received my card.")
            self.assertEqual(a.state.pending_slot, "card")
            a.handle("reset")
            self.assertIsNone(a.state.pending_slot)
            t = a.handle("Hi, I still have not received my card. The reference is CARD-1001.")
            self.assertEqual([x["tool"] for x in t.tools], ["get_card_delivery_status"])
        finally:
            a.close()

    def test_confirmation_correction_and_ownership(self):
        a = self.agent()
        try:
            a.handle("Please cancel my transfer TXN-2001.")
            t = a.handle("No, use TXN-2002 instead.")
            self.assertIn("TXN-2002", t.response)
            t = a.handle("yes")
            self.assertEqual([x["tool"] for x in t.tools], ["cancel_transfer"])
            self.assertEqual(next(x for x in a.snapshot()["transfers"] if x["transfer_id"] == "TXN-2002")["status"], "cancelled")
            self.assertEqual(next(x for x in a.snapshot()["transfers"] if x["transfer_id"] == "TXN-2001")["status"], "pending")
        finally:
            a.close()
        b = self.agent(StubRouter("activate_my_card"))
        try:
            t = b.handle("Please activate CARD-2001 now.")
            self.assertEqual([x["tool"] for x in t.tools], ["handoff_to_human"])
            self.assertEqual(next(x for x in b.snapshot()["cards"] if x["card_id"] == "CARD-2001")["status"], "active")
        finally:
            b.close()

    def test_topic_switch_does_not_swallow_slot(self):
        a = self.agent()
        try:
            a.handle("Hi, I still have not received my card.")
            t = a.handle("What is the policy on transfer fees?")
            self.assertEqual(t.tools, [])
            self.assertTrue(t.citations)
            self.assertIsNone(a.state.pending_slot)
        finally:
            a.close()

    def test_dispute_status_gate(self):
        a = self.agent()
        try:
            a.handle("I do not recognise a card payment. Can you dispute it?")
            t = a.handle("TXN-2003")
            self.assertEqual([x["tool"] for x in t.tools], ["get_transfer_status", "create_dispute"])
            self.assertEqual(len(a.snapshot()["disputes"]), 1)
        finally:
            a.close()
        b = self.agent()
        try:
            b.handle("I do not recognise a card payment. Can you dispute it?")
            t = b.handle("TXN-2001")
            self.assertEqual([x["tool"] for x in t.tools], ["get_transfer_status", "handoff_to_human"])
            self.assertEqual(len(b.snapshot()["disputes"]), 0)
        finally:
            b.close()

    def test_llm_schema_guard_and_blocked_status(self):
        class Bad:
            available = True

            def extract_json(self, *a, **k):
                return {"card_id": "EVIL-1"}

        assistant = LLMAssistant(client=Bad())
        with self.assertRaises(Exception):
            assistant.parse_slots("CARD-1001", ["card_arrival"])
        offline = LLMAssistant()
        self.assertFalse(offline.available)
        self.assertEqual(offline.status()["assist_mode"], "NOT RUN_API_BLOCKED")

    def test_benchmark_complete_when_run(self):
        p = RESULTS / "gate8_agent_results.csv"
        if not p.exists():
            self.skipTest("final Gate8 benchmark not run yet")
        d = pd.read_csv(p)
        self.assertEqual(len(d), 3 * 102)
        self.assertEqual(d.groupby("router").size().nunique(), 1)
        self.assertTrue((d.safe == 1).all())
        self.assertEqual(set(d.router), set(ROUTER_NAMES))

    def test_app_visible_provenance(self):
        text = (ROOT / "app_gate8.py").read_text(encoding="utf-8")
        self.assertIn("Offline Router-driven", text)
        self.assertIn("DeepSeek-assisted", text)
        self.assertIn("threshold_source", text)
        self.assertIn("Tool timeline", text)
        self.assertIn("KB citations", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
