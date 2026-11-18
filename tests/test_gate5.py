"""Gate-5 protocol, RAG, Agent isolation, LoRA and benchmark checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

from src.gate4.agent import StubRouter  # noqa: E402
from src.gate5.agent_v2 import BankAgentV2  # noqa: E402
from src.gate5.kb import KBRetriever  # noqa: E402
from src.gate5.llm_client import LLMClient  # noqa: E402


class Gate5Test(unittest.TestCase):
    def test_policy_kb_has_25_docs_and_citations(self) -> None:
        kb = KBRetriever()
        self.assertGreaterEqual(len(kb.docs), 25)
        self.assertEqual(len({d.doc_id for d in kb.docs}), len(kb.docs))
        hits = kb.search("What is the policy on transfer fees?", k=2)
        self.assertTrue(hits)
        self.assertTrue(hits[0]["citation"].startswith("KB-"))

    def test_natural_clarification_does_not_leak_internal_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = BankAgentV2(StubRouter("card_arrival", margin=0.0), session_id="clarify", db_dir=Path(tmp))
            turn = agent.handle("I have a problem with my card CARD-1001.")
            self.assertIn("not sure", turn.response.lower())
            self.assertNotIn("card_arrival", turn.response)
            self.assertNotIn("activate_my_card", turn.response)
            agent.close()

    def test_policy_fallback_without_router_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = BankAgentV2(StubRouter("OOS", accepted=False), session_id="policy", db_dir=Path(tmp))
            turn = agent.handle("What is the policy on transfer fees?")
            self.assertTrue(turn.citations)
            self.assertEqual(turn.citations[0]["doc_id"], "KB-013")
            self.assertEqual(turn.tools, [])
            agent.close()

    def test_session_isolation_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = BankAgentV2(StubRouter("cancel_transfer"), session_id="s-a", db_dir=Path(tmp))
            b = BankAgentV2(StubRouter("cancel_transfer"), session_id="s-b", db_dir=Path(tmp))
            a.handle("Please cancel TXN-2002.")
            a.handle("yes")
            self.assertEqual(next(t for t in a.snapshot()["transfers"] if t["transfer_id"] == "TXN-2002")["status"], "cancelled")
            self.assertEqual(next(t for t in b.snapshot()["transfers"] if t["transfer_id"] == "TXN-2002")["status"], "pending")
            a.close()
            b.close()

    def test_lora_adapter_and_vector_check(self) -> None:
        training = json.loads((RESULTS / "gate5_lora_training.json").read_text(encoding="utf-8"))
        self.assertEqual(training["epochs"], 2)
        self.assertEqual(training["params"]["trainable"], 1689600)
        self.assertTrue((RESULTS / "gate5_lora_adapter" / "adapter_model.safetensors").exists())
        self.assertGreater(training["vector_check"]["frozen_vs_lora_mean_abs"], 0.0)
        self.assertLess(training["vector_check"]["reload_max_abs"], 1e-6)
        self.assertLess(training["frozen_alignment"]["max_abs"], 0.01)

    def test_method_matrix_frozen_and_test_label(self) -> None:
        val = pd.read_csv(RESULTS / "gate5_method_matrix_val.csv")
        test = pd.read_csv(RESULTS / "gate5_method_matrix_historical_test.csv")
        self.assertEqual(len(val), 9)
        self.assertEqual(len(test), 9)
        self.assertTrue(test["historical_exploratory"].all())
        freeze = json.loads((RESULTS / "gate5_method_freeze.json").read_text(encoding="utf-8"))
        self.assertTrue(freeze["frozen"])
        self.assertTrue(freeze["test_evaluated"])

    def test_benchmark_120_and_three_routers(self) -> None:
        cases = [json.loads(x) for x in (ROOT / "data" / "gate5" / "benchmark_cases.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(cases), 120)
        self.assertTrue(all(isinstance(c["turns"], list) for c in cases))
        res = pd.read_csv(RESULTS / "gate5_benchmark_results.csv")
        self.assertEqual(len(res), 360)
        self.assertEqual(set(res["router"]), {"Frozen-B3H", "Frozen-B4Full", "LoRA-B4Full"})
        self.assertTrue((res["safe"] == 1).all())
        self.assertIn("arguments_correct", res.columns)
        self.assertTrue((ROOT / "results" / "gate5_benchmark_freeze.json").exists())

    def test_llm_api_blocked_is_explicit(self) -> None:
        status = LLMClient().status()
        payload = json.loads((RESULTS / "gate5_llm_api_results.json").read_text(encoding="utf-8"))
        if not status["available"]:
            self.assertEqual(payload["status"], "NOT RUN_API_BLOCKED")
            self.assertEqual(payload["n_completed"], 0)
        else:
            self.assertIn(payload["status"], {"COMPLETED", "NOT RUN_API_BLOCKED"})

    def test_reject_only_checkpoint_is_fixed_gate(self) -> None:
        payload = torch.load(RESULTS / "gate5_reject_only_frozen.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["meta"]["gate"], "fixed_0.5")
        self.assertEqual(payload["meta"]["encoder"], "frozen")


if __name__ == "__main__":
    unittest.main(verbosity=2)
