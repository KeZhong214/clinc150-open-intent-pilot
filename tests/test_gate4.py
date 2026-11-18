"""Gate-4 B4 checkpoint and Agent business/tool regression tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

from src.gate4.agent import BankAgent, StubRouter  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402
from src.gate4.db import BankingTools  # noqa: E402


class Gate4AgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "agent.db"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _tools(self, session: str = "TEST") -> BankingTools:
        return BankingTools(session_id=session, path=self.path)

    def _agent(self, intent: str, *, accepted: bool = True, margin: float = 0.2) -> tuple[BankAgent, BankingTools]:
        tools = self._tools()
        return BankAgent(StubRouter(intent, accepted=accepted, margin=margin), tools), tools

    def test_fictional_seed_and_no_real_bank_data(self) -> None:
        tools = self._tools()
        snap = tools.snapshot()
        self.assertTrue(all(row["customer_id"].startswith("CUST-") for row in snap["cards"]))
        self.assertTrue(all(row["card_id"].startswith("CARD-") for row in snap["cards"]))
        self.assertTrue(all(row["transfer_id"].startswith("TXN-") for row in snap["transfers"]))
        self.assertEqual(len(snap["cards"]), 3)
        tools.close()

    def test_card_status_and_ownership(self) -> None:
        tools = self._tools()
        self.assertEqual(tools.get_card_delivery_status("CARD-1001")["delivery_status"], "delivered")
        with self.assertRaises(Exception) as ctx:
            tools.get_card_delivery_status("CARD-2001")
        self.assertEqual(getattr(ctx.exception, "code", None), "ownership_denied")
        tools.close()

    def test_cancel_requires_confirmation_and_updates_db(self) -> None:
        tools = self._tools()
        first = tools.cancel_transfer("TXN-2002", confirmed=False)
        self.assertEqual(first["status"], "confirmation_required")
        self.assertEqual(tools.get_transfer_status("TXN-2002")["status"], "pending")
        second = tools.cancel_transfer("TXN-2002", confirmed=True)
        self.assertEqual(second["status"], "cancelled")
        self.assertEqual(tools.get_transfer_status("TXN-2002")["status"], "cancelled")
        self.assertEqual(len(tools.snapshot()["audit_events"]), 4)
        tools.close()

    def test_cancel_is_idempotent(self) -> None:
        tools = self._tools()
        tools.cancel_transfer("TXN-2002", confirmed=True)
        again = tools.cancel_transfer("TXN-2002", confirmed=True)
        self.assertEqual(again["status"], "already_cancelled")
        tools.close()

    def test_activate_requires_confirmation(self) -> None:
        tools = self._tools()
        first = tools.activate_card("CARD-1002", confirmed=False)
        self.assertEqual(first["status"], "confirmation_required")
        self.assertEqual(tools.snapshot()["cards"][1]["status"], "inactive")
        second = tools.activate_card("CARD-1002", confirmed=True)
        self.assertEqual(second["status"], "activated")
        self.assertEqual(tools.snapshot()["cards"][1]["status"], "active")
        tools.close()

    def test_agent_missing_slot_then_status(self) -> None:
        agent, tools = self._agent("card_arrival")
        turn = agent.handle("Where is my card?")
        self.assertIn("card ID", turn.response)
        self.assertEqual(turn.tools, [])
        turn = agent.handle("CARD-1001")
        self.assertIn("delivered", turn.response)
        self.assertEqual([t["tool"] for t in turn.tools], ["get_card_delivery_status"])
        tools.close()

    def test_agent_unknown_handoff(self) -> None:
        agent, tools = self._agent("OOS", accepted=False)
        turn = agent.handle("How do I buy Bitcoin?")
        self.assertIn("human", turn.response.lower())
        self.assertEqual([t["tool"] for t in turn.tools], ["handoff_to_human"])
        self.assertGreaterEqual(len(tools.snapshot()["tickets"]), 2)
        tools.close()

    def test_agent_ambiguous_clarification(self) -> None:
        agent, tools = self._agent("card_arrival", margin=0.0)
        turn = agent.handle("I have a problem with my card.")
        self.assertIn("not fully sure", turn.response.lower())
        self.assertEqual(turn.tools, [])
        turn = agent.handle("card_arrival")
        self.assertIn("card ID", turn.response)
        turn = agent.handle("CARD-1001")
        self.assertIn("delivered", turn.response)
        tools.close()

    def test_b4_checkpoints_are_loadable(self) -> None:
        for name in ("gate4_b4_fusion.pt", "gate4_b4_full.pt"):
            payload = torch.load(RESULTS / name, map_location="cpu", weights_only=False)
            model = B4AR()
            model.load_state_dict(payload["state_dict"])
            self.assertEqual(payload["meta"]["model"], "gte-modernbert-base")
            self.assertEqual(payload["meta"]["shot"], 10)
            self.assertEqual(len(payload["meta"]["classes"]), 40)

    def test_b4_metrics_and_freeze(self) -> None:
        freeze = json.loads((RESULTS / "gate4_b4_freeze.json").read_text(encoding="utf-8"))
        self.assertTrue(freeze["frozen"])
        self.assertTrue(freeze["test_evaluated"])
        test = pd.read_csv(RESULTS / "gate4_b4_test_metrics.csv")
        self.assertEqual(set(test["method"]), {"B0", "B3-H", "B4-Fusion", "B4-Full"})
        self.assertEqual(len(test), 4)
        self.assertTrue((RESULTS / "gate4_agent_cases.csv").exists())
        self.assertEqual(len(pd.read_csv(RESULTS / "gate4_agent_cases.csv")), 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
