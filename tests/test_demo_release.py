"""Focused regression tests for the final classroom demo.

These tests use StubRouter for deterministic state-machine checks; real model
preflight is performed separately by preflight_demo.py.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.gate4.agent import StubRouter
from src.gate5.agent_v2 import CARD_RE, TXN_RE
from src.gate8.agent import Gate8Agent
from src.gate8_1 import registry
from src.gate8_2.chinese_adapter import CARD_TXN
from src.gate8_2.deepseek_client import DeepSeekClient, DeepSeekUnavailable


class DemoReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def agent(self, intent: str, *, accepted: bool = True):
        return Gate8Agent(StubRouter(intent, accepted=accepted), session_id=f"t-{intent}-{accepted}", db_dir=self.dir)

    def test_01_tight_chinese_card_id_regex(self) -> None:
        self.assertEqual(CARD_RE.search("我的卡CARD-1002还没到").group(0), "CARD-1002")
        self.assertEqual(TXN_RE.search("请处理TXN-2002。").group(0), "TXN-2002")
        self.assertEqual(CARD_TXN.findall("我的卡CARD-1002还没到"), ["CARD-1002"])

    def test_02_card_query_reaches_read_only_tool(self) -> None:
        a = self.agent("card_arrival")
        turn = a.handle("我的卡CARD-1002还没到")
        self.assertTrue(turn.route.accepted)
        self.assertEqual(turn.route.intent, "card_arrival")
        self.assertIn("get_card_delivery_status", [t["tool"] for t in turn.tools])
        result = next(t["result"] for t in turn.tools if t["tool"] == "get_card_delivery_status")
        self.assertEqual(result["delivery_status"], "in_transit")
        self.assertEqual(result["status"], "inactive")
        a.close()

    def test_03_cancel_before_confirmation_does_not_change_db(self) -> None:
        a = self.agent("cancel_transfer")
        turn = a.handle("请取消转账 TXN-2002。")
        self.assertIn("cancel_transfer", [t["tool"] for t in turn.tools])
        self.assertTrue(turn.state["awaiting_confirmation"])
        transfer = next(x for x in a.snapshot()["transfers"] if x["transfer_id"] == "TXN-2002")
        self.assertEqual(transfer["status"], "pending")
        a.close()

    def test_04_chinese_confirmation_changes_pending_to_cancelled(self) -> None:
        a = self.agent("cancel_transfer")
        a.handle("请取消转账 TXN-2002。")
        turn = a.handle("确认。")
        transfer = next(x for x in a.snapshot()["transfers"] if x["transfer_id"] == "TXN-2002")
        self.assertEqual(transfer["status"], "cancelled")
        self.assertEqual(turn.tools[0]["result"]["status"], "cancelled")
        a.close()

    def test_05_repeated_cancel_is_idempotent(self) -> None:
        a = self.agent("cancel_transfer")
        a.handle("请取消转账 TXN-2002。")
        a.handle("确认")
        second = a.handle("请取消转账 TXN-2002。")
        self.assertEqual(second.tools[0]["result"]["status"], "already_cancelled")
        a.close()

    def test_06_ownership_denied_does_not_mutate(self) -> None:
        a = self.agent("card_arrival")
        turn = a.handle("请查 CARD-2001")
        self.assertIn("handoff_to_human", [t["tool"] for t in turn.tools])
        self.assertNotIn("get_card_delivery_status", [t["tool"] for t in turn.tools])
        a.close()

    def test_07_oos_never_calls_money_tools(self) -> None:
        a = self.agent("OOS", accepted=False)
        turn = a.handle("帮我订一张去巴黎的机票。")
        names = [t["tool"] for t in turn.tools]
        self.assertIn("handoff_to_human", names)
        self.assertNotIn("cancel_transfer", names)
        self.assertNotIn("activate_card", names)
        a.close()

    def test_08_topic_switch_does_not_swallow_new_policy_request(self) -> None:
        a = self.agent("card_arrival")
        a.handle("我的卡还没到")  # likely asks for card slot
        turn = a.handle("What is the transfer fee policy?")
        self.assertIsNotNone(turn)
        a.close()

    def test_09_reset_creates_independent_session(self) -> None:
        a = self.agent("cancel_transfer")
        a.handle("请取消转账 TXN-2002。")
        a.handle("确认")
        b = Gate8Agent(StubRouter("card_arrival"), session_id="independent", db_dir=self.dir)
        transfer = next(x for x in b.snapshot()["transfers"] if x["transfer_id"] == "TXN-2002")
        self.assertEqual(transfer["status"], "pending")
        a.close()
        b.close()

    def test_10_chinese_confirmation_tokens_and_punctuation(self) -> None:
        self.assertEqual(Gate8Agent._confirmation_command("确认。"), "yes")
        self.assertEqual(Gate8Agent._confirmation_command("是的"), "yes")
        self.assertEqual(Gate8Agent._confirmation_command("Confirm."), "yes")
        self.assertEqual(Gate8Agent._confirmation_command("取消"), "no")
        self.assertEqual(Gate8Agent._confirmation_command("不用了"), "no")
        self.assertIsNone(Gate8Agent._confirmation_command("取消转账 TXN-2002。"))

    def test_11_no_key_is_offline_not_success(self) -> None:
        c = DeepSeekClient(api_key="")
        self.assertFalse(c.available)
        self.assertEqual(c.status()["mode"], "NOT RUN_API_BLOCKED")
        self.assertFalse(c.status()["api_key_set"])

    def test_12_model_registry_and_threshold_pairing(self) -> None:
        entry = registry.validate("Qwen-Plain42-RejectOnly")
        self.assertTrue(entry["paired"])
        self.assertAlmostEqual(entry["tau"], 0.4168315291404724, places=9)
        self.assertEqual(entry["checkpoint"], "results/gate6_qwen_plain_reject.pt")

    def test_13_timeout_is_explicit_failure(self) -> None:
        import requests

        c = DeepSeekClient(api_key="k", base_url="https://api.deepseek.com", model="deepseek-flash")
        with patch("src.gate8_2.deepseek_client.requests.post", side_effect=requests.Timeout("slow")):
            with self.assertRaises(DeepSeekUnavailable) as ctx:
                c.chat([{"role": "user", "content": "hi"}])
        self.assertIn("timed out", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
