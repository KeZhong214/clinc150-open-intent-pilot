"""Gate 8.2 Chinese interface, templates and translation-safety tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from src.gate8_2.chinese_adapter import ChineseTranslationAdapter
from src.gate8_2.chinese_templates import chinese_reply
from src.gate8_2.deepseek_client import DeepSeekResult
from src.gate8_2.grounded_writer import GroundedReplyWriter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate8_2"


class FakeTranslate:
    available = True

    def __init__(self, text):
        self.text = text

    def chat(self, *a, **k):
        return DeepSeekResult(self.text, "deepseek-flash", 0.01, {"total_tokens": 12})


class Gate82ChineseTest(unittest.TestCase):
    def test_20_cases_and_categories(self):
        rows = json.loads((DATA / "chinese_cases.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(any("\u4e00" <= ch <= "\u9fff" for ch in " ".join(x["turns"])) for x in rows))
        self.assertTrue({"card_status", "transfer_status", "policy_faq", "dispute_ticket", "oos", "permission", "sensitive_refusal"} <= {x["family"] for x in rows})

    def test_direct_probe_60_rows_and_blocked_translation(self):
        d = pd.read_csv(RESULTS / "gate8_2_chinese_results.csv")
        self.assertEqual(len(d), 60)
        self.assertEqual(d.groupby("router").size().nunique(), 1)
        self.assertTrue((d.translation_status == "NOT RUN_API_BLOCKED").all())
        self.assertTrue(d.translated_intent.isna().all())

    def test_translation_preserves_ids(self):
        a = ChineseTranslationAdapter(client=FakeTranslate("I have not received my CARD-1001 card."))
        r = a.translate("我的银行卡 CARD-1001 还没有收到。")
        self.assertEqual(r.status, "ok")
        self.assertIn("CARD-1001", r.text)
        self.assertFalse(a.translate("我的卡还没到。").status == "ok" and "CARD" in a.translate("我的卡还没到。").text)

    def test_translation_rejects_new_or_missing_ids(self):
        a = ChineseTranslationAdapter(client=FakeTranslate("Please check CARD-9999."))
        r = a.translate("请查 CARD-1001。")
        self.assertEqual(r.status, "id_mismatch")
        b = ChineseTranslationAdapter(client=FakeTranslate("AMBIGUOUS"))
        self.assertEqual(b.translate("我的卡有问题。").status, "ambiguous")

    def test_chinese_offline_templates(self):
        # A minimal turn-like object is sufficient for the conservative fallback.
        from src.gate4.router import RouteResult
        from src.gate5.agent_v2 import TurnV2

        route = RouteResult("B4", "card_arrival", 0.9, 0.4, True, [("card_arrival", 0.9)])
        turn = TurnV2("English template", route, [{"tool": "get_card_delivery_status", "args": {"card_id": "CARD-1001"}, "result": {"card_id": "CARD-1001", "delivery_status": "delivered", "status": "active"}}], [], {"awaiting_confirmation": False})
        text = chinese_reply(turn)
        self.assertIn("CARD-1001", text)
        self.assertIn("已送达", text)
        self.assertNotIn("English", text)

    def test_chinese_writer_validation(self):
        class Good:
            available = True

            def chat(self, *a, **k):
                return DeepSeekResult("您的卡片 CARD-1001 已送达，状态为已激活。", "deepseek-flash", 0.02, {"total_tokens": 20})

        r = GroundedReplyWriter(client=Good()).rewrite(
            user_text="我的卡 CARD-1001 到了吗？",
            template="卡片 CARD-1001 的配送状态是已送达，卡片状态是已激活。",
            intent="card_arrival", accepted=True,
            tool_results=[{"tool": "get_card_delivery_status", "result": {"delivery_status": "delivered", "status": "active"}}],
            allowed_doc_ids=[], pending_confirmation=False, language="zh",
        )
        self.assertEqual(r.source, "deepseek")
        self.assertIn("CARD-1001", r.text)

        class Bad:
            available = True

            def chat(self, *a, **k):
                return DeepSeekResult("您的转账已取消。", "deepseek-flash", 0.02, {})

        r = GroundedReplyWriter(client=Bad()).rewrite(
            user_text="请查转账", template="转账 TXN-2001 待处理。", intent="transfer_timing", accepted=True,
            tool_results=[], allowed_doc_ids=[], pending_confirmation=False, language="zh",
        )
        self.assertEqual(r.source, "fallback")
        self.assertIn("unsupported_action_claim", r.error)

    def test_ui_has_chinese_mode_labels(self):
        text = (ROOT / "app_gate8.py").read_text(encoding="utf-8")
        for needle in ("银行智能客服工作台", "意图路由模型", "离线客服模式", "DeepSeek 增强模式", "中文翻译适配", "工具调用记录", "知识库引用"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
