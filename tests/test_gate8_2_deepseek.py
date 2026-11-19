"""Gate 8.2 B: DeepSeek client and grounded writer safety tests (Mock only)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.gate8_2.deepseek_client import DeepSeekClient, DeepSeekResult, DeepSeekUnavailable
from src.gate8_2.grounded_writer import GroundedReplyWriter

ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    available = True

    def __init__(self, text: str):
        self.text = text

    def chat(self, *a, **k):
        return DeepSeekResult(self.text, "deepseek-flash", 0.012, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})


class Gate82DeepSeekTest(unittest.TestCase):
    def writer(self, text):
        return GroundedReplyWriter(client=FakeClient(text))

    def test_offline_status_is_blocked_and_no_key(self):
        c = DeepSeekClient(api_key="")
        self.assertFalse(c.available)
        self.assertEqual(c.status()["mode"], "NOT RUN_API_BLOCKED")
        self.assertFalse(c.status()["api_key_set"])
        r = GroundedReplyWriter(c).rewrite(user_text="x", template="offline", intent=None, accepted=False, tool_results=[], allowed_doc_ids=[], pending_confirmation=False)
        self.assertEqual(r.source, "fallback")
        self.assertEqual(r.status, "NOT RUN_API_BLOCKED")

    def test_endpoint_and_no_openai_fallback(self):
        c = DeepSeekClient(api_key="k", base_url="https://api.deepseek.com", model="deepseek-flash")
        self.assertEqual(c._endpoint(), "https://api.deepseek.com/chat/completions")
        self.assertEqual(DeepSeekClient(api_key="k", base_url="https://api.deepseek.com/v1")._endpoint(), "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(c.status()["provider"], "deepseek")

    def test_valid_rewrite_and_usage(self):
        w = self.writer("Your CARD-1001 card has been delivered.")
        r = w.rewrite(
            user_text="Where is CARD-1001?",
            template="Card CARD-1001 has delivery status delivered.",
            intent="card_arrival",
            accepted=True,
            tool_results=[{"tool": "get_card_delivery_status", "result": {"status": "delivered"}}],
            allowed_doc_ids=[],
            pending_confirmation=False,
        )
        self.assertEqual(r.source, "deepseek")
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.usage["total_tokens"], 15)

    def test_new_business_id_rejected(self):
        r = self.writer("Your CARD-9999 is delivered.").rewrite(
            user_text="Where is CARD-1001?",
            template="Card CARD-1001 has delivery status delivered.",
            intent="card_arrival",
            accepted=True,
            tool_results=[],
            allowed_doc_ids=[],
            pending_confirmation=False,
        )
        self.assertEqual(r.source, "fallback")
        self.assertIn("new_business_id", r.error)

    def test_unsupported_action_claim_rejected(self):
        r = self.writer("Your transfer is now successfully cancelled.").rewrite(
            user_text="Cancel TXN-2002",
            template="Transfer TXN-2002 is pending. Do you confirm cancellation?",
            intent="cancel_transfer",
            accepted=True,
            tool_results=[{"tool": "cancel_transfer", "result": {"status": "confirmation_required"}}],
            allowed_doc_ids=[],
            pending_confirmation=True,
        )
        self.assertEqual(r.source, "fallback")
        self.assertIn("unsupported_action_claim", r.error)

    def test_unknown_kb_and_missing_confirmation_rejected(self):
        r = self.writer("See KB-999 for details. Card CARD-1002 is ready.")
        r = r.rewrite(
            user_text="Activate CARD-1002",
            template="Card CARD-1002 is inactive. Do you confirm activation?",
            intent="activate_my_card",
            accepted=True,
            tool_results=[{"tool": "activate_card", "result": {"status": "confirmation_required"}}],
            allowed_doc_ids=[],
            pending_confirmation=True,
        )
        self.assertEqual(r.source, "fallback")
        self.assertTrue(r.error)

    def test_api_error_falls_back(self):
        class Err:
            available = True

            def chat(self, *a, **k):
                raise DeepSeekUnavailable("HTTP 429")

        r = GroundedReplyWriter(client=Err()).rewrite(
            user_text="hello", template="Hello.", intent="OOS", accepted=False, tool_results=[], allowed_doc_ids=[], pending_confirmation=False
        )
        self.assertEqual(r.source, "fallback")
        self.assertEqual(r.status, "error")
        self.assertIn("429", r.error)

    def test_http_error_does_not_echo_key(self):
        class Resp:
            status_code = 401
            text = '{"error":"unauthorized"}'

        c = DeepSeekClient(api_key="secret-key-value", base_url="https://api.deepseek.com", model="deepseek-flash")
        with patch("src.gate8_2.deepseek_client.requests.post", return_value=Resp()):
            with self.assertRaises(DeepSeekUnavailable) as ctx:
                c.chat([{"role": "user", "content": "hi"}])
        self.assertNotIn("secret-key-value", str(ctx.exception))

    def test_key_shape_error_detected_without_request(self):
        c = DeepSeekClient(api_key=" <sk-test-key> ", base_url="https://api.deepseek.com", model="deepseek-flash")
        self.assertFalse(c.available)
        self.assertIn("config_error", c.status())
        self.assertEqual(c.status()["mode"], "config_error")
        with self.assertRaises(DeepSeekUnavailable) as ctx:
            c.chat([{"role": "user", "content": "hi"}])
        self.assertIn("DEEPSEEK_CONFIG_ERROR", str(ctx.exception))

    def test_app_modes_and_blocked_label(self):
        text = (ROOT / "app_gate8.py").read_text(encoding="utf-8")
        self.assertIn("Offline Router-driven", text)
        self.assertIn("DeepSeek-assisted", text)
        self.assertIn("NOT RUN_API_BLOCKED", text)
        self.assertIn("reply_source", text)
        self.assertIn("usage", text)
        self.assertIn("api_auth_error", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
