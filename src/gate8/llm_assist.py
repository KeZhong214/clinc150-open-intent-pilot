"""Gate-8 constrained LLM assistance.

The LLM is only a language/slot suggestion component.  Server-side route
acceptance, permissions, confirmation and tool execution remain in AgentV2.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.gate5.llm_client import LLMClient, LLMUnavailable

CARD_RE = re.compile(r"\bCARD-\d{4,}\b", re.IGNORECASE)
TXN_RE = re.compile(r"\bTXN-\d{4,}\b", re.IGNORECASE)
ALLOWED = {"card_id", "transfer_id", "intent_hint"}


class LLMAssistant:
    """Optional, schema-checked assist; no credentials means offline/blocked."""

    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return self.client.available

    def status(self) -> dict[str, Any]:
        return {**self.client.status(), "assist_mode": "constrained_json" if self.available else "NOT RUN_API_BLOCKED"}

    def parse_slots(self, text: str, allowed_intents: list[str]) -> dict[str, Any]:
        if not self.available:
            raise LLMUnavailable("LLM-assisted mode is not configured")
        schema = {
            "type": "object",
            "properties": {
                "card_id": {"type": "string"},
                "transfer_id": {"type": "string"},
                "intent_hint": {"type": "string", "enum": allowed_intents},
            },
            "additionalProperties": False,
        }
        try:
            data = self.client.extract_json(
                "Extract only explicit fictional IDs or an allowed intent hint. "
                "Return {} when unsure. Never invent IDs. Text: " + text,
                schema=schema,
            )
            if not isinstance(data, dict) or set(data) - ALLOWED:
                raise ValueError("schema violation")
            if "card_id" in data and not CARD_RE.fullmatch(str(data["card_id"])):
                raise ValueError("bad card_id")
            if "transfer_id" in data and not TXN_RE.fullmatch(str(data["transfer_id"])):
                raise ValueError("bad transfer_id")
            if "intent_hint" in data and data["intent_hint"] not in allowed_intents:
                raise ValueError("intent outside allowlist")
            return data
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise LLMUnavailable("LLM assist rejected; using offline parser") from exc
