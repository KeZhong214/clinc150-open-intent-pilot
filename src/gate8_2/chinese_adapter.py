"""Optional DeepSeek Chinese-to-English semantic adapter.

The translator never chooses an intent. It only returns English text for the
existing English router, preserving all CARD/TXN identifiers.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from src.gate8_2.deepseek_client import DeepSeekClient, DeepSeekUnavailable

CJK = re.compile(r"[\u4e00-\u9fff]")
CARD_TXN = re.compile(r"\b(?:CARD|TXN)-\d{4,}\b", re.IGNORECASE)


@dataclass
class TranslationResult:
    text: str
    source: str
    status: str
    elapsed_seconds: float = 0.0
    usage: dict = field(default_factory=dict)
    error: str | None = None


class ChineseTranslationAdapter:
    def __init__(self, client: DeepSeekClient | None = None):
        self.client = client or DeepSeekClient()

    @property
    def available(self) -> bool:
        return self.client.available

    @staticmethod
    def has_chinese(text: str) -> bool:
        return bool(CJK.search(text or ""))

    def translate(self, text: str) -> TranslationResult:
        if not self.has_chinese(text):
            return TranslationResult(text, "passthrough", "ok")
        status = self.client.status() if hasattr(self.client, "status") else {}
        if status.get("config_error"):
            return TranslationResult("", "config_error", "CONFIG_ERROR", error=status["config_error"])
        if not self.available:
            return TranslationResult("", "blocked", "NOT RUN_API_BLOCKED", error="DEEPSEEK_API_KEY not configured")
        ids = {x.upper() for x in CARD_TXN.findall(text)}
        prompt = (
            "Translate the user's Chinese bank-support request into concise English for an English intent router. "
            "Preserve every CARD-#### and TXN-#### identifier exactly; do not add or omit IDs. "
            "Do not decide an intent or answer the request. If the text is too ambiguous, return exactly: AMBIGUOUS. "
            f"User text: {text}"
        )
        try:
            r = self.client.chat(
                [{"role": "system", "content": "You are a faithful Chinese-to-English translation component."}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=180,
            )
        except DeepSeekUnavailable as exc:
            return TranslationResult("", "error", "error", error=str(exc))
        out = " ".join(r.text.split()).strip()
        if not out or out.upper() == "AMBIGUOUS":
            return TranslationResult("", "fallback", "ambiguous", elapsed_seconds=r.elapsed_seconds, usage=r.usage, error="translation ambiguous")
        out_ids = {x.upper() for x in CARD_TXN.findall(out)}
        if out_ids != ids:
            return TranslationResult("", "fallback", "id_mismatch", elapsed_seconds=r.elapsed_seconds, usage=r.usage, error=f"id mismatch: {sorted(ids)} -> {sorted(out_ids)}")
        if not re.search(r"[A-Za-z]", out):
            return TranslationResult("", "fallback", "not_english", elapsed_seconds=r.elapsed_seconds, usage=r.usage, error="translation is not English")
        return TranslationResult(out, "deepseek", "ok", elapsed_seconds=r.elapsed_seconds, usage=r.usage)
