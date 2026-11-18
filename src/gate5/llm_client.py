"""Optional OpenAI-compatible LLM adapter.

The project never logs or commits API keys.  If credentials are absent, the
client reports `available=False` and all Agent paths must use the offline
template mode; no fake API calls are recorded.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class LLMResult:
    text: str
    model: str
    elapsed_seconds: float
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model or os.environ.get("OPENAI_MODEL") or ""
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "base_url_set": bool(self.base_url),
            "api_key_set": bool(self.api_key),
            "model_set": bool(self.model),
            "model": self.model or None,
            "mode": "api" if self.available else "offline_or_blocked",
            "note": "Keys are read from environment variables only and never logged.",
        }

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0, max_tokens: int = 256) -> LLMResult:
        if not self.available:
            raise LLMUnavailable("OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL not configured")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        t0 = time.perf_counter()
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - t0
        if response.status_code >= 400:
            raise RuntimeError(f"LLM HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response has no choices")
        text = choices[0].get("message", {}).get("content", "")
        return LLMResult(
            text=text,
            model=self.model,
            elapsed_seconds=elapsed,
            usage=data.get("usage", {}) or {},
            raw=data,
        )

    def extract_json(self, prompt: str, *, schema: dict[str, Any] | None = None, max_tokens: int = 256) -> dict[str, Any]:
        instruction = prompt
        if schema:
            instruction += "\nReturn ONLY a JSON object matching this schema:\n" + json.dumps(schema, ensure_ascii=False)
        result = self.chat(
            [
                {"role": "system", "content": "You are a strict JSON extraction component. Never invent database results."},
                {"role": "user", "content": instruction},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text = result.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return json.loads(text)
