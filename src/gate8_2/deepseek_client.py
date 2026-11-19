"""Independent DeepSeek client; never falls back to OPENAI_* credentials."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests


class DeepSeekUnavailable(RuntimeError):
    pass


@dataclass
class DeepSeekResult:
    text: str
    model: str
    elapsed_seconds: float
    usage: dict[str, Any] = field(default_factory=dict)


class DeepSeekClient:
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, model: str | None = None, timeout: float | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url if base_url is not None else os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model if model is not None else os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        self.timeout = float(timeout if timeout is not None else os.environ.get("DEEPSEEK_TIMEOUT", "12"))

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def status(self) -> dict[str, Any]:
        # Deliberately never return the key or an Authorization header.
        return {
            "provider": "deepseek",
            "available": self.available,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_set": bool(self.api_key),
            "timeout_seconds": self.timeout,
            "mode": "deepseek" if self.available else "NOT RUN_API_BLOCKED",
        }

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 300, json_mode: bool = False) -> DeepSeekResult:
        if not self.available:
            raise DeepSeekUnavailable("DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL not configured")
        max_tokens = max(1, min(int(max_tokens), 512))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            response = requests.post(
                self._endpoint(),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise DeepSeekUnavailable("DeepSeek request timed out") from exc
        except requests.RequestException as exc:
            raise DeepSeekUnavailable(f"DeepSeek network error: {type(exc).__name__}") from exc
        elapsed = time.perf_counter() - t0
        if response.status_code >= 400:
            # Do not include headers or the request body in the error.
            raise DeepSeekUnavailable(f"DeepSeek HTTP {response.status_code}: {response.text[:180]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise DeepSeekUnavailable("DeepSeek returned non-JSON response") from exc
        choices = data.get("choices") or []
        if not choices:
            raise DeepSeekUnavailable("DeepSeek returned no choices")
        text = (choices[0].get("message") or {}).get("content") or ""
        if not str(text).strip():
            raise DeepSeekUnavailable("DeepSeek returned empty text")
        return DeepSeekResult(text=str(text).strip(), model=str(data.get("model") or self.model), elapsed_seconds=elapsed, usage=data.get("usage") or {})

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
            json_mode=True,
        )
        text = result.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DeepSeekUnavailable("DeepSeek JSON parse failed") from exc
        if not isinstance(data, dict):
            raise DeepSeekUnavailable("DeepSeek JSON is not an object")
        return data
