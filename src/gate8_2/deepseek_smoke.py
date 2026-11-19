"""One-shot, low-cost DeepSeek smoke test; never prints the key."""

from __future__ import annotations

import sys
import time

from src.gate8_2.deepseek_client import DeepSeekClient, DeepSeekUnavailable


def main() -> int:
    c = DeepSeekClient()
    status = c.status()
    print({k: status[k] for k in ("provider", "available", "base_url", "model", "api_key_set", "key_format_ok", "config_error", "timeout_seconds", "mode")})
    if status.get("config_error"):
        print(f"DEEPSEEK_CONFIG_ERROR: {status['config_error']}")
        return 2
    if not c.available:
        print("NOT RUN_API_BLOCKED: set DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL in this PowerShell session.")
        return 2
    t0 = time.perf_counter()
    try:
        r = c.chat(
            [
                {"role": "system", "content": "Reply briefly. Do not invent bank data."},
                {"role": "user", "content": "Return a short greeting for a fictional bank support demo."},
            ],
            temperature=0.0,
            max_tokens=256,
        )
    except DeepSeekUnavailable as exc:
        print(f"DEEPSEEK_SMOKE_FAILED: {type(exc).__name__}: {exc}")
        return 1
    print({"status": "ok", "model": r.model, "elapsed_seconds": round(time.perf_counter() - t0, 3), "usage": r.usage, "text": r.text[:200]})
    print("DEEPSEEK_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

