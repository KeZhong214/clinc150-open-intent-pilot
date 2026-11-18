"""Probe the optional OpenAI-compatible LLM endpoint without exposing keys."""

from __future__ import annotations

import json
from pathlib import Path

from src.gate5.llm_client import LLMClient

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def main() -> int:
    client = LLMClient()
    status = client.status()
    payload = {
        **status,
        "api_mode_result": "NOT RUN_API_BLOCKED" if not client.available else "READY_BUT_NOT_EXECUTED_BY_THIS_PROBE",
        "note": (
            "No OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL in this environment. "
            "The offline template Agent remains fully runnable; the 12-call API report is blocked."
            if not client.available
            else "Credentials are present. Run the dedicated API-case script to produce real calls."
        ),
    }
    (RESULTS / "gate5_llm_probe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("GATE5_LLM_PROBE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
