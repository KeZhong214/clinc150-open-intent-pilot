"""Run the optional 12-case API-mode report, or record an explicit block.

No API call is made unless OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL are
present.  The offline Agent benchmark is a separate result.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.gate5.llm_client import LLMClient

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate5"
CASE_IDS = ["B001", "B003", "B036", "B045", "B076", "B078", "B083", "B111", "B119", "B106", "B026", "B056"]


def main() -> int:
    client = LLMClient()
    status = client.status()
    cases = {x["case_id"]: x for x in map(json.loads, (DATA / "benchmark_cases.jsonl").read_text(encoding="utf-8").splitlines())}
    if not client.available:
        payload = {
            "status": "NOT RUN_API_BLOCKED",
            "available": False,
            "n_requested": len(CASE_IDS),
            "n_completed": 0,
            "cases": CASE_IDS,
            "reason": "Missing one or more of OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL.",
            "note": "Offline template Agent and the 120-case real-router benchmark remain valid; no API output is fabricated.",
            "client": status,
        }
        (RESULTS / "gate5_llm_api_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("GATE5_LLM_API_BLOCKED")
        return 0

    rows = []
    for cid in CASE_IDS:
        case = cases[cid]
        prompt = (
            "You are the language layer of a fictional bank customer-service agent. "
            "Do not invent tool results or perform actions. Given the user turns, return a short JSON object "
            "with keys intent_guess, slot_guess, clarification. Turns: " + json.dumps(case["turns"], ensure_ascii=False)
        )
        try:
            result = client.chat(
                [{"role": "system", "content": "Return only concise JSON. Never invent account data."}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=180,
            )
            rows.append({"case_id": cid, "ok": True, "model": result.model, "elapsed_seconds": result.elapsed_seconds, "usage": result.usage, "text": result.text})
        except Exception as exc:  # noqa: BLE001
            rows.append({"case_id": cid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    payload = {"status": "COMPLETED", "available": True, "n_requested": len(CASE_IDS), "n_completed": sum(r["ok"] for r in rows), "cases": rows, "client": status}
    (RESULTS / "gate5_llm_api_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("GATE5_LLM_API_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
