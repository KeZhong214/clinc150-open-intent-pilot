"""Gate 8.2 Chinese probe: direct untranslated routing is always measured.

Translated routing is only attempted with a real DeepSeek key and --live;
without a key it is explicitly NOT RUN_API_BLOCKED, never mocked as real.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CASES = ROOT / "data" / "gate8_2" / "chinese_cases.json"
import sys

sys.path.insert(0, str(ROOT))
from src.gate8.agent import Gate8Agent  # noqa: E402
from src.gate8_1.routers import ROUTER_NAMES, build  # noqa: E402
from src.gate8_2.chinese_adapter import ChineseTranslationAdapter  # noqa: E402

SLUG = {r: r.lower().replace("-", "_") for r in ROUTER_NAMES}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--router", choices=ROUTER_NAMES, required=True)
    p.add_argument("--live", action="store_true")
    p.add_argument("--tag", default="direct")
    args = p.parse_args(argv)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    router, meta = build(args.router)
    adapter = ChineseTranslationAdapter()
    rows = []
    for case in cases:
        session_dir = ROOT / "data" / "gate8_2" / "sessions" / (SLUG[args.router] + "_" + args.tag)
        agent = Gate8Agent(router, session_id=f"ZH-{case['case_id']}-{args.tag}", db_dir=session_dir)
        direct_turns, translated_turns = [], []
        try:
            for text in case["turns"]:
                turn = agent.handle(text)
                direct_turns.append({
                    "text": text,
                    "intent": turn.route.intent if turn.route else None,
                    "accepted": turn.route.accepted if turn.route else None,
                    "score": round(turn.route.score, 6) if turn.route else None,
                    "tools": [t["tool"] for t in turn.tools],
                    "response": turn.response,
                })
            direct_tools = [t for x in direct_turns for t in x["tools"]]
            direct_intent = direct_turns[0]["intent"] if direct_turns else None
            direct_accept = direct_turns[0]["accepted"] if direct_turns else None
            # A fresh DB is required for a fair translated run.
            if args.live and adapter.available:
                agent.close()
                agent = Gate8Agent(router, session_id=f"ZH-{case['case_id']}-live", db_dir=session_dir)
                for text in case["turns"]:
                    tr = adapter.translate(text)
                    if tr.status != "ok":
                        translated_turns.append({"text": text, "translation_status": tr.status, "error": tr.error})
                        break
                    turn = agent.handle(tr.text)
                    translated_turns.append({
                        "text": text,
                        "translated": tr.text,
                        "translation_status": tr.status,
                        "usage": tr.usage,
                        "intent": turn.route.intent if turn.route else None,
                        "accepted": turn.route.accepted if turn.route else None,
                        "tools": [t["tool"] for t in turn.tools],
                        "response": turn.response,
                    })
                tr_status = "ok" if translated_turns and all(x.get("translation_status") == "ok" for x in translated_turns) else "error"
            else:
                tr_status = "NOT RUN_API_BLOCKED"
            trans_intent = translated_turns[0].get("intent") if translated_turns else None
            trans_accept = translated_turns[0].get("accepted") if translated_turns else None
            rows.append({
                "router": args.router,
                "case_id": case["case_id"],
                "family": case["family"],
                "original_zh": " | ".join(case["turns"]),
                "direct_intent": direct_intent,
                "direct_accepted": direct_accept,
                "direct_tools": "|".join(direct_tools),
                "direct_response": direct_turns[-1]["response"] if direct_turns else "",
                "expected_intent_any": "|".join(case["expected_intent_any"]),
                "expected_tools": "|".join(case["expected_tools"]),
                "expected_oos": case["expected_oos"],
                "translation_status": tr_status,
                "translated_text": " | ".join(x.get("translated", "") for x in translated_turns),
                "translated_intent": trans_intent,
                "translated_accepted": trans_accept,
                "translated_tools": "|".join(t for x in translated_turns for t in x.get("tools", [])),
                "intent_changed": (direct_intent != trans_intent) if trans_intent else None,
            })
        finally:
            agent.close()
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / f"gate8_2_chinese_{args.tag}_{SLUG[args.router]}_results.csv", index=False)
    print(d.groupby("family").size().to_string())
    print(d[["case_id", "direct_intent", "direct_accepted", "translation_status"]].to_string(index=False))
    print("GATE8_2_CHINESE_PROBE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
