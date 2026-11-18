"""G6-D: real Qwen Agent replay demos with final database state."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate5"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5.agent_v2 import BankAgentV2  # noqa: E402
from src.gate6.router_qwen import load_qwen_router  # noqa: E402

DEMOS = [
    ("Q1", "Qwen-Plain-B3H", "B001"),
    ("Q2", "Qwen-Plain-RejectOnly", "B003"),
    ("Q3", "Qwen-Plain-RejectOnly", "B045"),
    ("Q4", "Qwen-Plain-B4Full", "B076"),
    ("Q5", "Qwen-Plain-B4Full", "B111"),
    ("Q6", "Qwen-Plain-B3H", "B106"),
]


def main() -> int:
    cases = {x["case_id"]: x for x in map(json.loads, (DATA / "benchmark_cases.jsonl").read_text(encoding="utf-8").splitlines())}
    root = ROOT / "data" / "gate6" / "demo_sessions"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    routers = {}
    encoder_cache = {}
    out = []
    for demo_id, router_name, case_id in DEMOS:
        if router_name not in routers:
            routers[router_name] = load_qwen_router(router_name, encoder_cache=encoder_cache)
        agent = BankAgentV2(routers[router_name], session_id=f"G6-{demo_id}", db_dir=root)
        case = cases[case_id]
        if case.get("setup", {}).get("pre_cancel"):
            agent.tools.cancel_transfer(case["setup"]["pre_cancel"], confirmed=True)
        turns = []
        for text in case["turns"]:
            turn = agent.handle(text)
            turns.append(
                {
                    "user": text,
                    "response": turn.response,
                    "intent": turn.route.intent if turn.route else None,
                    "accepted": turn.route.accepted if turn.route else None,
                    "score": round(turn.route.score, 6) if turn.route else None,
                    "threshold": round(turn.route.threshold, 6) if turn.route else None,
                    "top3": turn.route.top3[:3] if turn.route else [],
                    "tools": turn.tools,
                    "citations": turn.citations,
                }
            )
        snap = agent.snapshot()
        out.append({"demo_id": demo_id, "router": router_name, "case_id": case_id, "turns": turns, "snapshot": snap})
        print(demo_id, router_name, case_id, [(t["intent"], t["accepted"]) for t in turns], flush=True)
        agent.close()
    (RESULTS / "gate6_qwen_agent_demos.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("GATE6_QWEN_DEMOS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
