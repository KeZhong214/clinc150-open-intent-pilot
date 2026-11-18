"""Run ten real-router Gate-5 demo conversations (B3-H and B4-Full / LoRA)."""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate5"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5.agent_v2 import BankAgentV2  # noqa: E402
from src.gate5.kb import KBRetriever  # noqa: E402
from src.gate5.router_lora import FrozenEncoder, LoraEncoder, load_lora_b4_router  # noqa: E402
from src.gate4.router import load_b3h_router, load_b4_router  # noqa: E402

DEMOS = [
    ("D1", "B001", "Frozen-B3H"),
    ("D2", "B003", "Frozen-B4Full"),
    ("D3", "B036", "Frozen-B4Full"),
    ("D4", "B045", "Frozen-B4Full"),
    ("D5", "B076", "Frozen-B4Full"),
    ("D6", "B078", "Frozen-B4Full"),
    ("D7", "B083", "Frozen-B4Full"),
    ("D8", "B111", "Frozen-B4Full"),
    ("D9", "B119", "Frozen-B4Full"),
    ("D10", "B106", "Frozen-B3H"),
]


def main() -> int:
    cases = {x["case_id"]: x for x in map(json.loads, (DATA / "benchmark_cases.jsonl").read_text(encoding="utf-8").splitlines())}
    kb = KBRetriever()
    frozen = FrozenEncoder()
    routers = {
        "Frozen-B3H": load_b3h_router(frozen),
        "Frozen-B4Full": load_b4_router(frozen),
        "LoRA-B4Full": load_lora_b4_router(LoraEncoder()),
    }
    out = []
    for demo_id, case_id, router_name in DEMOS:
        case = cases[case_id]
        session_dir = DATA / "demo_sessions" / demo_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        agent = BankAgentV2(routers[router_name], session_id=f"DEMO-{demo_id}", db_dir=session_dir, kb=kb)
        if case.get("setup", {}).get("pre_cancel"):
            agent.tools.cancel_transfer(case["setup"]["pre_cancel"], confirmed=True)
        turns = []
        t0 = time.perf_counter()
        for user in case["turns"]:
            turn = agent.handle(user)
            turns.append(
                {
                    "user": user,
                    "response": turn.response,
                    "route": {
                        "method": turn.route.method if turn.route else None,
                        "intent": turn.route.intent if turn.route else None,
                        "accepted": turn.route.accepted if turn.route else None,
                        "score": round(turn.route.score, 6) if turn.route else None,
                        "threshold": round(turn.route.threshold, 6) if turn.route else None,
                        "top3": turn.route.top3 if turn.route else [],
                        "gate_top1": turn.route.gate_top1 if turn.route else None,
                    },
                    "tools": turn.tools,
                    "citations": turn.citations,
                }
            )
        snap = agent.snapshot()
        agent.close()
        out.append(
            {
                "demo_id": demo_id,
                "case_id": case_id,
                "router": router_name,
                "turns": turns,
                "snapshot": snap,
                "elapsed_seconds": round(time.perf_counter() - t0, 4),
            }
        )
        print(demo_id, router_name, "turns", len(turns), "last", turns[-1]["response"][:100])
    (RESULTS / "gate5_agent_demos.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("GATE5_AGENT_DEMOS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
