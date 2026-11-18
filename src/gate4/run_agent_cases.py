"""Run deterministic Agent tool/state cases and real-router demo scripts."""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate4"

import sys

sys.path.insert(0, str(ROOT))
from src.gate4.agent import BankAgent, StubRouter  # noqa: E402
from src.gate4.db import BankingTools  # noqa: E402


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")[:60]


def find_card(snap: dict, card_id: str) -> dict:
    return next(c for c in snap["cards"] if c["card_id"] == card_id)


def find_transfer(snap: dict, transfer_id: str) -> dict:
    return next(t for t in snap["transfers"] if t["transfer_id"] == transfer_id)


CASES = [
    {
        "id": "C01",
        "name": "card delivery status",
        "intent": "card_arrival",
        "users": ["Where is my card? I still have not received CARD-1001."],
        "expect_tools": ["get_card_delivery_status"],
        "check": lambda s: find_card(s, "CARD-1001")["delivery_status"] == "delivered",
    },
    {
        "id": "C02",
        "name": "card missing slot then supplied",
        "intent": "card_arrival",
        "users": ["Where is my card?", "CARD-1001"],
        "expect_tools": ["get_card_delivery_status"],
        "check": lambda s: find_card(s, "CARD-1001")["delivery_status"] == "delivered",
    },
    {
        "id": "C03",
        "name": "transfer status",
        "intent": "transfer_not_received_by_recipient",
        "users": ["My transfer TXN-2001 has not arrived."],
        "expect_tools": ["get_transfer_status"],
        "check": lambda s: find_transfer(s, "TXN-2001")["status"] == "pending",
    },
    {
        "id": "C04",
        "name": "transfer missing slot then supplied",
        "intent": "transfer_not_received_by_recipient",
        "users": ["My transfer has not arrived.", "TXN-2001"],
        "expect_tools": ["get_transfer_status"],
        "check": lambda s: find_transfer(s, "TXN-2001")["status"] == "pending",
    },
    {
        "id": "C05",
        "name": "cancel pending transfer with confirmation",
        "intent": "cancel_transfer",
        "users": ["Please cancel TXN-2002.", "yes"],
        "expect_tools": ["cancel_transfer", "cancel_transfer"],
        "check": lambda s: find_transfer(s, "TXN-2002")["status"] == "cancelled",
    },
    {
        "id": "C06",
        "name": "cancel pending transfer declined",
        "intent": "cancel_transfer",
        "users": ["Please cancel TXN-2001.", "no"],
        "expect_tools": ["cancel_transfer"],
        "check": lambda s: find_transfer(s, "TXN-2001")["status"] == "pending",
    },
    {
        "id": "C07",
        "name": "cancel completed transfer",
        "intent": "cancel_transfer",
        "users": ["Please cancel TXN-2003."],
        "expect_tools": ["cancel_transfer"],
        "check": lambda s: find_transfer(s, "TXN-2003")["status"] == "completed",
    },
    {
        "id": "C08",
        "name": "cancel already-cancelled transfer",
        "intent": "cancel_transfer",
        "users": ["Please cancel TXN-2002."],
        "expect_tools": ["cancel_transfer"],
        "pre": lambda tools: tools.cancel_transfer("TXN-2002", confirmed=True),
        "check": lambda s: find_transfer(s, "TXN-2002")["status"] == "cancelled",
    },
    {
        "id": "C09",
        "name": "activate inactive card with confirmation",
        "intent": "activate_my_card",
        "users": ["Please activate CARD-1002.", "yes"],
        "expect_tools": ["activate_card", "activate_card"],
        "check": lambda s: find_card(s, "CARD-1002")["status"] == "active",
    },
    {
        "id": "C10",
        "name": "activate inactive card declined",
        "intent": "activate_my_card",
        "users": ["Please activate CARD-1002.", "no"],
        "expect_tools": ["activate_card"],
        "check": lambda s: find_card(s, "CARD-1002")["status"] == "inactive",
    },
    {
        "id": "C11",
        "name": "already active card",
        "intent": "activate_my_card",
        "users": ["Please activate CARD-1001."],
        "expect_tools": ["activate_card"],
        "check": lambda s: find_card(s, "CARD-1001")["status"] == "active",
    },
    {
        "id": "C12",
        "name": "dispute creation",
        "intent": "card_payment_not_recognised",
        "users": ["I want to dispute TXN-2003 because I do not recognise it."],
        "expect_tools": ["create_dispute"],
        "check": lambda s: len(s["disputes"]) >= 1,
    },
    {
        "id": "C13",
        "name": "non-owned card escalated",
        "intent": "card_arrival",
        "users": ["Where is CARD-2001?"],
        "expect_tools": ["handoff_to_human"],
        "check": lambda s: len(s["tickets"]) >= 2,
    },
    {
        "id": "C14",
        "name": "unsupported request escalated",
        "intent": "OOS",
        "accepted": False,
        "users": ["How do I buy Bitcoin?"],
        "expect_tools": ["handoff_to_human"],
        "check": lambda s: len(s["tickets"]) >= 2,
    },
    {
        "id": "C15",
        "name": "safe known intent becomes ticket",
        "intent": "card_not_working",
        "users": ["My card is not working."],
        "expect_tools": ["create_ticket"],
        "check": lambda s: len(s["tickets"]) >= 2,
    },
    {
        "id": "C16",
        "name": "ambiguous clarification then card status",
        "intent": "card_arrival",
        "margin": 0.0,
        "users": ["I have a problem with my card.", "card_arrival", "CARD-1001"],
        "expect_tools": ["get_card_delivery_status"],
        "check": lambda s: find_card(s, "CARD-1001")["delivery_status"] == "delivered",
    },
]


def run_case(case: dict) -> dict[str, Any]:
    path = DATA / "cases" / f"{case['id']}.db"
    if path.exists():
        path.unlink()
    tools = BankingTools(session_id=f"CASE-{case['id']}", path=path)
    if "pre" in case:
        case["pre"](tools)
    router = StubRouter(case["intent"], accepted=case.get("accepted", True), margin=case.get("margin", 0.2))
    agent = BankAgent(router, tools)
    turns = []
    for user in case["users"]:
        turn = agent.handle(user)
        turns.append(
            {
                "user": user,
                "response": turn.response,
                "tools": turn.tools,
                "route": {
                    "intent": turn.route.intent if turn.route else None,
                    "accepted": turn.route.accepted if turn.route else None,
                    "method": turn.route.method if turn.route else None,
                },
            }
        )
    snap = tools.snapshot()
    used = [t["tool"] for turn in turns for t in turn["tools"]]
    tool_correct = used == case["expect_tools"]
    completed = bool(case["check"](snap))
    clarifications = sum(1 for turn in turns if "not fully sure" in turn["response"].lower())
    confirmations = sum(
        1
        for turn in turns
        for t in turn["tools"]
        if isinstance(t.get("result"), dict) and t["result"].get("status") == "confirmation_required"
    )
    escalations = sum(1 for x in used if x == "handoff_to_human")
    return {
        "id": case["id"],
        "name": case["name"],
        "intent": case["intent"],
        "turns": len(turns),
        "used_tools": "|".join(used),
        "expected_tools": "|".join(case["expect_tools"]),
        "tool_correct": tool_correct,
        "task_completed": completed,
        "clarifications": clarifications,
        "confirmations": confirmations,
        "escalations": escalations,
        "final_transfer_TXN_2002": find_transfer(snap, "TXN-2002")["status"],
        "final_card_CARD_1002": find_card(snap, "CARD-1002")["status"],
        "tickets": len(snap["tickets"]),
        "disputes": len(snap["disputes"]),
        "transcript": json.dumps(turns, ensure_ascii=False),
    }


def run_demos() -> list[dict[str, Any]]:
    from src.gate4.router import load_b3h_router, load_b4_router, load_gte

    encoder = load_gte()
    b3 = load_b3h_router(encoder)
    b4 = load_b4_router(encoder)
    demos = [
        ("D1", "B3-H", b3, ["Where is my card? I still have not received CARD-1001."]),
        ("D2", "B4-Full", b4, ["My transfer TXN-2001 has not arrived yet."]),
        ("D3", "B3-H", b3, ["Please cancel transfer TXN-2002.", "yes"]),
        ("D4", "B3-H", b3, ["How do I buy Bitcoin?"]),
        ("D5", "B3-H", b3, ["I still have not received my card CARD-2001."]),
        ("D6", "B3-H", b3, ["My transfer TXN-2001 has not arrived yet.", "transfer_not_received_by_recipient", "TXN-2001"]),
    ]
    out = []
    for demo_id, method, router, users in demos:
        path = DATA / "demos" / f"{demo_id}.db"
        if path.exists():
            path.unlink()
        tools = BankingTools(session_id=f"DEMO-{demo_id}", path=path)
        agent = BankAgent(router, tools)
        turns = []
        for user in users:
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
                        "top3": turn.route.top3 if turn.route else None,
                        "gate_top1": turn.route.gate_top1 if turn.route else None,
                    },
                    "tools": turn.tools,
                }
            )
        out.append(
            {
                "id": demo_id,
                "method": method,
                "turns": turns,
                "snapshot": tools.snapshot(),
            }
        )
    return out


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    case_rows = [run_case(c) for c in CASES]
    with (RESULTS / "gate4_agent_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [k for k in case_rows[0] if k != "transcript"] + ["transcript"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(case_rows)
    metrics = {
        "n_cases": len(case_rows),
        "task_completion_rate": sum(1 for r in case_rows if r["task_completed"]) / len(case_rows),
        "tool_call_correctness": sum(1 for r in case_rows if r["tool_correct"]) / len(case_rows),
        "clarification_rate": sum(1 for r in case_rows if r["clarifications"] > 0) / len(case_rows),
        "confirmation_rate": sum(1 for r in case_rows if r["confirmations"] > 0) / len(case_rows),
        "escalation_rate": sum(1 for r in case_rows if r["escalations"] > 0) / len(case_rows),
        "note": "16 scripted deterministic tool/state cases with StubRouter; real-router demos are separate.",
    }
    (RESULTS / "gate4_agent_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    demos = run_demos()
    (RESULTS / "gate4_agent_demos.json").write_text(json.dumps(demos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps([{"id": d["id"], "method": d["method"], "turns": len(d["turns"]), "last": d["turns"][-1]["response"]} for d in demos], ensure_ascii=False, indent=2))
    print("GATE4_AGENT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
