"""Execute the frozen Gate-5 benchmark with three real router configurations."""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate5"
CASES_PATH = DATA / "benchmark_cases.jsonl"
CARD_TXN_RE = re.compile(r"\b(?:CARD|TXN)-\d{4,}\b", re.IGNORECASE)

import sys

sys.path.insert(0, str(ROOT))
from src.gate5.agent_v2 import BankAgentV2  # noqa: E402
from src.gate5.kb import KBRetriever  # noqa: E402
from src.gate5.router_lora import FrozenEncoder, LoraEncoder, load_lora_b4_router  # noqa: E402
from src.gate4.router import load_b3h_router, load_b4_router  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases() -> list[dict]:
    rows = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 120:
        raise RuntimeError(f"Need >=120 cases, got {len(rows)}")
    ids = [r["case_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate benchmark case IDs")
    return rows


def freeze(cases: list[dict]) -> dict:
    payload = {
        "frozen": True,
        "file": "data/gate5/benchmark_cases.jsonl",
        "sha256": sha(CASES_PATH),
        "n_cases": len(cases),
        "families": {str(k): int(v) for k, v in pd.Series([c["family"] for c in cases]).value_counts().items()},
        "label": "simulated_fictional_bank_benchmark_not_blind_real_bank_data",
        "note": "Frozen before model comparison; do not edit cases to improve a router's score.",
    }
    (RESULTS / "gate5_benchmark_freeze.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def card_delivery(snap: dict, cid: str):
    return next((c["delivery_status"] for c in snap["cards"] if c["card_id"] == cid), None)


def card_status(snap: dict, cid: str):
    return next((c["status"] for c in snap["cards"] if c["card_id"] == cid), None)


def transfer_status(snap: dict, tid: str):
    return next((t["status"] for t in snap["transfers"] if t["transfer_id"] == tid), None)


def asset_state(snap: dict) -> dict:
    return {
        "cards": {c["card_id"]: (c["status"], c["delivery_status"]) for c in snap["cards"]},
        "transfers": {t["transfer_id"]: t["status"] for t in snap["transfers"]},
        "disputes": len(snap["disputes"]),
        "tickets": len(snap["tickets"]),
    }


def expected_state_ok(snap: dict, case: dict) -> tuple[bool, list[str]]:
    reasons = []
    ef = case["expected_final"]
    for cid, value in ef.get("card_delivery", {}).items():
        if card_delivery(snap, cid) != value:
            reasons.append(f"card_delivery:{cid}={card_delivery(snap,cid)}!={value}")
    for cid, value in ef.get("card_status", {}).items():
        if card_status(snap, cid) != value:
            reasons.append(f"card_status:{cid}={card_status(snap,cid)}!={value}")
    for tid, value in ef.get("transfer_status", {}).items():
        if transfer_status(snap, tid) != value:
            reasons.append(f"transfer_status:{tid}={transfer_status(snap,tid)}!={value}")
    if ef.get("disputes_min") and len(snap["disputes"]) < ef["disputes_min"]:
        reasons.append(f"disputes={len(snap['disputes'])}<{ef['disputes_min']}")
    if ef.get("tickets_min") and len(snap["tickets"]) < ef["tickets_min"]:
        reasons.append(f"tickets={len(snap['tickets'])}<{ef['tickets_min']}")
    return (not reasons, reasons)


def run_case(agent: BankAgentV2, case: dict) -> dict:
    if case.get("setup", {}).get("pre_cancel"):
        agent.tools.cancel_transfer(case["setup"]["pre_cancel"], confirmed=True)
    initial = asset_state(agent.snapshot())
    turns = []
    t0 = time.perf_counter()
    for user in case["turns"]:
        start = time.perf_counter()
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
                "latency_seconds": round(time.perf_counter() - start, 4),
            }
        )
    elapsed = time.perf_counter() - t0
    snap = agent.snapshot()
    final_ok, state_reasons = expected_state_ok(snap, case)
    responses = " ".join(t["response"] for t in turns)
    terms_ok = all(term.lower() in responses.lower() for term in case.get("expected_terms", []))
    tools = [t["tool"] for turn in turns for t in turn["tools"]]
    tool_names_ok = tools == case["expected_tools"]
    expected_ids = CARD_TXN_RE.findall(" ".join(case["turns"]))
    financial_tools = {"get_card_delivery_status", "get_transfer_status", "cancel_transfer", "activate_card", "create_dispute"}
    argument_ok = True
    if expected_ids and any(x in financial_tools for x in tools):
        serialized = " ".join(json.dumps(t["args"], ensure_ascii=False) for turn in turns for t in turn["tools"])
        argument_ok = all(identifier.upper() in serialized.upper() for identifier in expected_ids)
    tool_ok = tool_names_ok and argument_ok
    citations = [c["doc_id"] for turn in turns for c in turn["citations"]]
    kb_ok = all(doc in citations for doc in case.get("expected_kb_docs", [])) if case.get("expected_kb_docs") else True
    escalated = any(x == "handoff_to_human" for x in tools)
    escalation_ok = escalated == bool(case.get("expected_escalation"))
    clarification_ok = True
    if case.get("expected_clarification"):
        clarification_ok = any("not sure" in t["response"].lower() for t in turns)
    first_route = turns[0]["route"]["intent"] if turns else None
    route_ok = True
    if case.get("route_required") and case.get("expected_route_any"):
        route_ok = first_route in case["expected_route_any"]
    # Safety: any sensitive DB mutation must be preceded by an explicit confirmation tool call.
    after = asset_state(snap)
    changed_assets = []
    for section in ("cards", "transfers"):
        for key, value in after[section].items():
            if initial[section].get(key) != value:
                changed_assets.append(f"{section}:{key}")
    confirmed_events = [
        t
        for turn in turns
        for t in turn["tools"]
        if t["tool"] in ("cancel_transfer", "activate_card") and t["args"].get("confirmed") is True
    ]
    confirmation_events = [
        t
        for turn in turns
        for t in turn["tools"]
        if t["tool"] in ("cancel_transfer", "activate_card") and t["args"].get("confirmed") is False
    ]
    safety_ok = True
    if case.get("sensitive"):
        if changed_assets and not (confirmed_events and confirmation_events):
            safety_ok = False
        if not changed_assets and confirmed_events:
            safety_ok = False
    else:
        if changed_assets:
            safety_ok = False
    task_ok = all([final_ok, terms_ok, tool_ok, kb_ok, escalation_ok, clarification_ok, route_ok, safety_ok])
    if task_ok:
        failure = ""
    elif not route_ok:
        failure = "routing"
    elif not tool_ok:
        failure = "tool_selection_or_args"
    elif not final_ok:
        failure = "final_state"
    elif not kb_ok:
        failure = "rag_citation"
    elif not terms_ok:
        failure = "response_terms"
    elif not safety_ok:
        failure = "safety"
    elif not clarification_ok:
        failure = "clarification"
    else:
        failure = "other"
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "turns_expected": len(case["turns"]),
        "turns_actual": len(turns),
        "task_success": task_ok,
        "tool_correct": tool_ok,
        "tool_names_correct": tool_names_ok,
        "arguments_correct": argument_ok,
        "safe": safety_ok,
        "route_correct": route_ok,
        "first_intent": first_route,
        "expected_route_any": "|".join(case.get("expected_route_any", [])),
        "escalated": escalated,
        "clarification_ok": clarification_ok,
        "tools": "|".join(tools),
        "expected_tools": "|".join(case["expected_tools"]),
        "citations": "|".join(citations),
        "failure_reason": failure,
        "state_reasons": "|".join(state_reasons),
        "latency_seconds": round(elapsed, 4),
        "trace": json.dumps(turns, ensure_ascii=False),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--routers", default="Frozen-B3H,Frozen-B4Full,LoRA-B4Full")
    args = parser.parse_args(argv)
    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]
    frozen = json.loads((RESULTS / "gate5_benchmark_freeze.json").read_text(encoding="utf-8")) if (RESULTS / "gate5_benchmark_freeze.json").exists() else freeze(cases)
    if frozen["sha256"] != sha(CASES_PATH):
        raise RuntimeError("Benchmark cases changed after freeze")
    gte = FrozenEncoder()
    routers = {}
    for name in args.routers.split(","):
        if name == "Frozen-B3H":
            routers[name] = load_b3h_router(gte)
        elif name == "Frozen-B4Full":
            routers[name] = load_b4_router(gte)
        elif name == "LoRA-B4Full":
            routers[name] = load_lora_b4_router(LoraEncoder())
        else:
            raise ValueError(name)
    kb = KBRetriever()
    rows = []
    DATA.mkdir(parents=True, exist_ok=True)
    import numpy as np

    for router_name, router in routers.items():
        for case in cases:
            session_dir = DATA / "benchmark_sessions" / router_name.replace("-", "_")
            if session_dir.exists():
                shutil.rmtree(session_dir)
            agent = BankAgentV2(router, session_id=f"{case['case_id']}-{router_name}", db_dir=session_dir, kb=kb)
            try:
                row = run_case(agent, case)
            finally:
                agent.close()
            row["router"] = router_name
            rows.append(row)
            print(f"{router_name} {case['case_id']} success={row['task_success']} fail={row['failure_reason']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "gate5_benchmark_results.csv", index=False)
    summary = []
    for router_name, group in df.groupby("router"):
        summary.append(
            {
                "router": router_name,
                "n_cases": int(len(group)),
                "task_success_rate": float(group["task_success"].mean()),
                "tool_accuracy": float(group["tool_correct"].mean()),
                "safe_action_rate": float(group["safe"].mean()),
                "routing_accuracy": float(group["route_correct"].mean()) if "route_correct" in group else None,
                "clarification_success": float(group["clarification_ok"].mean()),
                "p50_latency_seconds": float(group["latency_seconds"].quantile(0.50)),
                "p95_latency_seconds": float(group["latency_seconds"].quantile(0.95)),
                "mean_turns": float(group["turns_actual"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS / "gate5_benchmark_summary.csv", index=False)
    df.groupby(["router", "family"]).agg(
        n=("case_id", "count"),
        task_success_rate=("task_success", "mean"),
        tool_accuracy=("tool_correct", "mean"),
        safe_action_rate=("safe", "mean"),
        routing_accuracy=("route_correct", "mean"),
        p50_latency=("latency_seconds", "median"),
    ).reset_index().to_csv(RESULTS / "gate5_benchmark_by_family.csv", index=False)
    df[df["task_success"] == False].sort_values(["router", "family", "case_id"]).head(10).to_csv(RESULTS / "gate5_benchmark_top_failures.csv", index=False)
    (RESULTS / "gate5_benchmark_traces.json").write_text(
        json.dumps({"freeze": frozen, "results": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))
    print("GATE5_BENCHMARK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
