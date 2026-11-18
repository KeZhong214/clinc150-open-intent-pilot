"""G6-D: the frozen 120-case Agent regression with real Qwen routers.

This is a historical/exploratory regression: the 120 tasks were already used in
Gate 5.  No rules or case expectations are modified here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CASES = ROOT / "data" / "gate5" / "benchmark_cases.jsonl"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5.agent_v2 import BankAgentV2  # noqa: E402
from src.gate5.benchmark import load_cases, run_case  # noqa: E402
from src.gate6.router_qwen import load_qwen_router  # noqa: E402

ROUTERS = ["Qwen-Frozen-B3H", "Qwen-Plain-B3H", "Qwen-Plain-RejectOnly", "Qwen-Plain-B4Full"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", choices=ROUTERS + [""], default="")
    args = parser.parse_args()
    cases = load_cases()
    gate5_freeze = json.loads((RESULTS / "gate5_benchmark_freeze.json").read_text(encoding="utf-8"))
    import hashlib

    if gate5_freeze["sha256"] != hashlib.sha256(CASES.read_bytes()).hexdigest():
        raise RuntimeError("Gate5 cases changed")
    session_root = ROOT / "data" / "gate6_qwen_benchmark_sessions"
    if session_root.exists():
        shutil.rmtree(session_root)
    session_root.mkdir(parents=True, exist_ok=True)
    encoder_cache = {}
    rows = []
    traces = []
    t_all = time.perf_counter()
    selected = [args.router] if args.router else ROUTERS
    for router_name in selected:
        router = load_qwen_router(router_name, encoder_cache=encoder_cache)
        db_dir = session_root / router_name
        db_dir.mkdir(parents=True, exist_ok=True)
        t_router = time.perf_counter()
        for i, case in enumerate(cases, 1):
            agent = BankAgentV2(router, session_id=case["case_id"], db_dir=db_dir)
            row = run_case(agent, case)
            row["router"] = router_name
            row["case_index"] = i
            rows.append(row)
            traces.append({"router": router_name, **{k: row[k] for k in ("case_id", "family", "task_success", "failure_reason", "trace")}})
            agent.close()
            if i % 20 == 0:
                print(f"{router_name} {i}/{len(cases)} elapsed={time.perf_counter()-t_router:.1f}s", flush=True)
        print(f"{router_name} done {len(cases)} cases in {time.perf_counter()-t_router:.1f}s", flush=True)
        del router
        encoder_cache.clear()
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    if args.router:
        slug = args.router.lower().replace("qwen-", "").replace("-", "_")
        df.to_csv(RESULTS / f"gate6_qwen_benchmark_{slug}_results.csv", index=False)
        (RESULTS / f"gate6_qwen_benchmark_{slug}_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(df.groupby("router").task_success.mean().to_string())
        print(f"total elapsed {time.perf_counter()-t_all:.1f}s")
        print("GATE6_QWEN_BENCHMARK_PART_OK")
        return 0
    df.to_csv(RESULTS / "gate6_qwen_benchmark_results.csv", index=False)
    (RESULTS / "gate6_qwen_benchmark_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = []
    for router, g in df.groupby("router"):
        cl = [x["case_id"] for x in cases if x.get("expected_clarification")]
        summary.append(
            {
                "router": router,
                "n_cases": len(g),
                "task_success_rate": round(float(g.task_success.mean()), 6),
                "tool_accuracy": round(float(g.tool_correct.mean()), 6),
                "safe_action_rate": round(float(g.safe.mean()), 6),
                "routing_accuracy": round(float(g.route_correct.mean()), 6),
                "clarification_success": round(float(g[g.case_id.isin(cl)].clarification_ok.mean()), 6),
                "p50_latency_seconds": round(float(g.latency_seconds.quantile(0.5)), 6),
                "p95_latency_seconds": round(float(g.latency_seconds.quantile(0.95)), 6),
                "mean_turns": round(float(g.turns_actual.mean()), 6),
            }
        )
    pd.DataFrame(summary).to_csv(RESULTS / "gate6_qwen_benchmark_summary.csv", index=False)
    byfam = df.groupby(["family", "router"]).task_success.mean().reset_index(name="task_success_rate")
    byfam.to_csv(RESULTS / "gate6_qwen_benchmark_by_family.csv", index=False)
    fails = df[df.task_success == False].groupby(["router", "failure_reason"]).size().reset_index(name="count")
    fails.to_csv(RESULTS / "gate6_qwen_benchmark_failures.csv", index=False)
    top = df[df.task_success == False].sort_values(["router", "latency_seconds"], ascending=[True, False]).groupby("router").head(10)
    top[["router", "case_id", "family", "failure_reason", "first_intent", "expected_route_any", "tools", "expected_tools", "state_reasons"]].to_csv(RESULTS / "gate6_qwen_benchmark_top_failures.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"total elapsed {time.perf_counter()-t_all:.1f}s")
    print("GATE6_QWEN_BENCHMARK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
