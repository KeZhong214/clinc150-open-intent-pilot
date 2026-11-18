"""Combine the four per-router Qwen Agent benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CASES = ROOT / "data" / "gate5" / "benchmark_cases.jsonl"

ROUTERS = {
    "Qwen-Frozen-B3H": "frozen_b3h",
    "Qwen-Plain-B3H": "plain_b3h",
    "Qwen-Plain-RejectOnly": "plain_rejectonly",
    "Qwen-Plain-B4Full": "plain_b4full",
}


def main() -> int:
    cases = [json.loads(x) for x in CASES.read_text(encoding="utf-8").splitlines() if x.strip()]
    frames = []
    traces = []
    for router, slug in ROUTERS.items():
        p = RESULTS / f"gate6_qwen_benchmark_{slug}_results.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        d = pd.read_csv(p)
        d["router"] = router
        frames.append(d)
        traces.extend(json.loads((RESULTS / f"gate6_qwen_benchmark_{slug}_traces.json").read_text(encoding="utf-8")))
    df = pd.concat(frames, ignore_index=True)
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
    print("GATE6_QWEN_BENCHMARK_COMBINE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
