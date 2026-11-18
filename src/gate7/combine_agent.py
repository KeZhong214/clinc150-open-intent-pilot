"""Combine Gate-7 per-router Agent benchmark partials."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CASES = ROOT / "data" / "gate5" / "benchmark_cases.jsonl"
ROUTERS = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Opt-RejectOnly"]


def main() -> int:
    cases = [json.loads(x) for x in CASES.read_text(encoding="utf-8").splitlines() if x.strip()]
    frames = []
    traces = []
    for r in ROUTERS:
        slug = r.lower().replace("-", "_")
        p = RESULTS / f"gate7_agent_{slug}_results.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_csv(p))
        traces.extend(json.loads((RESULTS / f"gate7_agent_{slug}_traces.json").read_text(encoding="utf-8")))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "gate7_agent_results.csv", index=False)
    (RESULTS / "gate7_agent_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for router, g in df.groupby("router"):
        cl = [x["case_id"] for x in cases if x.get("expected_clarification")]
        rows.append({
            "router": router,
            "n": len(g),
            "task_success_rate": round(float(g.task_success.mean()), 6),
            "tool_accuracy": round(float(g.tool_correct.mean()), 6),
            "safe_action_rate": round(float(g.safe.mean()), 6),
            "routing_accuracy": round(float(g.route_correct.mean()), 6),
            "clarification_success": round(float(g[g.case_id.isin(cl)].clarification_ok.mean()), 6),
            "p50_latency_seconds": round(float(g.latency_seconds.quantile(0.5)), 6),
            "p95_latency_seconds": round(float(g.latency_seconds.quantile(0.95)), 6),
        })
    pd.DataFrame(rows).to_csv(RESULTS / "gate7_agent_summary.csv", index=False)
    byfam = df.groupby(["family", "router"]).task_success.mean().reset_index(name="task_success_rate")
    byfam.to_csv(RESULTS / "gate7_agent_by_family.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("GATE7_AGENT_COMBINE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
