"""Combine Gate-8 three-router results and compute descriptive + bootstrap stats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ROUTERS = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly"]
SLUG = {r: r.lower().replace("-", "_") for r in ROUTERS}
METRICS = ["task_success", "tool_correct", "safe", "route_correct", "clarification_ok"]


def bootstrap_ci(values: np.ndarray, n=2000, seed=20261119) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    frames = []
    for r in ROUTERS:
        p = RESULTS / f"gate8_agent_{SLUG[r]}_results.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        d = pd.read_csv(p)
        if len(d) != 102:
            raise RuntimeError(f"{r}: expected 102 rows got {len(d)}")
        d["router"] = r
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "gate8_agent_results.csv", index=False)
    summary = []
    for r, g in df.groupby("router", sort=False):
        row = {
            "router": r,
            "n": len(g),
            "task_success_rate": float(g.task_success.mean()),
            "tool_accuracy": float(g.tool_correct.mean()),
            "safe_action_rate": float(g.safe.mean()),
            "routing_accuracy": float(g.route_correct.mean()),
            "clarification_success": float(g.clarification_ok.mean()),
            "p50_latency_seconds": float(g.latency_seconds.quantile(0.5)),
            "p95_latency_seconds": float(g.latency_seconds.quantile(0.95)),
            "citation_coverage": float((g.citations.fillna("") != "").mean()),
            "tsr_ci_low": bootstrap_ci(g.task_success.to_numpy().astype(float))[0],
            "tsr_ci_high": bootstrap_ci(g.task_success.to_numpy().astype(float))[1],
        }
        summary.append(row)
    s = pd.DataFrame(summary)
    s.to_csv(RESULTS / "gate8_agent_summary.csv", index=False)
    byfam = df.groupby(["router", "family"]).task_success.agg(["count", "mean"]).reset_index()
    byfam.to_csv(RESULTS / "gate8_agent_by_family.csv", index=False)
    fail = df[df.task_success == False].groupby(["router", "failure_reason"]).size().reset_index(name="n")
    fail.to_csv(RESULTS / "gate8_agent_failures.csv", index=False)
    # Paired bootstrap difference vs GTE on the same cases.
    base = df[df.router == ROUTERS[0]].set_index("case_id").task_success.astype(float)
    diffs = []
    for r in ROUTERS[1:]:
        cur = df[df.router == r].set_index("case_id").task_success.astype(float)
        d = (cur - base).dropna().to_numpy()
        if len(d):
            rng = np.random.default_rng(20261119)
            vals = [float(rng.choice(d, size=len(d), replace=True).mean()) for _ in range(2000)]
            diffs.append({"router": r, "n_pairs": int(len(d)), "delta_tsr": float(d.mean()), "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5))})
    pd.DataFrame(diffs).to_csv(RESULTS / "gate8_agent_paired_bootstrap.csv", index=False)
    # Compact traces (JSON) for the UI/qualitative analysis.
    traces = {}
    for r, g in df.groupby("router"):
        traces[r] = [json.loads(x) for x in g.trace]
    (RESULTS / "gate8_agent_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(s.round(4).to_string(index=False))
    print(pd.DataFrame(diffs).round(4).to_string(index=False))
    print("GATE8_COMBINE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
