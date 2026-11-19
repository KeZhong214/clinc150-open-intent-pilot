"""Combine Gate 8.1 results with case- and family-cluster bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ROUTERS = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly"]
SLUG = {r: r.lower().replace("-", "_") for r in ROUTERS}


def case_ci(v, n=2000, seed=20261119):
    rng = np.random.default_rng(seed)
    vals = [float(rng.choice(v, len(v), replace=True).mean()) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cluster_ci(df, n=2000, seed=20261119):
    fams = sorted(df.family.unique())
    groups = {f: df[df.family == f].task_success.astype(float).to_numpy() for f in fams}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        draw = rng.choice(fams, size=len(fams), replace=True)
        vals.append(float(np.concatenate([groups[f] for f in draw]).mean()))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    frames = []
    for r in ROUTERS:
        p = RESULTS / f"gate8_1_agent_{SLUG[r]}_final_results.csv"
        if not p.exists():
            p = RESULTS / f"gate8_1_agent_{SLUG[r]}_v1_results.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        d = pd.read_csv(p)
        if len(d) != 120:
            raise RuntimeError(f"{r}: {len(d)}")
        d["router"] = r
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "gate8_1_agent_results.csv", index=False)
    rows = []
    for r, g in df.groupby("router", sort=False):
        cilo, cihi = case_ci(g.task_success.to_numpy(float))
        cllo, clhi = cluster_ci(g)
        rows.append({
            "router": r, "n": len(g), "n_families": g.family.nunique(),
            "task_success_rate": float(g.task_success.mean()),
            "tool_accuracy": float(g.tool_correct.mean()),
            "safe_action_rate": float(g.safe.mean()),
            "routing_accuracy": float(g.route_correct.mean()),
            "clarification_success": float(g.clarification_ok.mean()),
            "citation_coverage": float((g.citations.fillna("") != "").mean()),
            "p50_latency_seconds": float(g.latency_seconds.quantile(.5)),
            "p95_latency_seconds": float(g.latency_seconds.quantile(.95)),
            "tsr_case_ci_low": cilo, "tsr_case_ci_high": cihi,
            "tsr_cluster_ci_low": cllo, "tsr_cluster_ci_high": clhi,
        })
    s = pd.DataFrame(rows)
    s.to_csv(RESULTS / "gate8_1_agent_summary.csv", index=False)
    df.groupby(["router", "family"]).task_success.agg(["count", "mean"]).reset_index().to_csv(RESULTS / "gate8_1_agent_by_family.csv", index=False)
    df.groupby(["router", "template_id"]).task_success.agg(["count", "mean"]).reset_index().to_csv(RESULTS / "gate8_1_agent_by_template.csv", index=False)
    df[~df.task_success].groupby(["router", "failure_reason"]).size().reset_index(name="n").to_csv(RESULTS / "gate8_1_agent_failures.csv", index=False)

    # paired family-cluster bootstrap vs GTE
    base = df[df.router == ROUTERS[0]]
    fams = sorted(base.family.unique())
    pairs = []
    for r in ROUTERS[1:]:
        cur = df[df.router == r]
        gs = []
        for f in fams:
            a = base[base.family == f].set_index("case_id").task_success.astype(float)
            b = cur[cur.family == f].set_index("case_id").task_success.astype(float)
            gs.append((b - a).dropna().to_numpy())
        rng = np.random.default_rng(20261119)
        vals = []
        for _ in range(2000):
            draw = rng.choice(len(fams), size=len(fams), replace=True)
            vals.append(float(np.concatenate([gs[i] for i in draw]).mean()))
        diff = float(np.concatenate(gs).mean())
        pairs.append({"router": r, "delta_tsr": diff, "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5)), "clusters": len(fams)})
    pd.DataFrame(pairs).to_csv(RESULTS / "gate8_1_agent_cluster_bootstrap.csv", index=False)
    traces = {r: [json.loads(x) for x in g.trace] for r, g in df.groupby("router")}
    (RESULTS / "gate8_1_agent_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(s.round(4).to_string(index=False))
    print(pd.DataFrame(pairs).round(4).to_string(index=False))
    print("GATE8_1_COMBINE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
