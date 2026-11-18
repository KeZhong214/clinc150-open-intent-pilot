"""Gate-8 independent frozen simulated benchmark: three routers, same cases."""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate8"
CASES = DATA / "benchmark_cases.jsonl"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5.benchmark import run_case  # noqa: E402
from src.gate5.kb import KBRetriever  # noqa: E402
from src.gate8.agent import Gate8Agent  # noqa: E402
from src.gate8.routers import ROUTER_NAMES, build  # noqa: E402

SLUG = {r: r.lower().replace("-", "_") for r in ROUTER_NAMES}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases() -> list[dict]:
    rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 100:
        raise RuntimeError("Gate8 needs >=100 cases")
    if len({r["case_id"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate case ids")
    return rows


def freeze(cases: list[dict]) -> dict:
    meta = {r: __import__("src.gate8.routers", fromlist=["metadata"]).metadata(r) for r in ROUTER_NAMES}
    payload = {
        "frozen": True,
        "label": "new_frozen_simulated_test_not_external_blind",
        "file": "data/gate8/benchmark_cases.jsonl",
        "sha256": sha(CASES),
        "n_cases": len(cases),
        "families": {str(k): int(v) for k, v in pd.Series([c["family"] for c in cases]).value_counts().items()},
        "routers": meta,
        "threshold_protocol": "Gate7 dev_threshold for all three routers",
        "note": "Cases/models/thresholds frozen before the final three-router run; no test-based tuning.",
    }
    (RESULTS / "gate8_benchmark_freeze.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--router", choices=ROUTER_NAMES, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)
    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]
    if not (RESULTS / "gate8_benchmark_freeze.json").exists():
        raise RuntimeError("Freeze the cases before running the final benchmark")
    frozen = json.loads((RESULTS / "gate8_benchmark_freeze.json").read_text(encoding="utf-8"))
    if frozen["sha256"] != sha(CASES):
        raise RuntimeError("Cases changed after Gate8 freeze")
    router, meta = build(args.router)
    kb = KBRetriever()
    rows = []
    session_root = DATA / "sessions" / SLUG[args.router]
    if session_root.exists():
        shutil.rmtree(session_root)
    for case in cases:
        agent = Gate8Agent(router, session_id=f"G8-{case['case_id']}", db_dir=session_root, kb=kb)
        try:
            row = run_case(agent, case)
        finally:
            agent.close()
        row["router"] = args.router
        row["router_meta"] = json.dumps(meta, ensure_ascii=False)
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / f"gate8_agent_{SLUG[args.router]}_results.csv", index=False)
    print(d.groupby("family").task_success.agg(["count", "mean"]).round(4).to_string())
    print(json.dumps({"router": args.router, "rows": len(d), "tsr": round(float(d.task_success.mean()), 6), "label": frozen["label"]}, ensure_ascii=False))
    print("GATE8_BENCHMARK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
