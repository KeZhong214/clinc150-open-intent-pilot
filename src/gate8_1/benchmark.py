"""Gate 8.1: evaluate the paired routers on the new frozen simulated set."""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "gate8_1"
CASES = DATA / "benchmark_cases.jsonl"
sys.path.insert(0, str(ROOT))

from src.gate5.benchmark import run_case  # noqa: E402
from src.gate5.kb import KBRetriever  # noqa: E402
from src.gate8.agent import Gate8Agent  # noqa: E402
from src.gate8_1.registry import validate  # noqa: E402
from src.gate8_1.routers import ROUTER_NAMES, build  # noqa: E402

SLUG = {r: r.lower().replace("-", "_") for r in ROUTER_NAMES}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_cases() -> list[dict]:
    rows = [json.loads(x) for x in CASES.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) < 100:
        raise RuntimeError(len(rows))
    if len({x["case_id"] for x in rows}) != len(rows):
        raise RuntimeError("duplicate case id")
    if len({x["template_id"] for x in rows}) < 60:
        raise RuntimeError("need >=60 distinct templates")
    return rows


def freeze(cases: list[dict]) -> dict:
    payload = {
        "frozen": True,
        "label": "new_frozen_simulated_test_not_external_blind",
        "file": "data/gate8_1/benchmark_cases.jsonl",
        "sha256": sha(CASES),
        "n_cases": len(cases),
        "n_templates": len({x["template_id"] for x in cases}),
        "families": {str(k): int(v) for k, v in pd.Series([x["family"] for x in cases]).value_counts().items()},
        "routers": {r: validate(r) for r in ROUTER_NAMES},
        "note": "Cases, paired model registry and code commit frozen before the final three-router run; no test-based tuning.",
    }
    (RESULTS / "gate8_1_benchmark_freeze.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--router", choices=ROUTER_NAMES, required=True)
    p.add_argument("--cases", default=str(CASES))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tag", default="v1")
    args = p.parse_args(argv)
    cases_path = Path(args.cases)
    rows_json = [json.loads(x) for x in cases_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        rows_json = rows_json[: args.limit]
    freeze_path = RESULTS / "gate8_1_benchmark_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("freeze the final cases before final run")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if args.cases == str(CASES) and frozen["sha256"] != sha(CASES):
        raise RuntimeError("final cases changed after freeze")
    router, meta = build(args.router)
    kb = KBRetriever()
    session_root = DATA / "sessions" / (SLUG[args.router] + "_" + args.tag)
    if session_root.exists():
        shutil.rmtree(session_root)
    rows = []
    for case in rows_json:
        agent = Gate8Agent(router, session_id=f"G81-{case['case_id']}", db_dir=session_root, kb=kb)
        try:
            row = run_case(agent, case)
        finally:
            agent.close()
        row["router"] = args.router
        row["router_meta"] = json.dumps(meta, ensure_ascii=False)
        row["template_id"] = case.get("template_id", case["case_id"])
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / f"gate8_1_agent_{SLUG[args.router]}_{args.tag}_results.csv", index=False)
    print(d.groupby("family").task_success.agg(["count", "mean"]).round(4).to_string())
    print(json.dumps({"router": args.router, "rows": len(d), "tsr": round(float(d.task_success.mean()), 6), "label": frozen["label"]}, ensure_ascii=False))
    print("GATE8_1_BENCHMARK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
