"""G7-D: same 120-case exploratory Agent regression for GTE/Qwen RejectOnly."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CASES = ROOT / "data" / "gate5" / "benchmark_cases.jsonl"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5.agent_v2 import BankAgentV2  # noqa: E402
from src.gate5.benchmark import load_cases, run_case  # noqa: E402
from src.gate4.router import load_bank_state  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402
from src.gate5.router_lora import FrozenEncoder  # noqa: E402
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate6.router_qwen import QwenEncoderAdapter, QwenRejectOnlyRouter  # noqa: E402
from src.gate7.router_opt import qwen_reject_router  # noqa: E402

ROUTERS = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly"]
LEGACY_ROUTER = {"Qwen-Opt-RejectOnly": "Qwen-Plain42-RejectOnly"}


def gte_reject():
    classes, X, y = load_bank_state()
    payload = torch.load(RESULTS / "gate5_reject_only_frozen.pt", map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tau = json.loads((RESULTS / "gate5_reject_only_frozen_threshold.json").read_text(encoding="utf-8"))["tau"]
    return QwenRejectOnlyRouter(classes, X, y, model, tau, encoder=FrozenEncoder())


def qwen_frozen_reject():
    cache = qd.load("frozen")
    classes = qe.classes()
    X, y = qd.support(cache)
    payload = torch.load(RESULTS / "gate6_qwen_frozen_reject.pt", map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tau = json.loads((RESULTS / "gate6_qwen_frozen_reject_threshold.json").read_text(encoding="utf-8"))["tau"]
    return QwenRejectOnlyRouter(classes, X, y, model, tau, encoder=QwenEncoderAdapter("frozen"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--router", choices=ROUTERS + list(LEGACY_ROUTER), required=True)
    p.add_argument("--opt-tag", default="p1")
    args = p.parse_args(argv)
    args.router = LEGACY_ROUTER.get(args.router, args.router)
    if args.router == "GTE-Frozen-RejectOnly":
        router = gte_reject()
    elif args.router == "Qwen-Frozen-RejectOnly":
        router = qwen_frozen_reject()
    else:
        if args.opt_tag == "plain42":
            router = qwen_reject_router("results/gate6_qwen_plain_training.json", "results/gate6_qwen_plain_reject.pt", "plain42")
        else:
            router = qwen_reject_router(f"results/gate7_qwen_{args.opt_tag}_training.json", f"results/gate7_qwen_{args.opt_tag}_reject.pt", args.opt_tag)
    cases = load_cases()
    root = ROOT / "data" / "gate7_agent_sessions" / args.router
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = []
    for i, case in enumerate(cases, 1):
        agent = BankAgentV2(router, session_id=case["case_id"], db_dir=root)
        row = run_case(agent, case)
        row["router"] = args.router
        rows.append(row)
        traces.append({"router": args.router, **{k: row[k] for k in ("case_id", "family", "task_success", "failure_reason", "trace")}})
        agent.close()
        if i % 20 == 0:
            print(f"{args.router} {i}/{len(cases)}", flush=True)
    slug = args.router.lower().replace("-", "_")
    pd.DataFrame(rows).to_csv(RESULTS / f"gate7_agent_{slug}_results.csv", index=False)
    (RESULTS / f"gate7_agent_{slug}_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).task_success.mean())
    print("GATE7_AGENT_PART_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
