"""Evaluate one Qwen adapter on the frozen Gate-7 two-level development split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate6.train_qwen_b4 import val_scores  # noqa: E402
from src.gate3.methods import evaluate, knn_scores, prototypes_multi, prototypes_single, scores_from_prototypes, select_threshold  # noqa: E402
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402


def load_cache(training_json: str, tag: str) -> dict:
    if not training_json:
        c = qd.load("frozen")
        c["encoder"] = "qwen_frozen"
        return c
    train = json.loads((ROOT / training_json).read_text(encoding="utf-8"))
    d = np.load(qe.lora_cache(train["adapter_sha256"]), allow_pickle=False)
    return {
        "vectors": d["vectors"].astype(np.float32),
        "sample_ids": d["sample_ids"].astype(str),
        "roles": d["roles"].astype(str),
        "labels": d["labels"].astype(str),
        "intents": d["intents"].astype(str),
        "encoder": f"qwen_{tag}_lora",
        "run": tag,
        "training": train,
    }


def split(name: str) -> tuple[list[str], np.ndarray]:
    rows = pd.read_csv(ROOT / "data" / "manifest" / f"gate7_{name}.csv")
    cmap = {c: i for i, c in enumerate(qe.classes())}
    return rows.sample_id.tolist(), np.asarray([cmap.get(x, -1) for x in rows.intent], dtype=int)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--training-json", default="")
    p.add_argument("--b4-checkpoint", default="")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    cache = load_cache(args.training_json, args.tag)
    classes = qe.classes()
    thr_ids, thr_y = split("dev_threshold")
    sel_ids, sel_y = split("dev_select")
    idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    q_thr = cache["vectors"][[idx[sid] for sid in thr_ids]]
    q_sel = cache["vectors"][[idx[sid] for sid in sel_ids]]
    X, y = qd.support(cache)
    rows = []
    for method in ("B0", "B3-H"):
        if method == "B0":
            protos = prototypes_single(X, y, classes)
            s_thr = scores_from_prototypes(q_thr, [p[None, :] for p in protos])
            s_sel = scores_from_prototypes(q_sel, [p[None, :] for p in protos])
        else:
            protos, _ = prototypes_multi(X, y, classes)
            p_thr = scores_from_prototypes(q_thr, protos)
            p_sel = scores_from_prototypes(q_sel, protos)
            n_thr = knn_scores(q_thr, X, y, classes, top_k=2)
            n_sel = knn_scores(q_sel, X, y, classes, top_k=2)
            s_thr = 0.5 * p_thr + 0.5 * n_thr
            s_sel = 0.5 * p_sel + 0.5 * n_sel
        tau, _, info = select_threshold(s_thr, thr_y, len(classes), classes)
        metrics, _, _ = evaluate(s_sel, sel_y, classes, tau)
        rows.append({"tag": args.tag, "encoder": cache["encoder"], "method": method, "tau": tau, "constraint_met": info["constraint_met"], **{f"select_{k}": v for k, v in metrics.items()}})
    if args.b4_checkpoint:
        payload = torch.load(ROOT / args.b4_checkpoint, map_location="cpu", weights_only=False)
        model = B4AR()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        s_thr, c_thr, _, _ = val_scores(model, cache, q_thr, classes, reject_only=True)
        s_sel, c_sel, _, _ = val_scores(model, cache, q_sel, classes, reject_only=True)
        tau, _, info = b4_core.select_threshold_custom(s_thr, c_thr, thr_y, classes)
        metrics, _, _ = b4_core.evaluate_custom(s_sel, c_sel, sel_y, tau, classes)
        rows.append({"tag": args.tag, "encoder": cache["encoder"], "method": "B4-RejectOnly", "tau": tau, "constraint_met": info["constraint_met"], **{f"select_{k}": v for k, v in metrics.items()}})
    out = pd.DataFrame(rows)
    path = ROOT / args.out if args.out else RESULTS / f"gate7_dev_{args.tag}.csv"
    out.to_csv(path, index=False)
    print(out.round(5).to_string(index=False))
    print("GATE7_DEV_EVAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
