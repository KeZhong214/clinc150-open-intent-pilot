"""G6-B: validation B0/B3-H for Qwen Plain/Anchor LoRA adapters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate3.methods import evaluate, knn_scores, prototypes_multi, prototypes_single, scores_from_prototypes, select_threshold  # noqa: E402


def main() -> int:
    frozen = qe.frozen_cache()
    rows = qe.support_rows()
    ids = [r["sample_id"] for r in rows]
    classes = qe.classes()
    cmap = {c: i for i, c in enumerate(classes)}
    support_y = np.asarray([r["intent"] for r in rows], dtype=object)
    f_idx = {sid: i for i, sid in enumerate(frozen["sample_ids"])}
    frozen_support = frozen["vectors"][[f_idx[sid] for sid in ids]]
    manifest = qe.manifest_rows()
    cal = [r for r in manifest if r["role"] == "calibration_known"]
    dev = [r for r in manifest if r["role"] == "dev_unknown"]
    val_ids = [r["sample_id"] for r in cal + dev]
    val_y = np.asarray([cmap[r["intent"]] for r in cal] + [-1] * len(dev), dtype=int)

    out_rows = []
    for run in ("plain", "anchor"):
        training = json.loads((RESULTS / f"gate6_qwen_{run}_training.json").read_text(encoding="utf-8"))
        path = qe.lora_cache(training["adapter_sha256"])
        data = np.load(path, allow_pickle=False)
        vectors = data["vectors"].astype(np.float32)
        sidx = {sid: i for i, sid in enumerate(data["sample_ids"].astype(str))}
        support = vectors[[sidx[sid] for sid in ids]]
        val = vectors[[sidx[sid] for sid in val_ids]]
        for method in ("B0", "B3-H"):
            if method == "B0":
                protos = prototypes_single(support, support_y, classes)
                sims = scores_from_prototypes(val, [p[None, :] for p in protos])
            else:
                protos, _ = prototypes_multi(support, support_y, classes)
                p = scores_from_prototypes(val, protos)
                n = knn_scores(val, support, support_y, classes, top_k=2)
                sims = 0.5 * p + 0.5 * n
            tau, _, info = select_threshold(sims, val_y, len(classes), classes)
            metrics, _, _ = evaluate(sims, val_y, classes, tau)
            out_rows.append(
                {
                    "encoder": "Qwen-LoRA",
                    "run": run,
                    "method": method,
                    "tau": tau,
                    "constraint_met": info["constraint_met"],
                    **{f"val_{k}": v for k, v in metrics.items()},
                    "adapter_sha256": training["adapter_sha256"],
                    "vector_shift_mean_cosine": training["vector_check"]["frozen_vs_lora_mean_cosine"],
                }
            )
    pd.DataFrame(out_rows).to_csv(RESULTS / "gate6_qwen_lora_validation.csv", index=False)
    g3 = pd.read_csv(RESULTS / "gate3_validation_metrics.csv")
    g3 = g3[(g3.dataset == "banking77") & (g3.model == "qwen") & (g3.shot == 10) & (g3.method.isin(["B0", "B3-H"]))]
    print(pd.DataFrame(out_rows)[["run", "method", "val_known_acceptance", "val_known_correct_routing", "val_oos_recall", "val_open_set_macro_f1", "vector_shift_mean_cosine"]].round(5).to_string(index=False))
    print("Frozen:")
    print(g3[["method", "val_known_acceptance", "val_known_correct_routing", "val_oos_recall", "val_open_set_macro_f1"]].round(5).to_string(index=False))
    print("GATE6_QWEN_LORA_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
