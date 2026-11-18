"""G5-A: validation B0/B3-H on the re-encoded LoRA-GTE BANKING77 cache."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "manifest"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5 import lora_common as lc  # noqa: E402
from src.gate3.methods import (  # noqa: E402
    evaluate,
    knn_scores,
    prototypes_multi,
    prototypes_single,
    scores_from_prototypes,
    select_threshold,
)


def main() -> int:
    training = json.loads((RESULTS / "gate5_lora_training.json").read_text(encoding="utf-8"))
    path = lc.lora_cache_path(training["adapter_sha256"])
    data = np.load(path, allow_pickle=False)
    vectors = data["vectors"].astype(np.float32)
    sample_ids = data["sample_ids"].astype(str)
    roles = data["roles"].astype(str)
    labels = data["labels"].astype(str)
    classes = lc.classes()
    support_rows = lc.support_rows()
    idx = {sid: i for i, sid in enumerate(sample_ids)}
    support_idx = [idx[r["sample_id"]] for r in support_rows]
    support_X = vectors[support_idx]
    support_y = np.asarray([r["intent"] for r in support_rows], dtype=object)

    def eval_rows(role: str):
        mask = roles == role
        lab = labels[mask]
        cmap = {c: i for i, c in enumerate(classes)}
        return vectors[mask], np.asarray([cmap.get(x, -1) for x in lab], dtype=int)

    cal_X, cal_y = eval_rows("calibration_known")
    dev_X, dev_y = eval_rows("dev_unknown")
    val_X = np.concatenate([cal_X, dev_X])
    val_y = np.concatenate([cal_y, dev_y])

    rows = []
    for method in ("B0", "B3-H"):
        if method == "B0":
            protos = prototypes_single(support_X, support_y, classes)
            sims = scores_from_prototypes(val_X, [p[None, :] for p in protos])
        else:
            protos, _ = prototypes_multi(support_X, support_y, classes)
            p = scores_from_prototypes(val_X, protos)
            n = knn_scores(val_X, support_X, support_y, classes, top_k=2)
            sims = 0.5 * p + 0.5 * n
        tau, _, info = select_threshold(sims, val_y, len(classes), classes)
        metrics, _, _ = evaluate(sims, val_y, classes, tau)
        rows.append(
            {
                "encoder": "LoRA-GTE",
                "method": method,
                "tau": tau,
                "constraint_met": info["constraint_met"],
                **{f"val_{k}": v for k, v in metrics.items()},
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "gate5_lora_validation.csv", index=False)

    frozen = pd.read_csv(RESULTS / "gate3_validation_metrics.csv")
    frozen = frozen[(frozen.dataset == "banking77") & (frozen.model == "gte") & (frozen.shot == 10) & (frozen.method.isin(["B0", "B3-H"]))]
    print("LoRA validation:")
    print(pd.DataFrame(rows)[["method", "tau", "val_known_acceptance", "val_oos_recall", "val_unknown_auroc", "val_open_set_macro_f1"]].to_string(index=False))
    print("Frozen Gate3 validation:")
    print(frozen[["method", "val_known_acceptance", "val_oos_recall", "val_unknown_auroc", "val_open_set_macro_f1"]].to_string(index=False))
    print("GATE5_LORA_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
