"""Run the Gate-3 3 datasets x 4 encoders x 2 shots x B0/B3-P/B3-H matrix.

Phase 1 (validation) selects thresholds on known calibration + dev-unknown only.
Phase 2 (test) applies frozen thresholds to known test + final-unknown once.

Usage:
    python -m src.gate3.run_matrix --phase validation
    python -m src.gate3.run_matrix --phase test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "manifest"
CACHE = PROJECT_ROOT / "data" / "cache_embeddings" / "gate3"
RESULTS = PROJECT_ROOT / "results"
DATASETS = ("clinc150", "banking77", "massive")
MODELS = ("bge", "qwen", "gte", "e5")
METHODS = ("B0", "B3-P", "B3-H")
SHOTS = (5, 10)

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.gate3.encode import MODELS as MODEL_INFO, output_paths, prompt_for  # noqa: E402
from src.gate3.methods import (  # noqa: E402
    evaluate,
    knn_scores,
    make_scores,
    prototypes_multi,
    prototypes_single,
    scores_from_prototypes,
    select_threshold,
)


def load_dataset(dataset: str) -> dict[str, Any]:
    path = MANIFEST / f"gate3_{dataset}_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"Run src.gate3.prepare first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_cache(model: str, dataset: str) -> dict[str, Any]:
    prompt, prompt_hash = prompt_for(model)
    npz_path, meta_path = output_paths(model, dataset, prompt_hash)
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing cache for {model}/{dataset}: {npz_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data = np.load(npz_path, allow_pickle=False)
    vectors = data["vectors"].astype(np.float32)
    sample_ids = data["sample_ids"].astype(str)
    roles = data["roles"].astype(str)
    labels = data["labels"].astype(str)
    intents = data["intents"].astype(str)
    if not (len(vectors) == len(sample_ids) == len(roles) == len(labels) == len(intents)):
        raise RuntimeError(f"Inconsistent cache arrays for {model}/{dataset}")
    return {
        "vectors": vectors,
        "sample_ids": sample_ids,
        "roles": roles,
        "labels": labels,
        "intents": intents,
        "meta": meta,
    }


def support_data(cache: dict[str, Any], dataset: str, shot: int, classes: Sequence[str]):
    path = MANIFEST / f"gate3_{dataset}_support_{shot}.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    ids = [row["sample_id"] for row in rows]
    id_to_idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    missing = [sid for sid in ids if sid not in id_to_idx]
    if missing:
        raise RuntimeError(f"Support IDs not in cache: {missing[:3]}")
    idx = np.asarray([id_to_idx[sid] for sid in ids], dtype=int)
    X = cache["vectors"][idx]
    y = np.asarray([row["intent"] for row in rows], dtype=object)
    if set(y) != set(classes) or any(np.sum(y == c) != shot for c in classes):
        raise RuntimeError(f"Bad support layout for {dataset}, shot={shot}")
    return X, y


def eval_data(cache: dict[str, Any], role: str, classes: Sequence[str]):
    mask = cache["roles"] == role
    labels = cache["labels"][mask]
    idx_map = {name: i for i, name in enumerate(classes)}
    true_idx = np.asarray([idx_map.get(label, -1) for label in labels], dtype=int)
    return cache["vectors"][mask], true_idx


def scores_for(
    method: str,
    support_X: np.ndarray,
    support_y: np.ndarray,
    vectors: np.ndarray,
    classes: Sequence[str],
    *,
    b3p_protos=None,
    b3p_details=None,
):
    if method == "B0":
        protos = prototypes_single(support_X, support_y, classes)
        return scores_from_prototypes(vectors, [p[None, :] for p in protos]), {}
    if method == "B3-P":
        protos, details = prototypes_multi(support_X, support_y, classes)
        return scores_from_prototypes(vectors, protos), details
    if method == "B3-H":
        if b3p_protos is None:
            b3p_protos, b3p_details = prototypes_multi(support_X, support_y, classes)
        p_scores = scores_from_prototypes(vectors, b3p_protos)
        n_scores = knn_scores(vectors, support_X, support_y, classes, top_k=2)
        return 0.5 * p_scores + 0.5 * n_scores, b3p_details or {}
    raise ValueError(method)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--force", action="store_true", help="Only for explicit test reruns.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.seed != 42:
        raise RuntimeError("Gate 3 is frozen to seed=42")

    if args.phase == "validation":
        if args.force:
            raise RuntimeError("--force is only for a deliberate test rerun")
        validation_rows = []
        thresholds: dict[str, Any] = {}
        details_all: dict[str, Any] = {}
        grids = []
        for dataset in DATASETS:
            info = load_dataset(dataset)
            classes = info["known_classes"]
            for model in MODELS:
                cache = load_cache(model, dataset)
                for shot in SHOTS:
                    support_X, support_y = support_data(cache, dataset, shot, classes)
                    b3p_protos, b3p_details = prototypes_multi(support_X, support_y, classes)
                    val_vectors = np.concatenate(
                        [
                            eval_data(cache, "calibration_known", classes)[0],
                            eval_data(cache, "dev_unknown", classes)[0],
                        ]
                    )
                    val_true = np.concatenate(
                        [
                            eval_data(cache, "calibration_known", classes)[1],
                            eval_data(cache, "dev_unknown", classes)[1],
                        ]
                    )
                    for method in METHODS:
                        sims, detail = scores_for(
                            method,
                            support_X,
                            support_y,
                            val_vectors,
                            classes,
                            b3p_protos=b3p_protos if method in ("B3-P", "B3-H") else None,
                            b3p_details=b3p_details if method in ("B3-P", "B3-H") else None,
                        )
                        tau, grid, select_info = select_threshold(sims, val_true, len(classes), classes)
                        metrics, _, _ = evaluate(sims, val_true, classes, tau)
                        key = f"{dataset}|{model}|{shot}|{method}"
                        thresholds[key] = {
                            "dataset": dataset,
                            "model": model,
                            "shot": shot,
                            "method": method,
                            "tau": tau,
                            "selection": select_info,
                        }
                        validation_rows.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "shot": shot,
                                "method": method,
                                "tau": tau,
                                "constraint_met": select_info["constraint_met"],
                                **{f"val_{k}": v for k, v in metrics.items()},
                            }
                        )
                        details_all[key] = detail
                        for row in grid:
                            grids.append(
                                {
                                    "dataset": dataset,
                                    "model": model,
                                    "shot": shot,
                                    "method": method,
                                    "tau": row["tau"],
                                    "known_acceptance_val": row["known_acceptance"],
                                    "oos_recall_val": row["oos_recall"],
                                    "open_set_macro_f1_val": row["open_set_macro_f1"],
                                    "constraint_met": row["constraint_met"],
                                }
                            )
        RESULTS.mkdir(parents=True, exist_ok=True)
        with (RESULTS / "gate3_validation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(validation_rows)
        (RESULTS / "gate3_validation_thresholds.json").write_text(
            json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (RESULTS / "gate3_method_details.json").write_text(
            json.dumps(details_all, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (RESULTS / "gate3_validation_grid.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(grids[0].keys()))
            writer.writeheader()
            writer.writerows(grids)
        print(f"Validation configs: {len(validation_rows)}")
        print(json.dumps(validation_rows[:3], ensure_ascii=False, indent=2))
        print("GATE3_VALIDATION_OK")
        return 0

    # test phase
    freeze_path = RESULTS / "gate3_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("Gate-3 is not frozen; run src.gate3.freeze first")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("frozen"):
        raise RuntimeError("Gate-3 freeze is not valid")
    if freeze.get("test_evaluated"):
        raise RuntimeError("Gate-3 test was already evaluated once; refusing a second read")
    threshold_path = RESULTS / "gate3_validation_thresholds.json"
    if not threshold_path.exists():
        raise RuntimeError("Run --phase validation first")
    if (RESULTS / "gate3_test_metrics.csv").exists() and not args.force:
        raise RuntimeError("Gate-3 test metrics already exist; refusing second test read")
    if freeze["validation_thresholds_sha256"] != hashlib.sha256(threshold_path.read_bytes()).hexdigest():
        raise RuntimeError("Validation thresholds changed after freeze")
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    test_rows = []
    predictions_by_dataset: dict[str, list[dict[str, Any]]] = {d: [] for d in DATASETS}
    for dataset in DATASETS:
        info = load_dataset(dataset)
        classes = info["known_classes"]
        for model in MODELS:
            cache = load_cache(model, dataset)
            for shot in SHOTS:
                support_X, support_y = support_data(cache, dataset, shot, classes)
                b3p_protos, b3p_details = prototypes_multi(support_X, support_y, classes)
                known_vectors, known_true = eval_data(cache, "known_test", classes)
                unknown_vectors, unknown_true = eval_data(cache, "final_unknown", classes)
                known_ids = cache["sample_ids"][cache["roles"] == "known_test"]
                unknown_ids = cache["sample_ids"][cache["roles"] == "final_unknown"]
                test_ids = np.concatenate([known_ids, unknown_ids])
                test_vectors = np.concatenate([known_vectors, unknown_vectors])
                test_true = np.concatenate([known_true, unknown_true])
                for method in METHODS:
                    key = f"{dataset}|{model}|{shot}|{method}"
                    if key not in thresholds:
                        raise RuntimeError(f"Missing threshold {key}")
                    tau = float(thresholds[key]["tau"])
                    sims, _ = scores_for(
                        method,
                        support_X,
                        support_y,
                        test_vectors,
                        classes,
                        b3p_protos=b3p_protos if method in ("B3-P", "B3-H") else None,
                        b3p_details=b3p_details if method in ("B3-P", "B3-H") else None,
                    )
                    metrics, pred, extra = evaluate(sims, test_true, classes, tau)
                    test_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "shot": shot,
                            "method": method,
                            "tau": tau,
                            "historical_clinc_test": int(dataset == "clinc150"),
                            **metrics,
                        }
                    )
                    scored = extra["scores"]
                    for index, true_idx in enumerate(test_true):
                        top = []
                        for rank_idx in scored.top3_idx[index]:
                            top.append(f"{classes[rank_idx]}:{float(sims[index, rank_idx]):.4f}")
                        predictions_by_dataset[dataset].append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "shot": shot,
                                "method": method,
                                "sample_id": test_ids[index],
                                "role": "known_test" if true_idx >= 0 else "final_unknown",
                                "true_label": classes[true_idx] if true_idx >= 0 else "OOS",
                                "pred_label": pred[index],
                                "accepted": int(extra["accept"][index]),
                                "tau": round(tau, 8),
                                "s1": round(float(scored.s1[index]), 8),
                                "s2": round(float(scored.s2[index]), 8),
                                "margin": round(float(scored.margin[index]), 8),
                                "decision_score": round(float(scored.s1[index] - tau), 8),
                                "top3": " | ".join(top),
                                "correct": int(pred[index] == (classes[true_idx] if true_idx >= 0 else "OOS")),
                            }
                        )
    with (RESULTS / "gate3_test_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)
    # Keep a compact name for the matrix.
    with (RESULTS / "gate3_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)
    for dataset, rows in predictions_by_dataset.items():
        with (RESULTS / f"gate3_predictions_{dataset}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    freeze["test_evaluated"] = True
    freeze["test_evaluated_at_utc"] = pd.Timestamp.utcnow().isoformat()
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Test configs: {len(test_rows)}")
    print(json.dumps(test_rows[:3], ensure_ascii=False, indent=2))
    print("GATE3_TEST_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"GATE3_MATRIX_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
