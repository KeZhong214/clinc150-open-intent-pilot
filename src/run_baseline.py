"""Run the fixed seed=42 / 5-shot B0 or B2 experiment for one embedding model.

B0: single prototype + global threshold.
B2: single prototype + class-aware threshold (10th percentile q_c) + margin gate.

Thresholds are selected on validation only.  The final test split is evaluated
once after the configuration is frozen.  Existing result files make the script
stop unless --force is passed, to avoid silent repeated test evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"
CACHE_DIR = PROJECT_ROOT / "data" / "cache_embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS = {
    "bge": {
        "id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "prompt_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "qwen": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "prompt_sha256": None,  # read from config
    },
}
OOS = "OOS"
OBJECTIVE_KNOWN_ACCEPTANCE = 0.90
N_TAU = 101


@dataclass
class Scores:
    sample_id: np.ndarray
    role: np.ndarray
    label: np.ndarray
    vectors: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    margin: np.ndarray
    pred_idx: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prompt_hash_for(model_key: str) -> str:
    if model_key == "bge":
        return MODELS["bge"]["prompt_sha256"]
    import yaml

    with (PROJECT_ROOT / "configs" / "pilot.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config["models"]["qwen"]["prompt_sha256"]


def cache_root(model_key: str, seed: int, shot: int, max_length: int = 128) -> Path:
    prompt_hash = prompt_hash_for(model_key)
    stem = f"seed{seed}_shot{shot}_L{max_length}_{prompt_hash[:16]}"
    return CACHE_DIR / model_key / MODELS[model_key]["revision"] / stem


def load_data(model_key: str, seed: int, shot: int) -> tuple[list[str], Scores]:
    npz_path = cache_root(model_key, seed, shot) / "embeddings.npz"
    metadata_path = npz_path.parent / "metadata.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing embedding cache: {npz_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing embedding metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["seed"] != seed or metadata["shot"] != shot:
        raise RuntimeError("Embedding cache seed/shot mismatch")
    if metadata["model_revision"] != MODELS[model_key]["revision"]:
        raise RuntimeError("Embedding cache revision mismatch")
    if metadata["prompt_sha256"] != prompt_hash_for(model_key):
        raise RuntimeError("Embedding cache prompt hash mismatch")

    data = np.load(npz_path, allow_pickle=False)
    vectors = data["vectors"].astype(np.float32)
    sample_ids = data["sample_ids"].astype(str)
    roles = data["roles"].astype(str)
    labels = data["labels"].astype(str)
    if len({len(sample_ids), len(roles), len(labels), len(vectors)}) != 1:
        raise RuntimeError("Embedding arrays have inconsistent lengths")

    selected = json.loads((MANIFEST_DIR / "selected_intents.json").read_text(encoding="utf-8"))
    known_classes = [item["intent"] for item in selected["intents"]]
    class_to_idx = {name: index for index, name in enumerate(known_classes)}

    support_mask = roles == "support"
    support_ids = sample_ids[support_mask]
    support_labels = labels[support_mask]
    per_class = Counter(support_labels.tolist())
    if set(per_class) != set(known_classes) or set(per_class.values()) != {shot}:
        raise RuntimeError(f"Bad support layout: {per_class}")

    prototypes = []
    for intent in known_classes:
        rows = vectors[support_mask][support_labels == intent]
        if len(rows) != shot:
            raise RuntimeError(f"{intent}: expected {shot} support vectors, got {len(rows)}")
        prototype = rows.mean(axis=0)
        norm = np.linalg.norm(prototype)
        if norm <= 0:
            raise RuntimeError(f"Zero prototype for {intent}")
        prototypes.append(prototype / norm)
    prototypes = np.stack(prototypes)

    sims = vectors @ prototypes.T
    order = np.argsort(-sims, axis=1)
    pred_idx = order[:, 0]
    s1 = sims[np.arange(len(sims)), pred_idx]
    s2 = sims[np.arange(len(sims)), order[:, 1]]
    margin = s1 - s2

    # Sanity: every eval label is known or OOS; support labels all known.
    known_set = set(known_classes) | {OOS}
    if not set(labels.tolist()) <= known_set:
        raise RuntimeError("Unknown label in embedding cache")
    if not set(labels[~support_mask].tolist()) <= known_set:
        raise RuntimeError("Unknown eval label")

    return known_classes, Scores(
        sample_id=sample_ids,
        role=roles,
        label=labels,
        vectors=vectors,
        s1=s1,
        s2=s2,
        margin=margin,
        pred_idx=pred_idx,
    )


def b0_threshold_grid(s1_known: np.ndarray) -> np.ndarray:
    return np.unique(np.quantile(s1_known, np.linspace(0.0, 1.0, N_TAU)))


def select_b0(s1_known: np.ndarray, s1_oos: np.ndarray) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    candidates = b0_threshold_grid(s1_known)
    rows: list[dict[str, Any]] = []
    for tau in candidates:
        known_acceptance = float(np.mean(s1_known >= tau))
        oos_recall = float(np.mean(s1_oos < tau))
        rows.append(
            {
                "tau": float(tau),
                "known_acceptance_val": known_acceptance,
                "oos_recall_val": oos_recall,
                "constraint_met": known_acceptance >= OBJECTIVE_KNOWN_ACCEPTANCE,
            }
        )

    valid = [row for row in rows if row["constraint_met"]]
    if valid:
        best = max(valid, key=lambda row: (row["oos_recall_val"], row["known_acceptance_val"], row["tau"]))
        constraint_met = True
    else:
        # This branch must be reported as infeasible; it is not silently treated
        # as satisfying the pre-registered 90% acceptance constraint.
        best = max(rows, key=lambda row: (row["oos_recall_val"] + row["known_acceptance_val"], row["tau"]))
        constraint_met = False
    info = {
        "tau": best["tau"],
        "known_acceptance_val": best["known_acceptance_val"],
        "oos_recall_val": best["oos_recall_val"],
        "constraint_met": constraint_met,
        "objective": (
            "max validation OOS recall subject to validation known acceptance >= 0.90; "
            "ties: known acceptance, then tau"
        ),
        "n_candidates": len(rows),
        "candidate_rule": (
            f"unique quantiles 0..1 ({N_TAU} points) of validation known top-1 cosine scores"
        ),
    }
    return float(best["tau"]), info, rows


def select_b2(
    known_classes: Sequence[str],
    scores: Scores,
    val_known_mask: np.ndarray,
    val_oos_mask: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, float]:
    known_set = set(known_classes)
    val_labels = scores.label[val_known_mask]
    val_true_idx = np.asarray([list(known_classes).index(label) for label in val_labels])
    val_s1 = scores.s1[val_known_mask]
    val_margin = scores.margin[val_known_mask]
    val_pred = scores.pred_idx[val_known_mask]

    # q_c uses only validation rows whose true label is c.
    q_c = np.zeros(len(known_classes), dtype=np.float64)
    for c in range(len(known_classes)):
        class_scores = scores.s1[val_known_mask][val_true_idx == c]
        if len(class_scores) != 20:
            raise RuntimeError(f"Expected 20 validation rows for {known_classes[c]}")
        q_c[c] = float(np.quantile(class_scores, 0.10))

    tau_b0, b0_info, _ = select_b0(val_s1, scores.s1[val_oos_mask])
    taus = b0_threshold_grid(val_s1)
    gamma_candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0]),
                np.quantile(val_margin, np.linspace(0.0, 0.9, 19)),
            ]
        )
    )
    gamma_candidates = np.round(gamma_candidates, 8)

    oos_s1 = scores.s1[val_oos_mask]
    oos_margin = scores.margin[val_oos_mask]
    oos_pred = scores.pred_idx[val_oos_mask]

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for lam in (0.0, 0.5, 1.0):
        for tau in taus:
            tau_c = (1.0 - lam) * tau + lam * q_c
            tau_pred_known = tau_c[val_pred]
            tau_pred_oos = tau_c[oos_pred]
            for gamma in gamma_candidates:
                known_accept = (val_s1 >= tau_pred_known) & (val_margin >= gamma)
                oos_accept = (oos_s1 >= tau_pred_oos) & (oos_margin >= gamma)
                known_acceptance = float(np.mean(known_accept))
                oos_recall = float(np.mean(~oos_accept))
                row = {
                    "lambda": float(lam),
                    "tau": float(tau),
                    "gamma": float(gamma),
                    "known_acceptance_val": known_acceptance,
                    "oos_recall_val": oos_recall,
                    "constraint_met": known_acceptance >= OBJECTIVE_KNOWN_ACCEPTANCE,
                }
                rows.append(row)
                if row["constraint_met"]:
                    key = (
                        row["oos_recall_val"],
                        row["known_acceptance_val"],
                        -row["gamma"],
                        row["lambda"],
                        -row["tau"],
                    )
                    if best is None or key > best["_key"]:
                        best = {**row, "_key": key}

    if best is None:
        raise RuntimeError(
            "No B2 configuration satisfies validation known acceptance >= 0.90; "
            "report infeasibility instead of silently relaxing the constraint."
        )

    selected = {
        "lambda": float(best["lambda"]),
        "tau": float(best["tau"]),
        "gamma": float(best["gamma"]),
        "known_acceptance_val": best["known_acceptance_val"],
        "oos_recall_val": best["oos_recall_val"],
        "constraint_met": True,
        "objective": (
            "max validation OOS recall subject to validation known acceptance >= 0.90; "
            "ties: known acceptance, lower gamma, lambda, higher tau"
        ),
        "n_candidates": len(rows),
        "tau_candidates": int(len(taus)),
        "gamma_candidates": int(len(gamma_candidates)),
        "lambda_values": [0.0, 0.5, 1.0],
        "b0_tau_reference": float(tau_b0),
        "b0_validation": b0_info,
    }
    return selected, rows, q_c, tau_b0


def predict(
    scores: Scores,
    known_classes: Sequence[str],
    *,
    method: str,
    tau: float,
    q_c: np.ndarray,
    lam: float,
    gamma: float,
) -> dict[str, Any]:
    if method == "B0":
        accept = scores.s1 >= tau
        tau_used = np.full(len(scores.s1), tau, dtype=np.float32)
        decision_score = scores.s1 - tau
    elif method == "B2":
        tau_c = (1.0 - lam) * tau + lam * q_c
        tau_used = tau_c[scores.pred_idx]
        accept = (scores.s1 >= tau_used) & (scores.margin >= gamma)
        decision_score = np.minimum(scores.s1 - tau_used, scores.margin - gamma)
    else:
        raise ValueError(method)

    pred_labels = np.asarray(
        [known_classes[index] if accepted else OOS for index, accepted in zip(scores.pred_idx, accept)],
        dtype=object,
    )
    return {
        "accept": accept,
        "tau_used": tau_used,
        "decision_score": decision_score,
        "pred_labels": pred_labels,
    }


def metrics(
    scores: Scores,
    known_classes: Sequence[str],
    prediction: dict[str, Any],
    eval_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    labels = scores.label[eval_mask]
    roles = scores.role[eval_mask]
    pred_labels = prediction["pred_labels"][eval_mask]
    s1 = scores.s1[eval_mask]
    margin = scores.margin[eval_mask]
    pred_idx = scores.pred_idx[eval_mask]
    decision_score = prediction["decision_score"][eval_mask]

    known_mask = roles == "calibration_known"
    known_mask |= roles == "final_known_test"
    oos_mask = roles == "calibration_oos"
    oos_mask |= roles == "final_oos_test"
    # When evaluating test, use only final_* roles; validation uses only calibration roles.
    if np.any(roles == "final_known_test"):
        selected_mask = (roles == "final_known_test") | (roles == "final_oos_test")
    else:
        selected_mask = (roles == "calibration_known") | (roles == "calibration_oos")
    if not np.all(selected_mask):
        raise RuntimeError(f"Unexpected role mix in evaluation: {Counter(roles.tolist())}")

    true_known = labels[known_mask]
    pred_known = pred_labels[known_mask]
    true_oos = labels[oos_mask]
    pred_oos = pred_labels[oos_mask]

    known_acceptance = float(np.mean(pred_known != OOS))
    known_correct_routing = float(np.mean(pred_known == true_known))
    top1_closed_set = float(np.mean(np.asarray(known_classes)[pred_idx[known_mask]] == true_known))
    known_macro_f1 = float(
        f1_score(true_known, pred_known, labels=list(known_classes), average="macro", zero_division=0)
    )
    oos_recall = float(np.mean(pred_oos == OOS))
    oos_far = 1.0 - oos_recall
    y_known_oos = np.concatenate([np.ones(known_mask.sum()), np.zeros(oos_mask.sum())])
    auroc = float(roc_auc_score(y_known_oos, decision_score[known_mask | oos_mask]))
    all_true = np.concatenate([true_known, true_oos])
    all_pred = np.concatenate([pred_known, pred_oos])
    open_set_macro_f1 = float(
        f1_score(
            all_true,
            all_pred,
            labels=list(known_classes) + [OOS],
            average="macro",
            zero_division=0,
        )
    )
    open_set_accuracy = float(np.mean(all_pred == all_true))

    row = {
        "n_known": int(known_mask.sum()),
        "n_oos": int(oos_mask.sum()),
        "known_acceptance": known_acceptance,
        "known_correct_routing": known_correct_routing,
        "known_macro_f1": known_macro_f1,
        "top1_closed_set_accuracy": top1_closed_set,
        "oos_recall": oos_recall,
        "oos_false_acceptance": oos_far,
        "oos_auroc": auroc,
        "open_set_macro_f1": open_set_macro_f1,
        "open_set_accuracy": open_set_accuracy,
        "known_rejected": int(np.sum(pred_known == OOS)),
        "oos_rejected": int(np.sum(pred_oos == OOS)),
        "oos_accepted": int(np.sum(pred_oos != OOS)),
    }

    label_order = list(known_classes) + [OOS]
    cm = confusion_matrix(all_true, all_pred, labels=label_order)

    per_class = []
    for c in known_classes:
        mask = true_known == c
        support = int(mask.sum())
        correct = int(np.sum(pred_known[mask] == c))
        rejected = int(np.sum(pred_known[mask] == OOS))
        per_class.append(
            {
                "intent": c,
                "support": support,
                "correct": correct,
                "rejected": rejected,
                "accuracy": correct / support if support else 0.0,
            }
        )
    return row, cm, per_class


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    scores: Scores,
    known_classes: Sequence[str],
    prediction: dict[str, Any],
    *,
    model_key: str,
    method: str,
    seed: int,
    shot: int,
) -> None:
    rows = []
    for index, sample_id in enumerate(scores.sample_id):
        rows.append(
            {
                "model": model_key,
                "method": method,
                "seed": seed,
                "shot": shot,
                "sample_id": sample_id,
                "role": scores.role[index],
                "true_label": scores.label[index],
                "pred_label": prediction["pred_labels"][index],
                "accepted": int(prediction["accept"][index]),
                "top1_class": known_classes[scores.pred_idx[index]],
                "s1": round(float(scores.s1[index]), 8),
                "s2": round(float(scores.s2[index]), 8),
                "margin": round(float(scores.margin[index]), 8),
                "tau_used": round(float(prediction["tau_used"][index]), 8),
                "decision_score": round(float(prediction["decision_score"][index]), 8),
                "correct": int(prediction["pred_labels"][index] == scores.label[index]),
            }
        )
    write_csv(path, rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--method", choices=("B0", "B2"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Overwrite existing result files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    known_classes, scores = load_data(args.model, args.seed, args.shot)
    known_set = set(known_classes)

    val_known_mask = scores.role == "calibration_known"
    val_oos_mask = scores.role == "calibration_oos"
    test_mask = (scores.role == "final_known_test") | (scores.role == "final_oos_test")
    if not (val_known_mask.sum() == 400 and val_oos_mask.sum() == 100 and test_mask.sum() == 1600):
        raise RuntimeError("Unexpected evaluation masks")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"metrics_{args.model}_{args.method.lower()}.csv"
    predictions_path = RESULTS_DIR / f"predictions_{args.model}_{args.method.lower()}.csv"
    threshold_path = RESULTS_DIR / f"thresholds_{args.model}_{args.method.lower()}.json"
    confusion_path = RESULTS_DIR / f"confusion_{args.model}_{args.method.lower()}.csv"
    per_class_path = RESULTS_DIR / f"per_class_{args.model}_{args.method.lower()}.csv"
    grid_path = RESULTS_DIR / f"val_grid_{args.model}_{args.method.lower()}.csv"
    outputs = [metrics_path, predictions_path, threshold_path, confusion_path, per_class_path, grid_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise RuntimeError(
            "Result files already exist; refusing a second test evaluation without --force: "
            + ", ".join(existing)
        )

    if args.method == "B0":
        tau, threshold_info, grid = select_b0(
            scores.s1[val_known_mask], scores.s1[val_oos_mask]
        )
        prediction = predict(
            scores, known_classes, method="B0", tau=tau, q_c=np.zeros(len(known_classes)), lam=0.0, gamma=0.0
        )
        threshold_payload = {
            "model": args.model,
            "model_id": MODELS[args.model]["id"],
            "model_revision": MODELS[args.model]["revision"],
            "seed": args.seed,
            "shot": args.shot,
            "method": "B0",
            "tau": tau,
            "validation": threshold_info,
        }
        grid_fields = ["tau", "known_acceptance_val", "oos_recall_val", "constraint_met"]
        grid_rows = grid
    else:
        b2_info, grid, q_c, b0_tau = select_b2(
            known_classes, scores, val_known_mask, val_oos_mask
        )
        prediction = predict(
            scores,
            known_classes,
            method="B2",
            tau=b2_info["tau"],
            q_c=q_c,
            lam=b2_info["lambda"],
            gamma=b2_info["gamma"],
        )
        threshold_payload = {
            "model": args.model,
            "model_id": MODELS[args.model]["id"],
            "model_revision": MODELS[args.model]["revision"],
            "seed": args.seed,
            "shot": args.shot,
            "method": "B2",
            "lambda": b2_info["lambda"],
            "tau": b2_info["tau"],
            "gamma": b2_info["gamma"],
            "q_c": {intent: float(value) for intent, value in zip(known_classes, q_c)},
            "validation": b2_info,
            "b0_control": {
                "lambda": 0.0,
                "gamma": 0.0,
                "tau": b0_tau,
                "note": "Equivalent to B0 acceptance; B0 metrics are reused, not re-evaluated.",
            },
        }
        grid_fields = [
            "lambda",
            "tau",
            "gamma",
            "known_acceptance_val",
            "oos_recall_val",
            "constraint_met",
        ]
        grid_rows = grid

    validation_metrics, _, _ = metrics(
        scores, known_classes, prediction, val_known_mask | val_oos_mask
    )
    test_metrics, cm, per_class = metrics(scores, known_classes, prediction, test_mask)
    row = {
        "model": args.model,
        "model_id": MODELS[args.model]["id"],
        "model_revision": MODELS[args.model]["revision"],
        "method": args.method,
        "seed": args.seed,
        "shot": args.shot,
    }
    row.update({f"val_{key}": value for key, value in validation_metrics.items()})
    row.update({key: value for key, value in test_metrics.items()})
    row["threshold"] = json.dumps(
        {
            key: threshold_payload[key]
            for key in ("tau", "lambda", "gamma")
            if key in threshold_payload
        },
        ensure_ascii=False,
    )
    write_csv(metrics_path, [row])
    write_predictions(
        predictions_path,
        scores,
        known_classes,
        prediction,
        model_key=args.model,
        method=args.method,
        seed=args.seed,
        shot=args.shot,
    )
    threshold_payload["selected_at_utc"] = utc_now()
    threshold_payload["validation_metrics"] = validation_metrics
    threshold_payload["test_protocol"] = (
        "Thresholds selected on validation only; test evaluated once after freezing."
    )
    threshold_path.write_text(
        json.dumps(threshold_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        confusion_path,
        [
            {"true_label": label, **{pred: int(cm[i, j]) for j, pred in enumerate(list(known_classes) + [OOS])}}
            for i, label in enumerate(list(known_classes) + [OOS])
        ],
    )
    write_csv(per_class_path, per_class)
    write_csv(grid_path, grid_rows, grid_fields)

    print(f"{args.model.upper()} {args.method} seed={args.seed} shot={args.shot}")
    print(f"Validation: {validation_metrics}")
    print(f"Test      : {test_metrics}")
    print(f"Threshold : {json.dumps({k: threshold_payload[k] for k in threshold_payload if k in ('tau','lambda','gamma')})}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {threshold_path}")
    print(f"Wrote {grid_path}")
    print("BASELINE_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"BASELINE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
