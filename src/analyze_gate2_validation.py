"""Validation-only diagnostics for the Gate-2 hard-OOS protocol.

Reads the frozen Qwen Gate-1 embeddings and the newly encoded hard-OOS
validation rows.  It never touches hard_oos_test.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "manifest"
CACHE = PROJECT_ROOT / "data" / "cache_embeddings"
RESULTS = PROJECT_ROOT / "results"
QWEN_REV = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PROMPT_HASH = "4bbdf1c169061f64ba742f64bca5e422d05e94ccc7261d29495c233ed9a90005"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {
        "vectors": data["vectors"].astype(np.float32),
        "sample_ids": data["sample_ids"].astype(str),
        "roles": data["roles"].astype(str),
        "labels": data["labels"].astype(str) if "labels" in data else data["intents"].astype(str),
    }


def score_matrix(vectors: np.ndarray, prototypes: np.ndarray) -> dict[str, np.ndarray]:
    sims = vectors @ prototypes.T
    order = np.argsort(-sims, axis=1)
    pred = order[:, 0]
    s1 = sims[np.arange(len(sims)), pred]
    s2 = sims[np.arange(len(sims)), order[:, 1]]
    return {
        "sims": sims,
        "pred": pred,
        "s1": s1,
        "s2": s2,
        "margin": s1 - s2,
    }


def metrics(
    known_classes: list[str],
    known_scores: dict[str, np.ndarray],
    known_true: np.ndarray,
    known_labels: np.ndarray,
    oos_scores: dict[str, np.ndarray],
    oos_labels: np.ndarray,
) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    accept_known = known_scores["accept"]
    accept_oos = oos_scores["accept"]
    known_acceptance = float(np.mean(accept_known))
    pred_known = np.where(accept_known, np.asarray(known_classes)[known_scores["pred"]], "OOS")
    known_routing = float(np.mean(pred_known == known_true))
    oos_recall = float(np.mean(~accept_oos))
    pred_oos = np.where(accept_oos, np.asarray(known_classes)[oos_scores["pred"]], "OOS")
    y = np.concatenate([known_true, np.full(len(oos_labels), "OOS")])
    p = np.concatenate([pred_known, pred_oos])
    f1 = float(f1_score(y, p, labels=known_classes + ["OOS"], average="macro", zero_division=0))
    scores = np.concatenate([known_scores["decision"], oos_scores["decision"]])
    ybin = np.concatenate([np.ones(len(known_true)), np.zeros(len(oos_labels))])
    return {
        "known_acceptance": known_acceptance,
        "known_routing": known_routing,
        "oos_recall": oos_recall,
        "open_set_macro_f1": f1,
        "oos_auroc": float(roc_auc_score(ybin, scores)),
        "n_known": int(len(known_true)),
        "n_oos": int(len(oos_labels)),
        "known_rejected": int(np.sum(~accept_known)),
        "oos_accepted": int(np.sum(accept_oos)),
    }


def main() -> int:
    selected = json.loads((MANIFEST / "selected_intents.json").read_text(encoding="utf-8"))
    known_classes = [item["intent"] for item in selected["intents"]]
    class_to_idx = {name: index for index, name in enumerate(known_classes)}

    known = load_npz(
        CACHE / "qwen" / QWEN_REV / f"seed42_shot5_L128_{PROMPT_HASH[:16]}" / "embeddings.npz"
    )
    hard = load_npz(
        CACHE / "qwen" / QWEN_REV / f"hard_oos_val_L128_{PROMPT_HASH[:16]}" / "embeddings.npz"
    )
    support_mask = known["roles"] == "support"
    support_labels = known["labels"][support_mask]
    support_vectors = known["vectors"][support_mask]
    proto = []
    for name in known_classes:
        rows = support_vectors[support_labels == name]
        if len(rows) != 5:
            raise RuntimeError(f"Expected 5 support rows for {name}, got {len(rows)}")
        center = rows.mean(axis=0)
        proto.append(center / np.linalg.norm(center))
    proto = np.stack(proto)

    known_mask = known["roles"] == "calibration_known"
    oos_mask = known["roles"] == "calibration_oos"
    known_scores = score_matrix(known["vectors"][known_mask], proto)
    oos_scores = score_matrix(known["vectors"][oos_mask], proto)
    hard_scores = score_matrix(hard["vectors"], proto)
    known_true = known["labels"][known_mask]
    known_idx = np.asarray([class_to_idx[name] for name in known_true])

    thresholds_b0 = json.loads((RESULTS / "thresholds_qwen_b0.json").read_text(encoding="utf-8"))
    thresholds_b2 = json.loads((RESULTS / "thresholds_qwen_b2.json").read_text(encoding="utf-8"))
    tau = thresholds_b0["tau"]
    lam = thresholds_b2["lambda"]
    b2_tau = thresholds_b2["tau"]
    gamma = thresholds_b2["gamma"]
    q_c_old = np.asarray([thresholds_b2["q_c"][name] for name in known_classes])
    # Correct definition: the score of the *true* class, s_c(x), not top-1 s1.
    q_c_new = np.asarray(
        [
            float(np.quantile(known_scores["sims"][known_idx == c, c], 0.10))
            for c in range(len(known_classes))
        ]
    )
    q_c_old_recomputed = np.asarray(
        [
            float(np.quantile(known_scores["s1"][known_idx == c], 0.10))
            for c in range(len(known_classes))
        ]
    )

    def evaluate(method: str) -> dict:
        if method == "B0":
            accept_known = known_scores["s1"] >= tau
            accept_oos = oos_scores["s1"] >= tau
            accept_hard = hard_scores["s1"] >= tau
            decision_known = known_scores["s1"] - tau
            decision_oos = oos_scores["s1"] - tau
            decision_hard = hard_scores["s1"] - tau
        else:
            tau_c = (1 - lam) * b2_tau + lam * q_c_old
            tau_c_new = (1 - lam) * b2_tau + lam * q_c_new
            accept_known = (known_scores["s1"] >= tau_c[known_scores["pred"]]) & (
                known_scores["margin"] >= gamma
            )
            accept_oos = (oos_scores["s1"] >= tau_c[oos_scores["pred"]]) & (
                oos_scores["margin"] >= gamma
            )
            accept_hard = (hard_scores["s1"] >= tau_c[hard_scores["pred"]]) & (
                hard_scores["margin"] >= gamma
            )
            decision_known = np.minimum(
                known_scores["s1"] - tau_c[known_scores["pred"]],
                known_scores["margin"] - gamma,
            )
            decision_oos = np.minimum(
                oos_scores["s1"] - tau_c[oos_scores["pred"]],
                oos_scores["margin"] - gamma,
            )
            decision_hard = np.minimum(
                hard_scores["s1"] - tau_c[hard_scores["pred"]],
                hard_scores["margin"] - gamma,
            )
        known_acceptance = float(np.mean(accept_known))
        pred_known = np.where(accept_known, np.asarray(known_classes)[known_scores["pred"]], "OOS")
        known_routing = float(np.mean(pred_known == known_true))
        return {
            "known_acceptance": known_acceptance,
            "known_routing": known_routing,
            "original_oos_recall": float(np.mean(~accept_oos)),
            "hard_oos_recall": float(np.mean(~accept_hard)),
            "hard_oos_accepted": int(np.sum(accept_hard)),
            "known_rejected": int(np.sum(~accept_known)),
            "known_accept": accept_known,
            "oos_accept": accept_oos,
            "hard_accept": accept_hard,
            "decision_known": decision_known,
            "decision_oos": decision_oos,
            "decision_hard": decision_hard,
        }

    b0 = evaluate("B0")
    b2 = evaluate("B2")
    for result in (b0, b2):
        result["hard_open_set_macro_f1"] = metrics(
            known_classes,
            {
                "accept": result["known_accept"],
                "pred": known_scores["pred"],
                "decision": result["decision_known"],
            },
            known_true,
            known_true,
            {
                "accept": result["hard_accept"],
                "pred": hard_scores["pred"],
                "decision": result["decision_hard"],
            },
            hard["labels"],
        )["open_set_macro_f1"]

    # How many evaluation rows have top1 != true label; q_c old vs corrected.
    top1_errors = int(np.sum(known_scores["pred"] != known_idx))
    qc_diff = q_c_new - q_c_old
    # Hard-OOS samples grouped by predicted class.
    class_stats = []
    for c, name in enumerate(known_classes):
        mask = hard_scores["pred"] == c
        if not np.any(mask):
            class_stats.append(
                {
                    "intent": name,
                    "n_hard_predicted": 0,
                    "s1_q50": None,
                    "s1_q90": None,
                    "known_q_c": float(q_c_new[c]),
                }
            )
            continue
        scores_c = hard_scores["s1"][mask]
        class_stats.append(
            {
                "intent": name,
                "n_hard_predicted": int(mask.sum()),
                "s1_q50": float(np.quantile(scores_c, 0.50)),
                "s1_q90": float(np.quantile(scores_c, 0.90)),
                "known_q_c": float(q_c_new[c]),
            }
        )

    payload = {
        "schema_version": "gate2-validation-diagnostics-v1",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": QWEN_REV,
        "seed": 42,
        "shot": 5,
        "thresholds_from_gate1": {
            "b0_tau": tau,
            "b2_tau": b2_tau,
            "b2_lambda": lam,
            "b2_gamma": gamma,
        },
        "known_val": {
            "n": int(len(known_true)),
            "top1_errors": top1_errors,
            "q_c_old_vs_corrected_max_abs_diff": float(np.max(np.abs(qc_diff))),
            "q_c_old_vs_corrected_mean_abs_diff": float(np.mean(np.abs(qc_diff))),
            "q_c_definition_old": "10th percentile of top-1 s1 among validation rows with true label c (Gate 1 historical code)",
            "q_c_definition_corrected": "10th percentile of s_c(x), the score of the true class c",
        },
        "original_oos_val": int(oos_mask.sum()),
        "hard_oos_val": int(len(hard["sample_ids"])),
        "hard_oos_val_classes": int(len(set(hard["labels"]))),
        "existing_thresholds_on_hard_val": {
            "B0": {
                key: value
                for key, value in b0.items()
                if key not in {"known_accept", "oos_accept", "hard_accept", "decision_known", "decision_oos", "decision_hard"}
            },
            "B2": {
                key: value
                for key, value in b2.items()
                if key not in {"known_accept", "oos_accept", "hard_accept", "decision_known", "decision_oos", "decision_hard"}
            },
        },
        "hard_class_by_predicted_class": class_stats,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "gate2_validation_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import csv

    with (RESULTS / "gate2_hard_class_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_stats[0].keys()))
        writer.writeheader()
        writer.writerows(class_stats)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("GATE2_VALIDATION_DIAGNOSTICS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
