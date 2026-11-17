"""Gate-3 prototype / neighbor methods and open-set threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score

OOS = "OOS"


@dataclass
class MethodScores:
    sims: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    margin: np.ndarray
    pred_idx: np.ndarray
    top3_idx: np.ndarray
    meta: dict[str, Any]


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero vector")
    return vectors / norms


def prototypes_single(X: np.ndarray, y: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    p = []
    for c in classes:
        rows = X[y == c]
        if len(rows) == 0:
            raise ValueError(f"No support for {c}")
        center = rows.mean(axis=0)
        p.append(center / np.linalg.norm(center))
    return np.stack(p)


def prototypes_multi(
    X: np.ndarray,
    y: np.ndarray,
    classes: Sequence[str],
    *,
    seed: int = 42,
    top_fraction: float = 0.25,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    single = prototypes_single(X, y, classes)
    dispersion = {}
    for c, name in enumerate(classes):
        rows = X[y == name]
        dispersion[name] = float(np.mean(1.0 - rows @ single[c]))
    n_multi = int(np.ceil(len(classes) * top_fraction))
    ranked = sorted(classes, key=lambda name: (-dispersion[name], name))
    multi_classes = set(ranked[:n_multi])
    proto_list: list[np.ndarray] = []
    details = []
    for c, name in enumerate(classes):
        rows = X[y == name]
        use_multi = name in multi_classes and len(rows) >= 2 and len(np.unique(rows, axis=0)) >= 2
        if use_multi:
            try:
                km = KMeans(n_clusters=2, random_state=seed, n_init=10)
                labels = km.fit_predict(rows)
                centers = []
                for k in range(2):
                    cluster = rows[labels == k]
                    if len(cluster) == 0:
                        raise ValueError("empty cluster")
                    centers.append(cluster.mean(axis=0))
                centers = normalize_rows(np.stack(centers))
                proto_list.append(centers)
                details.append(
                    {
                        "intent": name,
                        "dispersion": dispersion[name],
                        "n_prototypes": 2,
                        "multi_prototype": True,
                    }
                )
                continue
            except Exception:
                pass
        proto_list.append(single[c : c + 1])
        details.append(
            {
                "intent": name,
                "dispersion": dispersion[name],
                "n_prototypes": 1,
                "multi_prototype": False,
            }
        )
    return proto_list, {
        "n_classes": len(classes),
        "n_multi_classes": n_multi,
        "multi_classes": sorted(multi_classes),
        "dispersion": dispersion,
        "class_details": details,
    }


def scores_from_prototypes(vectors: np.ndarray, prototypes: Sequence[np.ndarray]) -> np.ndarray:
    sims = np.empty((len(vectors), len(prototypes)), dtype=np.float32)
    for c, proto in enumerate(prototypes):
        sims[:, c] = np.max(vectors @ np.asarray(proto).T, axis=1)
    return sims


def knn_scores(
    vectors: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    classes: Sequence[str],
    *,
    top_k: int = 2,
) -> np.ndarray:
    sims = np.empty((len(vectors), len(classes)), dtype=np.float32)
    for c, name in enumerate(classes):
        support = X[y == name]
        class_sims = vectors @ support.T
        k = min(top_k, class_sims.shape[1])
        if k <= 0:
            raise ValueError(f"No support for {name}")
        part = np.partition(class_sims, -k, axis=1)[:, -k:]
        sims[:, c] = part.mean(axis=1)
    return sims


def make_scores(sims: np.ndarray) -> MethodScores:
    order = np.argsort(-sims, axis=1)
    pred = order[:, 0]
    s1 = sims[np.arange(len(sims)), pred]
    s2 = sims[np.arange(len(sims)), order[:, 1]]
    top3 = order[:, :3]
    return MethodScores(
        sims=sims,
        s1=s1,
        s2=s2,
        margin=s1 - s2,
        pred_idx=pred,
        top3_idx=top3,
        meta={},
    )


def select_threshold(
    sims: np.ndarray,
    true_idx: np.ndarray,
    n_known: int,
    classes: Sequence[str],
    *,
    target_acceptance: float = 0.90,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """Select a global threshold on validation only.

    true_idx: known labels use class index; unknown rows use -1.
    """
    scores = make_scores(sims)
    is_known = true_idx >= 0
    known_s1 = scores.s1[is_known]
    unknown_s1 = scores.s1[~is_known]
    taus = np.unique(np.quantile(known_s1, np.linspace(0.0, 1.0, 101)))
    rows = []
    best = None
    for tau in taus:
        accept = scores.s1 >= tau
        known_acceptance = float(np.mean(accept[is_known]))
        oos_recall = float(np.mean(~accept[~is_known]))
        pred = np.where(accept, np.asarray(classes)[scores.pred_idx], OOS)
        y = np.asarray([classes[i] if i >= 0 else OOS for i in true_idx])
        f1 = float(f1_score(y, pred, labels=list(classes) + [OOS], average="macro", zero_division=0))
        row = {
            "tau": float(tau),
            "known_acceptance": known_acceptance,
            "oos_recall": oos_recall,
            "open_set_macro_f1": f1,
            "constraint_met": known_acceptance >= target_acceptance,
        }
        rows.append(row)
        if row["constraint_met"]:
            key = (row["open_set_macro_f1"], row["oos_recall"], row["known_acceptance"], row["tau"])
            if best is None or key > best["_key"]:
                best = {**row, "_key": key}
    if best is None:
        # Report infeasibility instead of silently relaxing the constraint.
        best = max(rows, key=lambda row: (row["open_set_macro_f1"], row["oos_recall"], row["known_acceptance"]))
        best["_key"] = None
    info = {
        "tau": float(best["tau"]),
        "known_acceptance_val": best["known_acceptance"],
        "oos_recall_val": best["oos_recall"],
        "open_set_macro_f1_val": best["open_set_macro_f1"],
        "constraint_met": bool(best["constraint_met"]),
        "n_candidates": len(rows),
        "target_acceptance": target_acceptance,
        "objective": "max validation open-set macro-F1 subject to known acceptance >= 0.90; tie OOS recall, known acceptance, tau",
    }
    return float(best["tau"]), rows, info


def evaluate(
    sims: np.ndarray,
    true_idx: np.ndarray,
    classes: Sequence[str],
    tau: float,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    scores = make_scores(sims)
    is_known = true_idx >= 0
    accept = scores.s1 >= tau
    pred = np.where(accept, np.asarray(classes)[scores.pred_idx], OOS)
    y = np.asarray([classes[i] if i >= 0 else OOS for i in true_idx])
    known_pred = pred[is_known]
    known_true = y[is_known]
    oos_pred = pred[~is_known]
    known_acceptance = float(np.mean(known_pred != OOS))
    known_routing = float(np.mean(known_pred == known_true))
    known_macro = float(f1_score(known_true, known_pred, labels=list(classes), average="macro", zero_division=0))
    oos_recall = float(np.mean(oos_pred == OOS))
    f1 = float(f1_score(y, pred, labels=list(classes) + [OOS], average="macro", zero_division=0))
    y_bin = is_known.astype(int)
    auroc = float(roc_auc_score(y_bin, scores.s1 - tau))
    top1 = float(np.mean(np.asarray(classes)[scores.pred_idx[is_known]] == known_true))
    metrics = {
        "known_acceptance": known_acceptance,
        "known_correct_routing": known_routing,
        "known_macro_f1": known_macro,
        "top1_closed_set_accuracy": top1,
        "oos_recall": oos_recall,
        "oos_false_acceptance": 1.0 - oos_recall,
        "unknown_auroc": auroc,
        "open_set_macro_f1": f1,
        "open_set_accuracy": float(np.mean(pred == y)),
        "n_known": int(is_known.sum()),
        "n_unknown": int((~is_known).sum()),
        "known_rejected": int(np.sum(known_pred == OOS)),
        "unknown_rejected": int(np.sum(oos_pred == OOS)),
        "unknown_accepted": int(np.sum(oos_pred != OOS)),
    }
    return metrics, pred, {"scores": scores, "accept": accept, "pred": pred}
