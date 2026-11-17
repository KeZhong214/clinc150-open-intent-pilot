"""Gate-2 hard-OOS validation, freeze, and one-shot hard-OOS test evaluation.

Protocol:
  * Gate-1 20 known classes and seed=42 / 5-shot support stay fixed.
  * 65 development unknown classes provide hard-OOS validation (val rows).
  * 65 disjoint held-out unknown classes provide hard-OOS test (test rows).
  * B0, B2-original and B2-v2 are calibrated on the same combined validation:
      400 known val + 100 original oos_val + 1300 hard-OOS val.
  * B2-v2 only raises class thresholds relative to the global tau, using the
    upper tail of hard-OOS scores received by each predicted class.

No model training. No hard-OOS test access in --phase validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "manifest"
CACHE = PROJECT_ROOT / "data" / "cache_embeddings"
RESULTS = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "configs" / "gate2.yaml"
QWEN_REV = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PROMPT_HASH = "4bbdf1c169061f64ba742f64bca5e422d05e94ccc7261d29495c233ed9a90005"
OOS = "OOS"
N_TAU = 101


@dataclass
class Scored:
    sample_ids: np.ndarray
    roles: np.ndarray
    intents: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    margin: np.ndarray
    pred_idx: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_known(seed: int, shot: int) -> dict[str, Any]:
    root = CACHE / "qwen" / QWEN_REV / f"seed{seed}_shot{shot}_L128_{PROMPT_HASH[:16]}"
    npz_path = root / "embeddings.npz"
    metadata_path = root / "metadata.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing Qwen cache: {npz_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["model_revision"] != QWEN_REV or metadata["prompt_sha256"] != PROMPT_HASH:
        raise RuntimeError("Known embedding cache metadata does not match the frozen Qwen model/prompt")
    data = np.load(npz_path, allow_pickle=False)
    return {
        "vectors": data["vectors"].astype(np.float32),
        "sample_ids": data["sample_ids"].astype(str),
        "roles": data["roles"].astype(str),
        "labels": data["labels"].astype(str),
        "metadata": metadata,
    }


def load_hard(split: str) -> dict[str, Any]:
    root = CACHE / "qwen" / QWEN_REV / f"{split}_L128_{PROMPT_HASH[:16]}"
    npz_path = root / "embeddings.npz"
    metadata_path = root / "metadata.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing hard-OOS cache: {npz_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["split"] != split:
        raise RuntimeError(f"Hard-OOS cache split mismatch: {metadata['split']} != {split}")
    if metadata["model_revision"] != QWEN_REV or metadata["prompt_sha256"] != PROMPT_HASH:
        raise RuntimeError("Hard-OOS cache model/prompt mismatch")
    data = np.load(npz_path, allow_pickle=False)
    return {
        "vectors": data["vectors"].astype(np.float32),
        "sample_ids": data["sample_ids"].astype(str),
        "roles": data["roles"].astype(str),
        "intents": data["intents"].astype(str),
        "metadata": metadata,
    }


def build_prototypes(known: dict[str, Any], known_classes: Sequence[str], shot: int) -> np.ndarray:
    support_mask = known["roles"] == "support"
    support_vectors = known["vectors"][support_mask]
    support_labels = known["labels"][support_mask]
    prototypes = []
    for intent in known_classes:
        rows = support_vectors[support_labels == intent]
        if len(rows) != shot:
            raise RuntimeError(f"{intent}: expected {shot} supports, got {len(rows)}")
        center = rows.mean(axis=0)
        prototypes.append(center / np.linalg.norm(center))
    return np.stack(prototypes)


def score(vectors: np.ndarray, prototypes: np.ndarray, sample_ids, roles, labels) -> Scored:
    sims = vectors @ prototypes.T
    order = np.argsort(-sims, axis=1)
    pred = order[:, 0]
    s1 = sims[np.arange(len(sims)), pred]
    s2 = sims[np.arange(len(sims)), order[:, 1]]
    return Scored(
        sample_ids=np.asarray(sample_ids),
        roles=np.asarray(roles),
        intents=np.asarray(labels),
        s1=s1,
        s2=s2,
        margin=s1 - s2,
        pred_idx=pred,
    )


def known_val_scores(known: dict[str, Any], prototypes: np.ndarray, known_classes: Sequence[str]):
    mask = known["roles"] == "calibration_known"
    scored = score(
        known["vectors"][mask],
        prototypes,
        known["sample_ids"][mask],
        known["roles"][mask],
        known["labels"][mask],
    )
    true_idx = np.asarray([list(known_classes).index(label) for label in scored.intents])
    return scored, true_idx


def oos_val_scores(known: dict[str, Any], prototypes: np.ndarray):
    mask = known["roles"] == "calibration_oos"
    return score(
        known["vectors"][mask],
        prototypes,
        known["sample_ids"][mask],
        known["roles"][mask],
        np.full(int(mask.sum()), OOS, dtype=object),
    )


def hard_scores(hard: dict[str, Any], prototypes: np.ndarray):
    return score(
        hard["vectors"],
        prototypes,
        hard["sample_ids"],
        hard["roles"],
        np.full(len(hard["sample_ids"]), OOS, dtype=object),
    ), hard["intents"]


def threshold_grid(s1_known: np.ndarray) -> np.ndarray:
    return np.unique(np.quantile(s1_known, np.linspace(0.0, 1.0, N_TAU)))


def gamma_grid(margin_known: np.ndarray) -> np.ndarray:
    values = np.concatenate([[0.0], np.quantile(margin_known, np.linspace(0.0, 0.9, 19))])
    return np.unique(np.round(values, 8))


def select_b0(
    s1_known: np.ndarray,
    s1_oos: np.ndarray,
    *,
    target_acceptance: float,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for tau in threshold_grid(s1_known):
        known_acceptance = float(np.mean(s1_known >= tau))
        oos_recall = float(np.mean(s1_oos < tau))
        rows.append(
            {
                "tau": float(tau),
                "known_acceptance_val": known_acceptance,
                "oos_recall_val": oos_recall,
                "constraint_met": known_acceptance >= target_acceptance,
            }
        )
    valid = [row for row in rows if row["constraint_met"]]
    if not valid:
        raise RuntimeError("B0: no threshold satisfies the known acceptance constraint")
    best = max(valid, key=lambda row: (row["oos_recall_val"], row["known_acceptance_val"], row["tau"]))
    info = {
        "tau": best["tau"],
        "known_acceptance_val": best["known_acceptance_val"],
        "oos_recall_val": best["oos_recall_val"],
        "constraint_met": True,
        "n_candidates": len(rows),
        "objective": (
            f"max combined validation OOS recall subject to known acceptance "
            f">= {target_acceptance:.2f}; tie: known acceptance, then tau"
        ),
    }
    return float(best["tau"]), info, rows


def select_b2_original(
    known_classes: Sequence[str],
    known_scored: Scored,
    known_true_idx: np.ndarray,
    oos_scored: Scored,
    hard_scored: Scored,
    *,
    target_acceptance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    q_c = np.asarray(
        [
            float(np.quantile(known_scored.s1[known_true_idx == c], 0.10))
            for c in range(len(known_classes))
        ]
    )
    taus = threshold_grid(known_scored.s1)
    gammas = gamma_grid(known_scored.margin)
    s1_oos = np.concatenate([oos_scored.s1, hard_scored.s1])
    margin_oos = np.concatenate([oos_scored.margin, hard_scored.margin])
    pred_oos = np.concatenate([oos_scored.pred_idx, hard_scored.pred_idx])

    rows: list[dict[str, Any]] = []
    best = None
    for lam in (0.0, 0.5, 1.0):
        for tau in taus:
            tau_c = (1.0 - lam) * tau + lam * q_c
            for gamma in gammas:
                known_accept = (known_scored.s1 >= tau_c[known_scored.pred_idx]) & (
                    known_scored.margin >= gamma
                )
                oos_accept = (s1_oos >= tau_c[pred_oos]) & (margin_oos >= gamma)
                known_acceptance = float(np.mean(known_accept))
                orig_accept = (oos_scored.s1 >= tau_c[oos_scored.pred_idx]) & (
                    oos_scored.margin >= gamma
                )
                hard_accept = (hard_scored.s1 >= tau_c[hard_scored.pred_idx]) & (
                    hard_scored.margin >= gamma
                )
                orig_recall = float(np.mean(~orig_accept))
                hard_recall = float(np.mean(~hard_accept))
                oos_recall = float(np.mean(~oos_accept))
                row = {
                    "lambda": float(lam),
                    "tau": float(tau),
                    "gamma": float(gamma),
                    "known_acceptance_val": known_acceptance,
                    "orig_oos_recall_val": orig_recall,
                    "hard_oos_recall_val": hard_recall,
                    "oos_recall_val": oos_recall,
                    "constraint_met": known_acceptance >= target_acceptance,
                }
                rows.append(row)
                if row["constraint_met"]:
                    key = (
                        row["oos_recall_val"],
                        row["known_acceptance_val"],
                        -row["gamma"],
                        -row["lambda"],
                        row["tau"],
                    )
                    if best is None or key > best["_key"]:
                        best = {**row, "_key": key}
    if best is None:
        raise RuntimeError("B2-original: no configuration satisfies the acceptance constraint")
    selected = {
        "lambda": float(best["lambda"]),
        "tau": float(best["tau"]),
        "gamma": float(best["gamma"]),
        "q_c": q_c,
        "known_acceptance_val": best["known_acceptance_val"],
        "oos_recall_val": best["oos_recall_val"],
        "orig_oos_recall_val": best["orig_oos_recall_val"],
        "hard_oos_recall_val": best["hard_oos_recall_val"],
        "n_candidates": len(rows),
        "objective": "max combined validation OOS recall subject to known acceptance >= target",
    }
    return selected, rows, q_c


def _select_b2_v2_impl(
    known_classes: Sequence[str],
    known_sims: np.ndarray,
    known_scored: Scored,
    known_true_idx: np.ndarray,
    oos_scored: Scored,
    hard_scored: Scored,
    *,
    kappa: int,
    u_quantile: float,
    target_acceptance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    q_c = np.asarray(
        [
            float(np.quantile(known_sims[known_true_idx == c, c], 0.10))
            for c in range(len(known_classes))
        ]
    )
    taus = threshold_grid(known_scored.s1)
    gammas = gamma_grid(known_scored.margin)

    u_c = np.full(len(known_classes), np.nan, dtype=float)
    n_c = np.zeros(len(known_classes), dtype=int)
    for c in range(len(known_classes)):
        mask = hard_scored.pred_idx == c
        n_c[c] = int(mask.sum())
        if n_c[c] > 0:
            u_c[c] = float(np.quantile(hard_scored.s1[mask], u_quantile))
    alpha = n_c / (n_c + kappa)

    s1_oos = np.concatenate([oos_scored.s1, hard_scored.s1])
    margin_oos = np.concatenate([oos_scored.margin, hard_scored.margin])
    pred_oos = np.concatenate([oos_scored.pred_idx, hard_scored.pred_idx])

    rows: list[dict[str, Any]] = []
    best = None
    for tau in taus:
        cap = np.maximum(tau, q_c)
        target = np.where(
            np.isnan(u_c),
            tau,
            np.maximum(tau, np.minimum(u_c, cap)),
        )
        tau_c = (1.0 - alpha) * tau + alpha * target
        for gamma in gammas:
            known_accept = (known_scored.s1 >= tau_c[known_scored.pred_idx]) & (
                known_scored.margin >= gamma
            )
            oos_accept = (s1_oos >= tau_c[pred_oos]) & (margin_oos >= gamma)
            orig_accept = (oos_scored.s1 >= tau_c[oos_scored.pred_idx]) & (
                oos_scored.margin >= gamma
            )
            hard_accept = (hard_scored.s1 >= tau_c[hard_scored.pred_idx]) & (
                hard_scored.margin >= gamma
            )
            known_acceptance = float(np.mean(known_accept))
            oos_recall = float(np.mean(~oos_accept))
            row = {
                "tau": float(tau),
                "gamma": float(gamma),
                "known_acceptance_val": known_acceptance,
                "orig_oos_recall_val": float(np.mean(~orig_accept)),
                "hard_oos_recall_val": float(np.mean(~hard_accept)),
                "oos_recall_val": oos_recall,
                "constraint_met": known_acceptance >= target_acceptance,
            }
            rows.append(row)
            if row["constraint_met"]:
                key = (
                    row["oos_recall_val"],
                    row["known_acceptance_val"],
                    -row["gamma"],
                    row["tau"],
                )
                if best is None or key > best["_key"]:
                    best = {**row, "_key": key}
    if best is None:
        raise RuntimeError("B2-v2: no configuration satisfies the acceptance constraint")

    cap = np.maximum(best["tau"], q_c)
    target = np.where(np.isnan(u_c), best["tau"], np.maximum(best["tau"], np.minimum(u_c, cap)))
    tau_c = (1.0 - alpha) * best["tau"] + alpha * target
    selected = {
        "method": "B2_v2",
        "kappa": int(kappa),
        "u_quantile": float(u_quantile),
        "tau": float(best["tau"]),
        "gamma": float(best["gamma"]),
        "q_c": q_c,
        "u_c": u_c,
        "n_c": n_c,
        "alpha_c": alpha,
        "target_c": target,
        "tau_c": tau_c,
        "known_acceptance_val": best["known_acceptance_val"],
        "orig_oos_recall_val": best["orig_oos_recall_val"],
        "hard_oos_recall_val": best["hard_oos_recall_val"],
        "oos_recall_val": best["oos_recall_val"],
        "n_candidates": len(rows),
        "objective": "max combined validation OOS recall subject to known acceptance >= target",
        "formula": (
            "q_c = q10(s_c on known val true c); "
            "u_c = q_u(s1 on hard-OOS val predicted c); "
            "target_c = max(tau, min(u_c, max(tau, q_c))); "
            "alpha_c = n_c/(n_c+kappa); "
            "tau_c = (1-alpha_c)*tau + alpha_c*target_c; "
            "accept iff s1 >= tau_c and margin >= gamma"
        ),
    }
    return selected, rows, {
        "q_c": q_c,
        "u_c": u_c,
        "n_c": n_c,
        "alpha_c": alpha,
        "target_c": target,
        "tau_c": tau_c,
    }


def apply_method(
    scored: Scored,
    known_classes: Sequence[str],
    method: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "B0":
        tau = float(config["tau"])
        accept = scored.s1 >= tau
        tau_used = np.full(len(scored.s1), tau)
        decision = scored.s1 - tau
    elif method == "B2":
        lam = float(config["lambda"])
        tau = float(config["tau"])
        gamma = float(config["gamma"])
        q_c = np.asarray([config["q_c"][name] for name in known_classes])
        tau_c = (1.0 - lam) * tau + lam * q_c
        tau_used = tau_c[scored.pred_idx]
        accept = (scored.s1 >= tau_used) & (scored.margin >= gamma)
        decision = np.minimum(scored.s1 - tau_used, scored.margin - gamma)
    elif method == "B2_v2":
        gamma = float(config["gamma"])
        tau_used = np.asarray([config["tau_c"][name] for name in known_classes])[scored.pred_idx]
        accept = (scored.s1 >= tau_used) & (scored.margin >= gamma)
        decision = np.minimum(scored.s1 - tau_used, scored.margin - gamma)
    else:
        raise ValueError(method)
    pred = np.where(accept, np.asarray(known_classes)[scored.pred_idx], OOS)
    return pred, accept, decision


def metric_payload(
    known_classes: Sequence[str],
    known_scored: Scored,
    known_true_idx: np.ndarray,
    orig_scored: Scored,
    hard_scored: Scored,
    pred_known: np.ndarray,
    pred_orig: np.ndarray,
    pred_hard: np.ndarray,
    decision_known: np.ndarray,
    decision_hard: np.ndarray,
) -> dict[str, Any]:
    known_labels = np.asarray(known_classes)[known_true_idx]
    known_acceptance = float(np.mean(pred_known != OOS))
    known_routing = float(np.mean(pred_known == known_labels))
    known_macro_f1 = float(
        f1_score(known_labels, pred_known, labels=list(known_classes), average="macro", zero_division=0)
    )
    top1 = float(np.mean(np.asarray(known_classes)[known_scored.pred_idx] == known_labels))
    hard_recall = float(np.mean(pred_hard == OOS))
    orig_recall = float(np.mean(pred_orig == OOS)) if len(pred_orig) else None
    y = np.concatenate([known_labels, np.full(len(pred_hard), OOS)])
    p = np.concatenate([pred_known, pred_hard])
    hard_open_f1 = float(
        f1_score(y, p, labels=list(known_classes) + [OOS], average="macro", zero_division=0)
    )
    y_bin = np.concatenate([np.ones(len(pred_known)), np.zeros(len(pred_hard))])
    scores = np.concatenate([decision_known, decision_hard])
    auroc = float(roc_auc_score(y_bin, scores))
    return {
        "known_acceptance": known_acceptance,
        "known_correct_routing": known_routing,
        "known_macro_f1": known_macro_f1,
        "top1_closed_set_accuracy": top1,
        "orig_oos_recall": orig_recall,
        "hard_oos_recall": hard_recall,
        "hard_oos_false_acceptance": 1.0 - hard_recall,
        "hard_open_set_macro_f1": hard_open_f1,
        "hard_oos_auroc": auroc,
        "open_set_accuracy": float(np.mean(p == y)),
        "n_known": int(len(pred_known)),
        "n_orig_oos": int(len(pred_orig)),
        "n_hard_oos": int(len(pred_hard)),
        "known_rejected": int(np.sum(pred_known == OOS)),
        "orig_oos_rejected": int(np.sum(pred_orig == OOS)),
        "hard_oos_rejected": int(np.sum(pred_hard == OOS)),
        "hard_oos_accepted": int(np.sum(pred_hard != OOS)),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    known_classes: Sequence[str],
    known_scored: Scored,
    orig_scored: Scored,
    hard_scored: Scored,
    hard_intents: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    split_name: str,
) -> None:
    rows = []
    for method, (pred_known, pred_orig, pred_hard) in predictions.items():
        payload = {
            "known": (known_scored, pred_known),
            "orig_oos": (orig_scored, pred_orig),
            "hard_oos": (hard_scored, pred_hard),
        }
        for role_name, (scored, pred) in payload.items():
            for index in range(len(scored.s1)):
                true_label = (
                    str(scored.intents[index]) if role_name == "known" else OOS
                )
                rows.append(
                    {
                        "split": split_name,
                        "method": method,
                        "sample_id": scored.sample_ids[index],
                        "role": role_name,
                        "true_label": true_label,
                        "true_hard_intent": hard_intents[index] if role_name == "hard_oos" else "",
                        "pred_label": pred[index],
                        "accepted": int(pred[index] != OOS),
                        "top1_class": known_classes[scored.pred_idx[index]],
                        "s1": round(float(scored.s1[index]), 8),
                        "s2": round(float(scored.s2[index]), 8),
                        "margin": round(float(scored.margin[index]), 8),
                        "correct": int(pred[index] == true_label),
                    }
                )
    write_csv(path, rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--freeze", action="store_true", help="Write results/gate2_freeze.json after validation.")
    parser.add_argument("--force", action="store_true", help="Only for explicit test reruns.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot", type=int, default=5)
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.seed != 42 or args.shot != 5:
        raise RuntimeError("Gate 2 is frozen to seed=42, shot=5")

    if args.phase == "validation":
        if args.force:
            raise RuntimeError("--force is only meaningful for a deliberate test rerun")
        freeze_path = RESULTS / "gate2_freeze.json"
        if freeze_path.exists():
            raise RuntimeError("Gate-2 freeze already exists; do not refreeze silently")

        selected = json.loads((MANIFEST / "selected_intents.json").read_text(encoding="utf-8"))
        known_classes = [item["intent"] for item in selected["intents"]]
        known = load_known(args.seed, args.shot)
        prototypes = build_prototypes(known, known_classes, args.shot)
        known_scored, known_true_idx = known_val_scores(known, prototypes, known_classes)
        orig_scored = oos_val_scores(known, prototypes)
        hard = load_hard("hard_oos_val")
        hard_scored, hard_intents = hard_scores(hard, prototypes)

        # Full similarity matrix is needed only for the corrected s_c q_c.
        known_sims = known["vectors"][known["roles"] == "calibration_known"] @ prototypes.T

        target_acceptance = float(cfg["validation"]["target_known_acceptance"])
        s1_all_val_oos = np.concatenate([orig_scored.s1, hard_scored.s1])

        b0_tau, b0_info, b0_grid = select_b0(
            known_scored.s1, s1_all_val_oos, target_acceptance=target_acceptance
        )
        b2, b2_grid, b2_qc = select_b2_original(
            known_classes,
            known_scored,
            known_true_idx,
            orig_scored,
            hard_scored,
            target_acceptance=target_acceptance,
        )
        b2v2, b2v2_grid, b2v2_arrays = _select_b2_v2_impl(
            known_classes,
            known_sims,
            known_scored,
            known_true_idx,
            orig_scored,
            hard_scored,
            kappa=int(cfg["b2_v2"]["kappa"]),
            u_quantile=float(cfg["b2_v2"]["u_quantile"]),
            target_acceptance=target_acceptance,
        )

        configs = {
            "B0": {"tau": b0_tau},
            "B2": {
                "lambda": b2["lambda"],
                "tau": b2["tau"],
                "gamma": b2["gamma"],
                "q_c": {name: float(value) for name, value in zip(known_classes, b2_qc)},
            },
            "B2_v2": {
                "kappa": b2v2["kappa"],
                "u_quantile": b2v2["u_quantile"],
                "tau": b2v2["tau"],
                "gamma": b2v2["gamma"],
                "q_c": {name: float(value) for name, value in zip(known_classes, b2v2["q_c"])},
                "u_c": {name: None if np.isnan(value) else float(value) for name, value in zip(known_classes, b2v2["u_c"])},
                "n_c": {name: int(value) for name, value in zip(known_classes, b2v2["n_c"])},
                "alpha_c": {name: float(value) for name, value in zip(known_classes, b2v2["alpha_c"])},
                "target_c": {name: float(value) for name, value in zip(known_classes, b2v2["target_c"])},
                "tau_c": {name: float(value) for name, value in zip(known_classes, b2v2["tau_c"])},
            },
        }

        validation_results = {}
        validation_predictions = {}
        for method, config in configs.items():
            pred_known, accept_known, dec_known = apply_method(
                known_scored, known_classes, method, config
            )
            pred_orig, _, _ = apply_method(orig_scored, known_classes, method, config)
            pred_hard, _, dec_hard = apply_method(hard_scored, known_classes, method, config)
            validation_predictions[method] = (pred_known, pred_orig, pred_hard)
            validation_results[method] = metric_payload(
                known_classes,
                known_scored,
                known_true_idx,
                orig_scored,
                hard_scored,
                pred_known,
                pred_orig,
                pred_hard,
                dec_known,
                dec_hard,
            )

        metrics_rows = []
        for method, metrics in validation_results.items():
            row = {"phase": "validation", "method": method, **metrics}
            row["threshold"] = json.dumps(configs[method], ensure_ascii=False)
            metrics_rows.append(row)
        write_csv(RESULTS / "gate2_validation_metrics.csv", metrics_rows)
        write_predictions(
            RESULTS / "gate2_validation_predictions.csv",
            known_classes,
            known_scored,
            orig_scored,
            hard_scored,
            hard_intents,
            validation_predictions,
            split_name="validation",
        )
        write_csv(RESULTS / "gate2_b0_validation_grid.csv", b0_grid)
        write_csv(RESULTS / "gate2_b2_validation_grid.csv", b2_grid)
        write_csv(RESULTS / "gate2_b2v2_validation_grid.csv", b2v2_grid)

        threshold_payload = {
            "schema_version": "gate2-validation-thresholds-v1",
            "selected_at_utc": utc_now(),
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "model_revision": QWEN_REV,
            "prompt_sha256": PROMPT_HASH,
            "seed": args.seed,
            "shot": args.shot,
            "validation": {
                "known": 400,
                "original_oos": 100,
                "hard_oos": 1300,
                "hard_dev_classes": 65,
                "source": "known calibration rows + original oos_val + hard_oos_val",
            },
            "objective": (
                f"max combined validation OOS recall with known acceptance "
                f">= {target_acceptance:.2f}"
            ),
            "B0": {"tau": b0_tau, "info": b0_info},
            "B2_original": {
                "lambda": b2["lambda"],
                "tau": b2["tau"],
                "gamma": b2["gamma"],
                "q_c_definition": "10th percentile of top-1 s1 among known val rows with true label c (Gate-1 historical formula)",
                "q_c": configs["B2"]["q_c"],
                "info": {key: value for key, value in b2.items() if key not in ("q_c",)},
            },
            "B2_v2": {
                **{k: v for k, v in configs["B2_v2"].items() if k not in ("q_c", "u_c", "n_c", "alpha_c", "target_c", "tau_c")},
                "q_c": configs["B2_v2"]["q_c"],
                "u_c": configs["B2_v2"]["u_c"],
                "n_c": configs["B2_v2"]["n_c"],
                "alpha_c": configs["B2_v2"]["alpha_c"],
                "target_c": configs["B2_v2"]["target_c"],
                "tau_c": configs["B2_v2"]["tau_c"],
                "formula": b2v2["formula"],
                "hyperparameter_note": (
                    "kappa=10 and u_quantile=0.99 were fixed after a validation-only "
                    "design check; they are not selected on hard_oos_test."
                ),
            },
            "validation_metrics": validation_results,
        }
        (RESULTS / "gate2_validation_thresholds.json").write_text(
            json.dumps(threshold_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Per-class B2-v2 threshold table.
        class_rows = []
        for c, name in enumerate(known_classes):
            class_rows.append(
                {
                    "intent": name,
                    "q_c": float(b2v2["q_c"][c]),
                    "u_c": None if np.isnan(b2v2["u_c"][c]) else float(b2v2["u_c"][c]),
                    "n_c": int(b2v2["n_c"][c]),
                    "alpha_c": float(b2v2["alpha_c"][c]),
                    "target_c": float(b2v2["target_c"][c]),
                    "tau_c": float(b2v2["tau_c"][c]),
                }
            )
        write_csv(RESULTS / "gate2_b2v2_class_thresholds.csv", class_rows)

        print(json.dumps(threshold_payload["validation_metrics"], ensure_ascii=False, indent=2))
        print(json.dumps({"B0": b0_info, "B2": {k: v for k, v in b2.items() if k != "q_c"}, "B2_v2": {k: v for k, v in b2v2.items() if k not in ("q_c", "u_c", "n_c", "alpha_c", "target_c", "tau_c")}}, ensure_ascii=False, indent=2))
        print("GATE2_VALIDATION_OK")

        if args.freeze:
            b0_metrics = validation_results["B0"]
            v2_metrics = validation_results["B2_v2"]
            if not (
                v2_metrics["hard_oos_recall"] > b0_metrics["hard_oos_recall"]
                and v2_metrics["known_correct_routing"] >= b0_metrics["known_correct_routing"]
            ):
                raise RuntimeError(
                    "B2-v2 did not improve hard-OOS recall while preserving known routing; "
                    "refusing to freeze/run the held-out test."
                )
            freeze = {
                "schema_version": "gate2-freeze-v1",
                "frozen": True,
                "frozen_at_utc": utc_now(),
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "model_revision": QWEN_REV,
                "prompt_sha256": PROMPT_HASH,
                "seed": args.seed,
                "shot": args.shot,
                "config_sha256": sha256_file(CONFIG_PATH),
                "hard_oos_split_sha256": sha256_file(MANIFEST / "gate2_hard_oos_split.json"),
                "validation_metrics": validation_results,
                "thresholds": configs,
                "selection": {
                    "objective": threshold_payload["objective"],
                    "b2_v2_hyperparameters": {
                        "kappa": int(cfg["b2_v2"]["kappa"]),
                        "u_quantile": float(cfg["b2_v2"]["u_quantile"]),
                    },
                    "q_c_definition_b2_v2": "10th percentile of s_c(x), the true-class score",
                },
                "test_policy": (
                    "hard_oos_test is 65 disjoint classes' official test rows (1950). "
                    "It is evaluated exactly once after this freeze. No threshold, method, "
                    "or hyperparameter may be changed after this file is written."
                ),
                "hard_oos_test_evaluated": False,
            }
            freeze_path.write_text(
                json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            # Write a human-readable copy for convenience.
            (RESULTS / "gate2_freeze_thresholds.json").write_text(
                json.dumps(configs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print("GATE2_FREEZE_OK")
        return 0

    # test phase
    freeze_path = RESULTS / "gate2_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("Gate-2 is not frozen; run validation with --freeze first")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("frozen"):
        raise RuntimeError("Gate-2 freeze is not valid")
    if freeze.get("hard_oos_test_evaluated"):
        raise RuntimeError("hard_oos_test was already evaluated once; refusing a second read")
    test_metrics_path = RESULTS / "gate2_test_metrics.csv"
    if test_metrics_path.exists() and not args.force:
        raise RuntimeError("Gate-2 test metrics already exist; refusing a second test read without --force")

    selected = json.loads((MANIFEST / "selected_intents.json").read_text(encoding="utf-8"))
    known_classes = [item["intent"] for item in selected["intents"]]
    known = load_known(args.seed, args.shot)
    prototypes = build_prototypes(known, known_classes, args.shot)
    test_mask = known["roles"] == "final_known_test"
    known_test = score(
        known["vectors"][test_mask],
        prototypes,
        known["sample_ids"][test_mask],
        known["roles"][test_mask],
        known["labels"][test_mask],
    )
    known_test_true_idx = np.asarray([list(known_classes).index(label) for label in known_test.intents])
    hard = load_hard("hard_oos_test")
    hard_test, hard_intents = hard_scores(hard, prototypes)

    configs = freeze["thresholds"]
    results = {}
    predictions = {}
    for method in ("B0", "B2", "B2_v2"):
        pred_known, accept_known, dec_known = apply_method(
            known_test, known_classes, method, configs[method]
        )
        pred_hard, accept_hard, dec_hard = apply_method(
            hard_test, known_classes, method, configs[method]
        )
        # For test, only known test + hard-OOS test; original OOS test is not
        # part of the new protocol and is not mixed into these metrics.
        orig_placeholder = Scored(
            sample_ids=np.asarray([], dtype=object),
            roles=np.asarray([], dtype=object),
            intents=np.asarray([], dtype=object),
            s1=np.asarray([]),
            s2=np.asarray([]),
            margin=np.asarray([]),
            pred_idx=np.asarray([], dtype=int),
        )
        payload = metric_payload(
            known_classes,
            known_test,
            known_test_true_idx,
            orig_placeholder,
            hard_test,
            pred_known,
            np.asarray([], dtype=object),
            pred_hard,
            dec_known,
            dec_hard,
        )
        # The payload includes original-OOS fields only if rows exist; remove
        # the undefined component for the new protocol.
        payload.pop("n_orig_oos", None)
        payload.pop("orig_oos_rejected", None)
        payload["orig_oos_recall"] = None
        results[method] = payload
        predictions[method] = (pred_known, np.asarray([], dtype=object), pred_hard)

    rows = []
    for method, payload in results.items():
        rows.append({"phase": "hard_oos_test", "method": method, **payload})
    write_csv(RESULTS / "gate2_test_metrics.csv", rows)
    write_predictions(
        RESULTS / "gate2_test_predictions.csv",
        known_classes,
        known_test,
        Scored(
            sample_ids=np.asarray([], dtype=object),
            roles=np.asarray([], dtype=object),
            intents=np.asarray([], dtype=object),
            s1=np.asarray([]),
            s2=np.asarray([]),
            margin=np.asarray([]),
            pred_idx=np.asarray([], dtype=int),
        ),
        hard_test,
        hard_intents,
        predictions,
        split_name="hard_oos_test",
    )

    for method in ("B0", "B2", "B2_v2"):
        pred_known, _, _ = apply_method(known_test, known_classes, method, configs[method])
        pred_hard, _, _ = apply_method(hard_test, known_classes, method, configs[method])
        labels = list(known_classes) + [OOS]
        y = np.concatenate([np.asarray(known_classes)[known_test_true_idx], np.full(len(pred_hard), OOS)])
        p = np.concatenate([pred_known, pred_hard])
        cm = confusion_matrix(y, p, labels=labels)
        write_csv(
            RESULTS / f"gate2_test_confusion_{method.lower()}.csv",
            [
                {"true_label": label, **{pred: int(cm[i, j]) for j, pred in enumerate(labels)}}
                for i, label in enumerate(labels)
            ],
        )

    run_payload = {
        "schema_version": "gate2-test-run-v1",
        "evaluated_at_utc": utc_now(),
        "freeze_path": str(freeze_path.resolve()),
        "hard_oos_test_rows": int(len(hard_test.s1)),
        "hard_oos_test_classes": int(len(set(hard_intents))),
        "known_test_rows": int(len(known_test.s1)),
        "test_metrics": results,
    }
    (RESULTS / "gate2_test_run.json").write_text(
        json.dumps(run_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    freeze["hard_oos_test_evaluated"] = True
    freeze["hard_oos_test_evaluated_at_utc"] = utc_now()
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("GATE2_TEST_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"GATE2_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
