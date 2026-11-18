"""B4-AR: adaptive prototype-neighbor gate and lightweight reject head.

The frozen encoder is never fine-tuned here.  All math operates on cached
BANKING77 GTE vectors.  The gate is shared across classes, so its input
features are per-class `[P_c, N_c, P_c-N_c, d_c]`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "manifest"
CACHE = PROJECT_ROOT / "data" / "cache_embeddings" / "gate3"
RESULTS = PROJECT_ROOT / "results"
DATASET = "banking77"
MODEL = "gte"
SHOT = 10
SEED = 42
OOS = "OOS"
GTE_REV = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
GTE_PROMPT_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

import sys

sys.path.insert(0, str(PROJECT_ROOT))
from src.gate3.methods import (  # noqa: E402
    knn_scores,
    prototypes_multi,
    scores_from_prototypes,
)


def load_info() -> dict:
    return json.loads((MANIFEST / f"gate3_{DATASET}_dataset.json").read_text(encoding="utf-8"))


def load_cache() -> dict[str, Any]:
    path = CACHE / MODEL / DATASET / GTE_REV / f"L128_{GTE_PROMPT_HASH[:16]}" / "embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    out = {
        "vectors": data["vectors"].astype(np.float32),
        "sample_ids": data["sample_ids"].astype(str),
        "roles": data["roles"].astype(str),
        "labels": data["labels"].astype(str),
        "intents": data["intents"].astype(str),
    }
    if not (len(out["vectors"]) == len(out["sample_ids"]) == len(out["roles"]) == len(out["labels"])):
        raise RuntimeError("Mismatched GTE cache arrays")
    return out


def support_arrays(cache: dict, shot: int = SHOT) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader((MANIFEST / f"gate3_{DATASET}_support_{shot}.csv").open("r", encoding="utf-8")))
    ids = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    idx = np.asarray([ids[row["sample_id"]] for row in rows], dtype=int)
    return cache["vectors"][idx], np.asarray([row["intent"] for row in rows], dtype=object)


def split_vectors(cache: dict, role: str, classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    mask = cache["roles"] == role
    labels = cache["labels"][mask]
    cmap = {c: i for i, c in enumerate(classes)}
    true = np.asarray([cmap.get(label, -1) for label in labels], dtype=int)
    return cache["vectors"][mask], true


def compute_pn(
    support_X: np.ndarray,
    support_y: np.ndarray,
    classes: Sequence[str],
    vectors: np.ndarray,
    *,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    protos, details = prototypes_multi(support_X, support_y, classes, seed=seed)
    p = scores_from_prototypes(vectors, protos)
    n = knn_scores(vectors, support_X, support_y, classes, top_k=2)
    disp = np.asarray([details["dispersion"][c] for c in classes], dtype=np.float32)
    return p.astype(np.float32), n.astype(np.float32), disp, details


@dataclass
class Episode:
    p_known: np.ndarray
    n_known: np.ndarray
    d_known: np.ndarray
    y_known: np.ndarray
    p_oos: np.ndarray
    n_oos: np.ndarray
    d_oos: np.ndarray
    n_classes: int
    fold: int
    mask: list[str]


def build_episodes(
    support_X: np.ndarray,
    support_y: np.ndarray,
    classes: Sequence[str],
    *,
    seed: int = SEED,
    n_masks: int = 12,
    mask_size: int = 6,
) -> list[Episode]:
    """Construct leave-fold / leave-class-out episodes from the 10-shot support.

    For each class, 5 samples build prototypes and the other 5 are queries; the
    two folds swap these roles.  Each episode masks `mask_size` classes and uses
    their query samples as pseudo-OOS, so a query is never its own prototype.
    """
    class_indices = {c: np.where(support_y == c)[0] for c in classes}
    for c in classes:
        if len(class_indices[c]) != 10:
            raise RuntimeError(f"Need exactly 10 support rows for {c}, got {len(class_indices[c])}")
    episodes: list[Episode] = []
    for fold in range(2):
        for m in range(n_masks):
            rng = np.random.default_rng(seed + fold * 1000 + m)
            masked_pos = rng.choice(len(classes), size=mask_size, replace=False)
            masked = {classes[i] for i in masked_pos}
            known = [c for c in classes if c not in masked]
            proto_rows = []
            query_known = []
            oos_rows = []
            for c in known:
                rows = class_indices[c]
                proto = rows[:5] if fold == 0 else rows[5:]
                query = rows[5:] if fold == 0 else rows[:5]
                proto_rows.extend(proto.tolist())
                query_known.extend(query.tolist())
            y_known = []
            for c in known:
                rows = class_indices[c]
                query = rows[5:] if fold == 0 else rows[:5]
                y_known.extend([known.index(c)] * len(query))
            for c in sorted(masked):
                rows = class_indices[c]
                query = rows[5:] if fold == 0 else rows[:5]
                oos_rows.extend(query.tolist())
            proto_X = support_X[proto_rows]
            proto_y = np.concatenate([[c] * (5) for c in known])
            query_X = support_X[query_known + oos_rows]
            p, n, d, _ = compute_pn(proto_X, proto_y, known, query_X, seed=seed + fold * 17 + m)
            nk = len(query_known)
            episodes.append(
                Episode(
                    p_known=p[:nk],
                    n_known=n[:nk],
                    d_known=d,
                    y_known=np.asarray(y_known, dtype=np.int64),
                    p_oos=p[nk:],
                    n_oos=n[nk:],
                    d_oos=d,
                    n_classes=len(known),
                    fold=fold,
                    mask=sorted(masked),
                )
            )
    return episodes


class B4AR(nn.Module):
    def __init__(self, gate_hidden: int = 32, reject_hidden: int = 16) -> None:
        super().__init__()
        self.gate_mlp = nn.Sequential(nn.Linear(4, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1))
        self.reject_mlp = nn.Sequential(nn.Linear(7, reject_hidden), nn.GELU(), nn.Linear(reject_hidden, 1))
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        nn.init.zeros_(self.reject_mlp[-1].weight)
        nn.init.zeros_(self.reject_mlp[-1].bias)

    def gate(self, p: torch.Tensor, n: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # p/n: [B, C], d: [C]
        c = p.shape[1]
        dmat = d.view(1, c, 1).expand(p.shape[0], c, 1)
        feat = torch.cat([p.unsqueeze(-1), n.unsqueeze(-1), (p - n).unsqueeze(-1), dmat], dim=-1)
        return torch.sigmoid(self.gate_mlp(feat)).squeeze(-1)

    def fused(self, p: torch.Tensor, n: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.gate(p, n, d)
        return g * p + (1.0 - g) * n, g

    def reject_features(self, p: torch.Tensor, n: torch.Tensor, s: torch.Tensor, g: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        top3 = torch.topk(s, k=3, dim=1)
        s1 = top3.values[:, 0]
        s2 = top3.values[:, 1]
        probs = torch.softmax(top3.values, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        best = top3.indices[:, 0]
        row = torch.arange(p.shape[0], device=p.device)
        disp_best = d[best]
        branch = torch.abs(p[row, best] - n[row, best])
        gate_best = g[row, best]
        return torch.stack([s1, s1 - s2, entropy, disp_best, branch, gate_best, s1 * (1.0 - s2)], dim=1)

    def reject_logit(self, feats: torch.Tensor) -> torch.Tensor:
        return self.reject_mlp(feats).squeeze(-1)


def select_threshold_custom(
    score: np.ndarray,
    class_idx: np.ndarray,
    true_idx: np.ndarray,
    classes: Sequence[str],
    *,
    target_acceptance: float = 0.90,
) -> tuple[float, list[dict], dict]:
    is_known = true_idx >= 0
    candidates = np.unique(np.quantile(score[is_known], np.linspace(0.0, 1.0, 101)))
    best = None
    rows = []
    y = np.asarray([classes[i] if i >= 0 else OOS for i in true_idx])
    for tau in candidates:
        accept = score >= tau
        pred = np.where(accept, np.asarray(classes)[class_idx], OOS)
        ka = float(np.mean(accept[is_known]))
        oos = float(np.mean(~accept[~is_known]))
        f1 = float(f1_score(y, pred, labels=list(classes) + [OOS], average="macro", zero_division=0))
        row = {
            "tau": float(tau),
            "known_acceptance": ka,
            "oos_recall": oos,
            "open_set_macro_f1": f1,
            "constraint_met": ka >= target_acceptance,
        }
        rows.append(row)
        if row["constraint_met"]:
            key = (row["open_set_macro_f1"], row["oos_recall"], row["known_acceptance"], row["tau"])
            if best is None or key > best["_key"]:
                best = {**row, "_key": key}
    if best is None:
        best = max(rows, key=lambda r: (r["open_set_macro_f1"], r["oos_recall"], r["known_acceptance"]))
        best["_key"] = None
    info = {
        "tau": float(best["tau"]),
        "known_acceptance_val": best["known_acceptance"],
        "oos_recall_val": best["oos_recall"],
        "open_set_macro_f1_val": best["open_set_macro_f1"],
        "constraint_met": bool(best["constraint_met"]),
        "n_candidates": len(rows),
        "target_acceptance": target_acceptance,
        "objective": "max validation open-set macro-F1 subject to known acceptance >= 0.90",
    }
    return float(best["tau"]), rows, info


def evaluate_custom(
    score: np.ndarray,
    class_idx: np.ndarray,
    true_idx: np.ndarray,
    tau: float,
    classes: Sequence[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    is_known = true_idx >= 0
    accept = score >= tau
    pred = np.where(accept, np.asarray(classes)[class_idx], OOS)
    y = np.asarray([classes[i] if i >= 0 else OOS for i in true_idx])
    known_pred = pred[is_known]
    known_true = y[is_known]
    oos_pred = pred[~is_known]
    metrics = {
        "known_acceptance": float(np.mean(accept[is_known])),
        "known_correct_routing": float(np.mean(known_pred == known_true)),
        "known_macro_f1": float(f1_score(known_true, known_pred, labels=list(classes), average="macro", zero_division=0)),
        "top1_closed_set_accuracy": float(np.mean(np.asarray(classes)[class_idx[is_known]] == known_true)),
        "oos_recall": float(np.mean(oos_pred == OOS)),
        "oos_false_acceptance": float(np.mean(oos_pred != OOS)),
        "unknown_auroc": float(roc_auc_score(is_known.astype(int), score)),
        "open_set_macro_f1": float(f1_score(y, pred, labels=list(classes) + [OOS], average="macro", zero_division=0)),
        "open_set_accuracy": float(np.mean(pred == y)),
        "n_known": int(is_known.sum()),
        "n_unknown": int((~is_known).sum()),
        "known_rejected": int(np.sum(known_pred == OOS)),
        "unknown_rejected": int(np.sum(oos_pred == OOS)),
        "unknown_accepted": int(np.sum(oos_pred != OOS)),
    }
    return metrics, pred, accept


def tensor_episode(ep: Episode, device: str = "cpu") -> dict[str, torch.Tensor]:
    return {
        "p_known": torch.from_numpy(ep.p_known).to(device),
        "n_known": torch.from_numpy(ep.n_known).to(device),
        "d": torch.from_numpy(ep.d_known).to(device),
        "y_known": torch.from_numpy(ep.y_known).to(device),
        "p_oos": torch.from_numpy(ep.p_oos).to(device),
        "n_oos": torch.from_numpy(ep.n_oos).to(device),
    }


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path: Path, model: nn.Module, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict_cpu(model), "meta": meta}, path)
