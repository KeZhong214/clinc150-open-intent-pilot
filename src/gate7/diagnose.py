"""G7-A: two-level development diagnostics for Gate6 Qwen Frozen/Plain/Anchor.

No training and no final-test labels are touched here.  Thresholds are selected
only on dev_threshold; all reported numbers are dev_select.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "manifest"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate6.train_qwen_b4 import val_scores  # noqa: E402
from src.gate3.methods import evaluate, knn_scores, prototypes_multi, prototypes_single, scores_from_prototypes, select_threshold  # noqa: E402
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402


def load_split(name: str) -> tuple[list[str], np.ndarray]:
    rows = pd.read_csv(MANIFEST / f"gate7_{name}.csv")
    classes = qe.classes()
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.asarray([cmap.get(x, -1) for x in rows.intent], dtype=int)
    return rows.sample_id.tolist(), y


def vectors_for(cache: dict, ids: list[str]) -> np.ndarray:
    idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    missing = [sid for sid in ids if sid not in idx]
    if missing:
        raise KeyError(missing[:3])
    return cache["vectors"][[idx[sid] for sid in ids]]


def sims_b0_b3(cache: dict, method: str, query: np.ndarray):
    X, y = qd.support(cache)
    classes = qe.classes()
    if method == "B0":
        protos = prototypes_single(X, y, classes)
        return scores_from_prototypes(query, [p[None, :] for p in protos])
    protos, _ = prototypes_multi(X, y, classes)
    p = scores_from_prototypes(query, protos)
    n = knn_scores(query, X, y, classes, top_k=2)
    return 0.5 * p + 0.5 * n


def b4_model(encoder: str) -> B4AR:
    payload = torch.load(RESULTS / f"gate6_qwen_{encoder}_reject.pt", map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def reject_scores(cache: dict, encoder: str, query: np.ndarray):
    model = b4_model(encoder)
    reject, class_idx, sims, _ = val_scores(model, cache, query, qe.classes(), reject_only=True)
    return reject, class_idx, sims


def score_summary(score: np.ndarray, y: np.ndarray, prefix: str) -> dict[str, Any]:
    known = score[y >= 0]
    oos = score[y < 0]
    return {
        f"{prefix}_known_mean": float(known.mean()),
        f"{prefix}_known_p10": float(np.quantile(known, 0.10)),
        f"{prefix}_known_p50": float(np.quantile(known, 0.50)),
        f"{prefix}_oos_mean": float(oos.mean()),
        f"{prefix}_oos_p50": float(np.quantile(oos, 0.50)),
        f"{prefix}_oos_p90": float(np.quantile(oos, 0.90)),
    }


def unknown_class_fpr(score: np.ndarray, y: np.ndarray, classes, tau: float, encoder: str, method: str, split: str):
    rows = []
    for ci, cname in enumerate(classes):
        pass
    # Unknown classes are not in the 40-class map; recover from manifest order.
    rows = []
    return rows


def main() -> int:
    classes = qe.classes()
    thr_ids, thr_y = load_split("dev_threshold")
    sel_ids, sel_y = load_split("dev_select")
    caches = {name: qd.load(name) for name in ("frozen", "plain", "anchor")}
    rows = []
    ng_rows = []
    curve_rows = []
    score_files = {}
    for enc_name, cache in caches.items():
        query_thr = vectors_for(cache, thr_ids)
        query_sel = vectors_for(cache, sel_ids)
        for method in ("B0", "B3-H", "B4-RejectOnly"):
            if method == "B4-RejectOnly" and enc_name == "anchor":
                continue
            if method == "B4-RejectOnly":
                s_thr, class_thr, sims_thr = reject_scores(cache, enc_name, query_thr)
                s_sel, class_sel, sims_sel = reject_scores(cache, enc_name, query_sel)
                tau, _, info = b4_core.select_threshold_custom(s_thr, class_thr, thr_y, classes)
                metrics, pred, accept = b4_core.evaluate_custom(s_sel, class_sel, sel_y, tau, classes)
                score = s_sel
                margin = np.array([np.sort(x)[-1] - np.sort(x)[-2] for x in sims_sel])
            else:
                sims_thr = sims_b0_b3(cache, method, query_thr)
                sims_sel = sims_b0_b3(cache, method, query_sel)
                tau, _, info = select_threshold(sims_thr, thr_y, len(classes), classes)
                metrics, _, _ = evaluate(sims_sel, sel_y, classes, tau)
                score = np.max(sims_sel, axis=1)
                order = np.argsort(-sims_sel, axis=1)
                margin = sims_sel[np.arange(len(sims_sel)), order[:, 0]] - sims_sel[np.arange(len(sims_sel)), order[:, 1]]
            row = {"encoder": f"qwen_{enc_name}", "method": method, "tau": float(tau), "dev_select_n": len(sel_y), **{f"select_{k}": v for k, v in metrics.items()}, **score_summary(score, sel_y, "select")}
            row["select_margin_mean"] = float(margin.mean())
            row["select_margin_p10"] = float(np.quantile(margin, 0.10))
            rows.append(row)
            score_files[f"{enc_name}_{method}"] = {"ids": sel_ids, "score": score.astype(np.float32), "y": sel_y, "tau": float(tau)}
            # acceptance curve on dev_select, for diagnosis only (not selection).
            for q in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0):
                t = float(np.quantile(score, q))
                curve_rows.append({"encoder": f"qwen_{enc_name}", "method": method, "score_quantile": q, "tau": t, "known_acceptance": float((score[sel_y >= 0] >= t).mean()), "oos_acceptance": float((score[sel_y < 0] >= t).mean())})
            # unknown-class acceptance on dev_select; manifest order gives class labels.
            sel_manifest = pd.read_csv(MANIFEST / "gate7_dev_select.csv")
            for cname, g in sel_manifest[sel_manifest.role == "dev_unknown"].groupby("intent"):
                idx = [sel_ids.index(sid) for sid in g.sample_id]
                ng_rows.append({"encoder": f"qwen_{enc_name}", "method": method, "unknown_class": cname, "n": len(idx), "fpr": float((score[idx] >= tau).mean())})
    pd.DataFrame(rows).to_csv(RESULTS / "gate7_dev_diagnosis.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(RESULTS / "gate7_dev_acceptance_curves.csv", index=False)
    pd.DataFrame(ng_rows).to_csv(RESULTS / "gate7_dev_unknown_class_fpr.csv", index=False)
    np.savez_compressed(RESULTS / "gate7_dev_scores.npz", **{f"{k}_score": v["score"] for k, v in score_files.items()}, **{f"{k}_y": v["y"] for k, v in score_files.items()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # Hard-negative diagnostics.
    hn = pd.read_csv(RESULTS / "gate6_qwen_hard_negatives.csv")
    hn["margin"] = hn["anchor_pos_cos"] - hn["anchor_neg_cos"]
    hn_stats = {
        "n": int(len(hn)),
        "hard_neg_cos_mean": float(hn.anchor_neg_cos.mean()),
        "hard_neg_cos_p90": float(hn.anchor_neg_cos.quantile(0.90)),
        "margin_mean": float(hn.margin.mean()),
        "margin_negative_count": int((hn.margin < 0).sum()),
        "margin_p10": float(hn.margin.quantile(0.10)),
        "duplicate_hard_negative_count": int(hn.hard_negative_id.duplicated().sum()),
        "unique_hard_negative_count": int(hn.hard_negative_id.nunique()),
        "top20": hn.sort_values("anchor_neg_cos", ascending=False).head(20).to_dict("records"),
    }
    (RESULTS / "gate7_hard_negative_stats.json").write_text(json.dumps(hn_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Toy proof: the current microbatch creates a 6x6 matrix; grad accumulation
    # does not concatenate microbatches into a 48x48 matrix.
    def matrix_shape(n: int) -> list[int]:
        z = torch.randn(n, 16)
        return list((z @ z.T).shape)
    toy = {
        "microbatch_2_three_roles_n": 6,
        "microbatch_similarity_shape": matrix_shape(6),
        "grad_accum_8_hypothetical_shape": matrix_shape(48),
        "actual_current_code": "each microbatch calls supcon_loss(z[6], y[6]); gradients accumulate, logits are never 48x48",
        "microbatches_per_epoch": 400 // 2,
        "optimizer_steps_per_epoch": (400 // 2) // 8,
        "leftover_microbatches": (400 // 2) % 8,
        "leftover_bug": "If microbatches % grad_accum != 0, the original loop drops the final partial accumulation; the Gate6 400/2/8 config has remainder 0.",
    }
    (RESULTS / "gate7_toy_batch.json").write_text(json.dumps(toy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"hard_negative": {k: v for k, v in hn_stats.items() if k != "top20"}, "toy": toy}, ensure_ascii=False, indent=2))
    print("GATE7_DIAGNOSE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
