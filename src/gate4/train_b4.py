"""Train B4-AR on frozen GTE BANKING77 10-shot support.

Stage 1: B4-Fusion (learned per-class gate only).
Stage 2: B4-Full (same gate + lightweight reject head).

Training uses only the 400 support vectors and leave-fold/leave-class-out
episodes constructed inside that support.  Calibration known + dev unknown are
used only for validation, threshold calibration and diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
MANIFEST = PROJECT_ROOT / "data" / "manifest"

import sys

sys.path.insert(0, str(PROJECT_ROOT))
from src.gate4.b4_core import B4AR  # noqa: E402
from src.gate4 import b4_core  # noqa: E402


def eval_gate(
    model: B4AR,
    p: torch.Tensor,
    n: torch.Tensor,
    d: torch.Tensor,
    y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    model.eval()
    with torch.no_grad():
        s, g = model.fused(p, n, d)
        loss = F.cross_entropy(s, y).item() if y is not None else float("nan")
        feats = model.reject_features(p, n, s, g, d)
        logit = model.reject_logit(feats)
    return s, g, {"ce": loss, "logit": logit.detach().cpu(), "feats": feats.detach().cpu()}


def train_fusion(model: B4AR, episodes, val_pack, epochs: int, lr: float) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(1000 + epoch)
        order = rng.permutation(len(episodes))
        total = 0.0
        gate_grads = []
        for idx in order:
            ep = b4_core.tensor_episode(episodes[idx])
            opt.zero_grad()
            s, g = model.fused(ep["p_known"], ep["n_known"], ep["d"])
            loss = F.cross_entropy(s, ep["y_known"])
            loss.backward()
            gate_grads.append(
                math.sqrt(sum(float(p.grad.pow(2).sum()) for p in model.gate_mlp.parameters() if p.grad is not None))
            )
            opt.step()
            total += float(loss.detach()) * len(ep["y_known"])
        with torch.no_grad():
            val_s, val_g, _ = eval_gate(model, val_pack["p"], val_pack["n"], val_pack["d"], None)
            val_ce = F.cross_entropy(val_s[val_pack["known_mask"]], val_pack["y_known"]).item()
        history.append(
            {
                "stage": "B4-Fusion",
                "epoch": epoch,
                "train_loss": total / sum(len(e.y_known) for e in episodes),
                "val_known_ce": val_ce,
                "val_reject_bce": "",
                "gate_grad_norm": float(np.mean(gate_grads)),
                "gate_mean": float(val_g.mean()),
                "gate_std": float(val_g.std()),
            }
        )
        if epoch in (1, 5, 10, 20, 40, epochs):
            print(f"Fusion epoch {epoch}: loss={history[-1]['train_loss']:.4f} val_ce={val_ce:.4f} gate={history[-1]['gate_mean']:.3f}")
    final = {k: v.detach().clone() for k, v in model.state_dict().items()}
    delta = max(float((final[k] - initial[k]).abs().max()) for k in initial if k.startswith("gate_mlp"))
    history[-1]["gate_max_abs_delta"] = delta
    return history


def train_full(model: B4AR, episodes, val_pack, epochs: int, lr: float) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    initial_gate = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("gate_mlp")}
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(3000 + epoch)
        order = rng.permutation(len(episodes))
        total = 0.0
        gate_grads = []
        n_seen = 0
        for idx in order:
            ep = b4_core.tensor_episode(episodes[idx])
            opt.zero_grad()
            s, g = model.fused(ep["p_known"], ep["n_known"], ep["d"])
            ce = F.cross_entropy(s, ep["y_known"])
            feats_k = model.reject_features(ep["p_known"], ep["n_known"], s, g, ep["d"])
            logit_k = model.reject_logit(feats_k)
            bce_k = F.binary_cross_entropy_with_logits(logit_k, torch.ones_like(logit_k))
            s_o, g_o = model.fused(ep["p_oos"], ep["n_oos"], ep["d"])
            feats_o = model.reject_features(ep["p_oos"], ep["n_oos"], s_o, g_o, ep["d"])
            logit_o = model.reject_logit(feats_o)
            bce_o = F.binary_cross_entropy_with_logits(logit_o, torch.zeros_like(logit_o))
            bce = 0.5 * (bce_k + bce_o)
            loss = ce + 0.5 * bce
            loss.backward()
            gate_grads.append(
                math.sqrt(sum(float(p.grad.pow(2).sum()) for p in model.gate_mlp.parameters() if p.grad is not None))
            )
            opt.step()
            total += float(loss.detach()) * (len(ep["y_known"]) + len(ep["p_oos"]))
            n_seen += len(ep["y_known"]) + len(ep["p_oos"])
        with torch.no_grad():
            val_s, val_g, _ = eval_gate(model, val_pack["p"], val_pack["n"], val_pack["d"], None)
            val_ce = F.cross_entropy(val_s[val_pack["known_mask"]], val_pack["y_known"]).item()
            feats_all = model.reject_features(val_pack["p"], val_pack["n"], val_s, val_g, val_pack["d"])
            logits_all = model.reject_logit(feats_all)
            bce_all = F.binary_cross_entropy_with_logits(logits_all, val_pack["y_all"]).item()
        history.append(
            {
                "stage": "B4-Full",
                "epoch": epoch,
                "train_loss": total / max(n_seen, 1),
                "val_known_ce": val_ce,
                "val_reject_bce": bce_all,
                "gate_grad_norm": float(np.mean(gate_grads)),
                "gate_mean": float(val_g.mean()),
                "gate_std": float(val_g.std()),
            }
        )
        if epoch in (1, 5, 10, 20, 40, epochs):
            print(f"Full epoch {epoch}: loss={history[-1]['train_loss']:.4f} val_ce={val_ce:.4f} val_bce={bce_all:.4f} gate={history[-1]['gate_mean']:.3f}")
    final_gate = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("gate_mlp")}
    history[-1]["gate_max_abs_delta"] = max(float((final_gate[k] - initial_gate[k]).abs().max()) for k in initial_gate)
    return history


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-fusion", type=int, default=60)
    parser.add_argument("--epochs-full", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--masks", type=int, default=12)
    parser.add_argument("--mask-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    info = b4_core.load_info()
    classes = info["known_classes"]
    cache = b4_core.load_cache()
    support_X, support_y = b4_core.support_arrays(cache, 10)
    if len(support_X) != 400:
        raise RuntimeError(f"Expected 400 support rows, got {len(support_X)}")
    if set(support_y) != set(classes):
        raise RuntimeError("Support/class mismatch")

    episodes = b4_core.build_episodes(
        support_X,
        support_y,
        classes,
        seed=args.seed,
        n_masks=args.masks,
        mask_size=args.mask_size,
    )
    print(f"episodes={len(episodes)} classes={len(classes)} support={len(support_X)}")
    print(f"mask_size={args.mask_size} known_per_episode={episodes[0].n_classes} oos_per_episode={len(episodes[0].p_oos)}")

    cal_X, cal_y = b4_core.split_vectors(cache, "calibration_known", classes)
    dev_X, dev_y = b4_core.split_vectors(cache, "dev_unknown", classes)
    val_vectors = np.concatenate([cal_X, dev_X])
    val_true = np.concatenate([cal_y, dev_y])
    val_p, val_n, val_d, _ = b4_core.compute_pn(support_X, support_y, classes, val_vectors)
    known_mask = val_true >= 0
    val_pack = {
        "p": torch.from_numpy(val_p),
        "n": torch.from_numpy(val_n),
        "d": torch.from_numpy(val_d),
        "y_known": torch.from_numpy(val_true[known_mask]),
        "known_mask": torch.from_numpy(known_mask),
        "y_all": torch.from_numpy((val_true >= 0).astype(np.float32)),
    }

    t0 = time.perf_counter()
    fusion = B4AR()
    hist_f = train_fusion(fusion, episodes, val_pack, args.epochs_fusion, args.lr)
    fusion_path = RESULTS / "gate4_b4_fusion.pt"
    b4_core.save_checkpoint(
        fusion_path,
        fusion,
        {
            "stage": "B4-Fusion",
            "classes": classes,
            "shot": 10,
            "model": "gte-modernbert-base",
            "seed": args.seed,
            "epochs": args.epochs_fusion,
            "lr": args.lr,
            "n_episodes": len(episodes),
        },
    )

    full = B4AR()
    full.load_state_dict(fusion.state_dict(), strict=True)
    hist_u = train_full(full, episodes, val_pack, args.epochs_full, args.lr)
    full_path = RESULTS / "gate4_b4_full.pt"
    b4_core.save_checkpoint(
        full_path,
        full,
        {
            "stage": "B4-Full",
            "classes": classes,
            "shot": 10,
            "model": "gte-modernbert-base",
            "seed": args.seed,
            "epochs": args.epochs_full,
            "lr": args.lr,
            "n_episodes": len(episodes),
            "alpha_oos": 0.5,
            "beta_weight_decay": 1e-4,
        },
    )
    train_seconds = time.perf_counter() - t0

    history = hist_f + hist_u
    pd.DataFrame(history).to_csv(RESULTS / "gate4_b4_training_history.csv", index=False)

    # Full-class validation scores for both checkpoints.
    with torch.no_grad():
        fusion.eval()
        full.eval()
        s_f, g_f = fusion.fused(torch.from_numpy(val_p), torch.from_numpy(val_n), torch.from_numpy(val_d))
        s_u, g_u = full.fused(torch.from_numpy(val_p), torch.from_numpy(val_n), torch.from_numpy(val_d))
        p_f = torch.softmax(s_f, dim=1).max(dim=1).values  # unused, keeps explicit score provenance
        p_u = torch.sigmoid(full.reject_logit(full.reject_features(torch.from_numpy(val_p), torch.from_numpy(val_n), s_u, g_u, torch.from_numpy(val_d))))
        s_f_np = s_f.numpy()
        s_u_np = s_u.numpy()
        g_f_np = g_f.numpy()
        g_u_np = g_u.numpy()
        p_u_np = p_u.numpy()

    # Thresholds: B4-Fusion on fused S1; B4-Full on reject-head known score.
    score_f = s_f_np.max(axis=1)
    class_f = s_f_np.argmax(axis=1)
    tau_f, grid_f, info_f = b4_core.select_threshold_custom(score_f, class_f, val_true, classes)
    metrics_f, pred_f, accept_f = b4_core.evaluate_custom(score_f, class_f, val_true, tau_f, classes)

    score_u = p_u_np
    class_u = s_u_np.argmax(axis=1)
    tau_u, grid_u, info_u = b4_core.select_threshold_custom(score_u, class_u, val_true, classes)
    metrics_u, pred_u, accept_u = b4_core.evaluate_custom(score_u, class_u, val_true, tau_u, classes)

    # Existing frozen B0/B3-H validation rows; no recomputation and no test read.
    g3v = pd.read_csv(RESULTS / "gate3_validation_metrics.csv")
    rows = []
    for method in ("B0", "B3-H"):
        row = g3v[(g3v.dataset == "banking77") & (g3v.model == "gte") & (g3v.shot == 10) & (g3v.method == method)]
        if len(row) != 1:
            raise RuntimeError(f"Missing Gate3 validation row for {method}")
        r = row.iloc[0]
        rows.append(
            {
                "method": method,
                "known_acceptance": r["val_known_acceptance"],
                "known_correct_routing": r["val_known_correct_routing"],
                "known_macro_f1": r["val_known_macro_f1"],
                "top1_closed_set_accuracy": r["val_top1_closed_set_accuracy"],
                "oos_recall": r["val_oos_recall"],
                "oos_false_acceptance": r["val_oos_false_acceptance"],
                "unknown_auroc": r["val_unknown_auroc"],
                "open_set_macro_f1": r["val_open_set_macro_f1"],
                "open_set_accuracy": r["val_open_set_accuracy"],
                "tau": r["tau"],
                "note": "Gate3 frozen validation baseline; not retrained.",
            }
        )
    rows.append(
        {
            "method": "B4-Fusion",
            "known_acceptance": metrics_f["known_acceptance"],
            "known_correct_routing": metrics_f["known_correct_routing"],
            "known_macro_f1": metrics_f["known_macro_f1"],
            "top1_closed_set_accuracy": metrics_f["top1_closed_set_accuracy"],
            "oos_recall": metrics_f["oos_recall"],
            "oos_false_acceptance": metrics_f["oos_false_acceptance"],
            "unknown_auroc": metrics_f["unknown_auroc"],
            "open_set_macro_f1": metrics_f["open_set_macro_f1"],
            "open_set_accuracy": metrics_f["open_set_accuracy"],
            "tau": tau_f,
            "note": "B4 gate only; threshold on fused S1.",
        }
    )
    rows.append(
        {
            "method": "B4-Full",
            "known_acceptance": metrics_u["known_acceptance"],
            "known_correct_routing": metrics_u["known_correct_routing"],
            "known_macro_f1": metrics_u["known_macro_f1"],
            "top1_closed_set_accuracy": metrics_u["top1_closed_set_accuracy"],
            "oos_recall": metrics_u["oos_recall"],
            "oos_false_acceptance": metrics_u["oos_false_acceptance"],
            "unknown_auroc": metrics_u["unknown_auroc"],
            "open_set_macro_f1": metrics_u["open_set_macro_f1"],
            "open_set_accuracy": metrics_u["open_set_accuracy"],
            "tau": tau_u,
            "note": "B4 gate + reject head; threshold on reject-head known score.",
        }
    )
    pd.DataFrame(rows).to_csv(RESULTS / "gate4_b4_validation_metrics.csv", index=False)

    thresholds = {
        "model": "gte-modernbert-base",
        "dataset": "banking77",
        "shot": 10,
        "B4-Fusion": {"tau": tau_f, "selection": info_f},
        "B4-Full": {"tau": tau_u, "selection": info_u},
        "B0": {"tau": float(g3v[(g3v.dataset == "banking77") & (g3v.model == "gte") & (g3v.shot == 10) & (g3v.method == "B0")]["tau"].iloc[0])},
        "B3-H": {"tau": float(g3v[(g3v.dataset == "banking77") & (g3v.model == "gte") & (g3v.shot == 10) & (g3v.method == "B3-H")]["tau"].iloc[0])},
        "test_evaluated": False,
        "test_policy": "BANKING77 test was already seen in Gate3; Gate4 test is exploratory, not a new blind test.",
    }
    (RESULTS / "gate4_b4_thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Gate distribution / gradient / parameter update evidence.
    gate_stats = {
        "train_seconds": train_seconds,
        "epochs_fusion": args.epochs_fusion,
        "epochs_full": args.epochs_full,
        "n_episodes": len(episodes),
        "known_queries_per_episode": int(len(episodes[0].y_known)),
        "pseudo_oos_per_episode": int(len(episodes[0].p_oos)),
        "fusion_gate_mean": float(g_f_np.mean()),
        "fusion_gate_std": float(g_f_np.std()),
        "full_gate_mean": float(g_u_np.mean()),
        "full_gate_std": float(g_u_np.std()),
        "fusion_gate_max_abs_delta": hist_f[-1].get("gate_max_abs_delta", 0.0),
        "full_gate_max_abs_delta": hist_u[-1].get("gate_max_abs_delta", 0.0),
        "first_fusion_gate_grad_norm": hist_f[0]["gate_grad_norm"],
        "last_fusion_gate_grad_norm": hist_f[-1]["gate_grad_norm"],
        "first_full_gate_grad_norm": hist_u[0]["gate_grad_norm"],
        "last_full_gate_grad_norm": hist_u[-1]["gate_grad_norm"],
    }
    (RESULTS / "gate4_b4_gate_stats.json").write_text(json.dumps(gate_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Gate by known class (top-1 gate) and by margin quartile.
    known_idx = np.where(known_mask)[0]
    top1 = s_u_np.argmax(axis=1)
    gate_top = g_u_np[np.arange(len(g_u_np)), top1]
    class_rows = []
    for ci, name in enumerate(classes):
        idx = known_idx[val_true[known_idx] == ci]
        class_rows.append(
            {
                "intent": name,
                "n_known_val": int(len(idx)),
                "gate_mean_top1": float(gate_top[idx].mean()) if len(idx) else None,
                "gate_std_top1": float(gate_top[idx].std()) if len(idx) else None,
            }
        )
    pd.DataFrame(class_rows).to_csv(RESULTS / "gate4_b4_gate_by_class.csv", index=False)
    margins = s_u_np.max(axis=1) - np.partition(s_u_np, -2, axis=1)[:, -2]
    q = np.quantile(margins[known_mask], [0.25, 0.5, 0.75])
    difficulty = np.digitize(margins, q)
    diff_rows = []
    for level in range(4):
        idx = np.where(known_mask & (difficulty == level))[0]
        diff_rows.append(
            {
                "margin_quartile": level + 1,
                "n": int(len(idx)),
                "gate_mean_top1": float(gate_top[idx].mean()) if len(idx) else None,
                "margin_mean": float(margins[idx].mean()) if len(idx) else None,
            }
        )
    pd.DataFrame(diff_rows).to_csv(RESULTS / "gate4_b4_gate_by_difficulty.csv", index=False)

    # Traceable validation cases from B4-Full.
    def find_ids(role: str):
        mask = cache["roles"] == role
        return cache["sample_ids"][mask], cache["labels"][mask]

    cal_ids, cal_labels = find_ids("calibration_known")
    dev_ids, _ = find_ids("dev_unknown")
    all_ids = np.concatenate([cal_ids, dev_ids])
    all_true = np.concatenate([cal_y, dev_y])
    all_pred = pred_u
    case_rows = []
    cats = [
        ("known_correct", (all_true >= 0) & (all_pred != "OOS") & (all_pred == np.asarray(classes)[all_true])),
        ("known_rejected", (all_true >= 0) & (all_pred == "OOS")),
        ("known_misclassified", (all_true >= 0) & (all_pred != "OOS") & (all_pred != np.asarray(classes)[all_true])),
        ("oos_correct", (all_true < 0) & (all_pred == "OOS")),
        ("oos_false_accept", (all_true < 0) & (all_pred != "OOS")),
    ]
    for category, mask in cats:
        idxs = np.where(mask)[0]
        if not len(idxs):
            continue
        i = int(idxs[0])
        top = np.argsort(-s_u_np[i])[:3]
        case_rows.append(
            {
                "category": category,
                "sample_id": all_ids[i],
                "true_label": classes[all_true[i]] if all_true[i] >= 0 else "OOS",
                "pred_label": all_pred[i],
                "score": float(score_u[i]),
                "tau": tau_u,
                "top3": " | ".join(f"{classes[j]}:{s_u_np[i, j]:.4f}" for j in top),
                "gate_top1": float(g_u_np[i, top[0]]),
                "margin": float(margins[i]),
            }
        )
    pd.DataFrame(case_rows).to_csv(RESULTS / "gate4_b4_validation_cases.csv", index=False)

    # Training config and checkpoint metadata.
    config = {
        "seed": args.seed,
        "encoder": "gte-modernbert-base",
        "dataset": "banking77",
        "shot": 10,
        "support_rows": len(support_X),
        "classes": len(classes),
        "episodes": len(episodes),
        "mask_size": args.mask_size,
        "n_masks_per_fold": args.masks,
        "folds": 2,
        "epochs_fusion": args.epochs_fusion,
        "epochs_full": args.epochs_full,
        "lr": args.lr,
        "gate_hidden": 32,
        "reject_hidden": 16,
        "alpha_oos": 0.5,
        "weight_decay": 1e-4,
        "train_seconds": train_seconds,
        "checkpoints": {
            "fusion": str(fusion_path.relative_to(PROJECT_ROOT)),
            "full": str(full_path.relative_to(PROJECT_ROOT)),
        },
        "test_evaluated": False,
    }
    (RESULTS / "gate4_b4_training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": rows, "gate": gate_stats}, ensure_ascii=False, indent=2))
    print("GATE4_TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
