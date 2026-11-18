"""One-shot exploratory Gate-4 B4 test on the already-seen BANKING77 test.

The test is intentionally marked exploratory: Gate3 exposed this test before
Gate4.  B4 was trained only on support episodes, so this script still checks
that the trained checkpoint can infer on real held-out-style BANKING text, but
it does not claim a new blind evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "manifest"

import sys

sys.path.insert(0, str(ROOT))
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: Path) -> tuple[B4AR, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta", {})


def main() -> int:
    freeze_path = RESULTS / "gate4_b4_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("frozen"):
        raise RuntimeError("Gate-4 is not frozen")
    for name, meta in freeze["files"].items():
        if sha(RESULTS / name) != meta["sha256"]:
            raise RuntimeError(f"Frozen file changed: {name}")
    if freeze.get("test_evaluated"):
        raise RuntimeError("Gate-4 B4 test was already evaluated once; refusing a second read")
    if (RESULTS / "gate4_b4_test_metrics.csv").exists():
        raise RuntimeError("Gate-4 test metrics already exist")

    info = b4_core.load_info()
    classes = info["known_classes"]
    cache = b4_core.load_cache()
    support_X, support_y = b4_core.support_arrays(cache, 10)

    known_X, known_y = b4_core.split_vectors(cache, "known_test", classes)
    oos_X, oos_y = b4_core.split_vectors(cache, "final_unknown", classes)
    test_X = np.concatenate([known_X, oos_X])
    test_true = np.concatenate([known_y, oos_y])
    test_ids = np.concatenate(
        [
            cache["sample_ids"][cache["roles"] == "known_test"],
            cache["sample_ids"][cache["roles"] == "final_unknown"],
        ]
    )
    p, n, d, _ = b4_core.compute_pn(support_X, support_y, classes, test_X)
    pt = torch.from_numpy(p)
    nt = torch.from_numpy(n)
    dt = torch.from_numpy(d)

    thresholds = json.loads((RESULTS / "gate4_b4_thresholds.json").read_text(encoding="utf-8"))
    tau_f = float(thresholds["B4-Fusion"]["tau"])
    tau_u = float(thresholds["B4-Full"]["tau"])
    fusion, _ = load_model(RESULTS / "gate4_b4_fusion.pt")
    full, _ = load_model(RESULTS / "gate4_b4_full.pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        s_f, g_f = fusion.fused(pt, nt, dt)
        s_u, g_u = full.fused(pt, nt, dt)
        p_u = torch.sigmoid(full.reject_logit(full.reject_features(pt, nt, s_u, g_u, dt)))
    s_f = s_f.numpy()
    s_u = s_u.numpy()
    g_f = g_f.numpy()
    g_u = g_u.numpy()
    p_u = p_u.numpy()
    score_f = s_f.max(axis=1)
    class_f = s_f.argmax(axis=1)
    score_u = p_u
    class_u = s_u.argmax(axis=1)
    metrics_f, pred_f, accept_f = b4_core.evaluate_custom(score_f, class_f, test_true, tau_f, classes)
    metrics_u, pred_u, accept_u = b4_core.evaluate_custom(score_u, class_u, test_true, tau_u, classes)

    g3 = pd.read_csv(RESULTS / "gate3_test_metrics.csv")
    rows = []
    for method in ("B0", "B3-H"):
        row = g3[(g3.dataset == "banking77") & (g3.model == "gte") & (g3.shot == 10) & (g3.method == method)]
        if len(row) != 1:
            raise RuntimeError(f"Missing Gate3 test row for {method}")
        r = row.iloc[0]
        rows.append(
            {
                "method": method,
                "known_acceptance": r["known_acceptance"],
                "known_correct_routing": r["known_correct_routing"],
                "known_macro_f1": r["known_macro_f1"],
                "top1_closed_set_accuracy": r["top1_closed_set_accuracy"],
                "oos_recall": r["oos_recall"],
                "oos_false_acceptance": r["oos_false_acceptance"],
                "unknown_auroc": r["unknown_auroc"],
                "open_set_macro_f1": r["open_set_macro_f1"],
                "open_set_accuracy": r["open_set_accuracy"],
                "tau": r["tau"],
                "note": "Gate3 historical/exploratory test; same test used for Gate4.",
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
            "note": "Trained gate; threshold on fused S1.",
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
            "note": "Trained gate + reject head; threshold on reject score.",
        }
    )
    pd.DataFrame(rows).to_csv(RESULTS / "gate4_b4_test_metrics.csv", index=False)

    # Full B4-Full predictions with Top-3/gate values.
    top3 = np.argsort(-s_u, axis=1)[:, :3]
    pred_rows = []
    for i, sid in enumerate(test_ids):
        pred_rows.append(
            {
                "sample_id": sid,
                "role": "known_test" if test_true[i] >= 0 else "final_unknown",
                "true_label": classes[test_true[i]] if test_true[i] >= 0 else "OOS",
                "pred_label": pred_u[i],
                "accepted": int(accept_u[i]),
                "score": float(score_u[i]),
                "tau": tau_u,
                "top1": classes[top3[i, 0]],
                "top2": classes[top3[i, 1]],
                "top3": classes[top3[i, 2]],
                "s1": float(s_u[i, top3[i, 0]]),
                "s2": float(s_u[i, top3[i, 1]]),
                "gate_top1": float(g_u[i, top3[i, 0]]),
            }
        )
    pd.DataFrame(pred_rows).to_csv(RESULTS / "gate4_b4_test_predictions.csv", index=False)

    # Cases.
    text_by_id = {}
    for row in csv.DictReader((MANIFEST / "gate3_banking77_manifest.csv").open("r", encoding="utf-8")):
        if row["role"] in ("known_test", "final_unknown"):
            text_by_id[row["sample_id"]] = row["text"]
    case_rows = []
    categories = [
        ("known_correct", (test_true >= 0) & (pred_u != "OOS") & (pred_u == np.asarray(classes)[test_true])),
        ("known_rejected", (test_true >= 0) & (pred_u == "OOS")),
        ("known_misclassified", (test_true >= 0) & (pred_u != "OOS") & (pred_u != np.asarray(classes)[test_true])),
        ("oos_correct", (test_true < 0) & (pred_u == "OOS")),
        ("oos_false_accept", (test_true < 0) & (pred_u != "OOS")),
    ]
    for category, mask in categories:
        idxs = np.where(mask)[0]
        if not len(idxs):
            continue
        i = int(idxs[0])
        case_rows.append(
            {
                "category": category,
                "sample_id": test_ids[i],
                "text": text_by_id.get(test_ids[i], ""),
                "true_label": classes[test_true[i]] if test_true[i] >= 0 else "OOS",
                "pred_label": pred_u[i],
                "accepted": int(accept_u[i]),
                "reject_score": float(score_u[i]),
                "tau": tau_u,
                "top3": " | ".join(f"{classes[j]}:{s_u[i, j]:.4f}" for j in top3[i]),
                "gate_top1": float(g_u[i, top3[i, 0]]),
            }
        )
    pd.DataFrame(case_rows).to_csv(RESULTS / "gate4_b4_test_cases.csv", index=False)

    # Gate output by class / difficulty (known test rows).
    known_mask = test_true >= 0
    gate_top = g_u[np.arange(len(g_u)), class_u]
    class_rows = []
    for ci, name in enumerate(classes):
        idx = np.where(known_mask & (test_true == ci))[0]
        class_rows.append(
            {
                "intent": name,
                "n_test": int(len(idx)),
                "gate_mean_top1": float(gate_top[idx].mean()) if len(idx) else None,
                "gate_std_top1": float(gate_top[idx].std()) if len(idx) else None,
            }
        )
    pd.DataFrame(class_rows).to_csv(RESULTS / "gate4_b4_test_gate_by_class.csv", index=False)
    margins = s_u.max(axis=1) - np.partition(s_u, -2, axis=1)[:, -2]
    q = np.quantile(margins[known_mask], [0.25, 0.5, 0.75])
    levels = np.digitize(margins, q)
    difficulty_rows = []
    for level in range(4):
        idx = np.where(known_mask & (levels == level))[0]
        difficulty_rows.append(
            {
                "margin_quartile": level + 1,
                "n": int(len(idx)),
                "gate_mean_top1": float(gate_top[idx].mean()) if len(idx) else None,
                "margin_mean": float(margins[idx].mean()) if len(idx) else None,
            }
        )
    pd.DataFrame(difficulty_rows).to_csv(RESULTS / "gate4_b4_test_gate_by_difficulty.csv", index=False)

    test_seconds = time.perf_counter() - t0
    gate_stats = {
        "test_seconds": test_seconds,
        "n_known": int(known_mask.sum()),
        "n_unknown": int((~known_mask).sum()),
        "fusion_gate_mean": float(g_f.mean()),
        "fusion_gate_std": float(g_f.std()),
        "full_gate_mean": float(g_u.mean()),
        "full_gate_std": float(g_u.std()),
        "full_gate_top1_mean": float(gate_top[known_mask].mean()),
        "full_gate_top1_std": float(gate_top[known_mask].std()),
        "test_is_exploratory_not_blind": True,
    }
    (RESULTS / "gate4_b4_test_gate_stats.json").write_text(json.dumps(gate_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "gate4_b4_test_run.json").write_text(
        json.dumps(
            {
                "test_seconds": test_seconds,
                "counts": {"known": int(known_mask.sum()), "unknown": int((~known_mask).sum())},
                "note": "BANKING77 test was already seen in Gate3; this is exploratory, not a new blind test.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    freeze["test_evaluated"] = True
    freeze["test_metrics"] = "results/gate4_b4_test_metrics.csv"
    (RESULTS / "gate4_b4_freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(json.dumps(gate_stats, ensure_ascii=False, indent=2))
    print("GATE4_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
