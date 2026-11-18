"""G6-C: Qwen Frozen/LoRA × B0/B3-H/RejectOnly/B4-Full validation + one-shot test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate6.train_qwen_b4 import val_scores  # noqa: E402
from src.gate3.methods import evaluate, knn_scores, prototypes_multi, prototypes_single, scores_from_prototypes  # noqa: E402
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def b0_b3_sims(cache, method, support_X, support_y, classes, query):
    if method == "B0":
        protos = prototypes_single(support_X, support_y, classes)
        return scores_from_prototypes(query, [p[None, :] for p in protos])
    protos, _ = prototypes_multi(support_X, support_y, classes)
    p = scores_from_prototypes(query, protos)
    n = knn_scores(query, support_X, support_y, classes, top_k=2)
    return 0.5 * p + 0.5 * n


def load_b4(encoder: str, method: str) -> B4AR:
    ckpt = torch.load(RESULTS / f"gate6_qwen_{encoder}_{method}.pt", map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def test_b4(encoder: str, method: str, cache, classes, tau: float):
    X, y = qd.support(cache)
    test_X, test_y = qd.role(cache, "known_test", classes)
    unk_X, unk_y = qd.role(cache, "final_unknown", classes)
    query = np.concatenate([test_X, unk_X])
    y_test = np.concatenate([test_y, unk_y])
    model = load_b4(encoder, method)
    reject, class_idx, _, _ = val_scores(model, cache, query, classes, reject_only=(method == "reject"))
    metrics, pred, accept = b4_core.evaluate_custom(reject, class_idx, y_test, tau, classes)
    return metrics


def test_b0_b3(cache, method: str, classes, tau: float):
    X, y = qd.support(cache)
    kX, ky = qd.role(cache, "known_test", classes)
    uX, uy = qd.role(cache, "final_unknown", classes)
    query = np.concatenate([kX, uX])
    y_test = np.concatenate([ky, uy])
    sims = b0_b3_sims(cache, method, X, y, classes, query)
    metrics, _, _ = evaluate(sims, y_test, classes, tau)
    return metrics


def main() -> int:
    classes = qe.classes()
    fz = qd.load("frozen")
    pl = qd.load("plain")
    an = qd.load("anchor")
    freeze_path = RESULTS / "gate6_qwen_method_freeze.json"
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("test_evaluated"):
            print("Gate6 Qwen test already evaluated; refusing to rerun.")
            return 2
    else:
        freeze = {"stage": "gate6_qwen_methods", "test_evaluated": False, "label": "historical_exploratory", "files": {}}
        for p in [
            RESULTS / "gate6_qwen_plain_adapter" / "adapter_model.safetensors",
            RESULTS / "gate6_qwen_anchor_adapter" / "adapter_model.safetensors",
            *[RESULTS / f"gate6_qwen_{e}_{m}.pt" for e in ("frozen", "plain") for m in ("reject", "full")],
        ]:
            freeze["files"][str(p.relative_to(ROOT))] = sha(p)
        freeze["adapter_plain"] = json.loads((RESULTS / "gate6_qwen_plain_training.json").read_text(encoding="utf-8"))["adapter_sha256"]
        freeze["adapter_anchor"] = json.loads((RESULTS / "gate6_qwen_anchor_training.json").read_text(encoding="utf-8"))["adapter_sha256"]
        freeze["cases_sha256"] = sha(ROOT / "data" / "gate5" / "benchmark_cases.jsonl")
        freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Validation matrix.
    g3 = pd.read_csv(RESULTS / "gate3_validation_metrics.csv")
    g3 = g3[(g3.dataset == "banking77") & (g3.model == "qwen") & (g3.shot == 10)]
    rows = []
    for method in ("B0", "B3-H"):
        r = g3[g3.method == method].iloc[0]
        rows.append({"encoder": "qwen_frozen", "method": method, "open_set_macro_f1": float(r["val_open_set_macro_f1"]), "oos_recall": float(r["val_oos_recall"]), "known_correct_routing": float(r["val_known_correct_routing"]), "known_acceptance": float(r["val_known_acceptance"])})
    for method in ("B4-RejectOnly", "B4-Full"):
        r = pd.read_csv(RESULTS / f"gate6_qwen_frozen_{'reject' if method == 'B4-RejectOnly' else 'full'}_validation.csv").iloc[0]
        rows.append({"encoder": "qwen_frozen", "method": method, "open_set_macro_f1": float(r["val_open_set_macro_f1"]), "oos_recall": float(r["val_oos_recall"]), "known_correct_routing": float(r["val_known_correct_routing"]), "known_acceptance": float(r["val_known_acceptance"])})
    lora_val = pd.read_csv(RESULTS / "gate6_qwen_lora_validation.csv")
    for method in ("B0", "B3-H"):
        r = lora_val[(lora_val.run == "plain") & (lora_val.method == method)].iloc[0]
        rows.append({"encoder": "qwen_plain_lora", "method": method, "open_set_macro_f1": float(r["val_open_set_macro_f1"]), "oos_recall": float(r["val_oos_recall"]), "known_correct_routing": float(r["val_known_correct_routing"]), "known_acceptance": float(r["val_known_acceptance"])})
    for method in ("B4-RejectOnly", "B4-Full"):
        r = pd.read_csv(RESULTS / f"gate6_qwen_plain_{'reject' if method == 'B4-RejectOnly' else 'full'}_validation.csv").iloc[0]
        rows.append({"encoder": "qwen_plain_lora", "method": method, "open_set_macro_f1": float(r["val_open_set_macro_f1"]), "oos_recall": float(r["val_oos_recall"]), "known_correct_routing": float(r["val_known_correct_routing"]), "known_acceptance": float(r["val_known_acceptance"])})
    r = lora_val[(lora_val.run == "anchor") & (lora_val.method == "B3-H")].iloc[0]
    rows.append({"encoder": "qwen_anchor_lora", "method": "B3-H", "open_set_macro_f1": float(r["val_open_set_macro_f1"]), "oos_recall": float(r["val_oos_recall"]), "known_correct_routing": float(r["val_known_correct_routing"]), "known_acceptance": float(r["val_known_acceptance"])})
    val = pd.DataFrame(rows)
    val.to_csv(RESULTS / "gate6_qwen_method_matrix_val.csv", index=False)
    print(val.round(5).to_string(index=False))

    # One-shot historical test; all thresholds selected on validation.
    test_rows = []
    g3t = pd.read_csv(RESULTS / "gate3_test_metrics.csv")
    g3t = g3t[(g3t.dataset == "banking77") & (g3t.model == "qwen") & (g3t.shot == 10)]
    for method in ("B0", "B3-H"):
        r = g3t[g3t.method == method].iloc[0]
        test_rows.append({"encoder": "qwen_frozen", "method": method, **{k: float(r[k]) for k in ("known_acceptance", "known_correct_routing", "known_macro_f1", "top1_closed_set_accuracy", "oos_recall", "oos_false_acceptance", "unknown_auroc", "open_set_macro_f1", "open_set_accuracy")}})
    for method in ("B4-RejectOnly", "B4-Full"):
        key = "reject" if method == "B4-RejectOnly" else "full"
        thr = json.loads((RESULTS / f"gate6_qwen_frozen_{key}_threshold.json").read_text(encoding="utf-8"))["tau"]
        m = test_b4("frozen", key, fz, classes, thr)
        test_rows.append({"encoder": "qwen_frozen", "method": method, **m})
    for method in ("B0", "B3-H"):
        r = lora_val[(lora_val.run == "plain") & (lora_val.method == method)].iloc[0]
        m = test_b0_b3(pl, method, classes, float(r["tau"]))
        test_rows.append({"encoder": "qwen_plain_lora", "method": method, **m})
    for method in ("B4-RejectOnly", "B4-Full"):
        key = "reject" if method == "B4-RejectOnly" else "full"
        thr = json.loads((RESULTS / f"gate6_qwen_plain_{key}_threshold.json").read_text(encoding="utf-8"))["tau"]
        m = test_b4("plain", key, pl, classes, thr)
        test_rows.append({"encoder": "qwen_plain_lora", "method": method, **m})
    r = lora_val[(lora_val.run == "anchor") & (lora_val.method == "B3-H")].iloc[0]
    m = test_b0_b3(an, "B3-H", classes, float(r["tau"]))
    test_rows.append({"encoder": "qwen_anchor_lora", "method": "B3-H", **m})
    test = pd.DataFrame(test_rows)
    test["historical_exploratory"] = True
    test.to_csv(RESULTS / "gate6_qwen_method_matrix_historical_test.csv", index=False)
    print(test.round(5).to_string(index=False))
    freeze["test_evaluated"] = True
    freeze["val_sha256"] = sha(RESULTS / "gate6_qwen_method_matrix_val.csv")
    freeze["test_sha256"] = sha(RESULTS / "gate6_qwen_method_matrix_historical_test.csv")
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("GATE6_QWEN_MATRIX_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
