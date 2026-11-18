"""G5-B: Frozen/LoRA × B3-H/RejectOnly/B4-Full matrix.

Phase validation writes the comparison table and freeze hashes.  Phase test is
one-shot and explicitly labelled historical_exploratory because BANKING77 test
was already exposed in Gate 3/4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5 import encoder_data as ed  # noqa: E402
from src.gate5 import lora_common as lc  # noqa: E402
from src.gate3.methods import evaluate, knn_scores, prototypes_multi, prototypes_single, scores_from_prototypes, select_threshold  # noqa: E402
from src.gate4 import b4_core  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_from_metrics(encoder: str, method: str, tau: float, m: dict, note: str = "") -> dict:
    return {
        "encoder": encoder,
        "method": method,
        "tau": tau,
        "known_acceptance": m.get("known_acceptance", m.get("val_known_acceptance")),
        "known_correct_routing": m.get("known_correct_routing", m.get("val_known_correct_routing")),
        "known_macro_f1": m.get("known_macro_f1", m.get("val_known_macro_f1")),
        "top1_closed_set_accuracy": m.get("top1_closed_set_accuracy", m.get("val_top1_closed_set_accuracy")),
        "oos_recall": m.get("oos_recall", m.get("val_oos_recall")),
        "oos_false_acceptance": m.get("oos_false_acceptance", m.get("val_oos_false_acceptance")),
        "unknown_auroc": m.get("unknown_auroc", m.get("val_unknown_auroc")),
        "open_set_macro_f1": m.get("open_set_macro_f1", m.get("val_open_set_macro_f1")),
        "open_set_accuracy": m.get("open_set_accuracy", m.get("val_open_set_accuracy")),
        "n_known": m.get("n_known", m.get("val_n_known")),
        "n_unknown": m.get("n_unknown", m.get("val_n_unknown")),
        "historical_exploratory": False,
        "note": note,
    }


def load_simple_scores(cache: dict, classes, support_X, support_y, query):
    p0 = prototypes_single(support_X, support_y, classes)
    b0 = scores_from_prototypes(query, [p[None, :] for p in p0])
    protos, _ = prototypes_multi(support_X, support_y, classes)
    p = scores_from_prototypes(query, protos)
    n = knn_scores(query, support_X, support_y, classes, top_k=2)
    return b0, 0.5 * p + 0.5 * n


def validation_rows() -> list[dict]:
    rows = []
    g3 = pd.read_csv(RESULTS / "gate3_validation_metrics.csv")
    for method in ("B0", "B3-H"):
        r = g3[(g3.dataset == "banking77") & (g3.model == "gte") & (g3.shot == 10) & (g3.method == method)].iloc[0]
        rows.append(row_from_metrics("frozen", method, float(r["tau"]), r.to_dict(), "Gate3 frozen validation"))
    g4 = pd.read_csv(RESULTS / "gate4_b4_validation_metrics.csv")
    for method in ("B4-Fusion", "B4-Full"):
        r = g4[g4.method == method].iloc[0]
        rows.append(row_from_metrics("frozen", method, float(r["tau"]), r.to_dict(), "Gate4 frozen validation" if method == "B4-Full" else "Gate4 fusion ablation"))
    for method in ("B0", "B3-H"):
        r = pd.read_csv(RESULTS / "gate5_lora_validation.csv")
        r = r[r.method == method].iloc[0]
        rows.append(row_from_metrics("lora", method, float(r["tau"]), r.to_dict(), "LoRA-GTE validation"))
    r = pd.read_csv(RESULTS / "gate5_reject_only_frozen_validation.csv").iloc[0]
    rows.append(row_from_metrics("frozen", "B4-RejectOnly", float(r["tau"]), r.to_dict(), "new RejectOnly"))
    r = pd.read_csv(RESULTS / "gate5_reject_only_lora_validation.csv").iloc[0]
    rows.append(row_from_metrics("lora", "B4-RejectOnly", float(r["tau"]), r.to_dict(), "new RejectOnly"))
    r = pd.read_csv(RESULTS / "gate5_b4_lora_validation.csv").iloc[0]
    rows.append(row_from_metrics("lora", "B4-Full", float(r["tau"]), r.to_dict(), "new LoRA B4-Full"))
    return rows


def freeze_files() -> list[Path]:
    return [
        RESULTS / "gate5_lora_training.json",
        RESULTS / "gate5_lora_adapter" / "adapter_model.safetensors",
        RESULTS / "gate5_lora_validation.csv",
        RESULTS / "gate5_reject_only_frozen.pt",
        RESULTS / "gate5_reject_only_lora.pt",
        RESULTS / "gate5_reject_only_frozen_validation.csv",
        RESULTS / "gate5_reject_only_lora_validation.csv",
        RESULTS / "gate5_reject_only_frozen_threshold.json",
        RESULTS / "gate5_reject_only_lora_threshold.json",
        RESULTS / "gate5_b4_full_lora.pt",
        RESULTS / "gate5_b4_lora_validation.csv",
        RESULTS / "gate5_b4_lora_threshold.json",
        RESULTS / "gate5_lora_adapter" / "adapter_config.json",
        RESULTS / "gate4_b4_full.pt",
        RESULTS / "gate4_b4_thresholds.json",
        RESULTS / "gate3_validation_metrics.csv",
        RESULTS / "gate3_test_metrics.csv",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    args = parser.parse_args(argv)
    if args.phase == "validation":
        rows = validation_rows()
        df = pd.DataFrame(rows)
        order = {"frozen": 0, "lora": 1}
        df["_order"] = df["encoder"].map(order)
        df = df.sort_values(["_order", "method"]).drop(columns="_order")
        df.to_csv(RESULTS / "gate5_method_matrix_val.csv", index=False)
        payload = {
            "frozen": True,
            "test_evaluated": False,
            "label": "historical_exploratory",
            "note": "BANKING77 test was already seen in Gate3/4. Gate5 test is a one-shot historical exploratory comparison.",
            "files": {str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in freeze_files()},
        }
        (RESULTS / "gate5_method_freeze.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(df[["encoder", "method", "known_acceptance", "known_correct_routing", "oos_recall", "open_set_macro_f1"]].to_string(index=False))
        print("GATE5_METHOD_VALIDATION_OK")
        return 0

    freeze = json.loads((RESULTS / "gate5_method_freeze.json").read_text(encoding="utf-8"))
    if not freeze.get("frozen"):
        raise RuntimeError("Gate5 method matrix is not frozen")
    if freeze.get("test_evaluated") or (RESULTS / "gate5_method_matrix_historical_test.csv").exists():
        raise RuntimeError("Gate5 historical test was already evaluated once")
    for name, meta in freeze["files"].items():
        if sha(ROOT / name) != meta["sha256"]:
            raise RuntimeError(f"Frozen file changed: {name}")

    classes = lc.classes()
    rows = []
    # Existing frozen rows.
    g3 = pd.read_csv(RESULTS / "gate3_test_metrics.csv")
    for method in ("B0", "B3-H"):
        r = g3[(g3.dataset == "banking77") & (g3.model == "gte") & (g3.shot == 10) & (g3.method == method)].iloc[0]
        rows.append(row_from_metrics("frozen", method, float(r["tau"]), r.to_dict(), "Gate3 historical test"))
    g4 = pd.read_csv(RESULTS / "gate4_b4_test_metrics.csv")
    for method in ("B4-Fusion", "B4-Full"):
        r = g4[g4.method == method].iloc[0]
        rows.append(row_from_metrics("frozen", method, float(r["tau"]), r.to_dict(), "Gate4 historical exploratory test"))

    # Frozen RejectOnly test.
    cache_f = ed.load_encoder("frozen")
    X_f, y_f = ed.support(cache_f)
    test_X, test_y = ed.role(cache_f, "known_test", classes)
    test_oos, test_oos_y = ed.role(cache_f, "final_unknown", classes)
    test_all = np.concatenate([test_X, test_oos])
    test_true = np.concatenate([test_y, test_oos_y])
    ro_f = json.loads((RESULTS / "gate5_reject_only_frozen_threshold.json").read_text(encoding="utf-8"))["tau"]
    model_ro_f = ed.load_b4(RESULTS / "gate5_reject_only_frozen.pt")
    sc = ed.b4_scores(model_ro_f, cache_f, test_all, classes, reject_only=True)
    m, _, _ = b4_core.evaluate_custom(sc["reject"], sc["class_idx"], test_true, float(ro_f), classes)
    rows.append(row_from_metrics("frozen", "B4-RejectOnly", float(ro_f), m, "new RejectOnly historical test"))

    # LoRA B0/B3-H test.
    cache_l = ed.load_encoder("lora")
    X_l, y_l = ed.support(cache_l)
    lora_test_X, lora_test_y = ed.role(cache_l, "known_test", classes)
    lora_oos_X, lora_oos_y = ed.role(cache_l, "final_unknown", classes)
    lora_all = np.concatenate([lora_test_X, lora_oos_X])
    lora_true = np.concatenate([lora_test_y, lora_oos_y])
    b0, b3 = load_simple_scores(cache_l, classes, X_l, y_l, lora_all)
    val_l = pd.read_csv(RESULTS / "gate5_lora_validation.csv").set_index("method")
    for method, sims in (("B0", b0), ("B3-H", b3)):
        tau = float(val_l.loc[method, "tau"])
        m, _, _ = evaluate(sims, lora_true, classes, tau)
        rows.append(row_from_metrics("lora", method, tau, m, "LoRA historical test"))

    # LoRA RejectOnly / Full.
    ro_l = json.loads((RESULTS / "gate5_reject_only_lora_threshold.json").read_text(encoding="utf-8"))["tau"]
    model_ro_l = ed.load_b4(RESULTS / "gate5_reject_only_lora.pt")
    sc = ed.b4_scores(model_ro_l, cache_l, lora_all, classes, reject_only=True)
    m, _, _ = b4_core.evaluate_custom(sc["reject"], sc["class_idx"], lora_true, float(ro_l), classes)
    rows.append(row_from_metrics("lora", "B4-RejectOnly", float(ro_l), m, "LoRA RejectOnly historical test"))

    bfl = json.loads((RESULTS / "gate5_b4_lora_threshold.json").read_text(encoding="utf-8"))["tau"]
    model_bf_l = ed.load_b4(RESULTS / "gate5_b4_full_lora.pt")
    sc = ed.b4_scores(model_bf_l, cache_l, lora_all, classes, reject_only=False)
    m, _, _ = b4_core.evaluate_custom(sc["reject"], sc["class_idx"], lora_true, float(bfl), classes)
    rows.append(row_from_metrics("lora", "B4-Full", float(bfl), m, "LoRA B4-Full historical test"))

    df = pd.DataFrame(rows)
    df["historical_exploratory"] = True
    df.to_csv(RESULTS / "gate5_method_matrix_historical_test.csv", index=False)
    freeze["test_evaluated"] = True
    freeze["test_file"] = "results/gate5_method_matrix_historical_test.csv"
    (RESULTS / "gate5_method_freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(df[["encoder", "method", "known_acceptance", "known_correct_routing", "oos_recall", "open_set_macro_f1"]].to_string(index=False))
    print("GATE5_METHOD_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
