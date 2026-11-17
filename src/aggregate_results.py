"""Aggregate the per-model Gate-1 result files into compact deliverables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
MANIFEST = PROJECT_ROOT / "data" / "manifest"

METRIC_COLUMNS = [
    "model",
    "method",
    "seed",
    "shot",
    "threshold",
    "known_acceptance",
    "known_correct_routing",
    "known_macro_f1",
    "top1_closed_set_accuracy",
    "oos_recall",
    "oos_false_acceptance",
    "oos_auroc",
    "open_set_macro_f1",
    "open_set_accuracy",
    "known_rejected",
    "oos_rejected",
    "oos_accepted",
    "n_known",
    "n_oos",
    "note",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    all_metrics: list[dict[str, Any]] = []
    predictions: dict[str, list[dict[str, str]]] = {"B0": [], "B2": []}

    for model in ("bge", "qwen"):
        for method in ("B0", "B2"):
            path = RESULTS / f"metrics_{model}_{method.lower()}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Run {model.upper()} {method} first: {path}")
            row = read_csv(path)[0]
            payload = {
                "model": row["model"],
                "method": method,
                "seed": int(row["seed"]),
                "shot": int(row["shot"]),
                "threshold": row["threshold"],
                "known_acceptance": float(row["known_acceptance"]),
                "known_correct_routing": float(row["known_correct_routing"]),
                "known_macro_f1": float(row["known_macro_f1"]),
                "top1_closed_set_accuracy": float(row["top1_closed_set_accuracy"]),
                "oos_recall": float(row["oos_recall"]),
                "oos_false_acceptance": float(row["oos_false_acceptance"]),
                "oos_auroc": float(row["oos_auroc"]),
                "open_set_macro_f1": float(row["open_set_macro_f1"]),
                "open_set_accuracy": float(row["open_set_accuracy"]),
                "known_rejected": int(row["known_rejected"]),
                "oos_rejected": int(row["oos_rejected"]),
                "oos_accepted": int(row["oos_accepted"]),
                "n_known": int(row["n_known"]),
                "n_oos": int(row["n_oos"]),
                "note": "",
            }
            all_metrics.append(payload)
            predictions[method].extend(
                read_csv(RESULTS / f"predictions_{model}_{method.lower()}.csv")
            )

            # λ=0, γ=0 is exactly B0 acceptance; copy the already-evaluated B0
            # test metrics rather than touching the test set a second time.
            if method == "B0":
                control = {
                    **payload,
                    "method": "B2_control",
                    "note": "lambda=0, gamma=0; equivalent to B0 acceptance, test metrics reused.",
                }
                all_metrics.append(control)

    all_metrics.sort(key=lambda row: (row["model"], row["method"]))
    write_csv(RESULTS / "gate1_metrics.csv", all_metrics, METRIC_COLUMNS)
    write_csv(
        RESULTS / "b0_metrics.csv",
        [row for row in all_metrics if row["method"] == "B0"],
        METRIC_COLUMNS,
    )
    write_csv(
        RESULTS / "b2_metrics.csv",
        [row for row in all_metrics if row["method"] in ("B2", "B2_control")],
        METRIC_COLUMNS,
    )
    write_csv(
        RESULTS / "b0_baseline.csv",
        [row for row in all_metrics if row["method"] == "B0"],
        METRIC_COLUMNS,
    )
    write_csv(
        RESULTS / "b2_dynamic_rejection.csv",
        [row for row in all_metrics if row["method"] == "B2"],
        METRIC_COLUMNS,
    )

    prediction_fields = [
        "model",
        "method",
        "seed",
        "shot",
        "sample_id",
        "role",
        "true_label",
        "pred_label",
        "accepted",
        "top1_class",
        "s1",
        "s2",
        "margin",
        "tau_used",
        "decision_score",
        "correct",
    ]
    for method, rows in predictions.items():
        write_csv(RESULTS / f"predictions_{method.lower()}.csv", rows, prediction_fields)

    # Error cases from Qwen B2, joined to the frozen eval text.
    eval_rows = {row["sample_id"]: row for row in read_csv(MANIFEST / "eval_manifest.csv")}
    qwen_b2 = [row for row in predictions["B2"] if row["model"] == "qwen"]
    test_rows = [row for row in qwen_b2 if row["role"].startswith("final_")]
    categories: list[tuple[str, dict[str, str]]] = []
    for row in test_rows:
        true_known = row["true_label"] != "OOS"
        accepted = row["accepted"] == "1"
        if true_known and accepted and row["pred_label"] == row["true_label"]:
            category = "correct_known"
        elif true_known and not accepted:
            category = "known_rejected"
        elif true_known and accepted and row["pred_label"] != row["true_label"]:
            category = "known_misclassified"
        elif not true_known and not accepted:
            category = "correct_oos_rejection"
        else:
            category = "oos_false_accept"
        categories.append((category, row))
    picked: list[dict[str, Any]] = []
    for wanted in (
        "correct_known",
        "known_misclassified",
        "known_rejected",
        "correct_oos_rejection",
        "oos_false_accept",
    ):
        for category, row in categories:
            if category == wanted:
                picked.append(
                    {
                        "category": category,
                        "sample_id": row["sample_id"],
                        "true_label": row["true_label"],
                        "pred_label": row["pred_label"],
                        "s1": row["s1"],
                        "margin": row["margin"],
                        "tau_used": row["tau_used"],
                        "text": eval_rows[row["sample_id"]]["text"],
                    }
                )
                break
    write_csv(
        RESULTS / "error_cases_qwen_b2.csv",
        picked,
        ["category", "sample_id", "true_label", "pred_label", "s1", "margin", "tau_used", "text"],
    )

    print(f"Wrote {RESULTS / 'gate1_metrics.csv'}")
    for row in all_metrics:
        print(
            f"{row['model']:4s} {row['method']:10s} "
            f"known_acc={row['known_acceptance']:.4f} "
            f"known_route={row['known_correct_routing']:.4f} "
            f"oos_recall={row['oos_recall']:.4f} "
            f"open_f1={row['open_set_macro_f1']:.4f}"
        )
    print("AGGREGATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
