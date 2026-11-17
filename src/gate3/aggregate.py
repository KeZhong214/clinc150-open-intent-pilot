"""Aggregate Gate-3 matrix into compact comparison tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATASETS = ["clinc150", "banking77", "massive"]
METHODS = ["B0", "B3-P", "B3-H"]


def main() -> int:
    metrics = pd.read_csv(RESULTS / "gate3_test_metrics.csv")
    validation = pd.read_csv(RESULTS / "gate3_validation_metrics.csv")

    metrics.groupby(["dataset", "method"])[
        ["known_acceptance", "known_correct_routing", "oos_recall", "unknown_auroc", "open_set_macro_f1", "open_set_accuracy"]
    ].mean().reset_index().to_csv(RESULTS / "gate3_summary_by_dataset_method.csv", index=False)
    metrics.groupby(["model", "method"])[
        ["known_acceptance", "known_correct_routing", "oos_recall", "unknown_auroc", "open_set_macro_f1", "open_set_accuracy"]
    ].mean().reset_index().to_csv(RESULTS / "gate3_summary_by_model.csv", index=False)

    rows = []
    for dataset in DATASETS:
        for model in sorted(metrics["model"].unique()):
            for shot in [5, 10]:
                b0 = metrics[(metrics.dataset == dataset) & (metrics.model == model) & (metrics.shot == shot) & (metrics.method == "B0")]
                b3 = metrics[(metrics.dataset == dataset) & (metrics.model == model) & (metrics.shot == shot) & (metrics.method == "B3-H")]
                if len(b0) != 1 or len(b3) != 1:
                    continue
                b0 = b0.iloc[0]
                b3 = b3.iloc[0]
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "shot": shot,
                        "B0_known_routing": b0["known_correct_routing"],
                        "B3H_known_routing": b3["known_correct_routing"],
                        "B0_oos_recall": b0["oos_recall"],
                        "B3H_oos_recall": b3["oos_recall"],
                        "B0_open_f1": b0["open_set_macro_f1"],
                        "B3H_open_f1": b3["open_set_macro_f1"],
                        "delta_known_routing_pp": (b3["known_correct_routing"] - b0["known_correct_routing"]) * 100,
                        "delta_oos_recall_pp": (b3["oos_recall"] - b0["oos_recall"]) * 100,
                        "delta_open_f1_pp": (b3["open_set_macro_f1"] - b0["open_set_macro_f1"]) * 100,
                    }
                )
    delta = pd.DataFrame(rows)
    delta.to_csv(RESULTS / "gate3_b3h_vs_b0.csv", index=False)

    best = metrics.sort_values("open_set_macro_f1", ascending=False).groupby("dataset").head(1)
    best.to_csv(RESULTS / "gate3_best_configs.csv", index=False)

    # Validation/test comparison for the F1-oriented threshold rule.
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            v = validation[(validation.dataset == dataset) & (validation.method == method)]
            t = metrics[(metrics.dataset == dataset) & (metrics.method == method)]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "val_known_acceptance": v["val_known_acceptance"].mean(),
                    "test_known_acceptance": t["known_acceptance"].mean(),
                    "val_oos_recall": v["val_oos_recall"].mean(),
                    "test_oos_recall": t["oos_recall"].mean(),
                    "val_open_f1": v["val_open_set_macro_f1"].mean(),
                    "test_open_f1": t["open_set_macro_f1"].mean(),
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS / "gate3_validation_vs_test.csv", index=False)

    costs = []
    for model in ("bge", "qwen", "gte", "e5"):
        data = json.loads((RESULTS / f"gate3_encoding_{model}.json").read_text(encoding="utf-8"))
        datasets = data["datasets"]
        total = sum(v["encode"]["elapsed_seconds"] for v in datasets.values())
        texts = sum(v["n_texts"] for v in datasets.values())
        dims = {v["embedding_dim"] for v in datasets.values()}
        vram = max((v["encode"].get("cuda_peak_allocated_mib") or 0) for v in datasets.values())
        costs.append(
            {
                "model": model,
                "encoding_seconds": total,
                "texts": texts,
                "texts_per_second": texts / total,
                "embedding_dim": sorted(dims)[0],
                "peak_vram_mib": vram if vram else None,
            }
        )
    pd.DataFrame(costs).to_csv(RESULTS / "gate3_cost.csv", index=False)

    # Training comparison (SetFit + adapter + BGE-B0).
    setfit = pd.read_csv(RESULTS / "gate3_setfit_metrics.csv").rename(columns={"model": "method"})
    adapter = pd.read_csv(RESULTS / "gate3_adapter_metrics.csv")
    adapter = adapter.rename(columns={"method": "method"})
    training = pd.concat(
        [
            setfit[["method", "known_acceptance", "known_correct_routing", "oos_recall", "open_set_macro_f1", "note"]],
            adapter[adapter["method"].str.contains("Adapter")][["method", "known_acceptance", "known_correct_routing", "oos_recall", "open_set_macro_f1", "note"]],
        ]
    )
    training.to_csv(RESULTS / "gate3_training_summary.csv", index=False)

    print(delta.groupby("dataset")[["delta_known_routing_pp", "delta_oos_recall_pp", "delta_open_f1_pp"]].mean().round(2).to_string())
    print(best[["dataset", "model", "shot", "method", "open_set_macro_f1", "oos_recall"]].to_string(index=False))
    print("GATE3_AGGREGATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
