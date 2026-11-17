"""Build Gate-3 data profiles, error cases, confusion pairs and figures."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "manifest"
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

DATASETS = ("clinc150", "banking77", "massive")
MODELS = ("bge", "qwen", "gte", "e5")
METHODS = ("B0", "B3-P", "B3-H")
SHOTS = (5, 10)
COLORS = {"bge": "#7d9bb5", "qwen": "#c98b3c", "gte": "#2f6b57", "e5": "#8c5e8c"}
METHOD_COLORS = {"B0": "#7d9bb5", "B3-P": "#c98b3c", "B3-H": "#2f6b57"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def dataset_profiles() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        info = read_json(MANIFEST / f"gate3_{dataset}_dataset.json")
        rows.append(
            {
                "dataset": dataset,
                "known_classes": len(info["known_classes"]),
                "dev_unknown_classes": len(info["dev_unknown_classes"]),
                "final_unknown_classes": len(info["final_unknown_classes"]),
                "support_5": info["support_5_rows"],
                "support_10": info["support_10_rows"],
                "calibration_known": info["counts"]["calibration_known"],
                "known_test": info["counts"]["known_test"],
                "dev_unknown": info["counts"]["dev_unknown"],
                "final_unknown": info["counts"]["final_unknown"],
                "known_domains": len({row["domain"] for row in info["class_counts"]}),
                "historical": bool(info.get("historical", False)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "gate3_dataset_profile.csv", index=False)
    return frame


def dataset_figures() -> None:
    profile = pd.read_csv(RESULTS / "gate3_dataset_profile.csv")
    lengths = {}
    for dataset in DATASETS:
        rows = read_csv(MANIFEST / f"gate3_{dataset}_manifest.csv")
        lengths[dataset] = np.asarray([len(row["text"].split()) for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    for dataset in DATASETS:
        ax.hist(lengths[dataset], bins=40, density=True, alpha=0.45, label=dataset)
    ax.set_xlabel("Whitespace tokens")
    ax.set_ylabel("density")
    ax.set_title("Utterance length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    class_counts = []
    labels = []
    for dataset in DATASETS:
        info = read_json(MANIFEST / f"gate3_{dataset}_dataset.json")
        counts = [row["known_test"] for row in info["class_counts"]]
        class_counts.append(counts)
        labels.append(dataset)
    ax.boxplot(class_counts, tick_labels=labels, showmeans=True)
    ax.set_ylabel("known test rows per class")
    ax.set_title("Known-class test imbalance")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    x = np.arange(len(DATASETS))
    roles = ["support_5", "calibration_known", "known_test", "dev_unknown", "final_unknown"]
    width = 0.15
    for i, role in enumerate(roles):
        values = profile[role].tolist()
        ax.bar(x + (i - 2) * width, values, width=width, label=role)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("rows")
    ax.set_title("Role counts")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    values = profile["known_domains"].tolist()
    bars = ax.bar(DATASETS, values, color=["#7d9bb5", "#2f6b57", "#c98b3c"])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{bar.get_height():.0f}", ha="center")
    ax.set_ylabel("domains represented by known classes")
    ax.set_title("Known-domain coverage")
    ax.set_ylim(0, max(values) + 2)
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Gate 3 dataset profiles", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "gate3_dataset_overview.png", dpi=190)
    plt.close(fig)

    # Domain-level known-class distribution.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, dataset in zip(axes, DATASETS):
        info = read_json(MANIFEST / f"gate3_{dataset}_dataset.json")
        counts = Counter(row["domain"] for row in info["class_counts"])
        names = sorted(counts, key=lambda name: (-counts[name], name))
        ax.barh(names, [counts[name] for name in names], color="#4d6f8a")
        ax.set_xlabel("known classes")
        ax.set_title(dataset)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Known classes per domain", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate3_domain_distribution.png", dpi=190)
    plt.close(fig)


def heatmaps() -> None:
    metrics = pd.read_csv(RESULTS / "gate3_test_metrics.csv")
    b0 = metrics[(metrics["method"] == "B0") & (metrics["shot"] == 5)]
    for value, title, filename in (
        ("open_set_macro_f1", "B0 5-shot: open-set macro-F1", "gate3_heatmap_macro_f1.png"),
        ("oos_recall", "B0 5-shot: OOS recall", "gate3_heatmap_oos_recall.png"),
    ):
        pivot = b0.pivot(index="dataset", columns="model", values=value).loc[list(DATASETS), list(MODELS)]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        image = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=25)
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels(DATASETS)
        for i in range(len(DATASETS)):
            for j in range(len(MODELS)):
                ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=190)
        plt.close(fig)


def shot_curves() -> None:
    metrics = pd.read_csv(RESULTS / "gate3_test_metrics.csv")
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    for row, dataset in enumerate(DATASETS):
        for col, value in enumerate(("open_set_macro_f1", "oos_recall")):
            ax = axes[row, col]
            for model in MODELS:
                subset = metrics[(metrics["dataset"] == dataset) & (metrics["model"] == model) & (metrics["method"] == "B0")]
                subset = subset.sort_values("shot")
                ax.plot(subset["shot"], subset[value], marker="o", color=COLORS[model], label=model)
            ax.set_xticks([5, 10])
            ax.set_title(f"{dataset} — {'open-set F1' if col == 0 else 'OOS recall'}")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("shot")
    fig.suptitle("Gate 3 B0: 5-shot vs 10-shot", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES / "gate3_shot_curves.png", dpi=190)
    plt.close(fig)


def ablation() -> None:
    metrics = pd.read_csv(RESULTS / "gate3_test_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, value, title in zip(
        axes,
        ("open_set_macro_f1", "oos_recall"),
        ("Open-set macro-F1", "OOS recall"),
    ):
        x = np.arange(len(DATASETS))
        width = 0.25
        for i, method in enumerate(METHODS):
            means = []
            errors = []
            for dataset in DATASETS:
                subset = metrics[(metrics.dataset == dataset) & (metrics.method == method)][value]
                means.append(float(subset.mean()))
                errors.append(float(subset.std()))
            ax.bar(x + (i - 1) * width, means, width=width, yerr=errors, capsize=2,
                   label=method, color=METHOD_COLORS[method])
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS)
        ax.set_ylim(0, 1)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Gate 3 ablation: mean over 4 models x 2 shots (error bar = std)", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate3_ablation.png", dpi=190)
    plt.close(fig)


def validation_tradeoff() -> None:
    grid = pd.read_csv(RESULTS / "gate3_validation_grid.csv")
    thresholds = read_json(RESULTS / "gate3_validation_thresholds.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, dataset in zip(axes, DATASETS):
        for method in METHODS:
            subset = grid[(grid.dataset == dataset) & (grid.model == "qwen") & (grid.shot == 10) & (grid.method == method)]
            subset = subset.sort_values("known_acceptance_val")
            ax.plot(subset["known_acceptance_val"], subset["oos_recall_val"], label=method,
                    color=METHOD_COLORS[method], linewidth=1.4)
            key = f"{dataset}|qwen|10|{method}"
            val = thresholds[key]["selection"]
            ax.scatter([val["known_acceptance_val"]], [val["oos_recall_val"]],
                       color=METHOD_COLORS[method], s=45, zorder=3)
        ax.axvline(0.90, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Validation known acceptance")
        ax.set_ylabel("Validation OOS recall")
        ax.set_title(f"{dataset} — Qwen 10-shot")
        ax.set_xlim(0.82, 1.005)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Validation-only threshold tradeoff; dots are selected thresholds", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "gate3_validation_tradeoff.png", dpi=190)
    plt.close(fig)


def cost_scatter() -> None:
    metrics = pd.read_csv(RESULTS / "gate3_test_metrics.csv")
    meta = {}
    for model in MODELS:
        data = read_json(RESULTS / f"gate3_encoding_{model}.json")
        total = sum(v.get("encode", {}).get("elapsed_seconds", 0) for v in data["datasets"].values() if "encode" in v)
        meta[model] = total
    b0 = metrics[(metrics.method == "B0") & (metrics.shot == 10)]
    means = b0.groupby("model")["open_set_macro_f1"].mean().reindex(MODELS)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for model in MODELS:
        ax.scatter(meta[model], means[model], s=80, color=COLORS[model])
        ax.annotate(model, (meta[model], means[model]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Total encoding seconds for 3 datasets")
    ax.set_ylabel("Mean B0 10-shot open-set F1")
    ax.set_title("Model performance vs encoding cost")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate3_cost_scatter.png", dpi=190)
    plt.close(fig)


def training_figures() -> None:
    setfit = read_json(RESULTS / "gate3_setfit_training.json")
    adapter = pd.read_csv(RESULTS / "gate3_adapter_training.csv")
    history = setfit["history"]
    history = [row for row in history if "embedding_loss" in row]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.plot([row.get("step", i + 1) for i, row in enumerate(history)],
            [row["embedding_loss"] for row in history], color="#2f6b57")
    ax.set_xlabel("step")
    ax.set_ylabel("embedding loss")
    ax.set_title("SetFit-BGE embedding training")
    ax.grid(alpha=0.2)
    ax = axes[1]
    ax.plot(adapter["epoch"], adapter["loss"], color="#7d3c4a", label="total loss")
    ax.plot(adapter["epoch"], adapter["contrastive_identity"], color="#c98b3c", label="identity term")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Projection-adapter training")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.2)
    ax = axes[2]
    setfit_metrics = pd.read_csv(RESULTS / "gate3_setfit_metrics.csv").set_index("model")
    adapter_metrics = pd.read_csv(RESULTS / "gate3_adapter_metrics.csv").set_index("method")
    names = ["BGE-B0", "SetFit-BGE", "BGE-Adapter"]
    values = [
        setfit_metrics.loc["BGE-B0-5shot", "open_set_macro_f1"],
        setfit_metrics.loc["SetFit-BGE-5shot", "open_set_macro_f1"],
        adapter_metrics.loc["BGE-Adapter-5shot", "open_set_macro_f1"],
    ]
    ax.bar(names, values, color=["#7d9bb5", "#2f6b57", "#c98b3c"])
    ax.set_ylim(0.55, 0.85)
    ax.set_ylabel("BANKING77 test open-set F1")
    ax.set_title("Training-based comparisons")
    ax.grid(axis="y", alpha=0.2)
    for i, value in enumerate(values):
        ax.text(i, value + 0.005, f"{value:.3f}", ha="center", fontsize=8)
    fig.suptitle("Training-based methods (BANKING77, 5-shot; test already used by matrix)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate3_training.png", dpi=190)
    plt.close(fig)


def error_and_confusion() -> None:
    manifest_rows = {}
    for dataset in DATASETS:
        for row in read_csv(MANIFEST / f"gate3_{dataset}_manifest.csv"):
            manifest_rows[row["sample_id"]] = row
    metrics = pd.read_csv(RESULTS / "gate3_test_metrics.csv")
    best_b3h = (
        metrics[metrics.method == "B3-H"]
        .sort_values("open_set_macro_f1", ascending=False)
        .groupby("dataset")
        .head(1)
        .set_index("dataset")
    )
    error_cases = []
    for dataset in DATASETS:
        config = best_b3h.loc[dataset]
        preds = pd.read_csv(RESULTS / f"gate3_predictions_{dataset}.csv")
        preds = preds[
            (preds.model == config["model"])
            & (preds.shot == int(config["shot"]))
            & (preds.method == config["method"])
        ]
        for category in ("known_correct", "known_rejected", "known_misclassified", "oos_correct", "oos_false_accept"):
            if category == "known_correct":
                candidates = preds[(preds.role == "known_test") & (preds.correct == 1)]
            elif category == "known_rejected":
                candidates = preds[(preds.role == "known_test") & (preds.pred_label == "OOS")]
            elif category == "known_misclassified":
                candidates = preds[(preds.role == "known_test") & (preds.accepted == 1) & (preds.correct == 0)]
            elif category == "oos_correct":
                candidates = preds[(preds.role == "final_unknown") & (preds.pred_label == "OOS")]
            else:
                candidates = preds[(preds.role == "final_unknown") & (preds.pred_label != "OOS")]
            if len(candidates):
                row = candidates.iloc[0].to_dict()
                source = manifest_rows.get(row["sample_id"], {})
                error_cases.append(
                    {
                        "dataset": dataset,
                        "model": config["model"],
                        "shot": int(config["shot"]),
                        "method": config["method"],
                        "category": category,
                        "sample_id": row["sample_id"],
                        "text": source.get("text", ""),
                        "true_label": row["true_label"],
                        "pred_label": row["pred_label"],
                        "s1": row["s1"],
                        "margin": row["margin"],
                        "tau": row["tau"],
                        "top3": row["top3"],
                    }
                )
    pd.DataFrame(error_cases).to_csv(RESULTS / "gate3_error_cases.csv", index=False)

    # BANKING77 best B3-H confusion pairs.
    config = best_b3h.loc["banking77"]
    preds = pd.read_csv(RESULTS / "gate3_predictions_banking77.csv")
    preds = preds[
        (preds.model == config["model"])
        & (preds.shot == int(config["shot"]))
        & (preds.method == config["method"])
        & (preds.role == "known_test")
    ]
    pairs = Counter(
        (row.true_label, row.pred_label)
        for row in preds.itertuples()
        if row.true_label != row.pred_label and row.pred_label != "OOS"
    )
    examples = {}
    for row in preds.itertuples():
        if row.true_label != row.pred_label and row.pred_label != "OOS":
            examples.setdefault((row.true_label, row.pred_label), row._asdict())
    rows = []
    for (true_label, pred_label), count in pairs.most_common():
        if true_label == pred_label or pred_label == "OOS":
            continue
        ex = examples.get((true_label, pred_label))
        text = manifest_rows.get(ex["sample_id"], {}).get("text", "") if ex else ""
        rows.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "count": count,
                "example_text": text,
                "top3": ex["top3"] if ex else "",
            }
        )
        if len(rows) >= 15:
            break
    pd.DataFrame(rows).to_csv(RESULTS / "gate3_banking_confusions.csv", index=False)
    if rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{row['true_label']} → {row['pred_label']}" for row in rows]
        values = [row["count"] for row in rows]
        ax.barh(labels[::-1], values[::-1], color="#8c5e8c")
        ax.set_xlabel("test errors")
        ax.set_title(f"BANKING77 class confusions ({config['model']} {int(config['shot'])}-shot B3-H)", fontsize=10, loc="left")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURES / "gate3_banking_confusions.png", dpi=190)
        plt.close(fig)


def main() -> int:
    dataset_profiles()
    dataset_figures()
    heatmaps()
    shot_curves()
    ablation()
    validation_tradeoff()
    cost_scatter()
    training_figures()
    error_and_confusion()
    for path in sorted(FIGURES.glob("gate3_*.png")):
        print(f"Wrote {path} ({path.stat().st_size} bytes)")
    print("GATE3_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
