"""Create the three core Gate-1 figures from saved, real result files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(model: str, method: str) -> pd.Series:
    frame = pd.read_csv(RESULTS / "gate1_metrics.csv")
    rows = frame[(frame["model"] == model) & (frame["method"] == method)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for {model}/{method}, got {len(rows)}")
    return rows.iloc[0]


def plot_comparison() -> None:
    metrics = [
        ("known_acceptance", "Known acceptance"),
        ("known_correct_routing", "Known correct routing"),
        ("oos_recall", "OOS recall"),
        ("open_set_macro_f1", "Open-set macro-F1"),
    ]
    models = ["bge", "qwen"]
    methods = ["B0", "B2"]
    colors = {
        ("bge", "B0"): "#8ca7c4",
        ("bge", "B2"): "#2f5d8a",
        ("qwen", "B0"): "#e6b566",
        ("qwen", "B2"): "#b56b1f",
    }
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, (model, method) in zip((-1.5, -0.5, 0.5, 1.5), [(m, meth) for m in models for meth in methods]):
        row = metric_row(model, method)
        values = [float(row[key]) for key, _ in metrics]
        bars = ax.bar(
            x + offset * width,
            values,
            width=width,
            label=f"{model.upper()} {method}",
            color=colors[(model, method)],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0.78, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Gate 1: B0 vs B2 on the fixed final test (seed=42, 5-shot)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "gate1_b0_b2_comparison.png", dpi=190)
    plt.close(fig)


def plot_score_distributions() -> None:
    preds = pd.read_csv(RESULTS / "predictions_b0.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    for row_index, model in enumerate(("bge", "qwen")):
        model_preds = preds[(preds["model"] == model) & (preds["role"].str.startswith("final_"))]
        for col_index, role in enumerate(("final_known_test", "final_oos_test")):
            ax = axes[row_index, col_index]
            values = model_preds.loc[model_preds["role"] == role, "s1"].astype(float)
            color = "#3d7a5b" if role == "final_known_test" else "#b24a4a"
            ax.hist(values, bins=30, density=True, alpha=0.65, color=color, edgecolor="white")
            threshold = read_json(RESULTS / f"thresholds_{model}_b0.json")["tau"]
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"B0 tau={threshold:.3f}")
            ax.set_title(f"{model.upper()} — {'known' if 'known' in role else 'OOS'} test")
            ax.set_xlabel("Top-1 cosine")
            ax.set_ylabel("density")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Top-1 score distributions and the validation-selected B0 threshold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "gate1_score_distribution.png", dpi=190)
    plt.close(fig)


def plot_validation_tradeoff() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, model in zip(axes, ("bge", "qwen")):
        grid = pd.read_csv(RESULTS / f"val_grid_{model}_b0.csv")
        grid = grid.sort_values("known_acceptance_val")
        ax.plot(
            grid["known_acceptance_val"],
            grid["oos_recall_val"],
            color="#355c7d",
            linewidth=1.5,
            label="B0 threshold sweep",
        )
        b2 = read_json(RESULTS / f"thresholds_{model}_b2.json")
        ax.scatter(
            [b2["validation"]["known_acceptance_val"]],
            [b2["validation"]["oos_recall_val"]],
            color="#c06c3a",
            s=70,
            zorder=4,
            label=f"B2 selected (λ={b2['lambda']}, γ={b2['gamma']})",
        )
        ax.axvline(0.90, color="gray", linestyle=":", linewidth=1, label="90% acceptance")
        ax.set_xlabel("Validation known acceptance")
        ax.set_ylabel("Validation OOS recall")
        ax.set_title(model.upper())
        ax.set_xlim(0.82, 1.005)
        ax.set_ylim(min(0.90, grid["oos_recall_val"].min() - 0.01), 1.005)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Validation-only threshold selection", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate1_validation_tradeoff.png", dpi=190)
    plt.close(fig)


def plot_qwen_confusion(method: str, output_name: str) -> None:
    frame = pd.read_csv(RESULTS / f"confusion_qwen_{method.lower()}.csv")
    labels = frame["true_label"].tolist()
    matrix = frame[labels].to_numpy(dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Qwen {method}: row-normalized test confusion matrix (includes OOS)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / output_name, dpi=190)
    plt.close(fig)


def plot_per_class_delta() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, model in zip(axes, ("bge", "qwen")):
        b0 = pd.read_csv(RESULTS / f"per_class_{model}_b0.csv").set_index("intent")
        b2 = pd.read_csv(RESULTS / f"per_class_{model}_b2.csv").set_index("intent")
        delta = (b2["accuracy"] - b0["accuracy"]).sort_values()
        colors = ["#a34a4a" if value < 0 else "#3d7a5b" for value in delta]
        ax.barh(delta.index, delta.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("B2 known routing accuracy − B0")
        ax.set_title(model.upper())
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Per-class effect of B2 vs B0 on final known test", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate1_per_class_delta.png", dpi=190)
    plt.close(fig)


def plot_qc_vs_delta() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, model in zip(axes, ("bge", "qwen")):
        b0 = pd.read_csv(RESULTS / f"per_class_{model}_b0.csv").set_index("intent")
        b2 = pd.read_csv(RESULTS / f"per_class_{model}_b2.csv").set_index("intent")
        threshold = read_json(RESULTS / f"thresholds_{model}_b2.json")
        q_c = pd.Series(threshold["q_c"])
        delta = (b2["accuracy"] - b0["accuracy"]).reindex(q_c.index) * 100
        ax.axhline(0, color="black", linewidth=0.8)
        ax.scatter(q_c.values, delta.values, color="#6b4c7a", s=45)
        for intent, x_value, y_value in zip(q_c.index, q_c.values, delta.values):
            if abs(y_value) >= 5:
                ax.annotate(intent, (x_value, y_value), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Validation 10th percentile q_c")
        ax.set_ylabel("Known routing delta (percentage points)")
        ax.set_title(model.upper())
        ax.grid(alpha=0.2)
    fig.suptitle("Class-aware threshold strength versus B2 effect", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate1_qc_vs_delta.png", dpi=190)
    plt.close(fig)


def main() -> int:
    plot_comparison()
    plot_score_distributions()
    plot_validation_tradeoff()
    plot_qwen_confusion("B0", "baseline_confusion.png")
    plot_qwen_confusion("B2", "gate1_qwen_b2_confusion.png")
    plot_per_class_delta()
    plot_qc_vs_delta()
    shutil.copyfile(
        FIGURES / "gate1_score_distribution.png",
        FIGURES / "score_distribution.png",
    )
    for path in sorted(FIGURES.glob("*.png")):
        print(f"Wrote {path} ({path.stat().st_size} bytes)")
    print("FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
