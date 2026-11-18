"""Gate-5 figures: LoRA, six-method attribution, and Agent benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

COLORS = {"frozen": "#2f6b57", "lora": "#c98b3c"}
METHOD_COLORS = {"B0": "#7d9bb5", "B3-H": "#c98b3c", "B4-Fusion": "#8c5e8c", "B4-RejectOnly": "#2f6b57", "B4-Full": "#b05a5a"}


def lora_figures():
    hist = pd.read_csv(RESULTS / "gate5_lora_training_history.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(hist.epoch, hist.train_loss, marker="o", color="#2f6b57")
    axes[0].set_title("LoRA training loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.2)
    axes[1].plot(hist.epoch, hist.supcon, marker="o", label="SupCon", color="#7d9bb5")
    axes[1].plot(hist.epoch, hist.triplet, marker="o", label="hard triplet", color="#c98b3c")
    axes[1].set_title("Loss components")
    axes[1].set_xlabel("epoch")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)
    axes[2].plot(hist.epoch, hist.grad_norm, marker="o", color="#8c5e8c")
    axes[2].set_title("Gradient norm")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.2)
    fig.suptitle("GTE-ModernBERT LoRA: BANKING77 10-shot support-only training")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate5_lora_training.png", dpi=190)
    plt.close(fig)

    frozen = pd.read_csv(RESULTS / "gate3_validation_metrics.csv")
    frozen = frozen[(frozen.dataset == "banking77") & (frozen.model == "gte") & (frozen.shot == 10) & (frozen.method.isin(["B0", "B3-H"]))]
    lora = pd.read_csv(RESULTS / "gate5_lora_validation.csv")
    methods = ["B0", "B3-H"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = range(len(methods))
    width = 0.36
    for i, (frame, label, color) in enumerate(((frozen, "Frozen GTE", "#7d9bb5"), (lora, "LoRA-GTE", "#c98b3c"))):
        vals = [float(frame[frame.method == m]["val_open_set_macro_f1"].iloc[0]) for m in methods]
        oos = [float(frame[frame.method == m]["val_oos_recall"].iloc[0]) for m in methods]
        for ax, values, title in ((axes[0], vals, "Validation open-set F1"), (axes[1], oos, "Validation OOS recall")):
            bars = ax.bar([j + (i - 0.5) * width for j in x], values, width=width, label=label, color=color)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
            ax.set_xticks(list(x))
            ax.set_xticklabels(methods)
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Frozen vs LoRA-GTE validation (same 800 known + 720 dev unknown)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "gate5_lora_validation.png", dpi=190)
    plt.close(fig)


def matrix_figures():
    val = pd.read_csv(RESULTS / "gate5_method_matrix_val.csv")
    test = pd.read_csv(RESULTS / "gate5_method_matrix_historical_test.csv")
    methods = ["B0", "B3-H", "B4-Fusion", "B4-RejectOnly", "B4-Full"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, (frame, label) in enumerate(((val, "Validation"), (test, "Historical exploratory test"))):
        for col, metric in enumerate(("open_set_macro_f1", "oos_recall")):
            ax = axes[row, col]
            labels = []
            values = []
            colors = []
            for encoder in ("frozen", "lora"):
                for method in methods:
                    r = frame[(frame.encoder == encoder) & (frame.method == method)]
                    if len(r) == 0:
                        continue
                    labels.append(f"{encoder}\n{method}")
                    values.append(float(r[metric].iloc[0]))
                    colors.append(METHOD_COLORS[method])
            bars = ax.bar(range(len(values)), values, color=colors)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=6)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_title(f"{label} — {'open-set F1' if col == 0 else 'OOS recall'}")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Frozen/LoRA × B3-H/RejectOnly/B4-Full attribution matrix")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate5_method_matrix.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    subset = val[val.encoder == "frozen"].set_index("method").loc[["B0", "B3-H", "B4-RejectOnly", "B4-Full"]]
    x = range(len(subset))
    width = 0.36
    ax.bar([i - width / 2 for i in x], subset["open_set_macro_f1"], width=width, label="open-set F1", color="#2f6b57")
    ax.bar([i + width / 2 for i in x], subset["oos_recall"], width=width, label="OOS recall", color="#c98b3c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(subset.index, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title("Module attribution on Frozen GTE (validation)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate5_module_attribution.png", dpi=190)
    plt.close(fig)


def benchmark_figures():
    summary = pd.read_csv(RESULTS / "gate5_benchmark_summary.csv")
    byfam = pd.read_csv(RESULTS / "gate5_benchmark_by_family.csv")
    res = pd.read_csv(RESULTS / "gate5_benchmark_results.csv")
    routers = list(summary["router"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(routers))
    width = 0.36
    ax.bar([i - width / 2 for i in x], summary["task_success_rate"], width=width, label="Task Success Rate", color="#2f6b57")
    ax.bar([i + width / 2 for i in x], summary["tool_accuracy"], width=width, label="Tool+argument accuracy", color="#c98b3c")
    ax.plot(x, summary["safe_action_rate"], color="#8c5e8c", marker="o", label="Safe action rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(routers)
    ax.set_ylim(0, 1.05)
    ax.set_title("Gate-5 Agent benchmark (120 frozen tasks)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate5_benchmark_overall.png", dpi=190)
    plt.close(fig)

    pivot = byfam.pivot(index="family", columns="router", values="task_success_rate")
    fig, ax = plt.subplots(figsize=(11, 5))
    idx = list(pivot.index)
    x = range(len(idx))
    width = 0.26
    for i, router in enumerate(routers):
        vals = [float(pivot.loc[f, router]) if router in pivot else 0 for f in idx]
        ax.bar([j + (i - 1) * width for j in x], vals, width=width, label=router)
    ax.set_xticks(list(x))
    ax.set_xticklabels(idx, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Task Success Rate by business family")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate5_benchmark_by_family.png", dpi=190)
    plt.close(fig)

    fails = res[res.task_success == False].groupby(["router", "failure_reason"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    reasons = list(fails["failure_reason"].unique())
    x = range(len(reasons))
    width = 0.26
    for i, router in enumerate(routers):
        sub = fails[fails.router == router].set_index("failure_reason")
        vals = [int(sub.loc[r, "count"]) if r in sub.index else 0 for r in reasons]
        ax.bar([j + (i - 1) * width for j in x], vals, width=width, label=router)
    ax.set_xticks(list(x))
    ax.set_xticklabels(reasons, rotation=20, ha="right")
    ax.set_ylabel("failed cases")
    ax.set_title("Failure reasons across routers")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate5_benchmark_failures.png", dpi=190)
    plt.close(fig)


def main() -> int:
    lora_figures()
    matrix_figures()
    benchmark_figures()
    for p in sorted(FIGURES.glob("gate5_*.png")):
        print(p, p.stat().st_size)
    print("GATE5_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
