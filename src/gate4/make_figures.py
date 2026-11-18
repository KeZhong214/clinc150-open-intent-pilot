"""Gate-4 B4 and Agent figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def main() -> int:
    hist = pd.read_csv(RESULTS / "gate4_b4_training_history.csv")
    val = pd.read_csv(RESULTS / "gate4_b4_validation_metrics.csv")
    test = pd.read_csv(RESULTS / "gate4_b4_test_metrics.csv")
    pred = pd.read_csv(RESULTS / "gate4_b4_test_predictions.csv")
    cls = pd.read_csv(RESULTS / "gate4_b4_test_gate_by_class.csv")
    diff = pd.read_csv(RESULTS / "gate4_b4_test_gate_by_difficulty.csv")
    agent = json.loads((RESULTS / "gate4_agent_metrics.json").read_text(encoding="utf-8"))

    # Training curves.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for stage, color in (("B4-Fusion", "#7d9bb5"), ("B4-Full", "#2f6b57")):
        sub = hist[hist.stage == stage]
        axes[0].plot(sub.epoch, sub.train_loss, color=color, label=f"{stage} train")
        axes[1].plot(sub.epoch, sub.val_known_ce, color=color, label=f"{stage} known CE")
        axes[2].plot(sub.epoch, sub.gate_mean, color=color, label=f"{stage} gate mean")
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation known CE")
    axes[2].set_title("Gate mean")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("B4-AR training (GTE BANKING77 10-shot, support-only episodes)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate4_b4_training.png", dpi=190)
    plt.close(fig)

    # Validation/test comparison.
    methods = ["B0", "B3-H", "B4-Fusion", "B4-Full"]
    colors = ["#7d9bb5", "#c98b3c", "#8c5e8c", "#2f6b57"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (frame, label) in enumerate(((val, "Validation"), (test, "Test (exploratory)"))):
        frame = frame.set_index("method").loc[methods]
        for col, metric in enumerate(("open_set_macro_f1", "oos_recall")):
            ax = axes[row, col]
            bars = ax.bar(methods, frame[metric], color=colors)
            for bar, value in zip(bars, frame[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006, f"{value:.3f}", ha="center", fontsize=7)
            ax.set_ylim(0.45 if metric == "open_set_macro_f1" else 0.0, 1.0)
            ax.set_title(f"{label} — {'open-set F1' if col == 0 else 'OOS recall'}")
            ax.grid(axis="y", alpha=0.2)
            ax.tick_params(axis="x", rotation=15)
    fig.suptitle("B0 vs B3-H vs B4 (same support / validation budget; test already seen)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate4_b4_comparison.png", dpi=190)
    plt.close(fig)

    # Gate by class.
    cls = cls.dropna(subset=["gate_mean_top1"]).sort_values("gate_mean_top1")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(cls["intent"], cls["gate_mean_top1"], color="#2f6b57")
    ax.set_xlabel("mean gate weight on Top-1 class")
    ax.set_title("B4-Full gate weight by known class (test)")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate4_b4_gate_by_class.png", dpi=190)
    plt.close(fig)

    # Gate by difficulty.
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(diff["margin_quartile"].astype(str), diff["gate_mean_top1"], color="#c98b3c")
    ax.set_xlabel("margin quartile (1 = hardest)")
    ax.set_ylabel("mean gate on Top-1")
    ax.set_ylim(0, 1)
    ax.set_title("B4 gate responds to sample difficulty")
    ax.grid(axis="y", alpha=0.2)
    for i, row in diff.iterrows():
        ax.text(i, row["gate_mean_top1"] + 0.01, f"{row['gate_mean_top1']:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate4_b4_gate_by_difficulty.png", dpi=190)
    plt.close(fig)

    # Gate vs margin scatter.
    fig, ax = plt.subplots(figsize=(8, 5))
    known = pred["role"] == "known_test"
    ax.scatter(pred.loc[known, "s1"].to_numpy() - pred.loc[known, "s2"].to_numpy(), pred.loc[known, "gate_top1"], s=8, alpha=0.35, label="known")
    ax.scatter(pred.loc[~known, "s1"].to_numpy() - pred.loc[~known, "s2"].to_numpy(), pred.loc[~known, "gate_top1"], s=8, alpha=0.35, label="OOS")
    ax.set_xlabel("S1 - S2 (fused margin)")
    ax.set_ylabel("gate weight on Top-1")
    ax.set_title("B4-Full gate varies with input difficulty")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate4_b4_gate_scatter.png", dpi=190)
    plt.close(fig)

    # Agent metrics.
    fig, ax = plt.subplots(figsize=(7, 4.3))
    labels = ["task\ncompletion", "tool-call\ncorrectness", "clarification\nrate", "confirmation\nrate", "escalation\nrate"]
    values = [
        agent["task_completion_rate"],
        agent["tool_call_correctness"],
        agent["clarification_rate"],
        agent["confirmation_rate"],
        agent["escalation_rate"],
    ]
    bars = ax.bar(labels, values, color=["#2f6b57", "#7d9bb5", "#c98b3c", "#8c5e8c", "#b05a5a"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("rate")
    ax.set_title(f"Agent scripted cases (n={agent['n_cases']})")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate4_agent_metrics.png", dpi=190)
    plt.close(fig)

    print("GATE4_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
