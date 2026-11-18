"""Gate-6 figures: alignment, drift, Qwen matrix, and Agent regression."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def figure_alignment():
    a = json.loads((RESULTS / "gate6_qwen_alignment.json").read_text(encoding="utf-8"))
    v = a["vector_alignment"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["mean cosine", "min cosine", "top-1 consistency"]
    values = [v["mean_cosine"], v["min_cosine"], v["top1_intent_consistency"]]
    bars = ax.bar(labels, values, color=["#2f6b57", "#c98b3c", "#7d9bb5"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0005, f"{value:.6f}", ha="center", fontsize=9)
    ax.set_ylim(0.99, 1.0015)
    ax.set_title("Qwen frozen encoder vs Gate-3 cache (32 probes)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "gate6_qwen_alignment.png", dpi=190)
    plt.close(fig)


def figure_training():
    p = pd.read_csv(RESULTS / "gate6_qwen_plain_history.csv")
    a = pd.read_csv(RESULTS / "gate6_qwen_anchor_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(p.epoch, p.loss, marker="o", label="Plain", color="#2f6b57")
    axes[0].plot(a.epoch, a.loss, marker="o", label="Anchor", color="#c98b3c")
    axes[0].set_title("Qwen LoRA training loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].plot(a.epoch, a.anchor, marker="o", color="#8c5e8c", label="anchor loss")
    axes[1].set_title("Anchor cosine penalty")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("Qwen3-Embedding LoRA: 2 epochs, 400 support rows, microbatch 2")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "gate6_qwen_lora_training.png", dpi=190)
    plt.close(fig)


def figure_drift():
    plain = json.loads((RESULTS / "gate6_qwen_plain_training.json").read_text(encoding="utf-8"))
    anchor = json.loads((RESULTS / "gate6_qwen_anchor_training.json").read_text(encoding="utf-8"))
    test = pd.read_csv(RESULTS / "gate6_qwen_method_matrix_historical_test.csv")
    val = pd.read_csv(RESULTS / "gate6_qwen_method_matrix_val.csv")
    def get(frame, enc, method, col):
        return float(frame[(frame.encoder == enc) & (frame.method == method)][col].iloc[0])
    points = [
        ("Frozen", 1.0, get(val, "qwen_frozen", "B3-H", "open_set_macro_f1"), get(test, "qwen_frozen", "B3-H", "open_set_macro_f1")),
        ("Plain", plain["vector_check"]["frozen_vs_lora_mean_cosine"], get(val, "qwen_plain_lora", "B3-H", "open_set_macro_f1"), get(test, "qwen_plain_lora", "B3-H", "open_set_macro_f1")),
        ("Anchor", anchor["vector_check"]["frozen_vs_lora_mean_cosine"], get(val, "qwen_anchor_lora", "B3-H", "open_set_macro_f1"), get(test, "qwen_anchor_lora", "B3-H", "open_set_macro_f1")),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, x, v, t in points:
        ax.scatter(x, t, s=90, label=f"{name} test", color="#b05a5a")
        ax.scatter(x, v, s=90, marker="^", label=f"{name} val", color="#2f6b57")
        ax.annotate(name, (x, t), textcoords="offset points", xytext=(5, 6), fontsize=8)
        ax.plot([x, x], [v, t], color="#999999", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("support cosine to frozen Qwen (1.0 = no drift)")
    ax.set_ylabel("B3-H open-set F1")
    ax.set_title("Drift vs validation/test F1 (Qwen, B3-H)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "gate6_qwen_drift.png", dpi=190)
    plt.close(fig)


def figure_matrix():
    val = pd.read_csv(RESULTS / "gate6_qwen_method_matrix_val.csv")
    test = pd.read_csv(RESULTS / "gate6_qwen_method_matrix_historical_test.csv")
    labels = []
    for enc in ("qwen_frozen", "qwen_plain_lora", "qwen_anchor_lora"):
        for method in ("B0", "B3-H", "B4-RejectOnly", "B4-Full"):
            if ((val.encoder == enc) & (val.method == method)).any():
                labels.append(f"{enc.replace('qwen_', '').replace('_lora','')}\n{method}")
    vals, tests = [], []
    for label in labels:
        enc, method = label.split("\n")
        enc = f"qwen_{enc}_lora" if enc in ("plain", "anchor") else "qwen_frozen"
        vals.append(float(val[(val.encoder == enc) & (val.method == method)]["open_set_macro_f1"].iloc[0]))
        tests.append(float(test[(test.encoder == enc) & (test.method == method)]["open_set_macro_f1"].iloc[0]))
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar([i - w / 2 for i in x], vals, width=w, label="validation", color="#7d9bb5")
    ax.bar([i + w / 2 for i in x], tests, width=w, label="historical test", color="#c98b3c")
    for i, (v, t) in enumerate(zip(vals, tests)):
        ax.text(i - w / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=6)
        ax.text(i + w / 2, t + 0.006, f"{t:.3f}", ha="center", fontsize=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("Qwen3-Embedding Frozen/LoRA × B0/B3-H/RejectOnly/Full")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "gate6_qwen_matrix.png", dpi=190)
    plt.close(fig)


def figure_benchmark():
    if not (RESULTS / "gate6_qwen_benchmark_summary.csv").exists():
        return
    s = pd.read_csv(RESULTS / "gate6_qwen_benchmark_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(s))
    w = 0.28
    ax.bar([i - w for i in x], s.task_success_rate, width=w, label="Task Success", color="#2f6b57")
    ax.bar([i for i in x], s.tool_accuracy, width=w, label="Tool+arg", color="#c98b3c")
    ax.bar([i + w for i in x], s.safe_action_rate, width=w, label="Safe", color="#8c5e8c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(s.router, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Gate-6 Qwen Agent regression on 120 frozen tasks (exploratory)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "gate6_qwen_benchmark.png", dpi=190)
    plt.close(fig)
    if (RESULTS / "gate6_qwen_benchmark_by_family.csv").exists():
        b = pd.read_csv(RESULTS / "gate6_qwen_benchmark_by_family.csv")
        pivot = b.pivot(index="family", columns="router", values="task_success_rate")
        fig, ax = plt.subplots(figsize=(11, 5))
        idx = list(pivot.index)
        x = range(len(idx))
        w = 0.2
        for i, router in enumerate(s.router):
            vals = [float(pivot.loc[f, router]) if router in pivot else 0 for f in idx]
            ax.bar([j + (i - 1.5) * w for j in x], vals, width=w, label=router)
        ax.set_xticks(list(x))
        ax.set_xticklabels(idx, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("Qwen Agent task success by business family")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "gate6_qwen_by_family.png", dpi=190)
        plt.close(fig)


def main() -> int:
    figure_alignment()
    figure_training()
    figure_drift()
    figure_matrix()
    figure_benchmark()
    for p in sorted(FIG.glob("gate6_qwen_*.png")):
        print(p.name, p.stat().st_size)
    print("GATE6_QWEN_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
