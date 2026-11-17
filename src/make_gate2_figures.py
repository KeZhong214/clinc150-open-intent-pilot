"""Figures for the Gate-2 hard-OOS protocol."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["B0", "B2", "B2_v2"]
METHOD_LABELS = {"B0": "B0", "B2": "B2 original", "B2_v2": "B2-v2"}
COLORS = {"B0": "#7d9bb5", "B2": "#c98b3c", "B2_v2": "#2f6b57"}
METRICS = [
    ("known_acceptance", "Known acceptance"),
    ("known_correct_routing", "Known routing"),
    ("hard_oos_recall", "Hard-OOS recall"),
    ("hard_open_set_macro_f1", "Hard open-set F1"),
    ("open_set_accuracy", "Open-set accuracy"),
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def comparison(path: Path, title: str, output: str) -> None:
    frame = pd.read_csv(path).set_index("method")
    x = np.arange(len(METRICS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for offset, method in zip((-1, 0, 1), METHOD_ORDER):
        values = [float(frame.loc[method, key]) for key, _ in METRICS]
        bars = ax.bar(
            x + offset * width,
            values,
            width=width,
            color=COLORS[method],
            label=METHOD_LABELS[method],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS], fontsize=8)
    ax.set_ylim(0.84, 1.005)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / output, dpi=190)
    plt.close(fig)


def threshold_correction() -> None:
    table = pd.read_csv(RESULTS / "gate2_b2v2_class_thresholds.csv")
    tau = float(read_json(RESULTS / "gate2_freeze.json")["thresholds"]["B2_v2"]["tau"])
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter(
        table["q_c"],
        table["tau_c"],
        s=np.clip(table["n_c"] * 1.3, 20, 160),
        color="#6b4c7a",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.6,
    )
    low = min(table["q_c"].min(), tau) - 0.01
    high = max(table["q_c"].max(), table["tau_c"].max()) + 0.01
    ax.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1, label="tau_c = q_c")
    ax.axhline(tau, color="#2f6b57", linestyle=":", linewidth=1.2, label=f"global tau={tau:.3f}")
    for row in table[table["tau_c"] > tau + 1e-9].itertuples():
        ax.annotate(
            row.intent,
            (row.q_c, row.tau_c),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("q_c: known-class 10th percentile")
    ax.set_ylabel("B2-v2 class threshold tau_c")
    ax.set_title("B2-v2 only raises thresholds; point size = hard-OOS validation count n_c")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "gate2_threshold_correction.png", dpi=190)
    plt.close(fig)


def transfer_figure() -> None:
    val = pd.read_csv(RESULTS / "gate2_validation_metrics.csv").set_index("method")
    test = pd.read_csv(RESULTS / "gate2_test_metrics.csv").set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, frame, label in (
        (axes[0], val, "Validation"),
        (axes[1], test, "Held-out hard-OOS test"),
    ):
        for method in METHOD_ORDER:
            row = frame.loc[method]
            ax.scatter(
                row["known_correct_routing"],
                row["hard_oos_recall"],
                s=80,
                color=COLORS[method],
                label=METHOD_LABELS[method],
                zorder=3,
            )
            ax.annotate(
                METHOD_LABELS[method],
                (row["known_correct_routing"], row["hard_oos_recall"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel("Known correct routing")
        ax.set_ylabel("Hard-OOS recall")
        ax.set_title(label)
        ax.grid(alpha=0.2)
        ax.set_xlim(0.875, 0.915)
        ax.set_ylim(0.93, 0.992)
    fig.suptitle("Validation improvement does not fully transfer to the held-out unknown classes", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate2_transfer.png", dpi=190)
    plt.close(fig)


def score_distribution() -> None:
    val = pd.read_csv(RESULTS / "gate2_validation_predictions.csv")
    test = pd.read_csv(RESULTS / "gate2_test_predictions.csv")
    val = val[val["method"] == "B0"]
    test = test[test["method"] == "B0"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, frame, label, tau in (
        (axes[0], val, "Validation", 0.790428638458252),
        (axes[1], test, "Held-out hard-OOS test", 0.790428638458252),
    ):
        for role, color, name in (
            ("known", "#3d7a5b", "known"),
            ("hard_oos", "#b24a4a", "hard OOS"),
        ):
            values = frame.loc[frame["role"] == role, "s1"].astype(float)
            ax.hist(values, bins=35, density=True, alpha=0.55, color=color, label=name, edgecolor="white")
        ax.axvline(tau, color="black", linestyle="--", linewidth=1.2, label="B0 tau")
        ax.set_title(label)
        ax.set_xlabel("Top-1 cosine")
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Known vs hard-OOS top-1 scores", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "gate2_hard_score_distribution.png", dpi=190)
    plt.close(fig)


def main() -> int:
    comparison(
        RESULTS / "gate2_validation_metrics.csv",
        "Gate 2 validation: B0 vs B2 vs B2-v2",
        "gate2_validation_comparison.png",
    )
    comparison(
        RESULTS / "gate2_test_metrics.csv",
        "Gate 2 held-out hard-OOS test: B0 vs B2 vs B2-v2",
        "gate2_test_comparison.png",
    )
    threshold_correction()
    transfer_figure()
    score_distribution()
    for path in sorted(FIGURES.glob("gate2_*.png")):
        print(f"Wrote {path} ({path.stat().st_size} bytes)")
    print("GATE2_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
