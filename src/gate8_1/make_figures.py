"""Gate 8.1 figures from the frozen 120-case results only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG = ROOT / "figures"


def main() -> int:
    s = pd.read_csv(RESULTS / "gate8_1_agent_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(s))
    w = 0.25
    for i, (m, label) in enumerate([("task_success_rate", "TSR"), ("tool_accuracy", "Tool"), ("routing_accuracy", "Routing")]):
        ax.bar(x + (i - 1) * w, s[m], w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("-", "\n") for r in s.router], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Gate 8.1 — new frozen simulated test (n=120 per router)")
    ax.legend(ncol=3, loc="lower right")
    for i, r in s.iterrows():
        ax.text(i, 0.98, f"{r.task_success_rate:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "gate8_1_benchmark.png", dpi=160)
    plt.close(fig)

    fam = pd.read_csv(RESULTS / "gate8_1_agent_by_family.csv").pivot(index="family", columns="router", values="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 7))
    mat = ax.imshow(fam.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(fam.columns)))
    ax.set_xticklabels([c.replace("-", "\n") for c in fam.columns], fontsize=8)
    ax.set_yticks(range(len(fam.index)))
    ax.set_yticklabels(fam.index, fontsize=8)
    for i in range(fam.shape[0]):
        for j in range(fam.shape[1]):
            ax.text(j, i, f"{fam.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(mat, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Gate 8.1 TSR by family (new frozen simulated test)")
    fig.tight_layout()
    fig.savefig(FIG / "gate8_1_failure_heatmap.png", dpi=160)
    plt.close(fig)
    print("GATE8_1_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
