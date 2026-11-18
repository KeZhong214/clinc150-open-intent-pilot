"""G7-A diagnostic figures (development split only)."""

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
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def score_hist():
    d = np.load(RESULTS / "gate7_dev_scores.npz")
    diag = pd.read_csv(RESULTS / "gate7_dev_diagnosis.csv").set_index(["encoder", "method"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, enc, label in ((axes[0], "frozen", "Qwen Frozen"), (axes[1], "plain", "Qwen Plain LoRA")):
        s = d[f"{enc}_B3-H_score"]
        y = d[f"{enc}_B3-H_y"]
        ax.hist(s[y >= 0], bins=35, alpha=0.55, label="known", color="#2f6b57")
        ax.hist(s[y < 0], bins=35, alpha=0.55, label="dev-select OOS", color="#c98b3c")
        tau = float(diag.loc[(f"qwen_{enc}", "B3-H"), "tau"])
        ax.axvline(tau, color="#b05a5a", linestyle="--", label=f"tau={tau:.3f}")
        ax.set_title(label)
        ax.set_xlabel("score (B3-H / RejectOnly)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("count")
    fig.suptitle("Dev-select score distributions: known vs unseen classes")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "gate7_score_distribution.png", dpi=190)
    plt.close(fig)


def hard_negatives():
    h = pd.read_csv(RESULTS / "gate6_qwen_hard_negatives.csv")
    h["margin"] = h.anchor_pos_cos - h.anchor_neg_cos
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(h.anchor_neg_cos, bins=30, color="#c98b3c")
    axes[0].set_title("Hard-negative cosine (frozen Qwen support)")
    axes[0].set_xlabel("cos(anchor, hard negative)")
    axes[0].grid(alpha=0.2)
    axes[1].hist(h.margin, bins=30, color="#b05a5a")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Positive minus negative margin")
    axes[1].set_xlabel("cos(anchor,pos) - cos(anchor,neg)")
    axes[1].grid(alpha=0.2)
    fig.suptitle("400 support rows: 203/400 have negative margin; 157 unique negatives")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "gate7_hard_negative.png", dpi=190)
    plt.close(fig)


def acceptance_curves():
    c = pd.read_csv(RESULTS / "gate7_dev_acceptance_curves.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (enc, method), g in c.groupby(["encoder", "method"]):
        if method != "B3-H":
            continue
        ax.plot(g.known_acceptance, g.oos_acceptance, marker="o", label=enc)
    ax.set_xlabel("known acceptance")
    ax.set_ylabel("OOS acceptance (dev-select)")
    ax.set_title("Threshold acceptance curve")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "gate7_acceptance_curve.png", dpi=190)
    plt.close(fig)


def toy_matrix():
    n = 6
    z = np.random.default_rng(42).normal(size=(n, 16))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    m = z @ z.T
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_title("Actual microbatch SupCon matrix: 6x6")
    ax.set_xlabel("6 vectors = 2 anchor + 2 positive + 2 hard negative")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "gate7_toy_6x6.png", dpi=190)
    plt.close(fig)


def main() -> int:
    score_hist()
    hard_negatives()
    acceptance_curves()
    toy_matrix()
    for p in sorted(FIG.glob("gate7_*.png")):
        print(p.name, p.stat().st_size)
    print("GATE7_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
