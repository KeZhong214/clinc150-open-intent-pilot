"""Gate-7 stability and final-seed figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG = ROOT / "figures"


def stability():
    d = pd.read_csv(RESULTS / "gate7_stability_summary.csv")
    encs = list(dict.fromkeys(d.encoder))
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(encs))
    for j, (method, color) in enumerate((("B4-RejectOnly", "#2f6b57"), ("B4-Full", "#c98b3c"))):
        sub = d[d.method == method].set_index("encoder")
        mean = [float(sub.loc[e, "dev_open_set_macro_f1_mean"]) if e in sub.index else 0 for e in encs]
        std = [float(sub.loc[e, "dev_open_set_macro_f1_std"]) if e in sub.index else 0 for e in encs]
        ax.bar(x + (j - 0.5) * 0.36, mean, width=0.36, yerr=std, capsize=3, label=method, color=color)
        for i, (m, s) in enumerate(zip(mean, std)):
            if m:
                ax.text(i + (j - 0.5) * 0.36, m + s + 0.004, f"{m:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(encs, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.70, 0.87)
    ax.set_ylabel("dev-select open-set F1")
    ax.set_title("B4-RejectOnly vs B4-Full across seeds (mean ± std)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "gate7_stability.png", dpi=190)
    plt.close(fig)


def final_seeds():
    d = pd.read_csv(RESULTS / "gate7_final_summary.csv")
    methods = ["B3-H", "B4-RejectOnly"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(methods))
    w = 0.36
    for j, (metric, color, label) in enumerate((("open_set_macro_f1", "#2f6b57", "open-set F1"), ("oos_recall", "#c98b3c", "OOS recall"))):
        sub = d[d.metric == metric].set_index("method")
        mean = [float(sub.loc[m, "mean"]) for m in methods]
        std = [float(sub.loc[m, "std"]) for m in methods]
        frozen = [float(sub.loc[m, "frozen"]) for m in methods]
        ax.bar(x + (j - 0.5) * w, mean, width=w, yerr=std, capsize=3, color=color, label=f"Qwen Plain {label}")
        ax.scatter(x + (j - 0.5) * w, frozen, color="black", marker="_", s=220, label=f"Frozen {label}" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.55, 0.90)
    ax.set_ylabel("historical exploratory test")
    ax.set_title("Qwen Plain seeds 42/43/44 vs Frozen (mean ± std)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "gate7_final_test.png", dpi=190)
    plt.close(fig)


def main() -> int:
    stability()
    if (RESULTS / "gate7_final_summary.csv").exists():
        final_seeds()
    for p in sorted(FIG.glob("gate7_*.png")):
        print(p.name, p.stat().st_size)
    print("GATE7_STABILITY_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
