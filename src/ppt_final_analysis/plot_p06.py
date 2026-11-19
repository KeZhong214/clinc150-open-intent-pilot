"""P0-2: B4 module attribution on the Gate5 frozen development validation set."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.ppt_final_analysis.common import COMMIT, GRAY, LIGHT, NAVY, ORANGE, OUT, TEAL, BLUE, footer, notes, save, setup

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "gate5_method_matrix_val.csv"
METHODS = ["B0", "B3-H", "B4-Fusion", "B4-Full", "B4-RejectOnly"]
LABEL = {
    "B0": "B0 单原型基线",
    "B3-H": "B3-H 多原型 + Top-2 近邻",
    "B4-Fusion": "B4-Fusion 仅训练 gate",
    "B4-Full": "B4-Full gate + reject",
    "B4-RejectOnly": "B4-RejectOnly 只训练 reject",
}
COLOR = {"B0": GRAY, "B3-H": "#6C8194", "B4-Fusion": "#B8C2CB", "B4-Full": ORANGE, "B4-RejectOnly": TEAL}
EXPECTED_F1 = {"B0": .733685, "B3-H": .767048, "B4-Fusion": .740441, "B4-Full": .780876, "B4-RejectOnly": .790203}
EXPECTED_OOS = {"B0": .402778, "B3-H": .547222, "B4-Fusion": .470833, "B4-Full": .604167, "B4-RejectOnly": .606944}


def main() -> int:
    setup()
    d = pd.read_csv(SRC)
    x = d[d.encoder == "frozen"].set_index("method").loc[METHODS].reset_index()
    assert len(x) == 5 and x.method.is_unique
    for m in METHODS:
        assert abs(float(x[x.method == m].open_set_macro_f1.iloc[0]) - EXPECTED_F1[m]) < 1e-5
        assert abs(float(x[x.method == m].oos_recall.iloc[0]) - EXPECTED_OOS[m]) < 1e-5
    x["f1_pp"] = x.open_set_macro_f1 * 100
    x["oos_pp"] = x.oos_recall * 100
    x["delta_vs_b3h_pp"] = (x.open_set_macro_f1 - float(x[x.method == "B3-H"].open_set_macro_f1.iloc[0])) * 100
    x["delta_vs_full_pp"] = (x.open_set_macro_f1 - float(x[x.method == "B4-Full"].open_set_macro_f1.iloc[0])) * 100
    x[["encoder", "method", "open_set_macro_f1", "oos_recall", "known_correct_routing", "f1_pp", "oos_pp", "delta_vs_b3h_pp", "delta_vs_full_pp"]].to_csv(
        OUT / "P06_b4_module_ablation_dev_data.csv", index=False
    )
    y = np.arange(len(METHODS))[::-1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.333, 7.5), gridspec_kw={"width_ratios": [2.7, 1.0], "wspace": .05})
    for yy, m in zip(y, METHODS):
        val = float(x[x.method == m].f1_pp.iloc[0])
        ax.plot([70, val], [yy, yy], color="#D7DEE4", lw=2, zorder=1)
        ax.scatter([val], [yy], s=180, color=COLOR[m], edgecolor="white", linewidth=2, zorder=3)
        ax.text(val + .25, yy, f"{val:.1f}%", va="center", ha="left", fontsize=15, fontweight="bold", color=COLOR[m])
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in METHODS], fontsize=14)
    ax.set_xlim(70, 81.6)
    ax.set_xticks([70, 72, 74, 76, 78, 80])
    ax.set_xlabel("Development open-set Macro-F1（%）")
    ax.grid(axis="x", color="#E7ECF0", lw=.8)
    ax.set_axisbelow(True)
    ax.set_title("B4 模块归因：F1 与拒识头", loc="left", pad=12)
    # annotation of the two intended contrasts
    r = x[x.method == "B4-RejectOnly"].iloc[0]
    ax.text(0.98, 0.97, f"RejectOnly − B3-H：+{r.delta_vs_b3h_pp:.2f} pp\nRejectOnly − Full：+{r.delta_vs_full_pp:.2f} pp",
            transform=ax.transAxes, ha="right", va="top", fontsize=12, color=TEAL,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F1F8F7", edgecolor="none"))
    # OOS compact panel
    for yy, m in zip(y, METHODS):
        val = float(x[x.method == m].oos_pp.iloc[0])
        ax2.plot([0, val], [yy, yy], color="#D7DEE4", lw=2)
        ax2.scatter([val], [yy], s=110, color=COLOR[m], edgecolor="white", linewidth=1.5)
        ax2.text(val + 2, yy, f"{val:.1f}%", va="center", fontsize=12, color=NAVY)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 78)
    ax2.set_xticks([0, 25, 50, 75])
    ax2.set_xlabel("OOS Recall（%）")
    ax2.set_title("同批开发的 OOS", loc="left", pad=12)
    ax2.grid(axis="x", color="#E7ECF0", lw=.8)
    ax2.set_axisbelow(True)
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("B4-RejectOnly 的主要收益来自拒识头，而非更复杂的可训练门控", x=.035, y=.965, ha="left", fontsize=22, fontweight="bold", color=NAVY)
    fig.text(.035, .885, "BANKING77 / GTE Frozen / 10-shot / development validation；非最终盲测", fontsize=12, color="#4F5B67")
    fig.subplots_adjust(left=.25, right=.98, top=.80, bottom=.16)
    footer(fig, f"源数据：results/gate5_method_matrix_val.csv；Commit {COMMIT[:12]}；encoder=frozen；5 个方法")
    save(fig, "P06_b4_module_ablation_dev")
    notes("P06_b4_module_ablation_dev", {
        "commit": COMMIT, "input": "results/gate5_method_matrix_val.csv",
        "filter": "encoder == 'frozen'; 5 methods", "n": 5,
        "partition": "BANKING77 GTE Frozen development validation; not historical test",
        "values": {m: {"f1": EXPECTED_F1[m], "oos": EXPECTED_OOS[m]} for m in METHODS},
        "deltas_pp": {"RejectOnly_minus_B3H": .023154176011595085, "RejectOnly_minus_Full": .009327001705102367},
        "caption": "B4-RejectOnly 的主要收益来自拒识头，而非更复杂的可训练门控",
        "conclusion": "B3-H 带来基础提升；Fusion 单独训练 gate 不提升；RejectOnly 超过 Full，收益主要来自 reject head。",
    })
    print("P06_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
