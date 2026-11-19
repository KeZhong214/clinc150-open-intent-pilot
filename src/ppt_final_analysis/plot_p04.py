"""P0-1: B0 / 10-shot cross-dataset baseline heatmap (main + OOS appendix)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from src.ppt_final_analysis.common import BLUE, COMMIT, GRAY, LIGHT, NAVY, OUT, TEAL, footer, notes, save, setup

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "gate3_matrix.csv"
DATASETS = ["clinc150", "banking77", "massive"]
DS_LABEL = {"clinc150": "CLINC150（历史）", "banking77": "BANKING77", "massive": "MASSIVE en-US"}
MODELS = ["bge", "e5", "gte", "qwen"]
MODEL_LABEL = {"bge": "BGE-small", "e5": "E5-base", "gte": "GTE-ModernBERT", "qwen": "Qwen3-Emb-0.6B"}
HINTS = {"clinc150": "多领域、历史探索性", "banking77": "单领域、细粒度银行意图", "massive": "多领域、意图+槽位"}


def main() -> int:
    setup()
    d = pd.read_csv(SRC)
    assert len(d) == 72 and not d.duplicated(["dataset", "model", "shot", "method"]).any()
    x = d[(d.method == "B0") & (d.shot == 10)].copy()
    assert len(x) == 12
    x["dataset_label"] = x.dataset.map(DS_LABEL)
    x["model_label"] = x.model.map(MODEL_LABEL)
    x = x.set_index(["dataset", "model"]).loc[[(ds, m) for ds in DATASETS for m in MODELS]].reset_index()
    x["row_best"] = x.groupby("dataset").open_set_macro_f1.transform("max").eq(x.open_set_macro_f1)
    x["historical"] = x.dataset.eq("clinc150")
    x[["dataset", "dataset_label", "model", "model_label", "shot", "method", "open_set_macro_f1", "oos_recall", "row_best", "historical"]].to_csv(
        OUT / "P04_baseline_heatmap_10shot_B0_data.csv", index=False
    )
    x[["dataset", "dataset_label", "model", "model_label", "shot", "method", "oos_recall", "open_set_macro_f1", "historical"]].to_csv(
        OUT / "A01_oos_heatmap_10shot_B0_data.csv", index=False
    )
    # Explicit numeric contract from the task's reference values.
    expected = {
        ("banking77", "bge"): 0.8123, ("banking77", "e5"): 0.7684, ("banking77", "gte"): 0.8081, ("banking77", "qwen"): 0.7821,
        ("clinc150", "qwen"): 0.9226, ("massive", "qwen"): 0.7982,
    }
    for (ds, m), v in expected.items():
        got = float(x[(x.dataset == ds) & (x.model == m)].open_set_macro_f1.iloc[0])
        assert abs(got - v) < 0.0001, (ds, m, got, v)
    mat = x.pivot(index="dataset", columns="model", values="open_set_macro_f1").loc[DATASETS, MODELS].to_numpy()
    oos = x.pivot(index="dataset", columns="model", values="oos_recall").loc[DATASETS, MODELS].to_numpy()

    def draw(matrix, title, cbar_label, stem, vmin, vmax, read_note):
        fig, ax = plt.subplots(figsize=(13.333, 7.5))
        cmap = LinearSegmentedColormap.from_list("paper_teal", ["#F4F7F8", "#D9E6E4", TEAL])
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["BGE-\nsmall", "E5-\nbase", "GTE-\nModernBERT", "Qwen3-\nEmb-0.6B"], fontsize=13, linespacing=1.15)
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels([DS_LABEL[d] for d in DATASETS], fontsize=16)
        ax.tick_params(length=0)
        for i in range(3):
            for j in range(4):
                val = matrix[i, j]
                txt = f"{val*100:.1f}%"
                ax.text(j, i, txt, ha="center", va="center", fontsize=17, fontweight="bold",
                        color="white" if val >= (vmin + vmax) / 2 else NAVY)
                if i < 3 and j < 4:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="white", lw=2))
                    if abs(val - matrix[i].max()) < 1e-12:
                        ax.add_patch(Rectangle((j - .47, i - .47), .94, .94, fill=False, edgecolor=NAVY, lw=2.2))
        for i, ds in enumerate(DATASETS):
            ax.text(4.08, i, HINTS[ds], va="center", ha="left", fontsize=13, color="#4F5B67")
        ax.set_xlim(-.5, 5.55)
        ax.set_title("")
        fig.suptitle(title, x=.035, y=.965, ha="left", va="top", fontsize=23, fontweight="bold", color=NAVY)
        fig.text(.035, .885, "固定条件切片：method=B0 / shot=10；不是72组总排名", fontsize=12, color="#4F5B67")
        # Reserve breathing room for the two-line model labels, title and source note.
        fig.subplots_adjust(left=.16, right=.76, top=.80, bottom=.20)
        cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.12)
        cb.set_label(cbar_label, fontsize=12)
        cb.ax.tick_params(labelsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.text(.035, .055, read_note, fontsize=12, color="#4F5B67")
        footer(fig, f"源数据：results/gate3_matrix.csv；Commit {COMMIT[:12]}；历史探索性结果；行最优仅作行内对照")
        save(fig, stem)
        is_oos = stem.startswith("A01")
        value_col = "oos_recall" if is_oos else "open_set_macro_f1"
        value_name = "OOS Recall" if is_oos else "open_set_macro_f1"
        notes(stem, {
            "commit": COMMIT, "input": "results/gate3_matrix.csv",
            "filter": "method == 'B0' and shot == 10", "rows": 12,
            "key": f"dataset × model; values {value_name}",
            "row_best": {ds: MODEL_LABEL[x[x.dataset == ds].sort_values(value_col, ascending=False).model.iloc[0]] for ds in DATASETS},
            "verification": "12 unique (dataset,model); B0/10-shot; same 12 cells as the matrix; no new samples",
            "partition": "CLINC150 historical exploratory; BANKING77/MASSIVE validation matrix; not a test leaderboard",
            "caption": title,
            "conclusion": "任务域差异显著，模型并非绝对通吃；BANKING77 的细粒度银行意图更适合客服主线。" if not is_oos else "OOS Recall 也随任务域与模型变化；MASSIVE 的 BGE/GTE 接近 0，不能把 F1 单图当作全部能力。",
        })

    draw(mat, "四个编码器在三个数据集上的开放集 F1（B0 / 10-shot）", "Open-set Macro-F1", "P04_baseline_heatmap_10shot_B0", .65, .95,
         "读法：同色阶 65%–95%；深色=更高 F1。不同数据集不作总体平均。")
    draw(oos, "四个编码器在三个数据集上的 OOS Recall（B0 / 10-shot）", "OOS Recall", "A01_oos_heatmap_10shot_B0", 0.0, 1.0,
         "读法：同色阶 0%–100%；深色=更高 OOS Recall。不同数据集不作总体平均。")
    print("P04_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
