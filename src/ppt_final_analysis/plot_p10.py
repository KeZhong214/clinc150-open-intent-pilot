"""P0-4: Agent v1 corrected TSR, three routers, 120 simulated tasks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ppt_final_analysis.common import BLUE, COMMIT, GRAY, NAVY, ORANGE, OUT, TEAL, footer, notes, save, setup

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "gate8_2_v1_rescore_summary.csv"
LABEL = {
    "GTE-Frozen-RejectOnly": "GTE Frozen\nRejectOnly",
    "Qwen-Frozen-RejectOnly": "Qwen Frozen\nRejectOnly",
    "Qwen-Plain42-RejectOnly": "Qwen Plain42\nRejectOnly",
}
COLOR = {"GTE-Frozen-RejectOnly": GRAY, "Qwen-Frozen-RejectOnly": BLUE, "Qwen-Plain42-RejectOnly": TEAL}


def main() -> int:
    setup()
    d = pd.read_csv(SRC)
    assert len(d) == 3 and (d.n == 120).all()
    d = d.set_index("model")
    order = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly"]
    x = d.loc[order].reset_index()
    x["tsr_pct"] = x.v1_corrected_tsr * 100
    x["success_count"] = (x.v1_corrected_tsr * x.n).round().astype(int)
    x["original_count"] = (x.v1_original_tsr * x.n).round().astype(int)
    x["delta_vs_gte_pp"] = (x.v1_corrected_tsr - float(d.loc["GTE-Frozen-RejectOnly", "v1_corrected_tsr"])) * 100
    assert x.success_count.tolist() == [72, 72, 85]
    assert x.original_count.tolist() == [69, 69, 83]
    x[["model", "n", "v1_original_tsr", "v1_corrected_tsr", "success_count", "original_count", "delta_vs_gte_pp"]].to_csv(
        OUT / "P10_agent_v1_corrected_tsr_data.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ys = np.arange(3)[::-1]
    for y, row in zip(ys, x.itertuples()):
        ax.plot([0, row.tsr_pct], [y, y], color="#D7DEE4", lw=2.2, zorder=1)
        ax.scatter([row.tsr_pct], [y], s=210, color=COLOR[row.model], edgecolor="white", linewidth=2, zorder=3)
        ax.text(row.tsr_pct + 1.5, y, f"{row.tsr_pct:.1f}%", va="center", fontsize=20, color=COLOR[row.model], fontweight="bold")
        ax.text(97, y, f"{row.success_count}/120\n(v1 原 {row.original_count})", va="center", ha="right", fontsize=12, color="#4F5B67")
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[m] for m in order], fontsize=15, linespacing=1.2)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("任务成功率 TSR（%）")
    ax.grid(axis="x", color="#E7ECF0", lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Gate 8.2 · 120 条模拟业务任务 · 首次冻结轨迹修正评分", x=.035, y=.965, ha="left", fontsize=22, fontweight="bold", color=NAVY)
    fig.text(.035, .885, "v1 corrected TSR；非 DeepSeek API 模式效果，非真实银行线上数据；n=120/模型", fontsize=12, color="#4F5B67")
    fig.text(.64, .84, "Qwen Plain42 − GTE：+10.83 pp\n模拟任务，不宣称统计显著", fontsize=12, color=TEAL, ha="left", va="center")
    fig.subplots_adjust(left=.20, right=.98, top=.80, bottom=.16)
    footer(fig, f"源数据：results/gate8_2_v1_rescore_summary.csv；Commit {COMMIT[:12]}；v2 的 74.17% 未混入正式成绩")
    save(fig, "P10_agent_v1_corrected_tsr")
    notes("P10_agent_v1_corrected_tsr", {
        "commit": COMMIT, "input": "results/gate8_2_v1_rescore_summary.csv",
        "filter": "all 3 model rows; n=120", "metric": "v1_corrected_tsr",
        "values_pct": {m: float(d.loc[m, "v1_corrected_tsr"] * 100) for m in order},
        "success_counts": {m: int(round(d.loc[m, "v1_corrected_tsr"] * 120)) for m in order},
        "original_v1_counts": {m: int(round(d.loc[m, "v1_original_tsr"] * 120)) for m in order},
        "delta_qwen_plain_vs_gte_pp": 10.833333,
        "partition": "same-author frozen simulated test; v1 original traces, post-hoc scoring correction; not DeepSeek API",
        "caption": "Gate 8.2 · 120 条模拟业务任务 · 首次冻结轨迹修正评分",
        "conclusion": "Qwen Plain42 在这批模拟任务上的 TSR 更高（85/120 vs 72/120），但样本为模拟测试，不宣称显著优越。",
    })
    print("P10_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
