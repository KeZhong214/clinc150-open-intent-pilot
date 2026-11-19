"""P1-6 A03: v1-corrected failure attribution, mutually exclusive primary labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ppt_final_analysis.common import BLUE, COMMIT, GRAY, NAVY, ORANGE, OUT, TEAL, footer, notes, save, setup

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "gate8_2_v1_rescore_per_case.csv"
MODELS = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly"]
MLABEL = {"GTE-Frozen-RejectOnly": "GTE", "Qwen-Frozen-RejectOnly": "Qwen Frozen", "Qwen-Plain42-RejectOnly": "Qwen Plain42"}
CATS = ["路由/意图错误", "FAQ 事实/引用不足", "工具选择/参数"]
COLOR = {"GTE-Frozen-RejectOnly": "#5E7488", "Qwen-Frozen-RejectOnly": BLUE, "Qwen-Plain42-RejectOnly": TEAL}


def primary_category(row) -> str:
    c = str(row.correction_reason)
    if c.startswith("grounding_fail") or c.startswith("no_grounding"):
        return "FAQ 事实/引用不足"
    if row.v1_failure_reason == "routing":
        return "路由/意图错误"
    if row.v1_failure_reason == "tool_selection_or_args":
        return "工具选择/参数"
    return "其他"


def main() -> int:
    setup()
    d = pd.read_csv(SRC)
    f = d[d.v1_corrected_success == False].copy()
    assert len(f) == 131
    f["category"] = f.apply(primary_category, axis=1)
    assert set(f.category) <= set(CATS)
    counts = f.groupby(["model", "category"]).size().reset_index(name="count")
    counts.to_csv(OUT / "A03_agent_failure_attribution_data.csv", index=False)
    failed_n = f.groupby("model").size().to_dict()
    # Explicit count contract.
    expected = {
        ("GTE-Frozen-RejectOnly", "路由/意图错误"): 40,
        ("GTE-Frozen-RejectOnly", "FAQ 事实/引用不足"): 6,
        ("GTE-Frozen-RejectOnly", "工具选择/参数"): 2,
        ("Qwen-Frozen-RejectOnly", "路由/意图错误"): 39,
        ("Qwen-Frozen-RejectOnly", "FAQ 事实/引用不足"): 6,
        ("Qwen-Frozen-RejectOnly", "工具选择/参数"): 3,
        ("Qwen-Plain42-RejectOnly", "路由/意图错误"): 27,
        ("Qwen-Plain42-RejectOnly", "FAQ 事实/引用不足"): 7,
        ("Qwen-Plain42-RejectOnly", "工具选择/参数"): 1,
    }
    for (m, cat), n in expected.items():
        got = int(counts[(counts.model == m) & (counts.category == cat)]["count"].iloc[0])
        assert got == n, (m, cat, got, n)

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ys = np.arange(len(CATS))[::-1]
    h = .22
    for j, m in enumerate(MODELS):
        vals = [int(counts[(counts.model == m) & (counts.category == c)]["count"].sum()) for c in CATS]
        off = (j - 1) * h
        ax.barh(ys + off, vals, height=h * .9, color=COLOR[m], alpha=1.0, edgecolor="white", linewidth=1)
        for y, v in zip(ys + off, vals):
            ax.text(v + .5, y, str(v), va="center", fontsize=11, color=NAVY)
    # Model legend; failure category is already the y-axis label.
    from matplotlib.patches import Patch
    model_handles = [Patch(facecolor=COLOR[m], label=MLABEL[m]) for m in MODELS]
    fig.legend(handles=model_handles, loc="upper right", bbox_to_anchor=(.97, .845), frameon=False, fontsize=11, ncol=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(CATS, fontsize=14)
    ax.set_xlim(0, 47)
    ax.set_xlabel("失败案例数（单标签主因；GTE 48 / Qwen Frozen 48 / Qwen Plain42 35）")
    ax.grid(axis="x", color="#E7ECF0", lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("v1 校正后失败归因：路由/意图错误占 106/131", x=.035, y=.965, ha="left", fontsize=22, fontweight="bold", color=NAVY)
    fig.text(.035, .885, "Gate 8.2 v1 corrected；从同一 120 条轨迹重算；未使用 v2 聚合或旧 failures CSV", fontsize=12, color="#4F5B67")
    fig.subplots_adjust(left=.17, right=.97, top=.80, bottom=.16)
    footer(fig, f"源数据：results/gate8_2_v1_rescore_per_case.csv；Commit {COMMIT[:12]}；失败 n=131；互斥主因")
    save(fig, "A03_agent_failure_attribution")
    notes("A03_agent_failure_attribution", {
        "commit": COMMIT, "input": "results/gate8_2_v1_rescore_per_case.csv",
        "filter": "v1_corrected_success == False; n=131", "categories": CATS,
        "priority": "1) correction_reason grounding_fail/no_grounding -> FAQ; 2) v1_failure_reason routing -> route; 3) tool_selection_or_args -> tool; else other",
        "counts": {f"{m}|{c}": v for (m, c), v in expected.items()}, "failed_by_model": failed_n,
        "partition": "Gate8.2 v1 corrected failures only; single primary label, no multi-label double count",
        "caption": "v1 校正后失败归因：路由/意图错误占 106/131",
        "conclusion": "失败主要是路由/意图错误（106/131），工具选择仅 6 例；FAQ 事实/引用不足 19 例。",
    })
    print("A03_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
