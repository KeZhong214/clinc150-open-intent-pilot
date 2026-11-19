"""P1-6 A02: v1-corrected family-level success rates, 8 families with n>=5."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ppt_final_analysis.common import BLUE, COMMIT, GRAY, NAVY, OUT, TEAL, footer, notes, save, setup

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "gate8_2_v1_rescore_per_case.csv"
FAMILIES = ["card_status", "card_activation", "transfer_status", "cancel_transfer", "dispute_ticket", "permission", "policy_faq", "oos"]
LABEL = {
    "card_status": "卡片状态",
    "card_activation": "卡片激活",
    "transfer_status": "转账状态",
    "cancel_transfer": "取消转账",
    "dispute_ticket": "争议建单",
    "permission": "权限/越权",
    "policy_faq": "政策 FAQ",
    "oos": "OOS 转人工",
}
MODELS = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly"]
MLABEL = {"GTE-Frozen-RejectOnly": "GTE", "Qwen-Frozen-RejectOnly": "Qwen Frozen", "Qwen-Plain42-RejectOnly": "Qwen Plain42"}
COLOR = {"GTE-Frozen-RejectOnly": GRAY, "Qwen-Frozen-RejectOnly": BLUE, "Qwen-Plain42-RejectOnly": TEAL}


def main() -> int:
    setup()
    d = pd.read_csv(SRC)
    assert len(d) == 360
    assert d.groupby("model").case_id.nunique().eq(120).all()
    assert d.groupby("model").family.nunique().eq(18).all()
    sets = [set(g.case_id) for _, g in d.groupby("model")]
    assert sets[0] == sets[1] == sets[2]
    counts = d[d.family.isin(FAMILIES)].groupby(["model", "family"]).v1_corrected_success.agg(["sum", "count", "mean"]).reset_index()
    piv = counts.pivot(index="family", columns="model", values="mean").loc[FAMILIES, MODELS] * 100
    npiv = counts.pivot(index="family", columns="model", values="count").loc[FAMILIES, MODELS]
    assert (npiv >= 5).all().all()
    counts.to_csv(OUT / "A02_agent_family_regraded_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ys = np.arange(len(FAMILIES))[::-1]
    offsets = {MODELS[0]: -.16, MODELS[1]: 0, MODELS[2]: .16}
    for y, fam in zip(ys, FAMILIES):
        for m in MODELS:
            val = float(piv.loc[fam, m])
            ax.scatter([val], [y + offsets[m]], s=110, color=COLOR[m], edgecolor="white", linewidth=1.2, zorder=3, label=MLABEL[m] if y == ys[0] else None)
        n = int(npiv.loc[fam, MODELS[0]])
        ax.text(101, y, f"n={n}", va="center", ha="left", fontsize=11, color="#5D6B78")
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[f] for f in FAMILIES], fontsize=14)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("v1 corrected 任务成功率（%）")
    ax.grid(axis="x", color="#E7ECF0", lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.97, .845), frameon=False, fontsize=11, ncol=3)
    fig.suptitle("v1 校正后业务族成功率：差距集中在权限、政策 FAQ 与转账", x=.035, y=.965, ha="left", fontsize=22, fontweight="bold", color=NAVY)
    fig.text(.035, .885, "Gate 8.2 v1 corrected；8 个业务族，每族 n≥5；不是 18 族全量热力图；非真实银行数据", fontsize=12, color="#4F5B67")
    fig.subplots_adjust(left=.14, right=.93, top=.80, bottom=.20)
    footer(fig, f"源数据：results/gate8_2_v1_rescore_per_case.csv；Commit {COMMIT[:12]}；三模型均 120 个相同 case")
    save(fig, "A02_agent_family_regraded")
    notes("A02_agent_family_regraded", {
        "commit": COMMIT, "input": "results/gate8_2_v1_rescore_per_case.csv",
        "filter": "8 families with n>=5", "families": FAMILIES,
        "verification": "360 rows; each model 120 unique case_id, 18 families; case sets identical",
        "rates_pct": {f: {m: float(piv.loc[f, m]) for m in MODELS} for f in FAMILIES},
        "counts": {f: int(npiv.loc[f, MODELS[0]]) for f in FAMILIES},
        "partition": "Gate8.2 v1 same-trace corrected; 120 simulated tasks; not v2; not API demo",
        "caption": "v1 校正后业务族成功率：差距集中在权限、政策 FAQ 与转账",
        "conclusion": "政策 FAQ 与权限族差距最大；小样本族（如 cancel_idempotent n=1）未进入本图。",
    })
    print("A02_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
