"""P0-3: paired B4 seed deltas, RejectOnly minus Full, Gate7 dev stability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ppt_final_analysis.common import BLUE, COMMIT, GRAY, NAVY, ORANGE, OUT, TEAL, footer, notes, save, setup

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "gate7_stability_b4_results.csv"
ENCODERS = ["gte_frozen", "gte_lora42", "gte_lora43", "gte_lora44", "qwen_frozen", "qwen_plain42"]
LABEL = {
    "gte_frozen": "GTE Frozen",
    "gte_lora42": "GTE LoRA(42)",
    "gte_lora43": "GTE LoRA(43)",
    "gte_lora44": "GTE LoRA(44)",
    "qwen_frozen": "Qwen Frozen",
    "qwen_plain42": "Qwen Plain(42)",
}
COLOR = {"gte_frozen": TEAL, "gte_lora42": "#4FA79D", "gte_lora43": "#2B8C83", "gte_lora44": "#1E6F68", "qwen_frozen": BLUE, "qwen_plain42": "#6A95C9"}


def main() -> int:
    setup()
    d = pd.read_csv(SRC)
    assert len(d) == 36 and d.duplicated(["encoder", "method", "seed"]).sum() == 0
    assert set(d.encoder) == set(ENCODERS)
    assert set(d.method) == {"B4-Full", "B4-RejectOnly"} and set(d.seed) == {42, 43, 44}
    p = d.pivot_table(index=["encoder", "seed"], columns="method", values="dev_open_set_macro_f1").reset_index()
    p["delta_pp"] = (p["B4-RejectOnly"] - p["B4-Full"]) * 100
    p = p[p.encoder.isin(ENCODERS)].sort_values(["encoder", "seed"])
    p.to_csv(OUT / "P08_b4_paired_seed_deltas_data.csv", index=False)
    s = p.groupby("encoder").delta_pp.agg(["mean", "std", "count"]).reindex(ENCODERS)
    s.to_csv(OUT / "P08_b4_paired_seed_deltas_summary.csv")
    expected = [0.393823, 0.636996, 1.255517, 2.391287, 1.983939, 1.049985]
    for enc, exp in zip(ENCODERS, expected):
        assert abs(float(s.loc[enc, "mean"]) - exp) < 1e-5, (enc, s.loc[enc, "mean"], exp)
    assert (s["count"] == 3).all()

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ys = np.arange(len(ENCODERS))[::-1]
    for y, enc in zip(ys, ENCODERS):
        row = p[p.encoder == enc]
        vals = row.delta_pp.to_numpy()
        mean = float(s.loc[enc, "mean"])
        sd = float(s.loc[enc, "std"])
        ax.plot([mean - sd, mean + sd], [y, y], color=COLOR[enc], lw=2.0, alpha=.55, zorder=1)
        ax.scatter(vals, np.full(len(vals), y), s=62, color=COLOR[enc], edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter([mean], [y], marker="D", s=150, color=COLOR[enc], edgecolor="white", linewidth=1.6, zorder=4)
        label_x = max(mean + sd, float(vals.max())) + .10
        ax.text(label_x, y, f"{mean:+.2f} ± {sd:.2f} pp", va="center", fontsize=12, color=COLOR[enc], fontweight="bold")
    ax.axvline(0, color=NAVY, lw=1.1)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[e] for e in ENCODERS], fontsize=14)
    ax.set_xlim(-0.15, 3.15)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Δ Open-set Macro-F1（RejectOnly − Full，百分点）")
    ax.grid(axis="x", color="#E7ECF0", lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("B4-RejectOnly 相对 Full 的配对增益：6 组编码器均值为正", x=.035, y=.965, ha="left", fontsize=22, fontweight="bold", color=NAVY)
    fig.text(.035, .885, "Gate7 dev_select；12 个 B4 头配置 × 3 seeds；小点=每个 B4 头 seed，菱形=均值，短线=±1 sample std", fontsize=12, color="#4F5B67")
    fig.text(.035, .84, "注意：LoRA(42/43/44) 是编码器训练种子；横向 42/43/44 是 B4 头训练种子，两者不是同一层", fontsize=11, color="#6B7785")
    fig.subplots_adjust(left=.23, right=.97, top=.79, bottom=.16)
    footer(fig, f"源数据：results/gate7_stability_b4_results.csv；Commit {COMMIT[:12]}；每行 n=3；未画 95% CI")
    save(fig, "P08_b4_paired_seed_deltas")
    notes("P08_b4_paired_seed_deltas", {
        "commit": COMMIT, "input": "results/gate7_stability_b4_results.csv",
        "keys": "36 rows, 36 unique (encoder,method,seed); 6 encoders × 2 methods × 3 B4 seeds",
        "pairing": "same encoder × same B4 seed: delta_pp = (RejectOnly - Full) * 100",
        "summary": {e: {"mean_pp": float(s.loc[e, "mean"]), "sample_std_pp": float(s.loc[e, "std"]), "n": 3} for e in ENCODERS},
        "partition": "Gate7 dev_select; B4-head training seeds; support fixed",
        "caption": "B4-RejectOnly 相对 Full 的配对增益：6 组编码器均值为正",
        "conclusion": "6 组配置的 RejectOnly 均值为正，但效应大小随编码器和 seed 变化；不能夸大为跨支持集普遍稳定。",
    })
    print("P08_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
