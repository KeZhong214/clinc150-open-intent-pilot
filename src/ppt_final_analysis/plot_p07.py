"""P1-5: Qwen LoRA encoder training seeds on Gate7 historical exploratory test."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ppt_final_analysis.common import BLUE, COMMIT, GRAY, NAVY, OUT, TEAL, footer, notes, save, setup

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "gate7_final_historical_test_v2.csv"
ORDER = ["plain42", "plain43", "plain44"]
LABEL = {"plain42": "LoRA 42", "plain43": "LoRA 43", "plain44": "LoRA 44"}


def main() -> int:
    setup()
    d = pd.read_csv(SRC)
    q = d[(d.method == "B4-RejectOnly") & d.encoder.isin(["frozen", *ORDER])].copy()
    assert len(q) == 4 and q.encoder.is_unique
    frozen = float(q[q.encoder == "frozen"].open_set_macro_f1.iloc[0])
    lora = q[q.encoder.isin(ORDER)].set_index("encoder").loc[ORDER].reset_index()
    assert abs(frozen - .812998) < 1e-5
    vals = lora.open_set_macro_f1.to_numpy()
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    assert abs(mean - .8255657841471372) < 1e-9
    assert abs(sd - .014399303632915403) < 1e-9
    lora["delta_vs_frozen_pp"] = (lora.open_set_macro_f1 - frozen) * 100
    lora["frozen_baseline"] = frozen
    lora["lora_mean"] = mean
    lora["lora_sample_std"] = sd
    out = lora[["encoder", "seed", "open_set_macro_f1", "oos_recall", "delta_vs_frozen_pp", "frozen_baseline", "lora_mean", "lora_sample_std"]].copy()
    out["role"] = "LoRA"
    frozen_row = pd.DataFrame([{
        "encoder": "frozen", "seed": 42, "open_set_macro_f1": frozen,
        "oos_recall": float(q[q.encoder == "frozen"].oos_recall.iloc[0]),
        "delta_vs_frozen_pp": 0.0, "frozen_baseline": frozen, "lora_mean": mean, "lora_sample_std": sd, "role": "Frozen reference",
    }])
    pd.concat([frozen_row, out], ignore_index=True).to_csv(OUT / "P07_qwen_lora_three_seeds_historical_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    for i, (_, r) in enumerate(lora.iterrows()):
        ax.scatter([i], [r.open_set_macro_f1], s=230, color=BLUE, edgecolor="white", linewidth=2, zorder=2)
        ax.text(i, r.open_set_macro_f1 + .0016, f"{r.open_set_macro_f1*100:.2f}%", ha="center", fontsize=14, color=BLUE, fontweight="bold")
    # mean diamond at the right of the three seeds
    ax.scatter([2.48], [mean], marker="D", s=210, color=TEAL, edgecolor="white", linewidth=1.8, zorder=5)
    ax.errorbar([2.48], [mean], yerr=[sd], fmt="none", ecolor=TEAL, lw=1.8, capsize=5, zorder=4)
    ax.text(2.62, mean, f"三种子均值\n{mean*100:.2f}% ± {sd*100:.2f} pp", va="center", ha="left", fontsize=12, color=TEAL)
    ax.axhline(frozen, color=GRAY, ls=(0, (3, 3)), lw=2)
    ax.text(0.0, frozen + .0014, f"Frozen 参考 {frozen*100:.2f}%", fontsize=12, color="#5D6B78")
    ax.set_xlim(-.5, 3.7)
    ax.set_ylim(.795, .842)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["seed 42", "seed 43", "seed 44"], fontsize=14)
    ax.set_yticks([.80, .81, .82, .83, .84])
    ax.set_yticklabels(["80%", "81%", "82%", "83%", "84%"])
    ax.set_ylabel("Open-set Macro-F1")
    ax.set_xlabel("Qwen LoRA 编码器训练 seed（每点是一个 B4 头）")
    ax.grid(axis="y", color="#E7ECF0", lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Qwen LoRA 三个训练 seed 的 B4-RejectOnly 历史探索性结果", x=.035, y=.965, ha="left", fontsize=22, fontweight="bold", color=NAVY)
    fig.text(.035, .885, "Gate7 阈值修复后的 historical exploratory test；不是新盲测；保留 seed42 负结果，不删点", fontsize=12, color="#4F5B67")
    fig.subplots_adjust(left=.14, right=.97, top=.80, bottom=.16)
    footer(fig, f"源数据：results/gate7_final_historical_test_v2.csv；Commit {COMMIT[:12]}；B4-RejectOnly；n=3 LoRA seeds")
    save(fig, "P07_qwen_lora_three_seeds_historical")
    notes("P07_qwen_lora_three_seeds_historical", {
        "commit": COMMIT, "input": "results/gate7_final_historical_test_v2.csv",
        "filter": "method == 'B4-RejectOnly' and encoder in ['frozen','plain42','plain43','plain44']",
        "rows": 4, "frozen_f1": frozen,
        "seed_values": {r.encoder: {"seed": int(r.seed), "f1": float(r.open_set_macro_f1)} for r in lora.itertuples()},
        "lora_mean": mean, "lora_sample_std_ddof1": sd,
        "caption": "Qwen LoRA 三个训练 seed 的 B4-RejectOnly 历史探索性结果",
        "conclusion": "LoRA 三种子均值 82.56%±1.44 pp，略高于 Frozen 81.30%；seed42 是负结果，保留展示，不能删。",
    })
    print("P07_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
