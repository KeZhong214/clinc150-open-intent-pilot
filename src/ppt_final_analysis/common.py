"""Shared visual system for the final PPT analysis figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "ppt_final_analysis"
OUT.mkdir(parents=True, exist_ok=True)
COMMIT = "4c674cbc19ca3c623535033393878d20b5f04ad9"

NAVY = "#18304B"
TEAL = "#2B8C83"
ORANGE = "#D38C44"
GRAY = "#9AA5AF"
BLUE = "#386FB6"
LIGHT = "#F5F7F9"


def setup() -> None:
    preferred = "Microsoft YaHei"
    if not any(f.name == preferred for f in font_manager.fontManager.ttflist):
        preferred = "Noto Sans SC"
    plt.rcParams.update({
        "font.family": preferred,
        "font.size": 15,
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })


def save(fig: plt.Figure, stem: str) -> None:
    # Fixed 16:9 canvas (the PPT target); do not let tight bbox change aspect.
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches=None)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches=None)
    plt.close(fig)


def footer(fig: plt.Figure, source: str) -> None:
    fig.text(0.01, 0.012, source, fontsize=11, color="#6B7785", ha="left", va="bottom")


def notes(stem: str, payload: dict) -> None:
    (OUT / f"{stem}_notes.md").write_text(
        "# 图表说明与数据核验\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
