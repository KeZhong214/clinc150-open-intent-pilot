"""Recover completed stability rows from the interrupted stability log."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "logs" / "gate7_stability_all.log"
OUT = ROOT / "results" / "gate7_stability_b4_results.csv"


def main() -> int:
    rows = []
    seen = {}
    text = LOG.read_text(encoding="utf-8", errors="replace")
    if '"encoder"' not in text:
        text = LOG.read_text(encoding="utf-16", errors="replace")
    for line in text.splitlines():
        if '"encoder"' not in line:
            continue
        start = line.find("{")
        if start < 0:
            continue
        try:
            obj = json.loads(line[start:])
        except Exception:
            continue
        key = (obj["encoder"], obj["method"], obj["seed"])
        seen[key] = obj
    rows = list(seen.values())
    df = pd.DataFrame(rows).sort_values(["encoder", "method", "seed"])
    df.to_csv(OUT, index=False)
    print(df[["encoder", "method", "seed", "dev_open_set_macro_f1", "dev_oos_recall", "dev_known_correct_routing", "dev_unknown_auroc"]].round(5).to_string(index=False))
    print("GATE7_STABILITY_RECOVER_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
