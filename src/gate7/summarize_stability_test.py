"""Summarize the exploratory stability test across seeds."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def main() -> int:
    d = pd.read_csv(RESULTS / "gate7_stability_historical_test.csv")
    metrics = ["open_set_macro_f1", "oos_recall", "known_correct_routing", "unknown_auroc"]
    rows = []
    for (enc, method), g in d.groupby(["encoder", "method"]):
        row = {"encoder": enc, "method": method, "n_seeds": int(g.seed.nunique())}
        for m in metrics:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=1))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "gate7_stability_test_summary.csv", index=False)
    print(out.round(5).to_string(index=False))
    print("GATE7_STABILITY_TEST_SUMMARY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
