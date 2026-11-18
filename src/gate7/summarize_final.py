"""G7-C: mean/std and paired differences for the three Qwen Plain seeds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def main() -> int:
    test = pd.read_csv(RESULTS / "gate7_final_historical_test.csv")
    test = test[test.method.isin(["B3-H", "B4-RejectOnly"])]
    rows = []
    for method in ("B3-H", "B4-RejectOnly"):
        frozen = test[(test.encoder == "frozen") & (test.method == method)].iloc[0]
        seeds = test[test.encoder.isin(["plain42", "plain43", "plain44"]) & (test.method == method)]
        for metric in ("open_set_macro_f1", "oos_recall", "known_correct_routing", "unknown_auroc"):
            vals = seeds[metric].astype(float)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "frozen": float(frozen[metric]),
                    "seed42": float(vals.iloc[0]),
                    "seed43": float(vals.iloc[1]),
                    "seed44": float(vals.iloc[2]),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)),
                    "delta_mean_vs_frozen": float(vals.mean() - frozen[metric]),
                    "wins_vs_frozen": int((vals > frozen[metric]).sum()),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "gate7_final_summary.csv", index=False)
    print(df.round(5).to_string(index=False))
    print("GATE7_FINAL_SUMMARY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
