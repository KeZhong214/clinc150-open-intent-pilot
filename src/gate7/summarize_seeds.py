"""Summarize Gate-7 seeds and multi-seed stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", required=True, help="comma-separated dev_eval tags")
    p.add_argument("--group", default="")
    p.add_argument("--out", default="gate7_dev_seed_summary.csv")
    args = p.parse_args(argv)
    rows = []
    for tag in args.tags.split(","):
        path = RESULTS / f"gate7_dev_{tag}.csv"
        if not path.exists():
            print("MISSING", path)
            continue
        d = pd.read_csv(path)
        for _, r in d.iterrows():
            rows.append({"group": args.group or tag, "tag": tag, **r.to_dict()})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / args.out, index=False)
    metrics = ["select_open_set_macro_f1", "select_oos_recall", "select_known_correct_routing", "select_unknown_auroc"]
    agg = df.groupby(["group", "method"])[metrics].agg(["mean", "std"]).reset_index()
    agg.to_csv(RESULTS / "gate7_dev_seed_mean_std.csv", index=False)
    print(df.round(5).to_string(index=False))
    print("\nMEAN/STD\n", agg.round(5).to_string(index=False))
    print("GATE7_SEED_SUMMARY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
