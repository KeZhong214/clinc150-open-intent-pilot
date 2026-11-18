"""Compare the original Gate7 test with the protocol_fix_v2 run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def main() -> int:
    old = pd.read_csv(RESULTS / "gate7_final_historical_test_pre_fix.csv")
    new = pd.read_csv(RESULTS / "gate7_final_historical_test_v2.csv")
    keys = ["encoder", "method", "seed"]
    metrics = ["tau", "open_set_macro_f1", "oos_recall", "known_correct_routing", "unknown_auroc"]
    a = old[keys + metrics].rename(columns={m: f"old_{m}" for m in metrics})
    b = new[keys + metrics].rename(columns={m: f"new_{m}" for m in metrics})
    d = a.merge(b, on=keys, validate="one_to_one")
    for m in metrics:
        d[f"delta_{m}"] = d[f"new_{m}"] - d[f"old_{m}"]
    d.to_csv(RESULTS / "gate7_protocol_fix_diff.csv", index=False)
    payload = {
        "comparison": "gate7_final_historical_test_pre_fix.csv vs gate7_final_historical_test_v2.csv",
        "label": "historical_exploratory; protocol_fix_v2",
        "n_rows": len(d),
        "b4_threshold_rows": d[d.method == "B4-RejectOnly"][["encoder", "old_tau", "new_tau", "delta_tau", "old_open_set_macro_f1", "new_open_set_macro_f1"]].to_dict("records"),
    }
    (RESULTS / "gate7_protocol_fix_diff.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(d.round(6).to_string(index=False))
    print("GATE7_PROTOCOL_FIX_DIFF_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
