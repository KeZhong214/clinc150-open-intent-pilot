"""G7 stability summary: B4-RejectOnly / B4-Full across seeds."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def main() -> int:
    df = pd.read_csv(RESULTS / "gate7_stability_b4_results.csv")
    metrics = ["dev_open_set_macro_f1", "dev_oos_recall", "dev_known_correct_routing", "dev_unknown_auroc"]
    rows = []
    for (enc, method), g in df.groupby(["encoder", "method"]):
        row = {"encoder": enc, "method": method, "seeds": ",".join(str(x) for x in sorted(g.seed.tolist()))}
        for m in metrics:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=1))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "gate7_stability_summary.csv", index=False)
    shas = []
    for p in sorted(RESULTS.glob("gate7_stability_*.pt")):
        shas.append({"checkpoint": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    pd.DataFrame(shas).to_csv(RESULTS / "gate7_stability_checkpoint_hashes.csv", index=False)
    print(out.round(5).to_string(index=False))
    print("GATE7_STABILITY_SUMMARY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
