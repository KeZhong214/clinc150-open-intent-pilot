"""Re-select Gate-4 test cases from saved predictions (no model/test rerun)."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "manifest"


def main() -> int:
    preds = pd.read_csv(RESULTS / "gate4_b4_test_predictions.csv")
    text = {}
    for row in csv.DictReader((MANIFEST / "gate3_banking77_manifest.csv").open("r", encoding="utf-8")):
        if row["role"] in ("known_test", "final_unknown"):
            text[row["sample_id"]] = row["text"]
    categories = [
        ("known_correct", (preds.role == "known_test") & (preds.pred_label == preds.true_label)),
        ("known_rejected", (preds.role == "known_test") & (preds.pred_label == "OOS")),
        ("known_misclassified", (preds.role == "known_test") & (preds.pred_label != "OOS") & (preds.pred_label != preds.true_label)),
        ("oos_correct", (preds.role == "final_unknown") & (preds.pred_label == "OOS")),
        ("oos_false_accept", (preds.role == "final_unknown") & (preds.pred_label != "OOS")),
    ]
    rows = []
    for category, mask in categories:
        sub = preds[mask]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        rows.append(
            {
                "category": category,
                "sample_id": r["sample_id"],
                "text": text.get(r["sample_id"], ""),
                "true_label": r["true_label"],
                "pred_label": r["pred_label"],
                "accepted": int(r["accepted"]),
                "reject_score": r["score"],
                "tau": r["tau"],
                "top3": f"{r['top1']}:{r['s1']:.4f} | {r['top2']}:{r['s2']:.4f} | {r['top3']}",
                "gate_top1": r["gate_top1"],
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "gate4_b4_test_cases.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("GATE4_CASE_REFRESH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
