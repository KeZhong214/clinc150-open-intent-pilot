"""Combine Chinese direct probe results; translated columns stay blocked without API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODELS = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly"]
SLUG = {m: m.lower().replace("-", "_") for m in MODELS}


def main() -> int:
    frames = []
    for m in MODELS:
        p = RESULTS / f"gate8_2_chinese_direct_{SLUG[m]}_results.csv"
        d = pd.read_csv(p)
        if len(d) != 20:
            raise RuntimeError((m, len(d)))
        d["router"] = m
        d["direct_route_match"] = [
            (str(r.direct_intent) in str(r.expected_intent_any).split("|")) if r.expected_intent_any else None
            for r in d.itertuples()
        ]
        d["direct_oos_ok"] = [
            (not bool(r.direct_accepted)) if bool(r.expected_oos) else True
            for r in d.itertuples()
        ]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(RESULTS / "gate8_2_chinese_results.csv", index=False)
    rows = []
    for m, g in d.groupby("router", sort=False):
        rows.append({
            "router": m,
            "n_cases": len(g),
            "direct_accepted": int(g.direct_accepted.fillna(False).astype(bool).sum()),
            "direct_route_match": int(g.direct_route_match.fillna(False).astype(bool).sum()),
            "direct_oos_ok": int(g.direct_oos_ok.fillna(False).astype(bool).sum()),
            "translation_status": "NOT RUN_API_BLOCKED",
            "translated_route_match": None,
            "note": "Direct Chinese routing probe; translated routing requires a real DeepSeek key and was NOT RUN.",
        })
    s = pd.DataFrame(rows)
    s.to_csv(RESULTS / "gate8_2_chinese_summary.csv", index=False)
    print(s.to_string(index=False))
    print("GATE8_2_CHINESE_SUMMARY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
