"""G7-A: deterministic two-level development split.

dev_threshold: 10 known samples/class (400) + 9 whole unknown classes x 40 (360)
dev_select:    the other 10 known/class (400) + the other 9 unknown classes (360)

No text or sample_id overlaps. Unknown classes are disjoint. This is an internal
robustness split, not a new blind test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest"
RESULT = ROOT / "results"
SEED = 42


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    m = pd.read_csv(MANIFEST / "gate3_banking77_manifest.csv")
    known = m[m.role == "calibration_known"].copy()
    unknown = m[m.role == "dev_unknown"].copy()
    rng = np.random.default_rng(SEED)

    threshold_ids, select_ids = [], []
    for intent in sorted(known.intent.unique()):
        rows = known[known.intent == intent].sort_values("sample_id")
        ids = rows.sample_id.tolist()
        rng.shuffle(ids)
        threshold_ids.extend(ids[:10])
        select_ids.extend(ids[10:20])
    unknown_classes = sorted(unknown.intent.unique())
    rng.shuffle(unknown_classes)
    thr_classes = unknown_classes[:9]
    sel_classes = unknown_classes[9:]
    thr_unknown = unknown[unknown.intent.isin(thr_classes)]
    sel_unknown = unknown[unknown.intent.isin(sel_classes)]
    thr = pd.concat([known[known.sample_id.isin(threshold_ids)], thr_unknown]).sort_values(["role", "intent", "sample_id"])
    sel = pd.concat([known[known.sample_id.isin(select_ids)], sel_unknown]).sort_values(["role", "intent", "sample_id"])
    assert len(thr) == 760 and len(sel) == 760
    assert not set(thr.sample_id) & set(sel.sample_id)
    assert not set(thr.text_sha256) & set(sel.text_sha256)
    assert set(thr_unknown.intent).isdisjoint(set(sel_unknown.intent))
    assert thr.groupby("intent").size().max() == 40

    thr_path = MANIFEST / "gate7_dev_threshold.csv"
    sel_path = MANIFEST / "gate7_dev_select.csv"
    thr.to_csv(thr_path, index=False)
    sel.to_csv(sel_path, index=False)
    payload = {
        "seed": SEED,
        "label": "internal_two_level_development_not_blind",
        "dev_threshold": {
            "n": len(thr),
            "n_known": int((thr.role == "calibration_known").sum()),
            "n_unknown": int((thr.role == "dev_unknown").sum()),
            "known_per_class": 10,
            "unknown_classes": thr_classes,
            "path": str(thr_path.relative_to(ROOT)),
            "sha256": sha_file(thr_path),
            "sample_ids_sha256": hashlib.sha256("\n".join(thr.sample_id).encode()).hexdigest(),
        },
        "dev_select": {
            "n": len(sel),
            "n_known": int((sel.role == "calibration_known").sum()),
            "n_unknown": int((sel.role == "dev_unknown").sum()),
            "known_per_class": 10,
            "unknown_classes": sel_classes,
            "path": str(sel_path.relative_to(ROOT)),
            "sha256": sha_file(sel_path),
            "sample_ids_sha256": hashlib.sha256("\n".join(sel.sample_id).encode()).hexdigest(),
        },
        "checks": {
            "sample_id_overlap": 0,
            "text_sha256_overlap": 0,
            "unknown_class_overlap": 0,
            "n_known_classes": int(known.intent.nunique()),
            "n_unknown_classes_total": int(unknown.intent.nunique()),
        },
    }
    (RESULT / "gate7_dev_split.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("GATE7_DEV_SPLIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
