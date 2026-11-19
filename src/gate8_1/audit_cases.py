"""Audit Gate 8.1 cases for duplication and template diversity."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "gate8_1_case_audit.json"


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def norm(c):
    s = re.sub(r"[^a-z0-9 ]", " ", " ".join(c["turns"]).lower())
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    new = load(ROOT / "data" / "gate8_1" / "benchmark_cases.jsonl")
    old = load(ROOT / "data" / "gate5" / "benchmark_cases.jsonl") + load(ROOT / "data" / "gate8" / "benchmark_cases.jsonl")
    nn = [norm(c) for c in new]
    oo = [norm(c) for c in old]
    exact_old = sorted({c["case_id"] for c, t in zip(new, nn) if t in set(oo)})
    exact_within = sorted({c["case_id"] for i, (c, t) in enumerate(zip(new, nn)) if t in set(nn[:i])})
    max_old = 0.0
    max_old_pair = None
    for c, t in zip(new, nn):
        for o, u in zip(old, oo):
            r = SequenceMatcher(None, t, u).ratio()
            if r > max_old:
                max_old = r
                max_old_pair = [c["case_id"], o.get("case_id")]
    payload = {
        "file": "data/gate8_1/benchmark_cases.jsonl",
        "n_cases": len(new),
        "n_templates": len({c["template_id"] for c in new}),
        "n_families": len({c["family"] for c in new}),
        "exact_duplicates_against_gate5_or_gate8": exact_old,
        "exact_duplicates_within_new": exact_within,
        "max_char_similarity_to_old": round(max_old, 6),
        "max_similarity_pair": max_old_pair,
        "template_ids_unique": len({c["template_id"] for c in new}) == len(new),
        "label": "new_frozen_simulated_test_not_external_blind",
        "note": "Similarity is a diagnostic, not a claim of external independence; cases are authored by the same project author.",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("GATE8_1_CASE_AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
