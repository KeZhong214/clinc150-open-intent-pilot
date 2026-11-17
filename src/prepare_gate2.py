"""Freeze the Gate-2 hard-OOS class split and evaluation manifests.

Rule:
  * Keep the 20 Gate-1 known intents fixed.
  * The remaining 130 CLINC150 intents are unsupported/hard-OOS.
  * Rank all 130 by SHA-256("hard-oos|42|<intent>"); first 65 are development
    unknown classes, last 65 are held-out final-test unknown classes.
  * Development uses only the 65 development classes' official val rows.
  * Final test uses only the 65 held-out classes' official test rows.
  * No model score or result is read by this script.

This is a separate protocol from the original oos_val/oos_test experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
MANIFEST = PROJECT_ROOT / "data" / "manifest"
SEED = 42
N_DEV = 65
N_FINAL = 65


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"\s+", " ", value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    data = json.loads((RAW / "data_full.json").read_text(encoding="utf-8"))
    selected = json.loads((MANIFEST / "selected_intents.json").read_text(encoding="utf-8"))
    domains = json.loads((RAW / "domains.json").read_text(encoding="utf-8"))

    known = {item["intent"] for item in selected["intents"]}
    all_intents = {intent for values in domains.values() for intent in values}
    intent_to_domain = {
        intent: domain for domain, values in domains.items() for intent in values
    }
    if len(known) != 20:
        raise RuntimeError(f"Expected 20 known intents, got {len(known)}")
    if len(all_intents) != 150:
        raise RuntimeError(f"Expected 150 intents, got {len(all_intents)}")
    remaining = sorted(all_intents - known)
    if len(remaining) != 130:
        raise RuntimeError(f"Expected 130 remaining intents, got {len(remaining)}")

    ranked = sorted(
        remaining,
        key=lambda intent: (sha256_text(f"hard-oos|{SEED}|{intent}"), intent),
    )
    dev_classes = ranked[:N_DEV]
    final_classes = ranked[N_DEV:]
    if len(dev_classes) != N_DEV or len(final_classes) != N_FINAL:
        raise RuntimeError("Hard-OOS class split is not 65/65")
    if set(dev_classes) & set(final_classes):
        raise RuntimeError("Hard-OOS class groups overlap")

    def rows_for(classes: set[str], split: str, role: str) -> list[dict[str, Any]]:
        rows = []
        for index, (text, label) in enumerate(data[split]):
            if label not in classes:
                continue
            rows.append(
                {
                    "sample_id": f"clinc150:{split}:{index:05d}",
                    "source_split": split,
                    "role": role,
                    "class_group": "dev" if role == "hard_oos_val" else "final",
                    "intent": label,
                    "domain": intent_to_domain[label],
                    "text": text,
                    "text_sha256": sha256_text(text),
                    "text_norm_sha256": sha256_text(normalized_text(text)),
                }
            )
        return rows

    dev_rows = rows_for(set(dev_classes), "val", "hard_oos_val")
    final_rows = rows_for(set(final_classes), "test", "hard_oos_test")
    if len(dev_rows) != N_DEV * 20:
        raise RuntimeError(f"Expected {N_DEV * 20} hard-OOS val rows, got {len(dev_rows)}")
    if len(final_rows) != N_FINAL * 30:
        raise RuntimeError(f"Expected {N_FINAL * 30} hard-OOS test rows, got {len(final_rows)}")

    for rows, expected_classes, expected_per_class, name in (
        (dev_rows, set(dev_classes), 20, "hard_oos_val"),
        (final_rows, set(final_classes), 30, "hard_oos_test"),
    ):
        counts = Counter(row["intent"] for row in rows)
        if set(counts) != expected_classes or set(counts.values()) != {expected_per_class}:
            raise RuntimeError(f"Bad class/count layout in {name}: {counts}")

    # Text-level isolation against the known support/calibration/test and
    # original OOS sets.  This audit does not use labels or model scores.
    def hashes(rows: Sequence[dict[str, Any]]) -> set[str]:
        return {row["text_norm_sha256"] for row in rows}

    known_support = [
        row for row in csv.DictReader((MANIFEST / "support_manifest.csv").open("r", encoding="utf-8"))
        if int(row["support_seed"]) == SEED and int(row["shot"]) == 5
    ]
    known_eval = list(csv.DictReader((MANIFEST / "eval_manifest.csv").open("r", encoding="utf-8")))
    known_val_hashes = {
        row["text_norm_sha256"] for row in known_eval if row["role"] == "calibration_known"
    }
    known_test_hashes = {
        row["text_norm_sha256"] for row in known_eval if row["role"] == "final_known_test"
    }
    oos_val_hashes = {
        row["text_norm_sha256"] for row in known_eval if row["role"] == "calibration_oos"
    }
    oos_test_hashes = {
        row["text_norm_sha256"] for row in known_eval if row["role"] == "final_oos_test"
    }
    support_hashes = {row["text_norm_sha256"] for row in known_support}

    checks = {
        "schema_version": "gate2-hard-oos-checks-v1",
        "generated_at_utc": utc_now(),
        "seed": SEED,
        "n_remaining_intents": len(remaining),
        "n_dev_classes": len(dev_classes),
        "n_final_classes": len(final_classes),
        "n_hard_oos_val": len(dev_rows),
        "n_hard_oos_test": len(final_rows),
        "dev_classes": dev_classes,
        "final_classes": final_classes,
        "dev_domain_counts": dict(sorted(Counter(intent_to_domain[i] for i in dev_classes).items())),
        "final_domain_counts": dict(sorted(Counter(intent_to_domain[i] for i in final_classes).items())),
        "text_overlap": {
            "hard_val_vs_known_support": len(hashes(dev_rows) & support_hashes),
            "hard_val_vs_known_val": len(hashes(dev_rows) & known_val_hashes),
            "hard_val_vs_known_test": len(hashes(dev_rows) & known_test_hashes),
            "hard_val_vs_original_oos_val": len(hashes(dev_rows) & oos_val_hashes),
            "hard_val_vs_original_oos_test": len(hashes(dev_rows) & oos_test_hashes),
            "hard_test_vs_known_support": len(hashes(final_rows) & support_hashes),
            "hard_test_vs_known_val": len(hashes(final_rows) & known_val_hashes),
            "hard_test_vs_known_test": len(hashes(final_rows) & known_test_hashes),
            "hard_test_vs_original_oos_val": len(hashes(final_rows) & oos_val_hashes),
            "hard_test_vs_original_oos_test": len(hashes(final_rows) & oos_test_hashes),
            "hard_test_vs_hard_val": len(hashes(final_rows) & hashes(dev_rows)),
        },
        "no_model_results_used_for_split": True,
        "final_test_policy": (
            "hard_oos_test is held out. Do not use it for q_c, threshold selection, "
            "method choice, prompt choice, or any model comparison until the Gate-2 "
            "validation configuration is frozen."
        ),
    }
    if any(value != 0 for value in checks["text_overlap"].values()):
        raise RuntimeError(f"Hard-OOS text overlap is not zero: {checks['text_overlap']}")

    manifest_rows = [
        {
            "sample_id": row["sample_id"],
            "source_split": row["source_split"],
            "role": row["role"],
            "class_group": row["class_group"],
            "intent": row["intent"],
            "domain": row["domain"],
            "text": row["text"],
            "text_sha256": row["text_sha256"],
            "text_norm_sha256": row["text_norm_sha256"],
        }
        for row in dev_rows + final_rows
    ]
    manifest_fields = [
        "sample_id",
        "source_split",
        "role",
        "class_group",
        "intent",
        "domain",
        "text",
        "text_sha256",
        "text_norm_sha256",
    ]
    write_csv(MANIFEST / "gate2_hard_oos_manifest.csv", manifest_rows, manifest_fields)

    split_payload = {
        "schema_version": "gate2-hard-oos-split-v1",
        "frozen_at_utc": utc_now(),
        "dataset_revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
        "selection_seed": SEED,
        "rule": (
            "Rank the 130 non-selected intents by ascending "
            "SHA-256('hard-oos|42|<intent>'); first 65 = development, "
            "last 65 = held-out final test. No model result is used."
        ),
        "known_classes": sorted(known),
        "dev_classes": dev_classes,
        "final_classes": final_classes,
        "eval_manifest": {
            "hard_oos_val": {
                "source_split": "val",
                "classes": 65,
                "rows": len(dev_rows),
                "use": "hard-OOS validation/calibration only",
            },
            "hard_oos_test": {
                "source_split": "test",
                "classes": 65,
                "rows": len(final_rows),
                "use": "held out; only after validation freeze and one-shot evaluation",
            },
        },
        "checks": checks,
    }
    write_json(MANIFEST / "gate2_hard_oos_split.json", split_payload)
    write_json(MANIFEST / "gate2_hard_oos_checks.json", checks)

    print(f"Known classes : {len(known)}")
    print(f"Remaining     : {len(remaining)} -> dev={len(dev_classes)}, final={len(final_classes)}")
    print(f"hard_oos_val  : {len(dev_rows)} rows / {len({r['intent'] for r in dev_rows})} classes")
    print(f"hard_oos_test : {len(final_rows)} rows / {len({r['intent'] for r in final_rows})} classes")
    print(f"Text overlaps : {checks['text_overlap']}")
    print(f"Wrote {MANIFEST / 'gate2_hard_oos_manifest.csv'}")
    print(f"Wrote {MANIFEST / 'gate2_hard_oos_split.json'}")
    print("GATE2_HARD_OOS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
