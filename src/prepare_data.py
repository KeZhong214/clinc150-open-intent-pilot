"""Validate CLINC150, freeze a 20-intent pilot subset, and build manifests.

This is a Gate-0 script.  It does not encode text, train a classifier, fit a
threshold, or touch a model weight.

Protocol summary
----------------
* Known intents: 2 intents from each of the 10 official domains (20 total).
* Intent selection: deterministic SHA-256 ranking with ``--seed``.
* Support: deterministic SHA-256 ranking inside the official ``train`` split.
* Calibration: all official ``val`` rows for the 20 selected intents, plus
  all 100 official ``oos_val`` rows.
* Final test: all official ``test`` rows for the 20 intents (600), plus all
  1000 official ``oos_test`` rows.  Test rows are indexed, never used for
  selection or tuning.

Usage (from the project root):
    python -m src.download_data
    python -m src.prepare_data --seed 42 --known-classes 20 --per-domain 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_ORDER = ("train", "val", "test", "oos_train", "oos_val", "oos_test")
KNOWN_SPLITS = ("train", "val", "test")
OOS_SPLITS = ("oos_train", "oos_val", "oos_test")
EXPECTED_COUNTS = {
    "train": 15000,
    "val": 3000,
    "test": 4500,
    "oos_train": 100,
    "oos_val": 100,
    "oos_test": 1000,
}
EXPECTED_KNOWN_TRAIN = 100
EXPECTED_KNOWN_VAL = 20
EXPECTED_KNOWN_TEST = 30
OOS_RAW_LABEL = "oos"
OOS_OPEN_SET_LABEL = "OOS"


class ProtocolError(RuntimeError):
    """Raised when the official data or a generated manifest violates protocol."""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    index: int
    text: str
    label: str
    exact_sha256: str
    norm_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_text(text: str) -> str:
    """Conservative normalization used only for duplicate/leakage audits."""
    value = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", value)


def stable_rank(namespace: str, seed: int, *parts: str) -> str:
    """Return a seed-dependent, implementation-independent ranking key.

    Python's ``random`` module is intentionally not used here: SHA-256 ranking
    is independent of Python's RNG implementation and is easy to audit.
    """
    payload = "|".join((namespace, str(seed), *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_and_validate_data(path: Path) -> dict[str, list[Sample]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ProtocolError("data_full.json top level must be an object")
    if set(payload) != set(EXPECTED_COUNTS):
        raise ProtocolError(
            f"Unexpected top-level keys: {sorted(payload)}; expected {sorted(EXPECTED_COUNTS)}"
        )

    samples: dict[str, list[Sample]] = {}
    for split in SPLIT_ORDER:
        rows = payload[split]
        if not isinstance(rows, list):
            raise ProtocolError(f"{split} must be a list")
        if len(rows) != EXPECTED_COUNTS[split]:
            raise ProtocolError(
                f"{split} count {len(rows)} != expected {EXPECTED_COUNTS[split]}"
            )

        split_samples: list[Sample] = []
        for index, row in enumerate(rows):
            if not (isinstance(row, list) and len(row) == 2):
                raise ProtocolError(f"{split}[{index}] is not a [text, label] pair")
            text, label = row
            if not isinstance(text, str) or not text.strip():
                raise ProtocolError(f"{split}[{index}] has empty/non-string text")
            if not isinstance(label, str) or not label.strip():
                raise ProtocolError(f"{split}[{index}] has empty/non-string label")
            split_samples.append(
                Sample(
                    sample_id=f"clinc150:{split}:{index:05d}",
                    split=split,
                    index=index,
                    text=text,
                    label=label,
                    exact_sha256=sha256_text(text),
                    norm_sha256=sha256_text(normalized_text(text)),
                )
            )
        samples[split] = split_samples

    known_from_splits = {
        split: {sample.label for sample in samples[split]} for split in KNOWN_SPLITS
    }
    if not (
        known_from_splits["train"]
        == known_from_splits["val"]
        == known_from_splits["test"]
    ):
        raise ProtocolError("Known intent label sets differ across train/val/test")
    known_intents = known_from_splits["train"]
    if len(known_intents) != 150:
        raise ProtocolError(f"Expected 150 known intents, found {len(known_intents)}")
    if OOS_RAW_LABEL in known_intents:
        raise ProtocolError("The raw OOS label occurs in a known split")

    for split in OOS_SPLITS:
        labels = {sample.label for sample in samples[split]}
        if labels != {OOS_RAW_LABEL}:
            raise ProtocolError(f"{split} labels must all be 'oos', found {sorted(labels)}")

    for split in KNOWN_SPLITS:
        per_label = Counter(sample.label for sample in samples[split])
        for intent in known_intents:
            expected = {
                "train": EXPECTED_KNOWN_TRAIN,
                "val": EXPECTED_KNOWN_VAL,
                "test": EXPECTED_KNOWN_TEST,
            }[split]
            if per_label[intent] != expected:
                raise ProtocolError(
                    f"{split}/{intent} has {per_label[intent]} rows, expected {expected}"
                )

    return samples


def validate_domains(domains_path: Path, known_intents: set[str]) -> dict[str, list[str]]:
    raw = read_json(domains_path)
    if not isinstance(raw, dict):
        raise ProtocolError("domains.json top level must be an object")

    domains: dict[str, list[str]] = {}
    for domain, intents in raw.items():
        if not isinstance(domain, str) or not domain:
            raise ProtocolError("Domain names must be non-empty strings")
        if not isinstance(intents, list) or not intents:
            raise ProtocolError(f"Domain {domain!r} must contain a non-empty list")
        if any(not isinstance(intent, str) or not intent for intent in intents):
            raise ProtocolError(f"Domain {domain!r} contains an invalid intent name")
        domains[domain] = list(intents)

    flat = [intent for intents in domains.values() for intent in intents]
    if len(flat) != len(set(flat)):
        raise ProtocolError("domains.json assigns at least one intent to multiple domains")
    if set(flat) != known_intents:
        missing = sorted(known_intents - set(flat))
        extra = sorted(set(flat) - known_intents)
        raise ProtocolError(f"domains.json mismatch; missing={missing}, extra={extra}")
    return domains


def select_intents(
    domains: dict[str, list[str]],
    *,
    seed: int,
    per_domain: int,
    expected_total: int,
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    if per_domain < 1:
        raise ProtocolError("--per-domain must be >= 1")
    if per_domain * len(domains) != expected_total:
        raise ProtocolError(
            f"per-domain ({per_domain}) * domains ({len(domains)}) != "
            f"known-classes ({expected_total}); refusing to silently change the design"
        )

    selected_by_domain: dict[str, list[str]] = {}
    intent_to_domain: dict[str, str] = {}
    for domain in sorted(domains):
        ranked = sorted(
            domains[domain],
            key=lambda intent: (
                stable_rank("class-selection", seed, domain, intent),
                intent,
            ),
        )
        chosen = ranked[:per_domain]
        selected_by_domain[domain] = chosen
        for intent in chosen:
            intent_to_domain[intent] = domain

    selected = [intent for domain in sorted(selected_by_domain) for intent in selected_by_domain[domain]]
    if len(selected) != expected_total or len(set(selected)) != expected_total:
        raise ProtocolError(
            f"Selected {len(selected)} intents ({len(set(selected))} unique), "
            f"expected {expected_total} unique"
        )
    return selected, intent_to_domain, selected_by_domain


def select_support(
    samples: dict[str, list[Sample]],
    selected: Sequence[str],
    *,
    support_seeds: Sequence[int],
    shots: Sequence[int],
    excluded_norm_hashes: set[str],
    apply_filter: bool,
) -> tuple[dict[tuple[int, int], list[tuple[int, Sample]]], list[dict[str, Any]]]:
    max_shot = max(shots)
    train_by_intent: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples["train"]:
        train_by_intent[sample.label].append(sample)

    ranked_by_key: dict[tuple[int, str], list[tuple[int, Sample]]] = {}
    excluded: list[dict[str, Any]] = []

    for support_seed in support_seeds:
        for intent in sorted(selected):
            candidates = sorted(train_by_intent[intent], key=lambda sample: sample.sample_id)
            if not candidates:
                raise ProtocolError(f"No train rows for {intent}")

            filtered: list[Sample] = []
            for sample in candidates:
                if apply_filter and sample.norm_sha256 in excluded_norm_hashes:
                    excluded.append(
                        {
                            "support_seed": support_seed,
                            "intent": intent,
                            "sample_id": sample.sample_id,
                            "reason": "normalized text also occurs in an official eval split",
                        }
                    )
                    continue
                filtered.append(sample)

            if len(filtered) < max_shot:
                raise ProtocolError(
                    f"{intent}: only {len(filtered)} leakage-safe train rows; "
                    f"need {max_shot} for max shot"
                )

            ranked = sorted(
                filtered,
                key=lambda sample: (
                    stable_rank("support", support_seed, intent, sample.sample_id),
                    sample.sample_id,
                ),
            )
            ranked_by_key[(support_seed, intent)] = [
                (rank, sample) for rank, sample in enumerate(ranked[:max_shot], start=1)
            ]

    support_sets: dict[tuple[int, int], list[tuple[int, Sample]]] = {}
    for support_seed in support_seeds:
        for shot in shots:
            rows: list[tuple[int, Sample]] = []
            for intent in sorted(selected):
                rows.extend(
                    (rank, sample)
                    for rank, sample in ranked_by_key[(support_seed, intent)]
                    if rank <= shot
                )
            support_sets[(support_seed, shot)] = rows

    for (support_seed, shot), rows in support_sets.items():
        ids = [sample.sample_id for _, sample in rows]
        if len(ids) != len(set(ids)):
            raise ProtocolError(f"Duplicate support IDs for seed={support_seed}, shot={shot}")
        per_intent = Counter(sample.label for _, sample in rows)
        if set(per_intent) != set(selected):
            raise ProtocolError(f"Missing intents in support seed={support_seed}, shot={shot}")
        if any(count != shot for count in per_intent.values()):
            raise ProtocolError(f"Support count != {shot} for seed={support_seed}")

    return support_sets, excluded


def make_eval_rows(
    samples: dict[str, list[Sample]], selected: set[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, role in (
        ("val", "calibration_known"),
        ("test", "final_known_test"),
        ("oos_val", "calibration_oos"),
        ("oos_test", "final_oos_test"),
    ):
        for sample in samples[split]:
            if split in KNOWN_SPLITS and sample.label not in selected:
                continue
            is_oos = split in OOS_SPLITS
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_split": split,
                    "source_index": sample.index,
                    "role": role,
                    "is_oos": int(is_oos),
                    "raw_label": sample.label,
                    "open_set_label": OOS_OPEN_SET_LABEL if is_oos else sample.label,
                    "text": sample.text,
                    "text_sha256": sample.exact_sha256,
                    "text_norm_sha256": sample.norm_sha256,
                }
            )
    return rows


def leak_audit(samples: dict[str, list[Sample]]) -> dict[str, Any]:
    within: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        exact = Counter(sample.exact_sha256 for sample in samples[split])
        norm = Counter(sample.norm_sha256 for sample in samples[split])
        within[split] = {
            "rows": len(samples[split]),
            "exact_duplicate_groups": sum(1 for count in exact.values() if count > 1),
            "exact_extra_rows": sum(count - 1 for count in exact.values() if count > 1),
            "normalized_duplicate_groups": sum(1 for count in norm.values() if count > 1),
            "normalized_extra_rows": sum(count - 1 for count in norm.values() if count > 1),
        }

    index_by_norm: dict[str, list[Sample]] = defaultdict(list)
    for split in SPLIT_ORDER:
        for sample in samples[split]:
            index_by_norm[sample.norm_sha256].append(sample)

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for norm_hash, group in index_by_norm.items():
        splits = sorted({sample.split for sample in group}, key=SPLIT_ORDER.index)
        if len(splits) < 2:
            continue
        for i, split_a in enumerate(splits):
            for split_b in splits[i + 1 :]:
                key = (split_a, split_b)
                pairs.setdefault(
                    key,
                    {"split_a": split_a, "split_b": split_b, "hashes": [], "examples": []},
                )
                entry = pairs[key]
                entry["hashes"].append(norm_hash)
                if len(entry["examples"]) < 5:
                    entry["examples"].append(
                        [
                            {
                                "sample_id": sample.sample_id,
                                "label": sample.label,
                                "text": sample.text,
                            }
                            for sample in group
                            if sample.split in (split_a, split_b)
                        ]
                    )

    pair_list = []
    for key in sorted(pairs, key=lambda value: (SPLIT_ORDER.index(value[0]), SPLIT_ORDER.index(value[1]))):
        entry = pairs[key]
        pair_list.append(
            {
                "split_a": entry["split_a"],
                "split_b": entry["split_b"],
                "n_shared_normalized_hashes": len(set(entry["hashes"])),
                "examples": entry["examples"],
            }
        )

    return {
        "within_split": within,
        "cross_split_normalized_text": {
            "n_shared_hashes_total": sum(
                pair["n_shared_normalized_hashes"] for pair in pair_list
            ),
            "pairs": pair_list,
        },
        "id_overlap_possible": False,
        "id_note": (
            "Sample IDs embed the source split, so ID-set intersection is zero by "
            "construction. Text hashes are the meaningful leakage check."
        ),
    }


def support_eval_overlap(
    support_sets: dict[tuple[int, int], list[tuple[int, Sample]]],
    eval_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    eval_ids = {row["sample_id"] for row in eval_rows}
    eval_hashes = {row["text_norm_sha256"] for row in eval_rows}
    result: dict[str, Any] = {}
    for (support_seed, shot), rows in sorted(support_sets.items()):
        support_ids = {sample.sample_id for _, sample in rows}
        overlap_ids = sorted(support_ids & eval_ids)
        overlap_hashes = sorted(
            {sample.norm_sha256 for _, sample in rows} & eval_hashes
        )
        result[f"seed{support_seed}_shot{shot}"] = {
            "support_rows": len(rows),
            "id_overlap": len(overlap_ids),
            "normalized_text_overlap_hashes": len(overlap_hashes),
            "id_examples": overlap_ids[:5],
            "text_hash_examples": overlap_hashes[:5],
        }
    return result


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "data_full.json",
    )
    parser.add_argument(
        "--domains-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "domains.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifest",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--known-classes", type=int, default=20)
    parser.add_argument("--per-domain", type=int, default=2)
    parser.add_argument("--support-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 10])
    parser.add_argument(
        "--no-support-leakage-filter",
        action="store_true",
        help=(
            "Do not exclude train rows whose normalized text also occurs in an "
            "official eval split. Default is to exclude them and record every "
            "exclusion; official rows themselves are never deleted."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if sorted(args.support_seeds) != [42, 43, 44]:
        raise ProtocolError(
            "Gate 0 manifests are frozen for support seeds [42, 43, 44]; "
            f"got {args.support_seeds}"
        )
    if sorted(args.shots) != [5, 10]:
        raise ProtocolError(f"Gate 0 manifests are frozen for shots [5, 10]; got {args.shots}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data file    : {args.data_file}")
    print(f"Domains file : {args.domains_file}")
    print(f"Output dir   : {out_dir}")
    print(f"Seed         : {args.seed}")

    samples = load_and_validate_data(args.data_file)
    known_intents = {sample.label for sample in samples["train"]}
    domains = validate_domains(args.domains_file, known_intents)
    print(f"Official counts validated: { {split: len(samples[split]) for split in SPLIT_ORDER} }")
    print(f"Known intents: {len(known_intents)}; domains: {len(domains)}")

    selected, intent_to_domain, selected_by_domain = select_intents(
        domains,
        seed=args.seed,
        per_domain=args.per_domain,
        expected_total=args.known_classes,
    )
    selected_set = set(selected)
    print(f"Selected {len(selected)} intents (2 per domain):")
    for domain in sorted(selected_by_domain):
        print(f"  {domain}: {', '.join(selected_by_domain[domain])}")

    leakage = leak_audit(samples)
    n_cross = leakage["cross_split_normalized_text"]["n_shared_hashes_total"]
    print(f"Official cross-split normalized-text collisions: {n_cross}")
    for pair in leakage["cross_split_normalized_text"]["pairs"]:
        print(
            f"  {pair['split_a']} <-> {pair['split_b']}: "
            f"{pair['n_shared_normalized_hashes']} shared normalized text(s)"
        )

    # Deduplicate only for support *candidate* selection. Eval rows and raw data
    # stay exactly as published. This uses text identity, never labels/metrics.
    eval_norm_hashes = {
        sample.norm_sha256
        for split in ("val", "test", "oos_val", "oos_test")
        for sample in samples[split]
    }
    support_sets, excluded = select_support(
        samples,
        selected,
        support_seeds=args.support_seeds,
        shots=args.shots,
        excluded_norm_hashes=eval_norm_hashes,
        apply_filter=not args.no_support_leakage_filter,
    )
    print(
        f"Support sets built for seeds {args.support_seeds} and shots {args.shots}; "
        f"excluded leaky candidates: {len(excluded)}"
    )

    eval_rows = make_eval_rows(samples, selected_set)
    eval_counts = Counter(row["role"] for row in eval_rows)
    expected_roles = {
        "calibration_known": 400,
        "final_known_test": 600,
        "calibration_oos": 100,
        "final_oos_test": 1000,
    }
    if dict(eval_counts) != expected_roles:
        raise ProtocolError(f"Eval counts {dict(eval_counts)} != {expected_roles}")

    overlaps = support_eval_overlap(support_sets, eval_rows)
    for key, value in overlaps.items():
        if value["id_overlap"] or value["normalized_text_overlap_hashes"]:
            raise ProtocolError(
                f"Support/eval leakage remains for {key}: {value}. "
                "Investigate before Gate 1."
            )

    split_counts = []
    rows_by_split = {
        split: {
            "source_split": split,
            "total": len(samples[split]),
            "total_known": sum(1 for sample in samples[split] if sample.label != OOS_RAW_LABEL),
            "total_oos": sum(1 for sample in samples[split] if sample.label == OOS_RAW_LABEL),
            "selected_known": (
                sum(1 for sample in samples[split] if sample.label in selected_set)
                if split in KNOWN_SPLITS
                else 0
            ),
            "selected_oos": len(samples[split]) if split in ("oos_val", "oos_test") else 0,
        }
        for split in SPLIT_ORDER
    }
    roles = {
        "train": "support_pool",
        "val": "calibration_known",
        "test": "final_known_test",
        "oos_train": "not_used",
        "oos_val": "calibration_oos",
        "oos_test": "final_oos_test",
    }
    for split in SPLIT_ORDER:
        row = rows_by_split[split]
        row["selected_total"] = row["selected_known"] + row["selected_oos"]
        row["role"] = roles[split]
        split_counts.append(row)

    support_rows: list[dict[str, Any]] = []
    for (support_seed, shot), rows in sorted(support_sets.items()):
        for rank, sample in rows:
            support_rows.append(
                {
                    "support_seed": support_seed,
                    "shot": shot,
                    "intent": sample.label,
                    "domain": intent_to_domain[sample.label],
                    "sample_id": sample.sample_id,
                    "rank_within_intent": rank,
                    "source_split": sample.split,
                    "text": sample.text,
                    "text_sha256": sample.exact_sha256,
                    "text_norm_sha256": sample.norm_sha256,
                }
            )

    support_fields = [
        "support_seed",
        "shot",
        "intent",
        "domain",
        "sample_id",
        "rank_within_intent",
        "source_split",
        "text",
        "text_sha256",
        "text_norm_sha256",
    ]
    eval_fields = [
        "sample_id",
        "source_split",
        "source_index",
        "role",
        "is_oos",
        "raw_label",
        "open_set_label",
        "text",
        "text_sha256",
        "text_norm_sha256",
    ]
    split_fields = [
        "source_split",
        "role",
        "total",
        "total_known",
        "total_oos",
        "selected_known",
        "selected_oos",
        "selected_total",
    ]

    write_csv(out_dir / "support_manifest.csv", support_fields, support_rows)
    write_csv(out_dir / "eval_manifest.csv", eval_fields, eval_rows)
    write_csv(out_dir / "split_counts.csv", split_fields, split_counts)

    data_hash = hashlib.sha256(args.data_file.read_bytes()).hexdigest()
    domains_hash = hashlib.sha256(args.domains_file.read_bytes()).hexdigest()
    source_manifest = {
        "schema_version": "gate0-source-v1",
        "generated_at_utc": utc_now(),
        "dataset": {
            "name": "CLINC150 / OOS-Eval",
            "repository": "https://github.com/clinc/oos-eval",
            "revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
            "license": "CC BY 3.0",
            "data_file": {
                "path": str(args.data_file.resolve()),
                "bytes": args.data_file.stat().st_size,
                "sha256": data_hash,
            },
            "domains_file": {
                "path": str(args.domains_file.resolve()),
                "bytes": args.domains_file.stat().st_size,
                "sha256": domains_hash,
            },
        },
        "schema": {
            "top_level_keys": list(SPLIT_ORDER),
            "record_format": "[text: str, label: str]",
            "raw_oos_label": OOS_RAW_LABEL,
            "open_set_oos_label": OOS_OPEN_SET_LABEL,
        },
        "counts": {split: len(samples[split]) for split in SPLIT_ORDER},
        "known_intents": len(known_intents),
        "domains": len(domains),
        "selection": {
            "seed": args.seed,
            "known_classes": len(selected),
            "per_domain": args.per_domain,
            "rule": (
                "For each official domain, sort intents by ascending "
                "SHA-256('class-selection|<seed>|<domain>|<intent>'); take the "
                "first per_domain. No validation/test labels or metrics are used."
            ),
            "selected_intents": selected,
        },
        "support": {
            "seeds": args.support_seeds,
            "shots": args.shots,
            "source_split": "train",
            "rule": (
                "Within each intent, sort train rows by ascending "
                "SHA-256('support|<seed>|<intent>|<sample_id>'); take the first "
                "max(shots). K=5 is therefore a nested subset of K=10. Rows whose "
                "normalized text also occurs in an official eval split are excluded "
                "from the support candidate pool and listed in leakage_report.json."
            ),
            "leakage_filter_applied": not args.no_support_leakage_filter,
            "n_excluded_candidates": len(excluded),
        },
        "evaluation": {
            "calibration_known": 400,
            "calibration_oos": 100,
            "final_known_test": 600,
            "final_oos_test": 1000,
            "test_policy": (
                "Official test rows are indexed here only. They must not be used "
                "for model selection, prompt choice, threshold tuning, or seed choice."
            ),
        },
        "leakage_audit": {
            "official_cross_split_shared_normalized_hashes": n_cross,
            "within_split_duplicates": leakage["within_split"],
            "cross_split_pairs": leakage["cross_split_normalized_text"]["pairs"],
            "support_eval_overlap_after_filter": overlaps,
            "policy": (
                "Official rows are never deleted or relabeled. Cross-split text "
                "collisions are documented in leakage_report.json. Only support "
                "candidate selection excludes colliding train rows; val/test/OOS "
                "evaluation rows remain untouched."
            ),
        },
    }

    selected_manifest = {
        "schema_version": "gate0-selected-v1",
        "frozen_at_utc": utc_now(),
        "dataset": {
            "name": "CLINC150 / OOS-Eval",
            "revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
            "data_sha256": data_hash,
        },
        "selection": {
            "seed": args.seed,
            "known_classes": len(selected),
            "per_domain": args.per_domain,
            "rule": source_manifest["selection"]["rule"],
            "frozen": True,
            "reuse_across_methods": True,
        },
        "domains": {domain: sorted(selected_by_domain[domain]) for domain in sorted(selected_by_domain)},
        "intents": [
            {
                "intent": intent,
                "domain": intent_to_domain[intent],
                "train": sum(1 for sample in samples["train"] if sample.label == intent),
                "val": sum(1 for sample in samples["val"] if sample.label == intent),
                "test": sum(1 for sample in samples["test"] if sample.label == intent),
            }
            for intent in selected
        ],
        "support": {
            "seeds": args.support_seeds,
            "shots": args.shots,
            "source_split": "train",
            "k5_is_subset_of_k10": True,
            "leakage_filter_applied": not args.no_support_leakage_filter,
        },
        "evaluation": {
            "known_val": 400,
            "known_test": 600,
            "oos_val": 100,
            "oos_test": 1000,
            "extra_validation_labels_required_for_calibration": True,
        },
        "prohibited_uses": [
            "Do not change this list after observing validation or test metrics.",
            "Do not use final test rows for threshold or method selection.",
            "Do not train on OOS rows in this pilot.",
        ],
    }

    leakage_report = {
        "schema_version": "gate0-leakage-v1",
        "generated_at_utc": utc_now(),
        "official_data": leakage,
        "support_candidate_exclusions": excluded,
        "support_eval_overlap_after_filter": overlaps,
        "decision": (
            "No official row was deleted or relabeled. Cross-split text collisions "
            "are reported. Support candidates whose normalized text also occurs in "
            "an official eval split were excluded before K-shot ranking; official "
            "eval rows were untouched. With this filter, support/eval ID and "
            "normalized-text overlap is zero for every seed/shot."
        ),
        "limitations": [
            "The upstream dataset has no explicit duplicate policy across splits.",
            "Excluding a support candidate is a protocol safeguard, not a repair of the official splits.",
            "Normalized-text matching cannot detect paraphrase-level leakage.",
        ],
    }

    write_json(out_dir / "source_manifest.json", source_manifest)
    write_json(out_dir / "selected_intents.json", selected_manifest)
    write_json(out_dir / "leakage_report.json", leakage_report)
    write_json(
        out_dir / "support_leakage_exclusions.json",
        {
            "schema_version": "gate0-support-exclusions-v1",
            "n": len(excluded),
            "excluded": excluded,
        },
    )

    print("\nManifests written:")
    for name in (
        "selected_intents.json",
        "split_counts.csv",
        "source_manifest.json",
        "support_manifest.csv",
        "eval_manifest.csv",
        "leakage_report.json",
        "support_leakage_exclusions.json",
    ):
        path = out_dir / name
        print(f"  {path} ({path.stat().st_size} bytes)")

    print("\nSelected intents:")
    for intent in selected:
        print(f"  {intent_to_domain[intent]:18s} {intent}")
    print("\nPREPARE_DATA_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI must fail clearly
        print(f"PREPARE_DATA_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
