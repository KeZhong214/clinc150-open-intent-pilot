"""Build the Gate-3 unified manifests for CLINC150, BANKING77 and MASSIVE.

The existing Gate-0/1 CLINC150 pilot is reused as historical evidence.  New
BANKING77 and MASSIVE en-US splits are frozen here.

Selection rules
----------------
* BANKING77: 40 known / 18 dev-unknown / 19 final-unknown of 77 intents.
* MASSIVE en-US: 35 known / 12 dev-unknown / 13 final-unknown of 60 intents.
* Class order is deterministic SHA-256 of ``gate3|42|<dataset>|<intent>``.
* MASSIVE known classes require at least 10 official train rows and at least one
  dev row; remaining classes are allocated to dev/final with deterministic
  constraints based on actual row counts.
* Support = 10 train rows per known class, hashed; 5-shot is the first five.
* CLINC150 reuses its frozen 20 known classes and original OOS splits.

No model score is read by this script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW3 = PROJECT_ROOT / "data" / "gate3" / "raw"
MANIFEST = PROJECT_ROOT / "data" / "manifest"
CLINC_RAW = PROJECT_ROOT / "data" / "raw"
SEED = 42
MAX_SHOT = 10
SUPPORT_SHOTS = (5, 10)
DEV_FINAL = {
    "banking77": (40, 18, 19),
    "massive": (35, 12, 13),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"\s+", " ", value)


def h(prefix: str, dataset: str, *parts: str) -> str:
    return hashlib.sha256(("|".join((prefix, str(SEED), dataset, *parts))).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


class Builder:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.rows: list[dict[str, Any]] = []
        self.known: list[str] = []
        self.dev: list[str] = []
        self.final: list[str] = []
        self.class_domain: dict[str, str] = {}

    def add(
        self,
        sample_id: str,
        intent: str,
        domain: str,
        split: str,
        role: str,
        text: str,
        open_set_label: str,
        support_rank: int | None = None,
    ) -> None:
        self.rows.append(
            {
                "dataset": self.dataset,
                "intent": intent,
                "domain": domain,
                "split": split,
                "role": role,
                "sample_id": sample_id,
                "text": text,
                "open_set_label": open_set_label,
                "support_rank": "" if support_rank is None else support_rank,
                "text_sha256": sha256_text(text),
                "text_norm_sha256": sha256_text(norm_text(text)),
            }
        )

    def roles(self, role: str) -> list[dict[str, Any]]:
        return [row for row in self.rows if row["role"] == role]

    def all_unique(self) -> list[dict[str, Any]]:
        seen = set()
        result = []
        for row in self.rows:
            if row["sample_id"] in seen:
                continue
            seen.add(row["sample_id"])
            result.append(row)
        return result


def read_banking() -> dict[str, Any]:
    base = RAW3 / "banking77"
    train = pd.read_parquet(base / "train.parquet")
    test = pd.read_parquet(base / "test.parquet")
    labels = json.loads((base / "label_names.json").read_text(encoding="utf-8"))
    if len(labels) != 77:
        raise RuntimeError(f"Expected 77 BANKING77 labels, got {len(labels)}")

    def rows(frame: pd.DataFrame, split: str) -> list[dict[str, Any]]:
        result = []
        for index, row in frame.iterrows():
            label_idx = int(row["label"])
            result.append(
                {
                    "sample_id": f"banking77:{split}:{index:05d}",
                    "intent": labels[label_idx],
                    "domain": "banking",
                    "split": split,
                    "text": str(row["text"]),
                    "text_norm_sha256": sha256_text(norm_text(str(row["text"]))),
                }
            )
        return result

    return {
        "train": rows(train, "train"),
        "test": rows(test, "test"),
        "classes": labels,
        "source": {
            "id": "PolyAI/banking77",
            "revision": "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
            "license": "CC BY 4.0",
            "train_rows": len(train),
            "test_rows": len(test),
        },
    }


def read_massive() -> dict[str, Any]:
    path = RAW3 / "massive" / "1.1" / "data" / "en-US.jsonl"
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    class_domain: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            item = json.loads(line)
            if item.get("locale") != "en-US":
                continue
            partition = item["partition"]
            if partition not in by_split:
                continue
            intent = str(item["intent"])
            scenario = str(item["scenario"])
            class_domain[intent] = scenario
            by_split[partition].append(
                {
                    "sample_id": f"massive:en-US:{partition}:{index:05d}",
                    "intent": intent,
                    "domain": scenario,
                    "split": partition,
                    "text": str(item["utt"]),
                    "text_norm_sha256": sha256_text(norm_text(str(item["utt"]))),
                }
            )
    classes = sorted(class_domain)
    if len(classes) != 60:
        raise RuntimeError(f"Expected 60 MASSIVE intents, got {len(classes)}")
    return {
        "train": by_split["train"],
        "dev": by_split["dev"],
        "test": by_split["test"],
        "classes": classes,
        "class_domain": class_domain,
        "source": {
            "id": "AmazonScience/massive",
            "revision": "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
            "dataset_version": "1.1",
            "license": "CC BY 4.0",
            "train_rows": len(by_split["train"]),
            "dev_rows": len(by_split["dev"]),
            "test_rows": len(by_split["test"]),
        },
    }


def split_classes_hashed(classes: Sequence[str], dataset: str, sizes: tuple[int, int, int]) -> tuple[list[str], list[str], list[str]]:
    n_known, n_dev, n_final = sizes
    if len(classes) != n_known + n_dev + n_final:
        raise RuntimeError(f"{dataset}: class count {len(classes)} != {sizes}")
    return (
        list(classes[:n_known]),
        list(classes[n_known : n_known + n_dev]),
        list(classes[n_known + n_dev :]),
    )


def allocate_massive(
    data: dict[str, Any],
    clean_train_counts: Counter,
) -> tuple[list[str], list[str], list[str]]:
    counts = {
        split: Counter(row["intent"] for row in data[split])
        for split in ("train", "dev", "test")
    }
    ranked = sorted(
        data["classes"],
        key=lambda intent: (h("gate3-class", "massive", intent), intent),
    )
    eligible_known = [
        intent
        for intent in ranked
        if clean_train_counts[intent] >= MAX_SHOT
        and counts["dev"][intent] >= 1
        and counts["test"][intent] >= 1
    ]
    if len(eligible_known) < 35:
        raise RuntimeError(f"Only {len(eligible_known)} MASSIVE classes have >=10 train and >=1 dev")
    known = eligible_known[:35]
    remaining = [intent for intent in ranked if intent not in set(known)]
    if len(remaining) != 25:
        raise RuntimeError(f"Expected 25 remaining MASSIVE classes, got {len(remaining)}")

    # Classes with zero dev rows must be final; classes with zero test rows must
    # be dev.  Fill the rest by deterministic hash order.
    forced_dev = [intent for intent in remaining if counts["test"][intent] == 0]
    forced_final = [intent for intent in remaining if counts["dev"][intent] == 0]
    if set(forced_dev) & set(forced_final):
        raise RuntimeError("MASSIVE class has zero both dev and test rows")
    flexible = [intent for intent in remaining if intent not in set(forced_dev) | set(forced_final)]
    if len(forced_dev) > 12 or len(forced_final) > 13:
        raise RuntimeError("MASSIVE forced allocation exceeds target sizes")
    need_dev = 12 - len(forced_dev)
    if need_dev < 0 or need_dev > len(flexible):
        raise RuntimeError("MASSIVE cannot fill 12 dev classes")
    dev = forced_dev + flexible[:need_dev]
    final = forced_final + flexible[need_dev:]
    if len(dev) != 12 or len(final) != 13 or set(dev) & set(final):
        raise RuntimeError(f"Bad MASSIVE allocation: dev={len(dev)}, final={len(final)}")
    if any(counts["dev"][intent] == 0 for intent in dev):
        raise RuntimeError("MASSIVE dev class with zero dev rows")
    if any(counts["test"][intent] == 0 for intent in final):
        raise RuntimeError("MASSIVE final class with zero test rows")
    return known, dev, final


def support_for_class(rows: Sequence[dict[str, Any]], dataset: str, intent: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (h("gate3-support", dataset, intent, row["sample_id"]), row["sample_id"]))
    return ordered[:MAX_SHOT]


def main() -> int:
    MANIFEST.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}

    # ---- CLINC150 historical pilot -------------------------------------
    selected = json.loads((MANIFEST / "selected_intents.json").read_text(encoding="utf-8"))
    domains = json.loads((CLINC_RAW / "domains.json").read_text(encoding="utf-8"))
    intent_domain = {intent: domain for domain, values in domains.items() for intent in values}
    clinc = Builder("clinc150")
    clinc.known = [item["intent"] for item in selected["intents"]]
    clinc.dev = []
    clinc.final = []
    clinc.class_domain = {intent: intent_domain[intent] for intent in clinc.known}
    support = list(csv.DictReader((MANIFEST / "support_manifest.csv").open("r", encoding="utf-8")))
    for row in support:
        if int(row["support_seed"]) != SEED or int(row["shot"]) != 10:
            continue
        clinc.add(
            sample_id=row["sample_id"],
            intent=row["intent"],
            domain=row["domain"],
            split="train",
            role="support",
            text=row["text"],
            open_set_label=row["intent"],
            support_rank=int(row["rank_within_intent"]),
        )
    eval_rows = list(csv.DictReader((MANIFEST / "eval_manifest.csv").open("r", encoding="utf-8")))
    role_map = {
        "calibration_known": "calibration_known",
        "final_known_test": "known_test",
        "calibration_oos": "dev_unknown",
        "final_oos_test": "final_unknown",
    }
    for row in eval_rows:
        role = role_map[row["role"]]
        clinc.add(
            sample_id=row["sample_id"],
            intent=row["open_set_label"] if role != "dev_unknown" and role != "final_unknown" else "OOS",
            domain=clinc.class_domain.get(row["open_set_label"], "clinc150"),
            split=row["source_split"],
            role=role,
            text=row["text"],
            open_set_label=row["open_set_label"],
            support_rank=None,
        )
    summaries["clinc150"] = finalize(
        clinc,
        {
            "source": {
                "id": "clinc/oos-eval",
                "revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
                "license": "CC BY 3.0",
                "note": "Historical Gate 0/1 split; original oos_val/oos_test used. Test previously seen.",
            },
            "class_rule": "Gate-0 frozen 20 known intents; original OOS val/test as dev/final unknown.",
            "historical": True,
        },
    )

    # ---- BANKING77 ------------------------------------------------------
    banking = read_banking()
    bank_ranked = sorted(
        banking["classes"], key=lambda intent: (h("gate3-class", "banking77", intent), intent)
    )
    bank_known, bank_dev, bank_final = split_classes_hashed(
        bank_ranked, "banking77", DEV_FINAL["banking77"]
    )
    b = Builder("banking77")
    b.known, b.dev, b.final = bank_known, bank_dev, bank_final
    b.class_domain = {intent: "banking" for intent in banking["classes"]}
    train_by_intent = defaultdict(list)
    for row in banking["train"]:
        train_by_intent[row["intent"]].append(row)
    bank_test_hashes = {row["text_norm_sha256"] for row in banking["test"]}
    clean_train_by_intent = {
        intent: [row for row in rows if row["text_norm_sha256"] not in bank_test_hashes]
        for intent, rows in train_by_intent.items()
    }
    for intent in bank_known:
        chosen = support_for_class(clean_train_by_intent[intent], "banking77", intent)
        if len(chosen) != MAX_SHOT:
            raise RuntimeError(f"BANKING77 {intent}: only {len(chosen)} leakage-safe train rows")
        for rank, row in enumerate(chosen, 1):
            b.add(
                sample_id=row["sample_id"],
                intent=intent,
                domain="banking",
                split="train",
                role="support",
                text=row["text"],
                open_set_label=intent,
                support_rank=rank,
            )
    support_ids = {row["sample_id"] for row in b.roles("support")}
    support_hashes = {row["text_norm_sha256"] for row in b.roles("support")}
    for intent in bank_known:
        remaining = [
            row
            for row in clean_train_by_intent[intent]
            if row["sample_id"] not in support_ids
            and row["text_norm_sha256"] not in support_hashes
        ]
        remaining.sort(key=lambda row: h("gate3-cal", "banking77", intent, row["sample_id"]))
        if len(remaining) < 20:
            raise RuntimeError(f"BANKING77 {intent}: only {len(remaining)} calibration rows")
        for row in remaining[:20]:
            b.add(
                sample_id=row["sample_id"],
                intent=intent,
                domain="banking",
                split="train",
                role="calibration_known",
                text=row["text"],
                open_set_label=intent,
            )
    for row in banking["test"]:
        if row["intent"] in set(bank_known):
            b.add(row["sample_id"], row["intent"], "banking", "test", "known_test", row["text"], row["intent"])
        elif row["intent"] in set(bank_dev):
            b.add(row["sample_id"], row["intent"], "banking", "test", "dev_unknown", row["text"], "OOS")
        elif row["intent"] in set(bank_final):
            b.add(row["sample_id"], row["intent"], "banking", "test", "final_unknown", row["text"], "OOS")
        else:
            raise RuntimeError(f"Unassigned BANKING77 intent: {row['intent']}")
    summaries["banking77"] = finalize(
        b,
        {"source": banking["source"], "class_rule": "hash 40/18/19; all test rows used; train support/calibration"},
    )

    # ---- MASSIVE en-US --------------------------------------------------
    massive = read_massive()
    eval_hashes = {
        row["text_norm_sha256"] for split in ("dev", "test") for row in massive[split]
    }
    train_by_intent = defaultdict(list)
    for row in massive["train"]:
        train_by_intent[row["intent"]].append(row)
    clean_train_by_intent = {
        intent: [row for row in rows if row["text_norm_sha256"] not in eval_hashes]
        for intent, rows in train_by_intent.items()
    }
    clean_train_counts = Counter({intent: len(rows) for intent, rows in clean_train_by_intent.items()})
    mass_known, mass_dev, mass_final = allocate_massive(massive, clean_train_counts)
    m = Builder("massive")
    m.known, m.dev, m.final = mass_known, mass_dev, mass_final
    m.class_domain = massive["class_domain"]
    for intent in mass_known:
        chosen = support_for_class(clean_train_by_intent[intent], "massive", intent)
        if len(chosen) != MAX_SHOT:
            raise RuntimeError(f"MASSIVE {intent}: only {len(chosen)} leakage-safe train rows")
        for rank, row in enumerate(chosen, 1):
            m.add(
                sample_id=row["sample_id"],
                intent=intent,
                domain=row["domain"],
                split="train",
                role="support",
                text=row["text"],
                open_set_label=intent,
                support_rank=rank,
            )
    support_hashes = {row["text_norm_sha256"] for row in m.roles("support")}
    test_hashes = {row["text_norm_sha256"] for row in massive["test"]}
    unknown_dev_hashes = {
        row["text_norm_sha256"]
        for row in massive["dev"]
        if row["intent"] in set(mass_dev) | set(mass_final)
    }
    calibration_forbidden = support_hashes | test_hashes | unknown_dev_hashes
    for row in massive["dev"]:
        if row["intent"] in set(mass_known):
            if row["text_norm_sha256"] in calibration_forbidden:
                continue
            m.add(row["sample_id"], row["intent"], row["domain"], "dev", "calibration_known", row["text"], row["intent"])
        elif row["intent"] in set(mass_dev):
            m.add(row["sample_id"], row["intent"], row["domain"], "dev", "dev_unknown", row["text"], "OOS")
        elif row["intent"] in set(mass_final):
            # Held-out final classes' dev rows are not used; do not include.
            continue
        else:
            raise RuntimeError(f"Unassigned MASSIVE dev intent: {row['intent']}")
    for row in massive["test"]:
        if row["intent"] in set(mass_known):
            m.add(row["sample_id"], row["intent"], row["domain"], "test", "known_test", row["text"], row["intent"])
        elif row["intent"] in set(mass_final):
            m.add(row["sample_id"], row["intent"], row["domain"], "test", "final_unknown", row["text"], "OOS")
        elif row["intent"] in set(mass_dev):
            # Dev-unknown classes contribute only their official dev rows.
            continue
        else:
            raise RuntimeError(f"Unassigned MASSIVE test intent: {row['intent']}")
    summaries["massive"] = finalize(
        m,
        {"source": massive["source"], "class_rule": "hash with train>=10 for known; 35/12/13 constrained by dev/test availability"},
    )

    write_json(MANIFEST / "gate3_all_datasets.json", {"generated_at_utc": utc_now(), "datasets": summaries})
    print(json.dumps({key: {k: v for k, v in value.items() if k not in ("classes", "class_counts")} for key, value in summaries.items()}, ensure_ascii=False, indent=2))
    print("GATE3_PREPARE_OK")
    return 0


def finalize(builder: Builder, extra: dict[str, Any]) -> dict[str, Any]:
    unique = builder.all_unique()
    by_role = Counter(row["role"] for row in unique)
    if by_role["support"] != len(builder.known) * MAX_SHOT:
        raise RuntimeError(f"{builder.dataset}: support rows {by_role['support']}")
    # Support 5 is a subset of support 10.
    support = sorted(
        [row for row in unique if row["role"] == "support"],
        key=lambda row: (row["intent"], int(row["support_rank"])),
    )
    support10 = [{**row, "shot": 10} for row in support]
    support5 = [{**row, "shot": 5} for row in support if int(row["support_rank"]) <= 5]
    if len(support5) != len(builder.known) * 5:
        raise RuntimeError(f"{builder.dataset}: support5 rows {len(support5)}")
    # Required roles.
    for role in ("calibration_known", "known_test", "dev_unknown", "final_unknown"):
        if by_role[role] == 0:
            raise RuntimeError(f"{builder.dataset}: empty role {role}")

    def eval_rows(role: str) -> list[dict[str, Any]]:
        return sorted([row for row in unique if row["role"] == role], key=lambda row: row["sample_id"])

    # Text overlap audit (normalized).  Report; do not delete official rows.
    def hashes(rows: Iterable[dict[str, Any]]) -> set[str]:
        return {row["text_norm_sha256"] for row in rows}

    support_h = hashes(support)
    support5_h = hashes(support5)
    cal_h = hashes(eval_rows("calibration_known"))
    test_h = hashes(eval_rows("known_test"))
    dev_h = hashes(eval_rows("dev_unknown"))
    final_h = hashes(eval_rows("final_unknown"))
    overlaps = {
        "support10_vs_known_test": len(support_h & test_h),
        "support5_vs_known_test": len(support5_h & test_h),
        "support10_vs_calibration": len(support_h & cal_h),
        "support10_vs_dev_unknown": len(support_h & dev_h),
        "support10_vs_final_unknown": len(support_h & final_h),
        "calibration_vs_known_test": len(cal_h & test_h),
        "calibration_vs_dev_unknown": len(cal_h & dev_h),
        "calibration_vs_final_unknown": len(cal_h & final_h),
        "dev_unknown_vs_final_unknown": len(dev_h & final_h),
        "dev_unknown_vs_known_test": len(dev_h & test_h),
    }

    all_fields = [
        "dataset",
        "intent",
        "domain",
        "split",
        "role",
        "sample_id",
        "text",
        "open_set_label",
        "support_rank",
        "text_sha256",
        "text_norm_sha256",
    ]
    write_csv(MANIFEST / f"gate3_{builder.dataset}_manifest.csv", unique, all_fields)
    support_fields = ["shot", "support_rank", "sample_id", "intent", "domain", "text", "open_set_label", "text_sha256"]
    write_csv(
        MANIFEST / f"gate3_{builder.dataset}_support_10.csv",
        [{field: row[field] for field in support_fields} for row in support10],
        support_fields,
    )
    write_csv(
        MANIFEST / f"gate3_{builder.dataset}_support_5.csv",
        [{field: row[field] for field in support_fields} for row in support5],
        support_fields,
    )

    write_csv(MANIFEST / f"gate3_{builder.dataset}_calibration.csv", eval_rows("calibration_known"), all_fields)
    write_csv(MANIFEST / f"gate3_{builder.dataset}_known_test.csv", eval_rows("known_test"), all_fields)
    write_csv(MANIFEST / f"gate3_{builder.dataset}_dev_unknown.csv", eval_rows("dev_unknown"), all_fields)
    write_csv(MANIFEST / f"gate3_{builder.dataset}_final_unknown.csv", eval_rows("final_unknown"), all_fields)

    class_counts_train = Counter(
        row["intent"] for row in unique if row["role"] == "support"
    )
    class_counts_cal = Counter(
        row["intent"] for row in unique if row["role"] == "calibration_known"
    )
    class_counts_test = Counter(row["intent"] for row in unique if row["role"] == "known_test")
    class_rows = [
        {
            "intent": intent,
            "domain": builder.class_domain[intent],
            "support": class_counts_train[intent],
            "calibration": class_counts_cal[intent],
            "known_test": class_counts_test[intent],
        }
        for intent in builder.known
    ]
    write_csv(
        MANIFEST / f"gate3_{builder.dataset}_classes.csv",
        class_rows,
        ["intent", "domain", "support", "calibration", "known_test"],
    )
    split_rows = [
        {"role": role, "rows": by_role[role]} for role in ("support", "calibration_known", "known_test", "dev_unknown", "final_unknown")
    ]
    write_csv(MANIFEST / f"gate3_{builder.dataset}_split_counts.csv", split_rows, ["role", "rows"])

    summary = {
        "schema_version": "gate3-dataset-v1",
        "generated_at_utc": utc_now(),
        "dataset": builder.dataset,
        "seed": SEED,
        "known_classes": builder.known,
        "dev_unknown_classes": builder.dev,
        "final_unknown_classes": builder.final,
        "class_domain": {intent: builder.class_domain[intent] for intent in builder.known + builder.dev + builder.final},
        "counts": {role: by_role[role] for role in ("support", "calibration_known", "known_test", "dev_unknown", "final_unknown")},
        "support_5_rows": len(support5),
        "support_10_rows": len(support),
        "class_counts": class_rows,
        "text_overlap": overlaps,
        "fields": all_fields,
        **extra,
    }
    write_json(MANIFEST / f"gate3_{builder.dataset}_dataset.json", summary)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
