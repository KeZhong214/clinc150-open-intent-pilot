"""Encode the Gate-2 hard-OOS validation or final-test rows with Qwen.

Validation is encoded first.  The held-out hard-OOS test split is encoded only
after the Gate-2 configuration is frozen.  Encoding itself never uses labels to
choose a threshold; the split flag is explicit to keep that boundary visible.

Usage:
    python -m src.encode_gate2_hard_oos --split hard_oos_val
    # only after freeze:
    python -m src.encode_gate2_hard_oos --split hard_oos_test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "manifest"
CACHE = PROJECT_ROOT / "data" / "cache_embeddings"
MODEL_CACHE = PROJECT_ROOT / "data" / "model_cache"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PROMPT_HASH = "4bbdf1c169061f64ba742f64bca5e422d05e94ccc7261d29495c233ed9a90005"

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.encode_embeddings import (  # noqa: E402
    Row,
    encode_texts,
    load_model,
    prompt_contract,
    sha256_ids,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("hard_oos_val", "hard_oos_test"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = MANIFEST / "gate2_hard_oos_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Run `python -m src.prepare_gate2` first.")

    if args.split == "hard_oos_test":
        freeze_path = PROJECT_ROOT / "results" / "gate2_freeze.json"
        if not freeze_path.exists():
            raise RuntimeError(
                "hard_oos_test is held out; create and freeze "
                "results/gate2_freeze.json before encoding it."
            )
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if not freeze.get("frozen", False):
            raise RuntimeError("Gate-2 configuration is not marked frozen")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if row["role"] == args.split]
    if not rows:
        raise RuntimeError(f"No rows for {args.split}")

    expected = 1300 if args.split == "hard_oos_val" else 1950
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows for {args.split}, got {len(rows)}")

    prompt, prompt_hash = prompt_contract("qwen")
    if prompt_hash != PROMPT_HASH:
        raise RuntimeError("Qwen prompt hash differs from the frozen Gate-1 prompt")
    sample_rows = [
        Row(
            sample_id=row["sample_id"],
            text=row["text"],
            role=row["role"],
            label=row["intent"],
            text_sha256=row["text_sha256"],
        )
        for row in rows
    ]
    ids = [row.sample_id for row in sample_rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate hard-OOS sample IDs")

    stem = f"{args.split}_L{args.max_length}_{prompt_hash[:16]}"
    root = CACHE / "qwen" / QWEN_REVISION / stem
    npz_path = root / "embeddings.npz"
    metadata_path = root / "metadata.json"
    ids_hash = sha256_ids(ids)

    if npz_path.exists() and metadata_path.exists() and not args.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("ids_sha256") == ids_hash
            and metadata.get("prompt_sha256") == prompt_hash
            and metadata.get("split") == args.split
        ):
            print(f"CACHE_HIT {npz_path}")
            print("Encode skipped.")
            return 0
        raise RuntimeError("Existing hard-OOS cache does not match; pass --overwrite.")

    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np

    before = sum(
        path.stat().st_size
        for path in MODEL_CACHE.rglob("*")
        if path.is_file()
    ) if MODEL_CACHE.exists() else 0
    model, load_seconds = load_model(
        "qwen",
        "cuda",
        MODEL_CACHE,
        max_length=args.max_length,
        local_files_only=True,
    )
    after = sum(
        path.stat().st_size
        for path in MODEL_CACHE.rglob("*")
        if path.is_file()
    ) if MODEL_CACHE.exists() else 0
    vectors, info = encode_texts(model, sample_rows, prompt, args.batch_size, "cuda")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        vectors=vectors,
        sample_ids=np.asarray([row.sample_id for row in sample_rows], dtype="U32"),
        roles=np.asarray([row.role for row in sample_rows], dtype="U32"),
        intents=np.asarray([row.label for row in sample_rows], dtype="U64"),
        text_sha256=np.asarray([row.text_sha256 for row in sample_rows], dtype="U64"),
    )
    metadata = {
        "schema_version": "gate2-hard-oos-embeddings-v1",
        "encoded_at_utc": utc_now(),
        "split": args.split,
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": QWEN_REVISION,
        "prompt": prompt,
        "prompt_sha256": prompt_hash,
        "max_seq_length": args.max_length,
        "device": "cuda",
        "n_texts": len(sample_rows),
        "embedding_dim": info["embedding_dim"],
        "dtype": str(vectors.dtype),
        "ids_sha256": ids_hash,
        "source_manifest": str(manifest_path.resolve()),
        "npz_path": str(npz_path.resolve()),
        "load_seconds": round(load_seconds, 3),
        "cache_before_bytes": before,
        "cache_after_bytes": after,
        "encode": info,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"split": args.split, "shape": list(vectors.shape), "metadata": metadata}, ensure_ascii=False, indent=2))
    print("GATE2_HARD_OOS_ENCODE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
