"""Encode the frozen Gate-1 text set and cache unit-normalized vectors.

Only the seed=42, 5-shot support set and the fixed calibration/test rows are
encoded.  The final test rows are encoded, not used for threshold selection.

Usage:
    python -m src.encode_embeddings --model bge --device cpu
    python -m src.encode_embeddings --model qwen --device cuda --smoke
    python -m src.encode_embeddings --model qwen --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"
CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot.yaml"

# Must be set before importing torch.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODELS = {
    "bge": {
        "id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "default_device": "cpu",
        "default_batch": 32,
    },
    "qwen": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "default_device": "cuda",
        "default_batch": 4,
    },
}
ROLE_ORDER = (
    "calibration_known",
    "calibration_oos",
    "final_known_test",
    "final_oos_test",
)


@dataclass(frozen=True)
class Row:
    sample_id: str
    text: str
    role: str
    label: str
    text_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def read_rows(seed: int, shot: int) -> list[Row]:
    import csv

    support_path = MANIFEST_DIR / "support_manifest.csv"
    eval_path = MANIFEST_DIR / "eval_manifest.csv"
    if not support_path.exists() or not eval_path.exists():
        raise FileNotFoundError("Run `python -m src.prepare_data` first.")

    selected = json.loads((MANIFEST_DIR / "selected_intents.json").read_text(encoding="utf-8"))
    known = {item["intent"] for item in selected["intents"]}

    with support_path.open("r", encoding="utf-8", newline="") as handle:
        support = [
            row
            for row in csv.DictReader(handle)
            if int(row["support_seed"]) == seed and int(row["shot"]) == shot
        ]
    if len(support) != len(known) * shot:
        raise RuntimeError(f"Expected {len(known) * shot} support rows, got {len(support)}")

    with eval_path.open("r", encoding="utf-8", newline="") as handle:
        evaluation = list(csv.DictReader(handle))

    rows: list[Row] = []
    for row in sorted(support, key=lambda value: (value["intent"], value["sample_id"])):
        if row["intent"] not in known:
            raise RuntimeError(f"Unknown support intent: {row['intent']}")
        rows.append(
            Row(
                sample_id=row["sample_id"],
                text=row["text"],
                role="support",
                label=row["intent"],
                text_sha256=row["text_sha256"],
            )
        )

    eval_rank = {role: index for index, role in enumerate(ROLE_ORDER)}
    for row in sorted(
        evaluation,
        key=lambda value: (eval_rank[value["role"]], value["sample_id"]),
    ):
        rows.append(
            Row(
                sample_id=row["sample_id"],
                text=row["text"],
                role=row["role"],
                label=row["open_set_label"],
                text_sha256=row["text_sha256"],
            )
        )

    ids = [row.sample_id for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample_id in the encoding set")
    return rows


def prompt_contract(model_key: str) -> tuple[str, str]:
    import yaml

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if model_key == "bge":
        return "", sha256_text("")
    prompt = config["models"]["qwen"]["prompt"]
    expected = config["models"]["qwen"]["prompt_sha256"]
    actual = sha256_text(prompt)
    if actual != expected:
        raise RuntimeError(f"Qwen prompt hash mismatch: {actual} != {expected}")
    return prompt, actual


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def cache_paths(
    model_key: str,
    revision: str,
    prompt_hash: str,
    max_length: int,
    seed: int,
    shot: int,
    cache_dir: Path,
) -> tuple[Path, Path]:
    stem = f"seed{seed}_shot{shot}_L{max_length}_{prompt_hash[:16]}"
    root = cache_dir / model_key / revision / stem
    return root / "embeddings.npz", root / "metadata.json"


def load_model(
    model_key: str,
    device: str,
    cache_dir: Path,
    *,
    max_length: int = 128,
    local_files_only: bool = False,
):
    import torch
    from sentence_transformers import SentenceTransformer

    spec = MODELS[model_key]
    kwargs: dict[str, Any] = {
        "device": device,
        "cache_folder": str(cache_dir),
        "revision": spec["revision"],
        "trust_remote_code": False,
        "local_files_only": local_files_only,
    }
    if model_key == "qwen":
        kwargs["model_kwargs"] = {
            "torch_dtype": torch.float16,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        kwargs["processor_kwargs"] = {"padding_side": "left"}

    load_start = time.perf_counter()
    model = SentenceTransformer(spec["id"], **kwargs)
    load_seconds = time.perf_counter() - load_start
    model.max_seq_length = max_length
    model.eval()
    if model_key == "qwen" and device.startswith("cuda"):
        if next(model.parameters()).dtype != torch.float16:
            model.half()
    return model, load_seconds


def encode_texts(
    model,
    rows: Sequence[Row],
    prompt: str,
    batch_size: int,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    import psutil
    import torch

    process = psutil.Process()
    rss_before = process.memory_info().rss
    texts = [row.text for row in rows]
    encode_prompt = prompt or None

    attempts: list[int] = []
    current = max(1, batch_size)
    while True:
        attempts.append(current)
        try:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                vectors = model.encode(
                    texts,
                    prompt=encode_prompt,
                    batch_size=current,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current == 1:
                raise
            current = max(1, current // 2)
            print(f"[OOM] batch={attempts[-1]} failed; retrying batch={current}", flush=True)

    vectors = np.asarray(vectors, dtype=np.float32)
    # SentenceTransformer normalizes internally, but Qwen's FP16 output can
    # carry ~1e-4 norm error after the float32 cast.  Normalize explicitly in
    # float32 so all downstream cosine scores use unit vectors.
    raw_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(raw_norms <= 0):
        raise RuntimeError("Embedding matrix contains a zero vector")
    vectors = vectors / raw_norms
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embedding matrix contains NaN/Inf")
    if float(np.max(np.abs(norms - 1.0))) > 1e-5:
        raise RuntimeError(f"Embeddings are not unit-normalized: max error={np.max(np.abs(norms - 1.0))}")

    info = {
        "elapsed_seconds": round(elapsed, 3),
        "texts_per_second": round(len(rows) / elapsed, 3) if elapsed > 0 else None,
        "batch_size_requested": batch_size,
        "batch_size_used": current,
        "oom_retries": attempts[:-1],
        "rss_before_mib": round(rss_before / (1024**2), 1),
        "rss_after_mib": round(process.memory_info().rss / (1024**2), 1),
        "cuda_peak_allocated_mib": (
            round(torch.cuda.max_memory_allocated() / (1024**2), 1)
            if device.startswith("cuda")
            else None
        ),
        "cuda_peak_reserved_mib": (
            round(torch.cuda.max_memory_reserved() / (1024**2), 1)
            if device.startswith("cuda")
            else None
        ),
        "embedding_dim": int(vectors.shape[1]),
        "n_texts": int(vectors.shape[0]),
        "norm_max_abs_error": float(np.max(np.abs(norms - 1.0))),
    }
    return vectors, info


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults per model")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache_embeddings",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "model_cache",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Load the model and encode the first 2 rows only; do not write a cache.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail instead of downloading a missing model.",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    spec = MODELS[args.model]
    device = args.device or spec["default_device"]
    batch_size = args.batch_size or spec["default_batch"]
    prompt, prompt_hash = prompt_contract(args.model)
    args.model_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.model_cache.resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str(args.model_cache.resolve()))

    rows = read_rows(args.seed, args.shot)
    print(f"Model       : {spec['id']} @ {spec['revision']}")
    print(f"Device      : {device}; batch={batch_size}; max_length={args.max_length}")
    print(f"Prompt hash : {prompt_hash}")
    print(f"Rows        : {len(rows)} = {sum(row.role == 'support' for row in rows)} support + "
          f"{sum(row.role != 'support' for row in rows)} eval")

    if args.smoke:
        sample = rows[:2]
        model, load_seconds = load_model(
            args.model,
            device,
            args.model_cache,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
        )
        vectors, info = encode_texts(model, sample, prompt, batch_size, device)
        print(
            json.dumps(
                {
                    "status": "smoke_ok",
                    "model": args.model,
                    "device": device,
                    "load_seconds": round(load_seconds, 3),
                    "shape": list(vectors.shape),
                    "info": info,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print("ENCODE_SMOKE_OK")
        return 0

    npz_path, metadata_path = cache_paths(
        args.model,
        spec["revision"],
        prompt_hash,
        args.max_length,
        args.seed,
        args.shot,
        args.cache_dir,
    )
    expected_ids = [row.sample_id for row in rows]
    expected_ids_hash = sha256_ids(expected_ids)

    if npz_path.exists() and metadata_path.exists() and not args.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("ids_sha256") == expected_ids_hash
            and metadata.get("prompt_sha256") == prompt_hash
            and metadata.get("max_seq_length") == args.max_length
            and metadata.get("model_revision") == spec["revision"]
        ):
            print(f"CACHE_HIT {npz_path}")
            print("Encode skipped.")
            return 0
        raise RuntimeError(
            "Existing cache does not match this protocol; pass --overwrite to rebuild it."
        )

    cache_before = directory_size_bytes(args.model_cache)
    model, load_seconds = load_model(
        args.model,
        device,
        args.model_cache,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )
    cache_after = directory_size_bytes(args.model_cache)
    vectors, info = encode_texts(model, rows, prompt, batch_size, device)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez(
        npz_path,
        vectors=vectors,
        sample_ids=np.asarray([row.sample_id for row in rows], dtype="U32"),
        roles=np.asarray([row.role for row in rows], dtype="U32"),
        labels=np.asarray([row.label for row in rows], dtype="U64"),
        text_sha256=np.asarray([row.text_sha256 for row in rows], dtype="U64"),
    )

    metadata = {
        "schema_version": "gate1-embeddings-v1",
        "encoded_at_utc": utc_now(),
        "model_key": args.model,
        "model_id": spec["id"],
        "model_revision": spec["revision"],
        "device": device,
        "prompt": prompt,
        "prompt_sha256": prompt_hash,
        "max_seq_length": args.max_length,
        "seed": args.seed,
        "shot": args.shot,
        "n_texts": len(rows),
        "n_support": sum(row.role == "support" for row in rows),
        "n_eval": sum(row.role != "support" for row in rows),
        "embedding_dim": info["embedding_dim"],
        "dtype": str(vectors.dtype),
        "ids_sha256": expected_ids_hash,
        "npz_path": str(npz_path.resolve()),
        "load_seconds": round(load_seconds, 3),
        "cache_before_bytes": cache_before,
        "cache_after_bytes": cache_after,
        "cache_download_bytes_approx": max(0, cache_after - cache_before),
        "encode": info,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {npz_path} ({npz_path.stat().st_size} bytes)")
    print(f"Wrote {metadata_path}")
    print(json.dumps(metadata["encode"], ensure_ascii=False, indent=2))
    print("ENCODE_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ENCODE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
