"""Encode the three Gate-3 datasets with one frozen text encoder.

Models:
  bge  -> BAAI/bge-small-en-v1.5              (no prefix)
  qwen -> Qwen/Qwen3-Embedding-0.6B            (fixed intent instruction)
  gte  -> Alibaba-NLP/gte-modernbert-base      (no prefix)
  e5   -> intfloat/e5-base-v2                  ("query: " prefix)

Usage:
    python -m src.gate3.encode --model bge
    python -m src.gate3.encode --model gte --device cuda --batch-size 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "manifest"
CACHE = PROJECT_ROOT / "data" / "cache_embeddings" / "gate3"
MODEL_CACHE = PROJECT_ROOT / "data" / "model_cache"
GATE2_CONFIG = PROJECT_ROOT / "configs" / "gate2.yaml"

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODELS = {
    "bge": {
        "id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "device": "cpu",
        "batch": 32,
        "prompt": "",
    },
    "qwen": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "device": "cuda",
        "batch": 8,
    },
    "gte": {
        "id": "Alibaba-NLP/gte-modernbert-base",
        "revision": "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        "device": "cuda",
        "batch": 16,
        "prompt": "",
    },
    "e5": {
        "id": "intfloat/e5-base-v2",
        "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        "device": "cuda",
        "batch": 32,
        "prompt": "query: ",
    },
}
DATASETS = ("clinc150", "banking77", "massive")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def prompt_for(model_key: str) -> tuple[str, str]:
    if model_key == "qwen":
        cfg = yaml.safe_load(GATE2_CONFIG.read_text(encoding="utf-8"))
        # Use the same frozen prompt as Gate 1/2.
        prompt = (
            "Instruct: Given a customer service message, identify the supported "
            "intent it expresses.\nQuery: "
        )
        expected = "4bbdf1c169061f64ba742f64bca5e422d05e94ccc7261d29495c233ed9a90005"
        if sha256_text(prompt) != expected:
            raise RuntimeError("Qwen prompt differs from the frozen Gate-1/2 prompt")
        return prompt, expected
    prompt = MODELS[model_key].get("prompt", "")
    return prompt, sha256_text(prompt)


def output_paths(model_key: str, dataset: str, prompt_hash: str) -> tuple[Path, Path]:
    root = (
        CACHE
        / model_key
        / dataset
        / MODELS[model_key]["revision"]
        / f"L128_{prompt_hash[:16]}"
    )
    return root / "embeddings.npz", root / "metadata.json"


def load_rows(dataset: str) -> list[dict[str, Any]]:
    path = MANIFEST / f"gate3_{dataset}_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run src.gate3.prepare first: {path}")
    rows = pd.read_csv(path, keep_default_na=False).to_dict("records")
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate sample IDs in {path}")
    return rows


def load_model(model_key: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    info = MODELS[model_key]
    kwargs: dict[str, Any] = {
        "device": device,
        "cache_folder": str(MODEL_CACHE),
        "revision": info["revision"],
        "trust_remote_code": False,
    }
    if device.startswith("cuda"):
        kwargs["model_kwargs"] = {
            "torch_dtype": torch.float16,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
    t0 = time.perf_counter()
    model = SentenceTransformer(info["id"], **kwargs)
    load_seconds = time.perf_counter() - t0
    model.max_seq_length = 128
    model.eval()
    if device.startswith("cuda"):
        if next(model.parameters()).dtype != torch.float16:
            model.half()
    return model, load_seconds


def encode(model, rows: list[dict[str, Any]], prompt: str, device: str, batch_size: int):
    import psutil
    import torch

    texts = [str(row["text"]) for row in rows]
    process = psutil.Process()
    rss_before = process.memory_info().rss
    attempts = []
    current = batch_size
    while True:
        attempts.append(current)
        try:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                vectors = model.encode(
                    texts,
                    prompt=prompt or None,
                    batch_size=current,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current == 1:
                raise
            current = max(1, current // 2)
            print(f"[OOM] batch={attempts[-1]} failed; retry batch={current}", flush=True)

    vectors = np.asarray(vectors, dtype=np.float32)
    raw_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(raw_norms <= 0):
        raise RuntimeError("Zero embedding vector")
    vectors = vectors / raw_norms
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all():
        raise RuntimeError("NaN/Inf embedding")
    if float(np.max(np.abs(norms - 1.0))) > 1e-5:
        raise RuntimeError(f"Norm error too large: {np.max(np.abs(norms - 1.0))}")
    return vectors, {
        "elapsed_seconds": round(elapsed, 3),
        "texts_per_second": round(len(rows) / elapsed, 3),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    info = MODELS[args.model]
    device = args.device or info["device"]
    batch_size = args.batch_size or info["batch"]
    prompt, prompt_hash = prompt_for(args.model)
    print(f"Model={args.model} id={info['id']} revision={info['revision']}")
    print(f"Device={device} batch={batch_size} prompt_hash={prompt_hash}")

    model, load_seconds = load_model(args.model, device)
    model_cache_size = sum(p.stat().st_size for p in MODEL_CACHE.rglob("*") if p.is_file())
    all_meta = {
        "schema_version": "gate3-encoding-v1",
        "encoded_at_utc": utc_now(),
        "model_key": args.model,
        "model_id": info["id"],
        "model_revision": info["revision"],
        "device": device,
        "prompt": prompt,
        "prompt_sha256": prompt_hash,
        "max_seq_length": 128,
        "load_seconds": round(load_seconds, 3),
        "model_cache_bytes": model_cache_size,
        "datasets": {},
    }

    for dataset in args.datasets:
        rows = load_rows(dataset)
        ids = [str(row["sample_id"]) for row in rows]
        ids_hash = sha256_ids(ids)
        npz_path, meta_path = output_paths(args.model, dataset, prompt_hash)
        if npz_path.exists() and meta_path.exists() and not args.overwrite:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("ids_sha256") == ids_hash
                and meta.get("prompt_sha256") == prompt_hash
                and meta.get("model_revision") == info["revision"]
            ):
                print(f"CACHE_HIT {npz_path}")
                all_meta["datasets"][dataset] = {"cache_hit": True, "npz": str(npz_path)}
                continue
            raise RuntimeError(f"Stale cache for {dataset}; pass --overwrite")
        vectors, enc = encode(model, rows, prompt, device, batch_size)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            vectors=vectors,
            sample_ids=np.asarray(ids, dtype="U48"),
            roles=np.asarray([str(row["role"]) for row in rows], dtype="U32"),
            labels=np.asarray([str(row["open_set_label"]) for row in rows], dtype="U64"),
            intents=np.asarray([str(row["intent"]) for row in rows], dtype="U64"),
            text_sha256=np.asarray([str(row["text_sha256"]) for row in rows], dtype="U64"),
        )
        meta = {
            "schema_version": "gate3-dataset-embeddings-v1",
            "encoded_at_utc": utc_now(),
            "dataset": dataset,
            "model_key": args.model,
            "model_id": info["id"],
            "model_revision": info["revision"],
            "device": device,
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "max_seq_length": 128,
            "n_texts": len(rows),
            "role_counts": {
                role: sum(1 for row in rows if row["role"] == role)
                for role in ("support", "calibration_known", "known_test", "dev_unknown", "final_unknown")
            },
            "embedding_dim": enc["embedding_dim"],
            "ids_sha256": ids_hash,
            "npz_path": str(npz_path.resolve()),
            "encode": enc,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_meta["datasets"][dataset] = meta
        print(
            f"{dataset}: n={len(rows)} dim={enc['embedding_dim']} "
            f"time={enc['elapsed_seconds']}s "
            f"peak={enc['cuda_peak_allocated_mib']}MiB "
            f"batch={enc['batch_size_used']}"
        )

    out = PROJECT_ROOT / "results" / f"gate3_encoding_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print("GATE3_ENCODE_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"GATE3_ENCODE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
