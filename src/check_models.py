"""Record model metadata and run a CPU-only BGE smoke test.

Gate 0 only checks that the lightweight comparison encoder can be loaded and
that its vectors are finite/unit-normalized.  Qwen3-Embedding-0.6B metadata is
queried, but its weights are NEVER downloaded or loaded here.

Usage:
    python -m src.check_models --bge-smoke
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "data" / "model_cache"
CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot.yaml"

# Exact Hub revisions observed during Gate 0.  These are recorded in the
# manifest and should be used for the actual embedding pass.
MODEL_SPECS = {
    "bge": {
        "id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "role": "lightweight comparison model",
    },
    "qwen": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "role": "primary embedding model",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def qwen_prompt_contract() -> dict[str, str]:
    """Read the fixed Qwen instruction from the single config file."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    prompt = config["models"]["qwen"]["prompt"]
    expected_hash = config["models"]["qwen"]["prompt_sha256"]
    actual_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Qwen prompt hash mismatch: config={expected_hash}, actual={actual_hash}"
        )
    return {
        "prompt": prompt,
        "prompt_sha256": actual_hash,
        "scope": "support/query/validation/test all receive the same prefix",
    }


def fetch_json(url: str, timeout: int = 60) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "intent-open-set-gate0/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def model_metadata(spec: dict[str, str]) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{spec['id']}?blobs=true"
    payload = fetch_json(url)
    siblings = payload.get("siblings") or []
    files = []
    total_bytes = 0
    for sibling in siblings:
        size = int(sibling.get("size") or 0)
        total_bytes += size
        files.append(
            {
                "path": sibling.get("rfilename"),
                "bytes": size,
                "mib": round(size / (1024**2), 2),
            }
        )

    weights = [
        file
        for file in files
        if file["path"] and (
            file["path"].endswith(".safetensors")
            or file["path"].endswith(".bin")
            or file["path"].endswith(".onnx")
        )
    ]
    card = payload.get("cardData") or {}
    return {
        "id": spec["id"],
        "role": spec["role"],
        "requested_revision": spec["revision"],
        "hub_revision_at_check": payload.get("sha"),
        "revision_matches_pin": payload.get("sha") == spec["revision"],
        "license": card.get("license"),
        "pipeline_tag": payload.get("pipeline_tag"),
        "library_name": payload.get("library_name"),
        "total_repo_mib": round(total_bytes / (1024**2), 2),
        "total_repo_gb_decimal": round(total_bytes / (1000**3), 3),
        "weight_files": weights,
        "n_files": len(files),
        "metadata_url": url,
    }


def cache_presence(cache_dir: Path) -> dict[str, Any]:
    # Hugging Face cache directory naming convention.
    bge_dir = cache_dir / "models--BAAI--bge-small-en-v1.5"
    qwen_dir = cache_dir / "models--Qwen--Qwen3-Embedding-0.6B"
    return {
        "cache_root": str(cache_dir),
        "bge_cache_present": bge_dir.exists(),
        "bge_cache_bytes": directory_size_bytes(bge_dir),
        "qwen_cache_present": qwen_dir.exists(),
        "qwen_cache_bytes": directory_size_bytes(qwen_dir),
    }


def bge_smoke(cache_dir: Path, revision: str) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import psutil
    import torch
    from sentence_transformers import SentenceTransformer

    texts = [
        "I need to reset my card PIN.",
        "My card was swallowed by the ATM.",
        "What is the weather in Paris?",
    ]

    process = psutil.Process()
    rss_before = process.memory_info().rss
    before_cache = directory_size_bytes(cache_dir)
    load_start = time.perf_counter()
    model = SentenceTransformer(
        MODEL_SPECS["bge"]["id"],
        revision=revision,
        device="cpu",
        cache_folder=str(cache_dir),
    )
    load_seconds = time.perf_counter() - load_start
    model.max_seq_length = 128
    model.eval()
    rss_after_load = process.memory_info().rss
    after_cache = directory_size_bytes(cache_dir)

    token_lengths = []
    try:
        encoded = model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_length=True,
        )
        token_lengths = [int(value) for value in encoded["length"]]
    except Exception:  # noqa: BLE001 - tokenizer introspection is optional
        token_lengths = []

    encode_start = time.perf_counter()
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    encode_seconds = time.perf_counter() - encode_start
    rss_after_encode = process.memory_info().rss

    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    similarities = embeddings @ embeddings.T
    finite = bool(np.isfinite(embeddings).all())
    dim = int(embeddings.shape[1])
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    result = {
        "status": "ok",
        "model_id": MODEL_SPECS["bge"]["id"],
        "revision": revision,
        "device": "cpu",
        "dtype": str(embeddings.dtype),
        "max_seq_length": int(model.max_seq_length),
        "embedding_dim": dim,
        "parameter_count": parameter_count,
        "n_texts": len(texts),
        "texts": texts,
        "token_lengths": token_lengths,
        "shape": list(embeddings.shape),
        "finite": finite,
        "norms": [round(float(value), 6) for value in norms],
        "norm_max_abs_error_from_1": round(float(np.max(np.abs(norms - 1.0))), 8),
        "cosine_matrix": [
            [round(float(value), 6) for value in row] for row in similarities
        ],
        "load_seconds": round(load_seconds, 3),
        "encode_seconds": round(encode_seconds, 3),
        "texts_per_second": round(len(texts) / encode_seconds, 3) if encode_seconds else None,
        "cache_bytes_before": before_cache,
        "cache_bytes_after": after_cache,
        "cache_downloaded_bytes_approx": max(0, after_cache - before_cache),
        "rss_before_mib": round(rss_before / (1024**2), 1),
        "rss_after_load_mib": round(rss_after_load / (1024**2), 1),
        "rss_after_encode_mib": round(rss_after_encode / (1024**2), 1),
        "cuda_used": False,
        "cuda_peak_mib": None,
        "torch_version": torch.__version__,
        "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
        "transformers_version": importlib.metadata.version("transformers"),
    }
    if not finite or dim <= 0:
        raise RuntimeError(f"BGE smoke test produced invalid embeddings: {result}")
    if not math.isclose(float(np.mean(norms)), 1.0, abs_tol=1e-4):
        raise RuntimeError(f"BGE embeddings are not normalized: norms={norms.tolist()}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Hugging Face cache root; defaults to data/model_cache on D:.",
    )
    parser.add_argument(
        "--bge-smoke",
        action="store_true",
        help="Download/load BGE-small and encode 3 texts on CPU.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifest" / "model_resource_manifest.json",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    print(f"HF cache: {cache_dir}")
    print("Querying Hub metadata (no weights for Qwen are downloaded)...")
    models: dict[str, Any] = {}
    for key, spec in MODEL_SPECS.items():
        metadata = model_metadata(spec)
        models[key] = metadata
        print(
            f"{key}: {metadata['id']} revision={metadata['hub_revision_at_check']} "
            f"license={metadata['license']} repo={metadata['total_repo_mib']} MiB "
            f"match={metadata['revision_matches_pin']}"
        )

    cache = cache_presence(cache_dir)
    print(
        f"Cache before smoke: BGE={cache['bge_cache_present']} "
        f"({cache['bge_cache_bytes']} bytes), Qwen={cache['qwen_cache_present']} "
        f"({cache['qwen_cache_bytes']} bytes)"
    )

    smoke: dict[str, Any] | None = None
    if args.bge_smoke:
        print("Running BGE CPU smoke test (max_seq_length=128, 3 texts)...")
        smoke = bge_smoke(cache_dir, MODEL_SPECS["bge"]["revision"])
        print(
            "BGE smoke: "
            f"shape={smoke['shape']} dim={smoke['embedding_dim']} "
            f"finite={smoke['finite']} norm_error={smoke['norm_max_abs_error_from_1']} "
            f"load={smoke['load_seconds']}s encode={smoke['encode_seconds']}s"
        )
    else:
        print("BGE smoke skipped (pass --bge-smoke to run it).")

    cache_after = cache_presence(cache_dir)
    prompt_contract = qwen_prompt_contract()
    manifest = {
        "schema_version": "gate0-models-v1",
        "generated_at_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "cache": {
            "before": cache,
            "after": cache_after,
            "policy": (
                "All model files and embeddings must stay on D:. Qwen weights are "
                "not downloaded in Gate 0."
            ),
        },
        "models": models,
        "bge_smoke_test": smoke,
        "qwen": {
            "status": "metadata_only_not_downloaded",
            "download_weights": False,
            "weights_cached": cache_after["qwen_cache_present"],
            "load_test": False,
            "planned_device": "cuda:0 (RTX 5070 Laptop, 8 GiB) after Gate 1 approval",
            "planned_dtype": "float16",
            "planned_max_seq_length": 128,
            "planned_batch_sizes": [4, 8],
            "vram_planning_range_mib": [3072, 6144],
            "prompt_contract": prompt_contract,
            "note": (
                "These are planning values, not measured Qwen memory use. The "
                "fixed task instruction is frozen in configs/pilot.yaml and is "
                "applied identically to support and query text. It is not the "
                "model card's web-search prompt."
            ),
        },
        "risks": [
            "System RAM is only ~15.2 GiB total and was ~1.1 GiB available during Gate 0.",
            "Qwen FP16 weights are ~1.14 GiB; model load plus CUDA context and tokenizer can exceed the free-RAM headroom even though 8 GiB VRAM is nominally enough.",
            "Windows WDDM can page/spill GPU memory; peak VRAM must be measured, not assumed.",
            "RTX 5070 Laptop is sm_120; the inherited torch 2.11.0+cu128 reports CUDA available, but a real Qwen forward pass is still untested.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out}")
    print("MODEL_CHECK_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI must fail clearly
        print(f"MODEL_CHECK_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
