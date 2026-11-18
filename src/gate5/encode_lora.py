"""G5-A: re-encode BANKING77 with base GTE + trained LoRA adapter."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ADAPTER_DIR = RESULTS / "gate5_lora_adapter"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5 import lora_common as lc  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    training = json.loads((RESULTS / "gate5_lora_training.json").read_text(encoding="utf-8"))
    adapter_hash = training["adapter_sha256"]
    rows = lc.manifest_rows()
    if args.limit:
        rows = rows[: args.limit]
    texts = [r["text"] for r in rows]
    tokenizer = lc.get_tokenizer()
    model = lc.load_lora_model(ADAPTER_DIR, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    vectors = lc.encode_texts(model, tokenizer, texts, device="cuda", batch_size=args.batch_size, show_progress=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise RuntimeError("Bad LoRA embedding")
    if float(np.max(np.abs(norms - 1.0))) > 1e-5:
        raise RuntimeError(f"Norm error {np.max(np.abs(norms - 1.0))}")

    # Re-load from disk and verify the same probe vectors.
    probe = texts[:8]
    with torch.no_grad():
        p1 = lc.encode_texts(model, tokenizer, probe, device="cuda", batch_size=8)
        reloaded = lc.load_lora_model(ADAPTER_DIR, device="cuda")
        p2 = lc.encode_texts(reloaded, tokenizer, probe, device="cuda", batch_size=8)

    out_path = lc.lora_cache_path(adapter_hash)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        vectors=vectors,
        sample_ids=np.asarray([r["sample_id"] for r in rows]),
        roles=np.asarray([r["role"] for r in rows]),
        labels=np.asarray([r["open_set_label"] for r in rows]),
        intents=np.asarray([r["intent"] for r in rows]),
    )
    meta = {
        "dataset": "banking77",
        "base_model": lc.BASE_ID,
        "base_revision": lc.BASE_REV,
        "adapter_path": str(ADAPTER_DIR.relative_to(ROOT)),
        "adapter_sha256": adapter_hash,
        "prompt": "",
        "max_length": lc.MAX_LEN,
        "n_texts": len(rows),
        "embedding_dim": int(vectors.shape[1]),
        "encode_seconds": round(elapsed, 3),
        "texts_per_second": round(len(rows) / elapsed, 3),
        "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "norm_max_abs_error": float(np.max(np.abs(norms - 1.0))),
        "reload_max_abs": float(np.max(np.abs(p1 - p2))),
        "cache_path": str(out_path.relative_to(ROOT)),
        "roles": {role: int(np.sum(np.asarray([r["role"] for r in rows]) == role)) for role in sorted({r["role"] for r in rows})},
    }
    (out_path.parent / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Compare support vectors against frozen Gate-3 cache.
    frozen = lc.load_frozen_cache()
    lora = {
        "vectors": vectors,
        "sample_ids": np.asarray([r["sample_id"] for r in rows]),
    }
    support_rows = lc.support_rows()
    lora_support, _ = lc.support_arrays(lora["vectors"], lora["sample_ids"], support_rows)
    frozen_support, _ = lc.support_arrays(frozen["vectors"], frozen["sample_ids"], support_rows)
    shift = {
        "support_max_abs": float(np.max(np.abs(frozen_support - lora_support))),
        "support_mean_abs": float(np.mean(np.abs(frozen_support - lora_support))),
        "support_mean_cosine": float(np.mean((frozen_support * lora_support).sum(axis=1))),
        "probe_reload_max_abs": float(np.max(np.abs(p1 - p2))),
    }
    (RESULTS / "gate5_lora_encoding.json").write_text(
        json.dumps({"metadata": meta, "vector_shift": shift}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"meta": meta, "shift": shift}, ensure_ascii=False, indent=2))
    print("GATE5_LORA_ENCODE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
