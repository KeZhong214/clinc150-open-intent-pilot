"""G6-B: re-encode BANKING77 with a Qwen LoRA adapter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_encoder as qe  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("plain", "anchor"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    train = json.loads((RESULTS / f"gate6_qwen_{args.run}_training.json").read_text(encoding="utf-8"))
    adapter_dir = ROOT / train["adapter_dir"]
    adapter_hash = train["adapter_sha256"]
    rows = qe.manifest_rows()
    texts = [r["text"] for r in rows]
    tokenizer = qe.get_tokenizer("right")
    model = qe.load_lora(adapter_dir, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    vectors, meta = qe.encode_texts(model, tokenizer, texts, device="cuda", batch_size=args.batch_size, show_progress=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise RuntimeError("Bad Qwen LoRA vectors")
    out = qe.lora_cache(adapter_hash)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vectors=vectors,
        sample_ids=np.asarray([r["sample_id"] for r in rows]),
        roles=np.asarray([r["role"] for r in rows]),
        labels=np.asarray([r["open_set_label"] for r in rows]),
        intents=np.asarray([r["intent"] for r in rows]),
    )
    frozen = qe.frozen_cache()
    idx = {sid: i for i, sid in enumerate(frozen["sample_ids"])}
    support_rows = qe.support_rows()
    lora_support = vectors[[{sid: i for i, sid in enumerate([r["sample_id"] for r in rows])}[r["sample_id"]] for r in support_rows]]
    frozen_support = frozen["vectors"][[idx[r["sample_id"]] for r in support_rows]]
    shift = {
        "support_mean_cosine": float(np.mean((frozen_support * lora_support).sum(axis=1))),
        "support_max_abs": float(np.max(np.abs(frozen_support - lora_support))),
        "support_mean_abs": float(np.mean(np.abs(frozen_support - lora_support))),
    }
    meta.update(
        {
            "run": args.run,
            "adapter_dir": str(adapter_dir.relative_to(ROOT)),
            "adapter_sha256": adapter_hash,
            "dataset": "banking77",
            "n_texts": len(rows),
            "embedding_dim": int(vectors.shape[1]),
            "prompt": qe.PROMPT,
            "prompt_sha256": qe.PROMPT_SHA,
            "pooling": "last_token",
            "padding_side": "right",
            "max_length": qe.MAX_LEN,
            "encode_seconds": round(elapsed, 3),
            "texts_per_second": round(len(rows) / elapsed, 3),
            "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "norm_max_abs_error": float(np.max(np.abs(norms - 1.0))),
            "cache_path": str(out.relative_to(ROOT)),
            "vector_shift": shift,
        }
    )
    (out.parent / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / f"gate6_qwen_{args.run}_encoding.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("GATE6_QWEN_ENCODE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
