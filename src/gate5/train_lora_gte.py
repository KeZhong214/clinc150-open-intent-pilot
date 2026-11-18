"""G5-A: train a real LoRA adapter on GTE-ModernBERT for BANKING77 10-shot.

Supervised contrastive loss + support-mined hard-negative triplet loss.  Only
the 400 support rows are used for gradients.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ADAPTER_DIR = RESULTS / "gate5_lora_adapter"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5 import lora_common as lc  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-triplets", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("G5-A requires CUDA for the recorded run")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    rows = lc.support_rows()
    texts = [r["text"] for r in rows]
    labels = np.asarray([r["intent"] for r in rows], dtype=object)
    ids = [r["sample_id"] for r in rows]
    classes = lc.classes()
    class_idx = {c: i for i, c in enumerate(classes)}
    y_all = np.asarray([class_idx[c] for c in labels], dtype=np.int64)
    if len(rows) != 400 or len(set(labels)) != 40:
        raise RuntimeError(f"Expected 400 support rows / 40 classes, got {len(rows)}/{len(set(labels))}")

    tokenizer = lc.get_tokenizer()
    base = lc.get_base(dtype=torch.float16, device=device)
    base.eval()
    frozen_direct = lc.encode_texts(base, tokenizer, texts, device=device, batch_size=16)
    frozen_cache = lc.load_frozen_cache()
    support_frozen, _ = lc.support_arrays(frozen_cache["vectors"], frozen_cache["sample_ids"], rows)
    diff = np.abs(frozen_direct - support_frozen)
    cos = (frozen_direct * support_frozen).sum(axis=1)
    alignment = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
        "note": "Direct AutoModel CLS pooling compared with Gate-3 SentenceTransformer cache.",
    }
    print(json.dumps({"frozen_alignment": alignment}, ensure_ascii=False))

    hard = lc.mine_hard_negatives(support_frozen, labels, ids, seed=args.seed)
    with (RESULTS / "gate5_lora_hard_negatives.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hard[0].keys()))
        writer.writeheader()
        writer.writerows(hard)
    hard_by_anchor = {r["anchor_id"]: r for r in hard}
    text_by_id = {r["sample_id"]: r["text"] for r in rows}
    label_by_id = {r["sample_id"]: r["intent"] for r in rows}

    # Freeze the base, then inject LoRA.
    model = lc.attach_lora(base, r=args.r, alpha=args.alpha, dropout=args.dropout)
    model.train()
    params = lc.count_params(model)
    print(json.dumps({"params": params, "targets": ["Wqkv", "Wo", "Wi"]}, ensure_ascii=False))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(args.seed + epoch)
        order = rng.permutation(len(rows))
        sums = {"loss": 0.0, "supcon": 0.0, "triplet": 0.0, "grad_norm": 0.0}
        n_batches = 0
        for start in range(0, len(order), args.batch_triplets):
            batch_idx = order[start : start + args.batch_triplets]
            if len(batch_idx) < 2:
                continue
            anchor_texts, pos_texts, neg_texts = [], [], []
            anchor_labels, pos_labels, neg_labels = [], [], []
            for i in batch_idx:
                row = rows[i]
                h = hard_by_anchor[row["sample_id"]]
                pos_id = h["positive_id"]
                neg_id = h["hard_negative_id"]
                # Map IDs through the support rows.
                anchor_texts.append(row["text"])
                pos_texts.append(text_by_id[pos_id])
                neg_texts.append(text_by_id[neg_id])
                anchor_labels.append(class_idx[row["intent"]])
                pos_labels.append(class_idx[label_by_id[pos_id]])
                neg_labels.append(class_idx[h["hard_negative_label"]])
            batch_texts = anchor_texts + pos_texts + neg_texts
            batch_labels = torch.tensor(anchor_labels + pos_labels + neg_labels, dtype=torch.long, device=device)
            token_batch = lc.tokenize_batch(tokenizer, batch_texts, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                last = model(**token_batch).last_hidden_state
                z = F.normalize(last[:, 0].float(), dim=1)
            b = len(batch_idx)
            sup = lc.supcon_loss(z, batch_labels)
            pos_sim = (z[:b] * z[b : 2 * b]).sum(dim=1)
            neg_sim = (z[:b] * z[2 * b :]).sum(dim=1)
            triplet = F.relu(neg_sim - pos_sim + args.margin).mean()
            loss = sup + 0.5 * triplet
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.detach())
            sums["supcon"] += float(sup.detach())
            sums["triplet"] += float(triplet.detach())
            sums["grad_norm"] += grad_norm
            n_batches += 1
            step += 1
            if step % 10 == 0:
                print(f"epoch={epoch} step={step} loss={float(loss.detach()):.4f} sup={float(sup.detach()):.4f} triplet={float(triplet.detach()):.4f} grad={grad_norm:.3f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        history.append(
            {
                "epoch": epoch,
                "steps": n_batches,
                "train_loss": sums["loss"] / max(n_batches, 1),
                "supcon": sums["supcon"] / max(n_batches, 1),
                "triplet": sums["triplet"] / max(n_batches, 1),
                "grad_norm": sums["grad_norm"] / max(n_batches, 1),
                "lr": args.lr,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
                "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    train_seconds = time.perf_counter() - t0
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    adapter_file = ADAPTER_DIR / "adapter_model.safetensors"
    adapter_hash = lc.sha256_file(adapter_file)
    adapter_bytes = adapter_file.stat().st_size

    # Reload base + adapter and verify the same-input vector is stable.
    model.eval()
    probe_texts = texts[:8]
    with torch.no_grad():
        lora_probe = lc.encode_texts(model, tokenizer, probe_texts, device=device, batch_size=8)
        frozen_probe = lc.encode_texts(lc.get_base(dtype=torch.float16, device=device).eval(), tokenizer, probe_texts, device=device, batch_size=8)
    reload_model = lc.load_lora_model(ADAPTER_DIR, device=device)
    with torch.no_grad():
        reload_probe = lc.encode_texts(reload_model, tokenizer, probe_texts, device=device, batch_size=8)
    vector_check = {
        "n_probe": len(probe_texts),
        "dim": int(lora_probe.shape[1]),
        "lora_norm_max_abs_error": float(np.max(np.abs(np.linalg.norm(lora_probe, axis=1) - 1.0))),
        "frozen_vs_lora_max_abs": float(np.max(np.abs(frozen_probe - lora_probe))),
        "frozen_vs_lora_mean_abs": float(np.mean(np.abs(frozen_probe - lora_probe))),
        "frozen_vs_lora_mean_cosine": float(np.mean((frozen_probe * lora_probe).sum(axis=1))),
        "reload_max_abs": float(np.max(np.abs(reload_probe - lora_probe))),
    }

    with (RESULTS / "gate5_lora_training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    payload = {
        "base_model": lc.BASE_ID,
        "base_revision": lc.BASE_REV,
        "dataset": "banking77",
        "shot": 10,
        "support_rows": len(rows),
        "classes": len(classes),
        "seed": args.seed,
        "target_modules": ["Wqkv", "Wo", "Wi"],
        "lora_r": args.r,
        "lora_alpha": args.alpha,
        "lora_dropout": args.dropout,
        "epochs": len(history),
        "batch_triplets": args.batch_triplets,
        "lr": args.lr,
        "margin": args.margin,
        "steps": step,
        "train_seconds": round(train_seconds, 3),
        "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "params": params,
        "adapter_path": str(ADAPTER_DIR.relative_to(ROOT)),
        "adapter_sha256": adapter_hash,
        "adapter_bytes": adapter_bytes,
        "frozen_alignment": alignment,
        "vector_check": vector_check,
        "hard_negatives": "results/gate5_lora_hard_negatives.csv",
        "history": "results/gate5_lora_training_history.csv",
        "test_evaluated": False,
    }
    (RESULTS / "gate5_lora_training.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_seconds": train_seconds, "adapter_sha256": adapter_hash, "vector_check": vector_check, "alignment": alignment}, ensure_ascii=False, indent=2))
    print("GATE5_LORA_TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
