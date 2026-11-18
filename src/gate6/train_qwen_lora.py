"""G6-B: real Qwen3-Embedding LoRA training (Plain / Anchor).

Training uses only the 400 BANKING77 10-shot support rows.  Validation is used
outside this script for candidate selection; no test labels are read here.
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

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_encoder as qe  # noqa: E402


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mask = (y[:, None] == y[None, :]) & (~self_mask)
    counts = mask.sum(dim=1).clamp_min(1)
    return -((log_prob * mask).sum(dim=1) / counts).mean()


def mine_hard_negatives(vectors: np.ndarray, labels: np.ndarray, ids: Sequence[str], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    out = []
    for i in range(len(v)):
        same = np.where(labels == labels[i])[0]
        same = same[same != i]
        pos = int(rng.choice(same))
        diff = np.where(labels != labels[i])[0]
        neg = int(diff[int(np.argmax(v[diff] @ v[i]))])
        out.append(
            {
                "anchor_id": ids[i],
                "anchor_label": labels[i],
                "positive_id": ids[pos],
                "hard_negative_id": ids[neg],
                "hard_negative_label": labels[neg],
                "anchor_pos_cos": float(v[i] @ v[pos]),
                "anchor_neg_cos": float(v[i] @ v[neg]),
            }
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("smoke", "plain", "anchor"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--lambda-anchor", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    rows = qe.support_rows()
    texts = [r["text"] for r in rows]
    labels = np.asarray([r["intent"] for r in rows], dtype=object)
    ids = [r["sample_id"] for r in rows]
    classes = qe.classes()
    cidx = {c: i for i, c in enumerate(classes)}
    y_all = np.asarray([cidx[c] for c in labels], dtype=np.int64)
    frozen = qe.frozen_cache()
    idx = {sid: i for i, sid in enumerate(frozen["sample_ids"])}
    frozen_support = frozen["vectors"][[idx[sid] for sid in ids]]
    hard = mine_hard_negatives(frozen_support, labels, ids, args.seed)
    hard_by = {r["anchor_id"]: r for r in hard}
    text_by = {r["sample_id"]: r["text"] for r in rows}
    label_by = {r["sample_id"]: r["intent"] for r in rows}
    (RESULTS / "gate6_qwen_hard_negatives.csv").write_text("")  # placeholder overwritten below
    with (RESULTS / "gate6_qwen_hard_negatives.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hard[0].keys()))
        writer.writeheader()
        writer.writerows(hard)

    tokenizer = qe.get_tokenizer("right")
    base = qe.get_base(device="cuda", dtype=torch.float16)
    model = qe.attach_lora(base, r=8, alpha=16, dropout=0.05)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    model.train()
    params = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    params["ratio"] = params["trainable"] / params["total"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    t0 = time.perf_counter()
    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(args.seed + epoch)
        order = rng.permutation(len(rows))
        sums = {"loss": 0.0, "supcon": 0.0, "triplet": 0.0, "anchor": 0.0, "grad": 0.0}
        n = 0
        for start in range(0, len(order), args.microbatch):
            batch_idx = order[start : start + args.microbatch]
            if len(batch_idx) < 2:
                continue
            anchor_texts, pos_texts, neg_texts = [], [], []
            anchor_labels, pos_labels, neg_labels = [], [], []
            anchor_teacher = []
            for i in batch_idx:
                row = rows[i]
                h = hard_by[row["sample_id"]]
                anchor_texts.append(row["text"])
                pos_texts.append(text_by[h["positive_id"]])
                neg_texts.append(text_by[h["hard_negative_id"]])
                anchor_labels.append(cidx[row["intent"]])
                pos_labels.append(cidx[label_by[h["positive_id"]]])
                neg_labels.append(cidx[h["hard_negative_label"]])
                anchor_teacher.append(frozen_support[i])
            batch_texts = anchor_texts + pos_texts + neg_texts
            batch_labels = torch.tensor(anchor_labels + pos_labels + neg_labels, dtype=torch.long, device="cuda")
            token_batch = qe.tokenize(tokenizer, [qe.PROMPT + t for t in batch_texts], "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                last = model(**token_batch).last_hidden_state
                z = F.normalize(qe.last_token_pool(last, token_batch["attention_mask"]).float(), dim=1)
            b = len(batch_idx)
            sup = supcon_loss(z, batch_labels)
            pos_sim = (z[:b] * z[b : 2 * b]).sum(dim=1)
            neg_sim = (z[:b] * z[2 * b :]).sum(dim=1)
            triplet = F.relu(neg_sim - pos_sim + args.margin).mean()
            anchor = torch.zeros((), device="cuda")
            if args.run == "anchor":
                teacher = torch.from_numpy(np.stack(anchor_teacher)).to("cuda").float()
                anchor = (1.0 - (z[:b] * teacher).sum(dim=1)).mean()
            loss = sup + 0.5 * triplet
            if args.run == "anchor":
                loss = loss + args.lambda_anchor * anchor
            scaler.scale(loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                grad = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                sums["loss"] += float(loss.detach())
                sums["supcon"] += float(sup.detach())
                sums["triplet"] += float(triplet.detach())
                sums["anchor"] += float(anchor.detach())
                sums["grad"] += grad
                n += 1
                if step % 5 == 0:
                    print(f"{args.run} epoch={epoch} step={step} loss={float(loss.detach()):.4f} sup={float(sup.detach()):.4f} trip={float(triplet.detach()):.4f} anchor={float(anchor.detach()):.4f} grad={grad:.3f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB", flush=True)
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break
        history.append(
            {
                "run": args.run,
                "seed": args.seed,
                "epoch": epoch,
                "optimizer_steps": n,
                "loss": sums["loss"] / max(n, 1),
                "supcon": sums["supcon"] / max(n, 1),
                "triplet": sums["triplet"] / max(n, 1),
                "anchor": sums["anchor"] / max(n, 1),
                "grad_norm": sums["grad"] / max(n, 1),
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
                "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if stop:
            break

    elapsed = time.perf_counter() - t0
    adapter_dir = RESULTS / f"gate6_qwen_{args.run}_adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(str(adapter_dir))
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_hash = qe.sha_file(adapter_file)
    adapter_bytes = adapter_file.stat().st_size

    # Reload base+adapter and compare a probe with frozen Qwen.
    model.eval()
    probe_rows = rows[:4]
    probe_texts = [r["text"] for r in probe_rows]
    frozen_probe = frozen["vectors"][[idx[r["sample_id"]] for r in probe_rows]]
    with torch.no_grad():
        lora_probe, _ = qe.encode_texts(model, tokenizer, probe_texts, device="cuda", batch_size=4)
    # Free the trained model before loading a second base for reload verification.
    del model, base
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    reload_model = qe.load_lora(adapter_dir, device="cuda")
    with torch.no_grad():
        reload_probe, _ = qe.encode_texts(reload_model, tokenizer, probe_texts, device="cuda", batch_size=4)
    vector_check = {
        "probe_n": len(probe_texts),
        "dim": int(lora_probe.shape[1]),
        "frozen_vs_lora_mean_cosine": float(np.mean((frozen_probe * lora_probe).sum(axis=1))),
        "frozen_vs_lora_max_abs": float(np.max(np.abs(frozen_probe - lora_probe))),
        "reload_max_abs": float(np.max(np.abs(reload_probe - lora_probe))),
        "norm_max_abs_error": float(np.max(np.abs(np.linalg.norm(lora_probe, axis=1) - 1.0))),
    }
    pd_history = Path(RESULTS / f"gate6_qwen_{args.run}_history.csv")
    import pandas as pd

    pd.DataFrame(history).to_csv(pd_history, index=False)
    payload = {
        "run": args.run,
        "seed": args.seed,
        "epochs": len(history),
        "optimizer_steps": step,
        "microbatch": args.microbatch,
        "grad_accum": args.grad_accum,
        "effective_triplets_per_step": args.microbatch * args.grad_accum,
        "lr": args.lr,
        "lambda_anchor": args.lambda_anchor if args.run == "anchor" else 0.0,
        "margin": args.margin,
        "params": params,
        "elapsed_seconds": round(elapsed, 3),
        "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "adapter_dir": str(adapter_dir.relative_to(ROOT)),
        "adapter_sha256": adapter_hash,
        "adapter_bytes": adapter_bytes,
        "vector_check": vector_check,
        "history_csv": str(pd_history.relative_to(ROOT)),
        "hard_negatives": "results/gate6_qwen_hard_negatives.csv",
    }
    (RESULTS / f"gate6_qwen_{args.run}_training.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("GATE6_QWEN_TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
