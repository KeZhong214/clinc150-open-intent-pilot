"""Shared utilities for Gate-5 LoRA-GTE training and encoding."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest"
GATE3_CACHE = ROOT / "data" / "cache_embeddings" / "gate3"
GATE5_CACHE = ROOT / "data" / "cache_embeddings" / "gate5"
MODEL_CACHE = ROOT / "data" / "model_cache"
RESULTS = ROOT / "results"
ADAPTER_DIR = RESULTS / "gate5_lora_adapter"
BASE_ID = "Alibaba-NLP/gte-modernbert-base"
BASE_REV = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
MAX_LEN = 128


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        BASE_ID,
        revision=BASE_REV,
        cache_dir=str(MODEL_CACHE),
        local_files_only=True,
        trust_remote_code=False,
    )


def get_base(*, dtype=torch.float16, device: str = "cuda"):
    model = AutoModel.from_pretrained(
        BASE_ID,
        revision=BASE_REV,
        cache_dir=str(MODEL_CACHE),
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.to(device)
    return model


def get_lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.05) -> LoraConfig:
    # Inspected with named_modules(): ModernBERT uses Wqkv/Wo (attention) and Wi/Wo (MLP).
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["Wqkv", "Wo", "Wi"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )


def attach_lora(base, *, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    return get_peft_model(base, get_lora_config(r=r, alpha=alpha, dropout=dropout))


def load_lora_model(adapter_dir: Path = ADAPTER_DIR, *, device: str = "cuda"):
    base = get_base(device=device)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.to(device)
    model.eval()
    return model


def count_params(model) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_ratio": float(trainable / total),
        "adapter_bytes": None,
    }


def tokenize_batch(tokenizer, texts: Sequence[str], device: str):
    batch = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    device: str = "cuda",
    batch_size: int = 16,
    show_progress: bool = False,
) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(texts), batch_size):
        batch = tokenize_batch(tokenizer, texts[start : start + batch_size], device)
        last = model(**batch).last_hidden_state
        # Gate-3 GTE SentenceTransformer uses CLS pooling and L2 normalization.
        z = F.normalize(last[:, 0].float(), dim=1)
        out.append(z.cpu().numpy().astype(np.float32))
        if show_progress:
            print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(out, axis=0)


def support_rows() -> list[dict]:
    return list(csv.DictReader((MANIFEST / "gate3_banking77_support_10.csv").open("r", encoding="utf-8")))


def manifest_rows() -> list[dict]:
    return list(csv.DictReader((MANIFEST / "gate3_banking77_manifest.csv").open("r", encoding="utf-8")))


def classes() -> list[str]:
    return read_json(MANIFEST / "gate3_banking77_dataset.json")["known_classes"]


def frozen_cache_path() -> Path:
    return (
        GATE3_CACHE
        / "gte"
        / "banking77"
        / BASE_REV
        / "L128_e3b0c44298fc1c14"
        / "embeddings.npz"
    )


def lora_cache_path(adapter_hash: str) -> Path:
    return GATE5_CACHE / "lora_gte" / "banking77" / f"{BASE_REV}_{adapter_hash}" / "L128" / "embeddings.npz"


def load_frozen_cache() -> dict:
    path = frozen_cache_path()
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return {
        "vectors": data["vectors"].astype(np.float32),
        "sample_ids": data["sample_ids"].astype(str),
        "roles": data["roles"].astype(str),
        "labels": data["labels"].astype(str),
        "intents": data["intents"].astype(str),
        "path": str(path),
    }


def support_arrays(vectors: np.ndarray, sample_ids: np.ndarray, rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    idx = {sid: i for i, sid in enumerate(sample_ids)}
    ids = [row["sample_id"] for row in rows]
    return vectors[[idx[sid] for sid in ids]], np.asarray([row["intent"] for row in rows], dtype=object)


def mine_hard_negatives(
    vectors: np.ndarray,
    labels: np.ndarray,
    sample_ids: Sequence[str],
    *,
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    unique = sorted(set(labels))
    rows = []
    for i in range(len(v)):
        same = np.where(labels == labels[i])[0]
        same = same[same != i]
        pos = int(rng.choice(same))
        diff = np.where(labels != labels[i])[0]
        sims = v[diff] @ v[i]
        neg = int(diff[int(np.argmax(sims))])
        rows.append(
            {
                "anchor_id": sample_ids[i],
                "anchor_label": labels[i],
                "positive_id": sample_ids[pos],
                "hard_negative_id": sample_ids[neg],
                "hard_negative_label": labels[neg],
                "anchor_pos_cos": float(v[i] @ v[pos]),
                "anchor_neg_cos": float(v[i] @ v[neg]),
                "hard_margin": float((v[i] @ v[pos]) - (v[i] @ v[neg])),
            }
        )
    return rows


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mask = (y[:, None] == y[None, :]) & (~self_mask)
    counts = mask.sum(dim=1).clamp_min(1)
    return -((log_prob * mask).sum(dim=1) / counts).mean()
