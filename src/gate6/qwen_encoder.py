"""Qwen3-Embedding encoder with exact last-token pooling and optional LoRA."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest"
MODEL_CACHE = ROOT / "data" / "model_cache"
GATE3_CACHE = ROOT / "data" / "cache_embeddings" / "gate3"
GATE6_CACHE = ROOT / "data" / "cache_embeddings" / "gate6"
RESULTS = ROOT / "results"
QWEN_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN_REV = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PROMPT = "Instruct: Given a customer service message, identify the supported intent it expresses.\nQuery: "
PROMPT_SHA = "4bbdf1c169061f64ba742f64bca5e422d05e94ccc7261d29495c233ed9a90005"
MAX_LEN = 128
assert hashlib.sha256(PROMPT.encode()).hexdigest() == PROMPT_SHA


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_tokenizer(padding_side: str = "right"):
    tok = AutoTokenizer.from_pretrained(
        QWEN_ID,
        revision=QWEN_REV,
        cache_dir=str(MODEL_CACHE),
        local_files_only=True,
        trust_remote_code=False,
    )
    tok.padding_side = padding_side
    return tok


def get_base(*, device: str = "cuda", dtype=torch.float16):
    model = AutoModel.from_pretrained(
        QWEN_ID,
        revision=QWEN_REV,
        cache_dir=str(MODEL_CACHE),
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.to(device)
    return model


def lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.05) -> LoraConfig:
    # Qwen3 uses q_proj/k_proj/v_proj/o_proj; printed by inspect_qwen before use.
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )


def attach_lora(base, *, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    return get_peft_model(base, lora_config(r, alpha, dropout))


def load_lora(adapter_dir: Path, *, device: str = "cuda"):
    base = get_base(device=device)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.to(device)
    model.eval()
    return model


def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Works for left or right padding: sum-1 is the last real token.
    idx = attention_mask.sum(dim=1).clamp_min(1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), idx]


def tokenize(tokenizer, texts: Sequence[str], device: str):
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
    prompt: str = PROMPT,
    device: str = "cuda",
    batch_size: int = 8,
    show_progress: bool = False,
) -> tuple[np.ndarray, dict]:
    model.eval()
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = [prompt + str(t) for t in texts[start : start + batch_size]]
        batch = tokenize(tokenizer, chunk, device)
        last = model(**batch).last_hidden_state
        pooled = last_token_pool(last, batch["attention_mask"]).float()
        z = F.normalize(pooled, dim=1)
        out.append(z.cpu().numpy().astype(np.float32))
        if show_progress:
            print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    vectors = np.concatenate(out, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise RuntimeError("Bad Qwen embedding")
    return vectors, {"dim": int(vectors.shape[1]), "norm_max_abs_error": float(np.max(np.abs(norms - 1.0)))}


def frozen_cache() -> dict:
    path = GATE3_CACHE / "qwen" / "banking77" / QWEN_REV / f"L128_{PROMPT_SHA[:16]}" / "embeddings.npz"
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


def lora_cache(adapter_hash: str) -> Path:
    return GATE6_CACHE / "qwen_lora" / "banking77" / f"{QWEN_REV}_{adapter_hash}" / "L128" / "embeddings.npz"


def support_rows() -> list[dict]:
    import csv

    return list(csv.DictReader((MANIFEST / "gate3_banking77_support_10.csv").open("r", encoding="utf-8")))


def manifest_rows() -> list[dict]:
    import csv

    return list(csv.DictReader((MANIFEST / "gate3_banking77_manifest.csv").open("r", encoding="utf-8")))


def classes() -> list[str]:
    return read_json(MANIFEST / "gate3_banking77_dataset.json")["known_classes"]
