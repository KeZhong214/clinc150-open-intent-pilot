"""Router helpers for Gate-7 Agent comparisons."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate6.router_qwen import QwenRejectOnlyRouter  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402
from src.gate4.router import B4Router  # noqa: E402


class TrainingAdapter:
    def __init__(self, training_json: str, *, device: str = "cuda"):
        training = json.loads((ROOT / training_json).read_text(encoding="utf-8"))
        self.tokenizer = qe.get_tokenizer("right")
        self.model = qe.load_lora(ROOT / training["adapter_dir"], device=device)
        self.device = device

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vectors, _ = qe.encode_texts(self.model, self.tokenizer, list(texts), device=self.device, batch_size=int(kwargs.get("batch_size", 8)))
        return vectors.astype(np.float32)


def load_b4(encoder: str, method: str):
    payload = torch.load(RESULTS / f"gate6_qwen_{encoder}_{method}.pt", map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def qwen_reject_router(training_json: str, b4_checkpoint: str, tag: str):
    train = json.loads((ROOT / training_json).read_text(encoding="utf-8"))
    d = np.load(qe.lora_cache(train["adapter_sha256"]), allow_pickle=False)
    cache = {"vectors": d["vectors"].astype(np.float32), "sample_ids": d["sample_ids"].astype(str), "roles": d["roles"].astype(str), "labels": d["labels"].astype(str), "intents": d["intents"].astype(str)}
    classes = qe.classes()
    X, y = qd.support(cache)
    payload = torch.load(ROOT / b4_checkpoint, map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tau = float(payload["meta"].get("tau", 0.0)) if "tau" in payload["meta"] else 0.0
    if not tau:
        # threshold file is the source of truth
        stem = Path(b4_checkpoint).stem
        tau = json.loads((RESULTS / f"{stem}_threshold.json").read_text(encoding="utf-8"))["tau"]
    return QwenRejectOnlyRouter(classes, X, y, model, tau, encoder=TrainingAdapter(training_json))
