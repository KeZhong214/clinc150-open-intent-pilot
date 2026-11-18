"""Gate-5 routers: frozen B3-H/B4-Full and LoRA-GTE B3-H/B4-Full."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5 import lora_common as lc  # noqa: E402
from src.gate5 import encoder_data as ed  # noqa: E402
from src.gate4.router import B3HRouter, B4Router, load_b3h_router, load_b4_router, load_gte  # noqa: E402


class FrozenEncoder:
    """Direct frozen GTE CLS encoder; avoids SentenceTransformer PEFT adapter probing."""

    def __init__(self):
        self.tokenizer = lc.get_tokenizer()
        self.model = lc.get_base(device="cuda").eval()

    def encode(self, texts, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
            return lc.encode_texts(self.model, self.tokenizer, list(texts), device="cuda", batch_size=len(texts))


class LoraEncoder:
    def __init__(self):
        self.tokenizer = lc.get_tokenizer()
        self.model = lc.load_lora_model(lc.ADAPTER_DIR, device="cuda")

    def encode(self, texts, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
            return lc.encode_texts(self.model, self.tokenizer, list(texts), device="cuda", batch_size=len(texts))


def load_lora_b3h_router(encoder: LoraEncoder) -> B3HRouter:
    classes, X, y = _lora_support()
    import pandas as pd

    row = pd.read_csv(RESULTS / "gate5_lora_validation.csv")
    tau = float(row[row.method == "B3-H"]["tau"].iloc[0])
    return B3HRouter(classes, X, y, tau=tau, encoder=encoder)


def load_lora_b4_router(encoder: LoraEncoder) -> B4Router:
    classes, X, y = _lora_support()
    payload = torch.load(RESULTS / "gate5_b4_full_lora.pt", map_location="cpu", weights_only=False)
    from src.gate4.b4_core import B4AR

    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    threshold = json.loads((RESULTS / "gate5_b4_lora_threshold.json").read_text(encoding="utf-8"))
    return B4Router(classes, X, y, model, float(threshold["tau"]), encoder=encoder)


def _lora_support():
    cache = ed.load_encoder("lora")
    X, y = ed.support(cache)
    return lc.classes(), X, y


def frozen_b3h():
    return load_b3h_router(FrozenEncoder())


def frozen_b4():
    return load_b4_router(FrozenEncoder())
