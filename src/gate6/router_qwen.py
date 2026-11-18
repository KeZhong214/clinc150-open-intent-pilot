"""Real Qwen3-Embedding routers for the Gate-5 Agent (frozen and LoRA)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402
from src.gate4.router import B3HRouter, B4Router, RouteResult  # noqa: E402
from src.gate3.methods import knn_scores, prototypes_multi, scores_from_prototypes  # noqa: E402


class QwenEncoderAdapter:
    def __init__(self, kind: str = "frozen", *, device: str = "cuda"):
        self.kind = kind
        self.tokenizer = qe.get_tokenizer("right")
        if kind == "frozen":
            self.model = qe.get_base(device=device)
        else:
            train = json.loads((RESULTS / f"gate6_qwen_{kind}_training.json").read_text(encoding="utf-8"))
            self.model = qe.load_lora(ROOT / train["adapter_dir"], device=device)
        self.device = device

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vectors, _ = qe.encode_texts(self.model, self.tokenizer, list(texts), device=self.device, batch_size=int(kwargs.get("batch_size", 8)))
        return vectors.astype(np.float32)

    def close(self):
        del self.model
        torch.cuda.empty_cache()


class QwenRejectOnlyRouter(B4Router):
    method = "B4-RejectOnly"

    def route_vec(self, vec: np.ndarray) -> RouteResult:
        p = scores_from_prototypes(vec, self.protos)
        n = knn_scores(vec, self.support_X, self.support_y, self.classes, top_k=2)
        s_np = 0.5 * p + 0.5 * n
        g_np = np.full_like(s_np, 0.5, dtype=np.float32)
        with torch.no_grad():
            reject = torch.sigmoid(
                self.model.reject_logit(
                    self.model.reject_features(torch.from_numpy(p), torch.from_numpy(n), torch.from_numpy(s_np), torch.from_numpy(g_np), torch.from_numpy(self.d))
                )
            ).numpy()
        score = float(reject[0])
        accepted = score >= self.tau
        order = np.argsort(-s_np[0])
        top3 = [(self.classes[i], float(s_np[0, i])) for i in order[:3]]
        return RouteResult(
            method=self.method,
            intent=self.classes[order[0]] if accepted else "OOS",
            score=score,
            threshold=self.tau,
            accepted=accepted,
            top3=top3,
            gate_top1=0.5,
            reject_score=score,
            margin=top3[0][1] - top3[1][1],
        )


def _load_b4(encoder: str, method: str) -> B4AR:
    payload = torch.load(RESULTS / f"gate6_qwen_{encoder}_{method}.pt", map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def load_qwen_router(name: str, *, encoder_cache: dict[str, QwenEncoderAdapter] | None = None):
    """name: Qwen-Frozen-B3H / Qwen-Plain-B3H / Qwen-Plain-RejectOnly / Qwen-Plain-B4Full."""
    cache_map = {"Qwen-Frozen": "frozen", "Qwen-Plain": "plain"}
    prefix = next(k for k in cache_map if name.startswith(k))
    kind = cache_map[prefix]
    cache = qd.load(kind)
    classes = qe.classes()
    X, y = qd.support(cache)
    holder = encoder_cache if encoder_cache is not None else {}
    if kind not in holder:
        holder[kind] = QwenEncoderAdapter(kind)
    encoder = holder[kind]
    if name.endswith("-B3H"):
        if kind == "frozen":
            thresholds = json.loads((RESULTS / "gate3_validation_thresholds.json").read_text(encoding="utf-8"))
            tau = float(thresholds["banking77|qwen|10|B3-H"]["tau"])
        else:
            val = pd.read_csv(RESULTS / "gate6_qwen_lora_validation.csv")
            tau = float(val[(val.run == "plain") & (val.method == "B3-H")]["tau"].iloc[0])
        return B3HRouter(classes, X, y, tau, encoder=encoder)
    suffix = name.split("-")[-1]
    if suffix == "RejectOnly":
        tau = json.loads((RESULTS / "gate6_qwen_plain_reject_threshold.json").read_text(encoding="utf-8"))["tau"]
        return QwenRejectOnlyRouter(classes, X, y, _load_b4("plain", "reject"), tau, encoder=encoder)
    if suffix == "B4Full":
        tau = json.loads((RESULTS / "gate6_qwen_plain_full_threshold.json").read_text(encoding="utf-8"))["tau"]
        return B4Router(classes, X, y, _load_b4("plain", "full"), tau, encoder=encoder)
    raise ValueError(name)
