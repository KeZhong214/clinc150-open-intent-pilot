"""Real B3-H / B4-Full routers for the customer-service agent."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest"
CACHE = ROOT / "data" / "cache_embeddings" / "gate3"
MODEL_CACHE = ROOT / "data" / "model_cache"
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate3.methods import knn_scores, prototypes_multi, scores_from_prototypes  # noqa: E402
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402

GTE_REV = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
GTE_PROMPT_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@dataclass
class RouteResult:
    method: str
    intent: str
    score: float
    threshold: float
    accepted: bool
    top3: list[tuple[str, float]]
    gate_top1: float | None = None
    reject_score: float | None = None
    margin: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def load_gte() -> SentenceTransformer:
    model = SentenceTransformer(
        "Alibaba-NLP/gte-modernbert-base",
        revision=GTE_REV,
        device="cpu",
        cache_folder=str(MODEL_CACHE),
        local_files_only=True,
    )
    model.max_seq_length = 128
    model.eval()
    return model


def load_bank_state() -> tuple[list[str], np.ndarray, np.ndarray]:
    info = json.loads((MANIFEST / "gate3_banking77_dataset.json").read_text(encoding="utf-8"))
    classes = info["known_classes"]
    rows = list(csv.DictReader((MANIFEST / "gate3_banking77_support_10.csv").open("r", encoding="utf-8")))
    cache = b4_core.load_cache()
    ids = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    idx = [ids[row["sample_id"]] for row in rows]
    X = cache["vectors"][idx].astype(np.float32)
    y = np.asarray([row["intent"] for row in rows], dtype=object)
    return classes, X, y


class B3HRouter:
    method = "B3-H"

    def __init__(self, classes, support_X, support_y, tau: float, encoder=None):
        self.classes = classes
        self.support_X = support_X
        self.support_y = support_y
        self.tau = float(tau)
        self.encoder = encoder
        self.protos, _ = prototypes_multi(support_X, support_y, classes)

    def encode(self, text: str) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded")
        with torch.inference_mode():
            return self.encoder.encode(
                [text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)

    def route_vec(self, vec: np.ndarray) -> RouteResult:
        p = scores_from_prototypes(vec, self.protos)
        n = knn_scores(vec, self.support_X, self.support_y, self.classes, top_k=2)
        s = 0.5 * p + 0.5 * n
        order = np.argsort(-s[0])
        top3 = [(self.classes[i], float(s[0, i])) for i in order[:3]]
        accepted = top3[0][1] >= self.tau
        return RouteResult(
            method=self.method,
            intent=top3[0][0] if accepted else "OOS",
            score=top3[0][1],
            threshold=self.tau,
            accepted=accepted,
            top3=top3,
            margin=top3[0][1] - top3[1][1],
        )

    def route(self, text: str) -> RouteResult:
        return self.route_vec(self.encode(text))


class B4Router:
    method = "B4-Full"

    def __init__(self, classes, support_X, support_y, model: B4AR, tau: float, encoder=None):
        self.classes = classes
        self.support_X = support_X
        self.support_y = support_y
        self.model = model.eval()
        self.tau = float(tau)
        self.encoder = encoder
        self.protos, details = prototypes_multi(support_X, support_y, classes)
        self.d = np.asarray([details["dispersion"][c] for c in classes], dtype=np.float32)
        self.p = None
        self.n = None

    def encode(self, text: str) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded")
        with torch.inference_mode():
            return self.encoder.encode(
                [text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)

    def route_vec(self, vec: np.ndarray) -> RouteResult:
        p = scores_from_prototypes(vec, self.protos)
        n = knn_scores(vec, self.support_X, self.support_y, self.classes, top_k=2)
        with torch.no_grad():
            s, g = self.model.fused(torch.from_numpy(p), torch.from_numpy(n), torch.from_numpy(self.d))
            reject = torch.sigmoid(
                self.model.reject_logit(
                    self.model.reject_features(
                        torch.from_numpy(p), torch.from_numpy(n), s, g, torch.from_numpy(self.d)
                    )
                )
            )
        s_np = s.numpy()
        g_np = g.numpy()
        score = float(reject.numpy()[0])
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
            gate_top1=float(g_np[0, order[0]]),
            reject_score=score,
            margin=top3[0][1] - top3[1][1],
        )

    def route(self, text: str) -> RouteResult:
        return self.route_vec(self.encode(text))


def load_b4_router(encoder=None) -> B4Router:
    classes, X, y = load_bank_state()
    payload = torch.load(RESULTS / "gate4_b4_full.pt", map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    thresholds = json.loads((RESULTS / "gate4_b4_thresholds.json").read_text(encoding="utf-8"))
    tau = float(thresholds["B4-Full"]["tau"])
    return B4Router(classes, X, y, model, tau, encoder=encoder)


def load_b3h_router(encoder=None) -> B3HRouter:
    classes, X, y = load_bank_state()
    thresholds = json.loads((RESULTS / "gate3_validation_thresholds.json").read_text(encoding="utf-8"))
    tau = float(thresholds["banking77|gte|10|B3-H"]["tau"])
    return B3HRouter(classes, X, y, tau, encoder=encoder)
