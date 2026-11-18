"""Shared Gate-5 encoder caches, support/validation/test views and B4 inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "manifest"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5 import lora_common as lc  # noqa: E402
from src.gate3.methods import knn_scores, prototypes_multi, scores_from_prototypes  # noqa: E402
from src.gate4.b4_core import B4AR, compute_pn  # noqa: E402


def load_encoder(encoder: str) -> dict[str, Any]:
    if encoder == "frozen":
        cache = lc.load_frozen_cache()
        cache["encoder"] = "frozen"
        cache["adapter_hash"] = None
        return cache
    if encoder == "lora":
        training = json.loads((RESULTS / "gate5_lora_training.json").read_text(encoding="utf-8"))
        path = lc.lora_cache_path(training["adapter_sha256"])
        data = np.load(path, allow_pickle=False)
        return {
            "vectors": data["vectors"].astype(np.float32),
            "sample_ids": data["sample_ids"].astype(str),
            "roles": data["roles"].astype(str),
            "labels": data["labels"].astype(str),
            "intents": data["intents"].astype(str),
            "path": str(path),
            "encoder": "lora",
            "adapter_hash": training["adapter_sha256"],
            "training": training,
        }
    raise ValueError(encoder)


def support(cache: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = lc.support_rows()
    idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    return cache["vectors"][[idx[r["sample_id"]] for r in rows]], np.asarray([r["intent"] for r in rows], dtype=object)


def role(cache: dict, name: str, classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    mask = cache["roles"] == name
    labels = cache["labels"][mask]
    cmap = {c: i for i, c in enumerate(classes)}
    return cache["vectors"][mask], np.asarray([cmap.get(x, -1) for x in labels], dtype=int)


def load_b4(path: Path) -> B4AR:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def pn_for(cache: dict, query: np.ndarray, classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = support(cache)
    return compute_pn(X, y, classes, query)


def b4_scores(
    model: B4AR,
    cache: dict,
    query: np.ndarray,
    classes: Sequence[str],
    *,
    reject_only: bool = False,
) -> dict[str, np.ndarray]:
    p, n, d, _ = pn_for(cache, query, classes)
    pt, nt, dt = torch.from_numpy(p), torch.from_numpy(n), torch.from_numpy(d)
    with torch.no_grad():
        if reject_only:
            s = 0.5 * pt + 0.5 * nt
            g = torch.full_like(s, 0.5)
        else:
            s, g = model.fused(pt, nt, dt)
        reject = torch.sigmoid(model.reject_logit(model.reject_features(pt, nt, s, g, dt)))
    return {
        "p": p,
        "n": n,
        "d": d,
        "s": s.numpy(),
        "g": g.numpy(),
        "reject": reject.numpy(),
        "class_idx": s.numpy().argmax(axis=1),
    }
