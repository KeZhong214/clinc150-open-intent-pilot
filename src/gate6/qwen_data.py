"""Qwen frozen/LoRA cache views for Gate-6 B4 experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_encoder as qe  # noqa: E402


def load(encoder: str) -> dict[str, Any]:
    if encoder == "frozen":
        c = qe.frozen_cache()
        c["encoder"] = "qwen_frozen"
        c["run"] = "frozen"
        return c
    if encoder in ("plain", "anchor"):
        train = json.loads((RESULTS / f"gate6_qwen_{encoder}_training.json").read_text(encoding="utf-8"))
        path = qe.lora_cache(train["adapter_sha256"])
        d = np.load(path, allow_pickle=False)
        return {
            "vectors": d["vectors"].astype(np.float32),
            "sample_ids": d["sample_ids"].astype(str),
            "roles": d["roles"].astype(str),
            "labels": d["labels"].astype(str),
            "intents": d["intents"].astype(str),
            "path": str(path),
            "encoder": f"qwen_{encoder}_lora",
            "run": encoder,
            "training": train,
        }
    raise ValueError(encoder)


def support(cache: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = qe.support_rows()
    idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    return cache["vectors"][[idx[r["sample_id"]] for r in rows]], np.asarray([r["intent"] for r in rows], dtype=object)


def role(cache: dict, name: str, classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    mask = cache["roles"] == name
    labels = cache["labels"][mask]
    cmap = {c: i for i, c in enumerate(classes)}
    return cache["vectors"][mask], np.asarray([cmap.get(x, -1) for x in labels], dtype=int)
