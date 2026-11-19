"""Gate 8.1 router factory backed by the paired model registry."""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402
from src.gate4.router import load_bank_state  # noqa: E402
from src.gate5.router_lora import FrozenEncoder  # noqa: E402
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate6.router_qwen import QwenEncoderAdapter, QwenRejectOnlyRouter  # noqa: E402
from src.gate8_1.registry import validate  # noqa: E402

ROUTER_NAMES = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly"]


def metadata(name: str) -> dict:
    e = validate(name)
    return {
        "name": name,
        "encoder": e["encoder"],
        "threshold": e["tau"],
        "threshold_source": e["threshold_file"],
        "checkpoint": e["checkpoint"],
        "checkpoint_sha256": e["checkpoint_sha256"],
        "adapter_sha256": e["adapter_sha256"],
        "calibration_split": e["calibration_split"],
        "paired": e["paired"],
    }


def build(name: str):
    e = validate(name)
    if name == "GTE-Frozen-RejectOnly":
        classes = qe.classes()
        cache = b4_core.load_cache()
        import pandas as pd

        rows = pd.read_csv(ROOT / "data" / "manifest" / "gate3_banking77_support_10.csv")
        idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
        X = cache["vectors"][[idx[s] for s in rows.sample_id]]
        y = rows.intent.to_numpy()
        payload = torch.load(ROOT / e["checkpoint"], map_location="cpu", weights_only=False)
        model = B4AR()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return QwenRejectOnlyRouter(classes, X, y, model, e["tau"], encoder=FrozenEncoder()), metadata(name)
    cache = qd.load("frozen" if name == "Qwen-Frozen-RejectOnly" else "plain")
    classes = qe.classes()
    X, y = qd.support(cache)
    payload = torch.load(ROOT / e["checkpoint"], map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    encoder = QwenEncoderAdapter("frozen" if name == "Qwen-Frozen-RejectOnly" else "plain")
    return QwenRejectOnlyRouter(classes, X, y, model, e["tau"], encoder=encoder), metadata(name)
