"""Gate-8 router factory using the unified Gate7 dev_threshold protocol."""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402
from src.gate4.router import load_bank_state  # noqa: E402
from src.gate5.router_lora import FrozenEncoder  # noqa: E402
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate6.router_qwen import QwenEncoderAdapter, QwenRejectOnlyRouter  # noqa: E402

ROUTER_NAMES = ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _threshold(encoder: str) -> float:
    d = pd.read_csv(RESULTS / "gate7_dev_diagnosis.csv")
    rows = d[(d.encoder == encoder) & (d.method == "B4-RejectOnly")]
    if len(rows):
        return float(rows.tau.iloc[0])
    # GTE Frozen is not in the Qwen dev table; use its Gate7 stability seed42 tau.
    s = pd.read_csv(RESULTS / "gate7_stability_b4_results.csv")
    r = s[(s.encoder == encoder) & (s.method == "B4-RejectOnly") & (s.seed == 42)]
    if not len(r):
        raise RuntimeError(f"No Gate7 dev_threshold for {encoder}")
    return float(r.tau.iloc[0])


def metadata(name: str) -> dict:
    if name == "GTE-Frozen-RejectOnly":
        return {
            "name": name,
            "encoder": "GTE-ModernBERT Frozen",
            "threshold_source": "gate7_stability_b4_results.csv:gte_frozen/reject/seed42",
            "threshold": _threshold("gte_frozen"),
            "checkpoint": "results/gate5_reject_only_frozen.pt",
            "checkpoint_sha256": sha(RESULTS / "gate5_reject_only_frozen.pt"),
            "adapter_sha256": None,
        }
    if name == "Qwen-Frozen-RejectOnly":
        return {
            "name": name,
            "encoder": "Qwen3-Embedding-0.6B Frozen",
            "threshold_source": "gate7_dev_frozen.csv:B4-RejectOnly",
            "threshold": _threshold("qwen_frozen"),
            "checkpoint": "results/gate6_qwen_frozen_reject.pt",
            "checkpoint_sha256": sha(RESULTS / "gate6_qwen_frozen_reject.pt"),
            "adapter_sha256": None,
        }
    if name == "Qwen-Plain42-RejectOnly":
        train = json.loads((RESULTS / "gate6_qwen_plain_training.json").read_text(encoding="utf-8"))
        return {
            "name": name,
            "encoder": "Qwen3-Embedding-0.6B Plain LoRA seed42",
            "threshold_source": "gate7_dev_plain42.csv:B4-RejectOnly",
            "threshold": _threshold("qwen_plain42"),
            "checkpoint": "results/gate6_qwen_plain_reject.pt",
            "checkpoint_sha256": sha(RESULTS / "gate6_qwen_plain_reject.pt"),
            "adapter_sha256": train["adapter_sha256"],
        }
    raise ValueError(name)


def build(name: str):
    meta = metadata(name)
    if name == "GTE-Frozen-RejectOnly":
        classes, X, y = load_bank_state()
        payload = torch.load(RESULTS / "gate5_reject_only_frozen.pt", map_location="cpu", weights_only=False)
        model = B4AR()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return QwenRejectOnlyRouter(classes, X, y, model, meta["threshold"], encoder=FrozenEncoder()), meta
    cache = qd.load("frozen" if name == "Qwen-Frozen-RejectOnly" else "plain")
    classes = qe.classes()
    X, y = qd.support(cache)
    ckpt = "gate6_qwen_frozen_reject.pt" if name == "Qwen-Frozen-RejectOnly" else "gate6_qwen_plain_reject.pt"
    payload = torch.load(RESULTS / ckpt, map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    encoder = QwenEncoderAdapter("frozen" if name == "Qwen-Frozen-RejectOnly" else "plain")
    return QwenRejectOnlyRouter(classes, X, y, model, meta["threshold"], encoder=encoder), meta
