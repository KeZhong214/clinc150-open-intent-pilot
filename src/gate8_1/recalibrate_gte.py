"""Gate 8.1: pair the Gate5 GTE reject-only checkpoint with a Gate7 dev tau."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "manifest"

import sys

sys.path.insert(0, str(ROOT))
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402
from src.gate7.stability_b4 import load_cache, b4_scores, support  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ckpt = RESULTS / "gate5_reject_only_frozen.pt"
    cache = load_cache("gte_frozen")
    classes = qe.classes()
    X, ysup = support(cache, classes)
    rows = pd.read_csv(MANIFEST / "gate7_dev_threshold.csv")
    idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    q = cache["vectors"][[idx[sid] for sid in rows.sample_id]]
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.asarray([cmap.get(c, -1) for c in rows.intent], dtype=int)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    reject, class_idx, _, _ = b4_scores(model, cache, q, classes, True)
    tau, candidates, info = b4_core.select_threshold_custom(reject, class_idx, y, classes, target_acceptance=0.90)
    metrics, _, _ = b4_core.evaluate_custom(reject, class_idx, y, tau, classes)
    out = {
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "checkpoint_sha256": sha(ckpt),
        "encoder": "GTE-ModernBERT Frozen",
        "calibration_split": "data/manifest/gate7_dev_threshold.csv",
        "calibration_split_sha256": sha(MANIFEST / "gate7_dev_threshold.csv"),
        "target_acceptance": 0.90,
        "tau": float(tau),
        "metrics": metrics,
        "selection": info,
        "n_candidates": len(candidates),
        "source": "Gate8.1 paired recalibration of Gate5 GTE weights",
        "not_used": "Gate7 stability gte_frozen seed42 checkpoint (different SHA) is not used by this router",
    }
    (RESULTS / "gate8_1_gte_recalibrated_threshold.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("GATE8_1_GTE_RECALIBRATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
