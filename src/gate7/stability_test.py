"""G7 stability: one historical-exploratory test pass for B4 seed checkpoints.

The test has already been viewed in earlier gates; this is explicitly labeled
exploratory. No seed/method is selected from these numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate7.stability_b4 import load_cache, split, b4_scores  # noqa: E402
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402


def main() -> int:
    rows = pd.read_csv(RESULTS / "gate7_stability_b4_results.csv")
    out = []
    classes = qe.classes()
    for _, r in rows.iterrows():
        enc = r.encoder
        method = "reject" if r.method == "B4-RejectOnly" else "full"
        ckpt = RESULTS / f"gate7_stability_{enc}_{method}_seed{int(r.seed)}.pt"
        if not ckpt.exists():
            continue
        cache = load_cache(enc)
        idx = {sid: i2 for i2, sid in enumerate(cache["sample_ids"])}
        m = pd.read_csv(ROOT / "data" / "manifest" / "gate3_banking77_manifest.csv")
        k = m[m.role == "known_test"]
        u = m[m.role == "final_unknown"]
        cmap = {c: i2 for i2, c in enumerate(classes)}
        q = np.concatenate([
            cache["vectors"][[idx[s] for s in k.sample_id]],
            cache["vectors"][[idx[s] for s in u.sample_id]],
        ])
        y = np.concatenate([
            np.asarray([cmap[c] for c in k.intent], dtype=int),
            np.asarray([cmap.get(c, -1) for c in u.intent], dtype=int),
        ])
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = B4AR()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        reject, class_idx, _, _ = b4_scores(model, cache, q, classes, method == "reject")
        metrics, _, _ = b4_core.evaluate_custom(reject, class_idx, y, float(r.tau), classes)
        out.append({"encoder": enc, "method": r.method, "seed": int(r.seed), "tau": float(r.tau), "historical_exploratory": True, **metrics})
    df = pd.DataFrame(out)
    df.to_csv(RESULTS / "gate7_stability_historical_test.csv", index=False)
    print(df.round(5).to_string(index=False))
    print("GATE7_STABILITY_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
