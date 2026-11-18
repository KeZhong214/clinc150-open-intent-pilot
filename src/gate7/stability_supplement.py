"""Safe one-config supplement for the missing qwen_plain42/B4-Full/seed44."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate7.stability_b4 import load_cache, support, split, b4_scores  # noqa: E402
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402


def main() -> int:
    path = RESULTS / "gate7_stability_b4_results.csv"
    existing = pd.read_csv(path)
    if len(existing) != 35:
        raise RuntimeError(f"Expected 35 existing rows, got {len(existing)}")
    key = ("qwen_plain42", "B4-Full", 44)
    if ((existing.encoder == key[0]) & (existing.method == key[1]) & (existing.seed == key[2])).any():
        print("Supplement already exists; refusing overwrite")
        return 2
    encoder, method, seed = key
    cache = load_cache(encoder)
    classes = None
    from src.gate6 import qwen_encoder as qe

    classes = qe.classes()
    X, y = support(cache, classes)
    qthr, ythr = split(cache, "dev_threshold")
    qsel, ysel = split(cache, "dev_select")
    torch.manual_seed(seed)
    np.random.seed(seed)
    episodes = b4_core.build_episodes(X, y, classes, seed=seed, n_masks=12, mask_size=6)
    model = B4AR()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    hist = []
    for epoch in range(1, 61):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(episodes))
        total, n = 0.0, 0
        for idx in order:
            te = b4_core.tensor_episode(episodes[idx])
            opt.zero_grad(set_to_none=True)
            sk, gk = model.fused(te["p_known"], te["n_known"], te["d"])
            so, go = model.fused(te["p_oos"], te["n_oos"], te["d"])
            ce = F.cross_entropy(sk, te["y_known"])
            k = model.reject_logit(model.reject_features(te["p_known"], te["n_known"], sk, gk, te["d"]))
            o = model.reject_logit(model.reject_features(te["p_oos"], te["n_oos"], so, go, te["d"]))
            bce = 0.5 * (F.binary_cross_entropy_with_logits(k, torch.ones_like(k)) + F.binary_cross_entropy_with_logits(o, torch.zeros_like(o)))
            loss = ce + 0.5 * bce
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            total += float(loss.detach()) * (len(te["y_known"]) + len(te["p_oos"]))
            n += len(te["y_known"]) + len(te["p_oos"])
        hist.append({"seed": seed, "method": "Full", "encoder": encoder, "epoch": epoch, "loss": total / max(n, 1)})
    model.eval()
    sth, cth, _, _ = b4_scores(model, cache, qthr, classes, False)
    sse, cse, _, _ = b4_scores(model, cache, qsel, classes, False)
    tau, _, info = b4_core.select_threshold_custom(sth, cth, ythr, classes)
    metrics, _, _ = b4_core.evaluate_custom(sse, cse, ysel, tau, classes)
    stem = f"gate7_stability_{encoder}_full_seed{seed}"
    b4_core.save_checkpoint(RESULTS / f"{stem}.pt", model, {"stage": "gate7-stability-supplement", "encoder": encoder, "method": "full", "seed": seed, "gate": "learned", "epochs": 60, "classes": classes})
    pd.DataFrame(hist).to_csv(RESULTS / f"{stem}_history.csv", index=False)
    row = {"encoder": encoder, "method": method, "seed": seed, "tau": tau, "constraint_met": info["constraint_met"], "elapsed_seconds": round(time.perf_counter() - t0, 3), **{f"dev_{k}": v for k, v in metrics.items()}}
    merged = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    if len(merged) != 36:
        raise RuntimeError(len(merged))
    if merged.duplicated(["encoder", "method", "seed"]).any():
        raise RuntimeError("duplicate stability key")
    merged.to_csv(path, index=False)
    (RESULTS / "gate7_stability_supplement.json").write_text(json.dumps({"added": row, "existing_rows": 35, "final_rows": len(merged), "checkpoint": f"results/{stem}.pt", "history": f"results/{stem}_history.csv"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print("GATE7_STABILITY_SUPPLEMENT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
