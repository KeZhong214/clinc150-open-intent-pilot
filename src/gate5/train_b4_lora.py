"""G5-B: retrain B4-Full on LoRA-GTE embeddings (support-only episodes)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate5 import encoder_data as ed  # noqa: E402
from src.gate5 import lora_common as lc  # noqa: E402
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache = ed.load_encoder("lora")
    classes = lc.classes()
    X, y = ed.support(cache)
    episodes = b4_core.build_episodes(X, y, classes, seed=args.seed, n_masks=12, mask_size=6)
    cal_X, cal_y = ed.role(cache, "calibration_known", classes)
    dev_X, dev_y = ed.role(cache, "dev_unknown", classes)
    val_X = np.concatenate([cal_X, dev_X])
    val_y = np.concatenate([cal_y, dev_y])

    model = B4AR()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(5000 + epoch)
        order = rng.permutation(len(episodes))
        total = 0.0
        n = 0
        grad_norms = []
        for idx in order:
            ep = episodes[idx]
            te = b4_core.tensor_episode(ep)
            optimizer.zero_grad(set_to_none=True)
            s_k, g_k = model.fused(te["p_known"], te["n_known"], te["d"])
            ce = F.cross_entropy(s_k, te["y_known"])
            k = model.reject_logit(model.reject_features(te["p_known"], te["n_known"], s_k, g_k, te["d"]))
            s_o, g_o = model.fused(te["p_oos"], te["n_oos"], te["d"])
            o = model.reject_logit(model.reject_features(te["p_oos"], te["n_oos"], s_o, g_o, te["d"]))
            bce = 0.5 * (F.binary_cross_entropy_with_logits(k, torch.ones_like(k)) + F.binary_cross_entropy_with_logits(o, torch.zeros_like(o)))
            loss = ce + 0.5 * bce
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            grad_norms.append(grad)
            total += float(loss.detach()) * (len(te["y_known"]) + len(te["p_oos"]))
            n += len(te["y_known"]) + len(te["p_oos"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(n, 1),
                "grad_norm": float(np.mean(grad_norms)),
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }
        )
        if epoch in (1, 10, 30, args.epochs):
            print(f"LoRA B4-Full epoch={epoch} loss={history[-1]['train_loss']:.4f} grad={history[-1]['grad_norm']:.3f}")

    scores = ed.b4_scores(model, cache, val_X, classes, reject_only=False)
    tau, grid, info = b4_core.select_threshold_custom(scores["reject"], scores["class_idx"], val_y, classes)
    metrics, pred, accept = b4_core.evaluate_custom(scores["reject"], scores["class_idx"], val_y, tau, classes)
    ckpt = RESULTS / "gate5_b4_full_lora.pt"
    b4_core.save_checkpoint(
        ckpt,
        model,
        {
            "stage": "B4-Full-LoRA",
            "encoder": "lora",
            "adapter_hash": cache["adapter_hash"],
            "classes": classes,
            "shot": 10,
            "seed": args.seed,
            "epochs": args.epochs,
            "alpha_oos": 0.5,
        },
    )
    pd.DataFrame(history).to_csv(RESULTS / "gate5_b4_lora_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "encoder": "lora",
                "method": "B4-Full",
                "tau": tau,
                "constraint_met": info["constraint_met"],
                **{f"val_{k}": v for k, v in metrics.items()},
            }
        ]
    ).to_csv(RESULTS / "gate5_b4_lora_validation.csv", index=False)
    (RESULTS / "gate5_b4_lora_threshold.json").write_text(
        json.dumps({"tau": tau, "selection": info, "gate_mean": float(scores["g"].mean()), "gate_top1_mean": float(scores["g"][np.arange(len(scores["s"])), scores["class_idx"]].mean())}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metrics": metrics, "tau": tau, "info": info, "gate_mean": float(scores["g"].mean()), "gate_top1_mean": float(scores["g"][np.arange(len(scores["s"])), scores["class_idx"]].mean())}, ensure_ascii=False, indent=2))
    print("GATE5_B4_LORA_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
