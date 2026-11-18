"""G5-B: B4-RejectOnly — fixed g=0.5, train only the reject head."""

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
    parser.add_argument("--encoder", choices=("frozen", "lora"), required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache = ed.load_encoder(args.encoder)
    classes = lc.classes()
    X, y = ed.support(cache)
    episodes = b4_core.build_episodes(X, y, classes, seed=args.seed, n_masks=12, mask_size=6)
    cal_X, cal_y = ed.role(cache, "calibration_known", classes)
    dev_X, dev_y = ed.role(cache, "dev_unknown", classes)
    val_X = np.concatenate([cal_X, dev_X])
    val_y = np.concatenate([cal_y, dev_y])

    model = B4AR()
    for p in model.gate_mlp.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in model.reject_mlp.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        grad_norms = []
        for ep in episodes:
            te = b4_core.tensor_episode(ep)
            s_k = 0.5 * te["p_known"] + 0.5 * te["n_known"]
            g_k = torch.full_like(s_k, 0.5)
            s_o = 0.5 * te["p_oos"] + 0.5 * te["n_oos"]
            g_o = torch.full_like(s_o, 0.5)
            logit_k = model.reject_logit(model.reject_features(te["p_known"], te["n_known"], s_k, g_k, te["d"]))
            logit_o = model.reject_logit(model.reject_features(te["p_oos"], te["n_oos"], s_o, g_o, te["d"]))
            bce_k = F.binary_cross_entropy_with_logits(logit_k, torch.ones_like(logit_k))
            bce_o = F.binary_cross_entropy_with_logits(logit_o, torch.zeros_like(logit_o))
            loss = 0.5 * (bce_k + bce_o)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_([p for p in model.reject_mlp.parameters()], 1.0))
            optimizer.step()
            grad_norms.append(grad)
            total += float(loss.detach()) * (len(te["y_known"]) + len(te["p_oos"]))
            n += len(te["y_known"]) + len(te["p_oos"])
        history.append(
            {
                "epoch": epoch,
                "train_bce": total / max(n, 1),
                "reject_grad_norm": float(np.mean(grad_norms)),
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }
        )
        if epoch in (1, 10, 30, args.epochs):
            print(f"{args.encoder} RejectOnly epoch={epoch} bce={history[-1]['train_bce']:.4f} grad={history[-1]['reject_grad_norm']:.4f}")

    scores = ed.b4_scores(model, cache, val_X, classes, reject_only=True)
    tau, grid, info = b4_core.select_threshold_custom(scores["reject"], scores["class_idx"], val_y, classes)
    metrics, pred, accept = b4_core.evaluate_custom(scores["reject"], scores["class_idx"], val_y, tau, classes)
    ckpt = RESULTS / f"gate5_reject_only_{args.encoder}.pt"
    b4_core.save_checkpoint(
        ckpt,
        model,
        {
            "stage": "B4-RejectOnly",
            "encoder": args.encoder,
            "classes": classes,
            "shot": 10,
            "gate": "fixed_0.5",
            "seed": args.seed,
            "epochs": args.epochs,
        },
    )
    pd.DataFrame(history).to_csv(RESULTS / f"gate5_reject_only_{args.encoder}_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "encoder": args.encoder,
                "method": "B4-RejectOnly",
                "tau": tau,
                "constraint_met": info["constraint_met"],
                **{f"val_{k}": v for k, v in metrics.items()},
            }
        ]
    ).to_csv(RESULTS / f"gate5_reject_only_{args.encoder}_validation.csv", index=False)
    (RESULTS / f"gate5_reject_only_{args.encoder}_threshold.json").write_text(
        json.dumps({"tau": tau, "selection": info}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"encoder": args.encoder, "metrics": metrics, "tau": tau, "info": info}, ensure_ascii=False, indent=2))
    print("GATE5_REJECT_ONLY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
