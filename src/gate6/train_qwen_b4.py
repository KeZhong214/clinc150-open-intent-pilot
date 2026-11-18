"""G6-C: train Qwen B4-RejectOnly / B4-Full on frozen or LoRA support."""

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
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402


def val_scores(model: B4AR, cache: dict, query: np.ndarray, classes, *, reject_only: bool):
    X, y = qd.support(cache)
    p, n, d, _ = b4_core.compute_pn(X, y, classes, query)
    pt, nt, dt = torch.from_numpy(p), torch.from_numpy(n), torch.from_numpy(d)
    with torch.no_grad():
        if reject_only:
            s = 0.5 * pt + 0.5 * nt
            g = torch.full_like(s, 0.5)
        else:
            s, g = model.fused(pt, nt, dt)
        reject = torch.sigmoid(model.reject_logit(model.reject_features(pt, nt, s, g, dt)))
    return reject.numpy(), s.numpy().argmax(axis=1), s.numpy(), g.numpy()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("frozen", "plain", "anchor"), required=True)
    parser.add_argument("--method", choices=("reject", "full"), required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache = qd.load(args.encoder)
    classes = qe.classes()
    X, y = qd.support(cache)
    cal_X, cal_y = qd.role(cache, "calibration_known", classes)
    dev_X, dev_y = qd.role(cache, "dev_unknown", classes)
    val_X = np.concatenate([cal_X, dev_X])
    val_y = np.concatenate([cal_y, dev_y])
    episodes = b4_core.build_episodes(X, y, classes, seed=args.seed, n_masks=12, mask_size=6)
    reject_only = args.method == "reject"
    model = B4AR()
    if reject_only:
        for p in model.gate_mlp.parameters():
            p.requires_grad = False
    params_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_train, lr=1e-3, weight_decay=1e-4)
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        order = rng.permutation(len(episodes))
        total = 0.0
        n = 0
        grads = []
        for idx in order:
            te = b4_core.tensor_episode(episodes[idx])
            optimizer.zero_grad(set_to_none=True)
            if reject_only:
                s_k = 0.5 * te["p_known"] + 0.5 * te["n_known"]
                g_k = torch.full_like(s_k, 0.5)
                s_o = 0.5 * te["p_oos"] + 0.5 * te["n_oos"]
                g_o = torch.full_like(s_o, 0.5)
                ce = torch.zeros((), device="cpu")
            else:
                s_k, g_k = model.fused(te["p_known"], te["n_known"], te["d"])
                s_o, g_o = model.fused(te["p_oos"], te["n_oos"], te["d"])
                ce = F.cross_entropy(s_k, te["y_known"])
            k = model.reject_logit(model.reject_features(te["p_known"], te["n_known"], s_k, g_k, te["d"]))
            o = model.reject_logit(model.reject_features(te["p_oos"], te["n_oos"], s_o, g_o, te["d"]))
            bce = 0.5 * (F.binary_cross_entropy_with_logits(k, torch.ones_like(k)) + F.binary_cross_entropy_with_logits(o, torch.zeros_like(o)))
            loss = (0.5 * bce) if reject_only else (ce + 0.5 * bce)
            loss.backward()
            grads.append(float(torch.nn.utils.clip_grad_norm_(params_train, 1.0)))
            optimizer.step()
            total += float(loss.detach()) * (len(te["y_known"]) + len(te["p_oos"]))
            n += len(te["y_known"]) + len(te["p_oos"])
        history.append(
            {
                "encoder": args.encoder,
                "method": args.method,
                "epoch": epoch,
                "loss": total / max(n, 1),
                "grad_norm": float(np.mean(grads)),
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }
        )
        if epoch in (1, 10, 30, args.epochs):
            print(f"{args.encoder} {args.method} epoch={epoch} loss={history[-1]['loss']:.4f} grad={history[-1]['grad_norm']:.4f}")

    reject, class_idx, s, g = val_scores(model, cache, val_X, classes, reject_only=reject_only)
    tau, _, info = b4_core.select_threshold_custom(reject, class_idx, val_y, classes)
    metrics, pred, accept = b4_core.evaluate_custom(reject, class_idx, val_y, tau, classes)
    stem = f"gate6_qwen_{args.encoder}_{args.method}"
    ckpt = RESULTS / f"{stem}.pt"
    b4_core.save_checkpoint(
        ckpt,
        model,
        {
            "stage": f"Qwen-B4-{args.method}",
            "encoder": args.encoder,
            "classes": classes,
            "shot": 10,
            "gate": "fixed_0.5" if reject_only else "learned",
            "seed": args.seed,
            "epochs": args.epochs,
        },
    )
    pd.DataFrame(history).to_csv(RESULTS / f"{stem}_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "encoder": f"qwen_{args.encoder}_lora" if args.encoder != "frozen" else "qwen_frozen",
                "method": "B4-RejectOnly" if reject_only else "B4-Full",
                "tau": tau,
                "constraint_met": info["constraint_met"],
                **{f"val_{k}": v for k, v in metrics.items()},
                "gate_mean": float(g.mean()),
                "gate_top1_mean": float(g[np.arange(len(s)), class_idx].mean()),
            }
        ]
    ).to_csv(RESULTS / f"{stem}_validation.csv", index=False)
    (RESULTS / f"{stem}_threshold.json").write_text(json.dumps({"tau": tau, "selection": info}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"encoder": args.encoder, "method": args.method, "metrics": metrics, "tau": tau, "gate_mean": float(g.mean())}, ensure_ascii=False, indent=2))
    print("GATE6_QWEN_B4_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
