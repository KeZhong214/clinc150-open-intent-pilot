"""G7 stability: B4-RejectOnly / B4-Full across seeds on Gate-7 dev splits.

Seeds change initialization, episode masks, sample order and pseudo-OOS sampling.
Thresholds are selected only on dev_threshold; dev_select is reported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "manifest"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate5 import lora_common as lc  # noqa: E402
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402


def load_cache(name: str) -> dict:
    if name == "gte_frozen":
        c = b4_core.load_cache()
        c["encoder"] = "gte_frozen"
        return c
    if name.startswith("gte_lora"):
        tag = name.replace("gte_lora", "")
        if tag == "42":
            train_path = RESULTS / "gate5_lora_training.json"
        else:
            train_path = RESULTS / f"gate7_gte{tag}_training.json"
        train = json.loads(train_path.read_text(encoding="utf-8"))
        path = lc.lora_cache_path(train["adapter_sha256"])
        d = np.load(path, allow_pickle=False)
        return {"vectors": d["vectors"].astype(np.float32), "sample_ids": d["sample_ids"].astype(str), "roles": d["roles"].astype(str), "labels": d["labels"].astype(str), "intents": d["intents"].astype(str), "encoder": name}
    if name == "qwen_frozen":
        c = qd.load("frozen")
        c["encoder"] = "qwen_frozen"
        return c
    tag = name.replace("qwen_", "")
    train_path = RESULTS / (f"gate7_qwen_{tag}_training.json" if tag != "plain42" else "gate6_qwen_plain_training.json")
    train = json.loads(train_path.read_text(encoding="utf-8"))
    d = np.load(qe.lora_cache(train["adapter_sha256"]), allow_pickle=False)
    return {"vectors": d["vectors"].astype(np.float32), "sample_ids": d["sample_ids"].astype(str), "roles": d["roles"].astype(str), "labels": d["labels"].astype(str), "intents": d["intents"].astype(str), "encoder": name}


def support(cache, classes):
    rows = list(pd.read_csv(MANIFEST / "gate3_banking77_support_10.csv").itertuples())
    idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    X = cache["vectors"][[idx[r.sample_id] for r in rows]]
    y = np.asarray([r.intent for r in rows], dtype=object)
    return X, y


def split(cache, name):
    rows = pd.read_csv(MANIFEST / f"gate7_{name}.csv")
    classes = qe.classes()
    cmap = {c: i for i, c in enumerate(classes)}
    idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    X = cache["vectors"][[idx[sid] for sid in rows.sample_id]]
    y = np.asarray([cmap.get(c, -1) for c in rows.intent], dtype=int)
    return X, y


def b4_scores(model, cache, query, classes, reject_only):
    X, y = support(cache, classes)
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoders", required=True, help="comma-separated gte_frozen,gte_lora43,qwen_frozen,qwen_plain42,qwen_p1")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args(argv)
    classes = qe.classes()
    encoders = args.encoders.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    for enc in encoders:
        cache = load_cache(enc)
        X, y = support(cache, classes)
        qthr, ythr = split(cache, "dev_threshold")
        qsel, ysel = split(cache, "dev_select")
        episodes = None
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            episodes = b4_core.build_episodes(X, y, classes, seed=seed, n_masks=12, mask_size=6)
            for method in ("reject", "full"):
                reject_only = method == "reject"
                model = B4AR()
                if reject_only:
                    for q in model.gate_mlp.parameters():
                        q.requires_grad = False
                opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-3, weight_decay=1e-4)
                t0 = time.perf_counter()
                hist = []
                for epoch in range(1, args.epochs + 1):
                    model.train()
                    rng = np.random.default_rng(seed + epoch)
                    order = rng.permutation(len(episodes))
                    total = 0.0
                    n = 0
                    for idx in order:
                        te = b4_core.tensor_episode(episodes[idx])
                        opt.zero_grad(set_to_none=True)
                        if reject_only:
                            sk = 0.5 * te["p_known"] + 0.5 * te["n_known"]
                            so = 0.5 * te["p_oos"] + 0.5 * te["n_oos"]
                            gk = torch.full_like(sk, 0.5)
                            go = torch.full_like(so, 0.5)
                            ce = torch.zeros(())
                        else:
                            sk, gk = model.fused(te["p_known"], te["n_known"], te["d"])
                            so, go = model.fused(te["p_oos"], te["n_oos"], te["d"])
                            ce = F.cross_entropy(sk, te["y_known"])
                        k = model.reject_logit(model.reject_features(te["p_known"], te["n_known"], sk, gk, te["d"]))
                        o = model.reject_logit(model.reject_features(te["p_oos"], te["n_oos"], so, go, te["d"]))
                        bce = 0.5 * (F.binary_cross_entropy_with_logits(k, torch.ones_like(k)) + F.binary_cross_entropy_with_logits(o, torch.zeros_like(o)))
                        loss = 0.5 * bce if reject_only else ce + 0.5 * bce
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
                        opt.step()
                        total += float(loss.detach()) * (len(te["y_known"]) + len(te["p_oos"]))
                        n += len(te["y_known"]) + len(te["p_oos"])
                    hist.append({"seed": seed, "method": method, "encoder": enc, "epoch": epoch, "loss": total / max(n, 1)})
                model.eval()
                sth, cth, _, _ = b4_scores(model, cache, qthr, classes, reject_only)
                sse, cse, _, _ = b4_scores(model, cache, qsel, classes, reject_only)
                tau, _, info = b4_core.select_threshold_custom(sth, cth, ythr, classes)
                metrics, _, _ = b4_core.evaluate_custom(sse, cse, ysel, tau, classes)
                stem = f"gate7_stability_{enc}_{method}_seed{seed}"
                b4_core.save_checkpoint(RESULTS / f"{stem}.pt", model, {"stage": "gate7-stability", "encoder": enc, "method": method, "seed": seed, "gate": "fixed_0.5" if reject_only else "learned", "epochs": args.epochs, "classes": classes})
                pd.DataFrame(hist).to_csv(RESULTS / f"{stem}_history.csv", index=False)
                rows.append({"encoder": enc, "method": "B4-RejectOnly" if reject_only else "B4-Full", "seed": seed, "tau": tau, "constraint_met": info["constraint_met"], "elapsed_seconds": round(time.perf_counter() - t0, 3), **{f"dev_{k}": v for k, v in metrics.items()}})
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "gate7_stability_b4_results.csv", index=False)
    print(df.round(5).to_string(index=False))
    print("GATE7_STABILITY_B4_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
