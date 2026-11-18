"""G7-C: freeze selected dev recipe, then one historical-exploratory test pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "manifest"
CASES = ROOT / "data" / "gate5" / "benchmark_cases.jsonl"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_data as qd  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate6.train_qwen_b4 import val_scores  # noqa: E402
from src.gate3.methods import evaluate, knn_scores, prototypes_multi, prototypes_single, scores_from_prototypes  # noqa: E402
from src.gate4 import b4_core  # noqa: E402
from src.gate4.b4_core import B4AR  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_cache(tag: str) -> dict:
    if tag == "frozen":
        return qd.load("frozen")
    if tag == "plain42":
        train = json.loads((RESULTS / "gate6_qwen_plain_training.json").read_text(encoding="utf-8"))
    else:
        train = json.loads((RESULTS / f"gate7_qwen_{tag}_training.json").read_text(encoding="utf-8"))
    d = np.load(qe.lora_cache(train["adapter_sha256"]), allow_pickle=False)
    return {"vectors": d["vectors"].astype(np.float32), "sample_ids": d["sample_ids"].astype(str), "roles": d["roles"].astype(str), "labels": d["labels"].astype(str), "intents": d["intents"].astype(str), "encoder": tag}


def split(cache, role):
    rows = pd.read_csv(MANIFEST / "gate3_banking77_manifest.csv")
    rows = rows[rows.role == role]
    classes = qe.classes()
    cmap = {c: i for i, c in enumerate(classes)}
    idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    X = cache["vectors"][[idx[sid] for sid in rows.sample_id]]
    y = np.asarray([cmap.get(c, -1) for c in rows.intent], dtype=int)
    return X, y


def b0_b3(cache, method, query):
    classes = qe.classes()
    rows = pd.read_csv(MANIFEST / "gate3_banking77_support_10.csv")
    idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    X = cache["vectors"][[idx[sid] for sid in rows.sample_id]]
    y = np.asarray([r.intent for r in rows.itertuples()], dtype=object)
    if method == "B0":
        protos = prototypes_single(X, y, classes)
        return scores_from_prototypes(query, [p[None, :] for p in protos])
    protos, _ = prototypes_multi(X, y, classes)
    p = scores_from_prototypes(query, protos)
    n = knn_scores(query, X, y, classes, top_k=2)
    return 0.5 * p + 0.5 * n


def b4_test(cache, checkpoint, tau):
    payload = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False)
    model = B4AR()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    X, y = split(cache, "known_test")
    uX, uy = split(cache, "final_unknown")
    q = np.concatenate([X, uX])
    yt = np.concatenate([y, uy])
    r, c, _, _ = val_scores(model, cache, q, qe.classes(), reject_only=True)
    return b4_core.evaluate_custom(r, c, yt, tau, qe.classes())[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-fix-v2", action="store_true", help="write a separate protocol_fix_v2 run; never overwrite the original freeze/test")
    args = parser.parse_args(argv)
    v2 = bool(args.protocol_fix_v2)
    original_freeze_path = RESULTS / "gate7_final_freeze.json"
    original_test_path = RESULTS / "gate7_final_historical_test.csv"
    freeze_path = RESULTS / "gate7_final_freeze_v2.json" if v2 else original_freeze_path
    test_path = RESULTS / "gate7_final_historical_test_v2.csv" if v2 else original_test_path
    if v2:
        if not original_freeze_path.exists() or not original_test_path.exists():
            raise RuntimeError("protocol_fix_v2 requires the original freeze and test files")
        parent = json.loads(original_freeze_path.read_text(encoding="utf-8"))
        if not parent.get("test_evaluated"):
            raise RuntimeError("protocol_fix_v2 requires the original test_evaluated=true record")
    else:
        if original_freeze_path.exists():
            f = json.loads(original_freeze_path.read_text(encoding="utf-8"))
            if f.get("test_evaluated"):
                print("Gate7 test already evaluated; refuse rerun (use --protocol-fix-v2 for a separate corrected run)")
                return 2
    tags = ["frozen", "plain42", "plain43", "plain44"]
    dev = {t: pd.read_csv(RESULTS / f"gate7_dev_{t}.csv") for t in tags if (RESULTS / f"gate7_dev_{t}.csv").exists()}
    if set(dev) != set(tags):
        raise RuntimeError(f"Missing dev results: {set(tags) - set(dev)}")
    freeze = {
        "selection": "P0 fixed plain recipe (P1/P2 did not beat P0 on dev_select; P3 completed but not selected)",
        "adapter_sha256": {},
        "b4_sha256": {},
        "dev_sha256": {},
        "test_evaluated": False,
        "label": "historical_exploratory",
        "protocol": "B0/B3-H/B4-RejectOnly all use Gate7 dev_threshold tau",
    }
    for t in tags:
        cache = load_cache(t)
        freeze["dev_sha256"][t] = sha(RESULTS / f"gate7_dev_{t}.csv")
        if t == "frozen":
            train = None
            b4_path = RESULTS / "gate6_qwen_frozen_reject.pt"
        elif t == "plain42":
            train = json.loads((RESULTS / "gate6_qwen_plain_training.json").read_text(encoding="utf-8"))
            b4_path = RESULTS / "gate6_qwen_plain_reject.pt"
        else:
            train = json.loads((RESULTS / f"gate7_qwen_{t}_training.json").read_text(encoding="utf-8"))
            b4_path = RESULTS / f"gate7_qwen_{t}_reject.pt"
        if train:
            freeze["adapter_sha256"][t] = train["adapter_sha256"]
        freeze["b4_sha256"][t] = sha(b4_path)
    freeze["cases_sha256"] = sha(CASES)
    freeze["dev_select_sha256"] = json.loads((RESULTS / "gate7_dev_split.json").read_text(encoding="utf-8"))["dev_select"]["sha256"]

    rows = []
    threshold_rows = []
    for t in tags:
        cache = load_cache(t)
        kX, ky = split(cache, "known_test")
        uX, uy = split(cache, "final_unknown")
        q = np.concatenate([kX, uX])
        yt = np.concatenate([ky, uy])
        for method in ("B0", "B3-H"):
            tau = float(dev[t][dev[t].method == method].tau.iloc[0])
            sims = b0_b3(cache, method, q)
            m, _, _ = evaluate(sims, yt, qe.classes(), tau)
            rows.append({"encoder": t, "method": method, "seed": 42 if t in ("frozen", "plain42") else int(t.replace("plain", "")), "tau": tau, **m})
            threshold_rows.append({"encoder": t, "method": method, "tau_used": tau, "dev_threshold_tau": tau})
        b4_path = RESULTS / ("gate6_qwen_frozen_reject.pt" if t == "frozen" else "gate6_qwen_plain_reject.pt" if t == "plain42" else f"gate7_qwen_{t}_reject.pt")
        # Important fix: B4 uses the same Gate7 dev_threshold tau as B0/B3-H.
        tau = float(dev[t][dev[t].method == "B4-RejectOnly"].tau.iloc[0])
        m = b4_test(cache, str(b4_path.relative_to(ROOT)), tau)
        rows.append({"encoder": t, "method": "B4-RejectOnly", "seed": 42 if t in ("frozen", "plain42") else int(t.replace("plain", "")), "tau": tau, **m})
        threshold_rows.append({"encoder": t, "method": "B4-RejectOnly", "tau_used": tau, "dev_threshold_tau": tau})
    for t in tags:
        for r in [x for x in threshold_rows if x["encoder"] == t]:
            assert abs(float(r["tau_used"]) - float(r["dev_threshold_tau"])) < 1e-12, r
    df = pd.DataFrame(rows)
    df["historical_exploratory"] = True
    df.to_csv(test_path, index=False)
    freeze["test_evaluated"] = True
    freeze["test_sha256"] = sha(test_path)
    freeze["thresholds"] = threshold_rows
    if v2:
        freeze["protocol_fix_v2"] = True
        freeze["parent_freeze_sha256"] = sha(original_freeze_path)
        freeze["parent_test_sha256"] = sha(original_test_path)
        freeze["parent_test_evaluated"] = True
        freeze["reason"] = "Prior B4 test used Gate6 threshold files; corrected run uses Gate7 dev_threshold for all methods."
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(df.round(5).to_string(index=False))
    print(json.dumps({"protocol_fix_v2": v2, "test_path": str(test_path.relative_to(ROOT)), "freeze_path": str(freeze_path.relative_to(ROOT)), "n_rows": len(df)}, ensure_ascii=False))
    print("GATE7_FINAL_TEST_V2_OK" if v2 else "GATE7_FINAL_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
