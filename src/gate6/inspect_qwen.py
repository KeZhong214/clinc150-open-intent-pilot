"""G6-A: environment, Qwen module inspection, frozen-cache alignment and B3-H replay."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate3.methods import evaluate, knn_scores, prototypes_multi, prototypes_single, scores_from_prototypes, select_threshold  # noqa: E402


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1) if torch.cuda.is_available() else None,
        "vram_free_mib": round(torch.cuda.mem_get_info()[0] / 1024**2, 1) if torch.cuda.is_available() else None,
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "sentence_transformers": __import__("sentence_transformers").__version__,
        "tokenizers": __import__("tokenizers").__version__,
        "qwen_id": qe.QWEN_ID,
        "qwen_revision": qe.QWEN_REV,
        "prompt_sha256": qe.PROMPT_SHA,
        "prompt": qe.PROMPT,
        "pooling": "last_token",
        "max_length": qe.MAX_LEN,
        "dtype": "float16",
    }
    (RESULTS / "gate6_environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = qe.get_tokenizer("right")
    t0 = time.perf_counter()
    model = qe.get_base(device="cuda", dtype=torch.float16).eval()
    load_s = time.perf_counter() - t0

    modules = []
    for name, module in model.named_modules():
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
            modules.append({"name": name, "class": type(module).__name__})
    (RESULTS / "gate6_qwen_modules.json").write_text(
        json.dumps({"n_linear_targets": len(modules), "examples": modules[:12], "all": modules}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    frozen = qe.frozen_cache()
    rows = qe.manifest_rows()
    text_by_id = {r["sample_id"]: r["text"] for r in rows}
    support_rows = qe.support_rows()
    cal_rows = [r for r in rows if r["role"] == "calibration_known"]
    probe_rows = support_rows[:20] + cal_rows[:12]
    probe_texts = [text_by_id[r["sample_id"]] for r in probe_rows]
    probe_ids = [r["sample_id"] for r in probe_rows]
    cache_idx = {sid: i for i, sid in enumerate(frozen["sample_ids"])}
    direct_probe, probe_meta = qe.encode_texts(model, tokenizer, probe_texts, device="cuda", batch_size=8)

    # Direct support / validation embeddings for the B0/B3-H replay.
    support_ids = [r["sample_id"] for r in support_rows]
    support_texts = [text_by_id[sid] for sid in support_ids]
    cal_ids = [r["sample_id"] for r in cal_rows]
    dev_rows = [r for r in rows if r["role"] == "dev_unknown"]
    dev_ids = [r["sample_id"] for r in dev_rows]
    direct_support, _ = qe.encode_texts(model, tokenizer, support_texts, device="cuda", batch_size=8)
    direct_cal, _ = qe.encode_texts(model, tokenizer, [text_by_id[sid] for sid in cal_ids], device="cuda", batch_size=8)
    direct_dev, _ = qe.encode_texts(model, tokenizer, [text_by_id[sid] for sid in dev_ids], device="cuda", batch_size=8)
    val_vectors = np.concatenate([direct_cal, direct_dev])
    classes = qe.classes()
    cmap = {c: i for i, c in enumerate(classes)}
    support_y = np.asarray([r["intent"] for r in support_rows], dtype=object)
    cal_y = np.asarray([cmap[r["intent"]] for r in cal_rows], dtype=int)
    dev_y = np.full(len(dev_rows), -1, dtype=int)
    val_y = np.concatenate([cal_y, dev_y])

    gate3_frozen_support = frozen["vectors"][[cache_idx[sid] for sid in support_ids]]
    cached_probe = frozen["vectors"][[cache_idx[sid] for sid in probe_ids]]
    cos = np.sum(direct_probe * cached_probe, axis=1)
    absdiff = np.abs(direct_probe - cached_probe)
    # Top-1 consistency using each representation's own support prototypes.
    p_direct = prototypes_single(direct_support, support_y, classes)
    p_frozen = prototypes_single(gate3_frozen_support, support_y, classes)
    top_direct = np.argmax(direct_probe @ p_direct.T, axis=1)
    top_frozen = np.argmax(cached_probe @ p_frozen.T, axis=1)
    top1_consistency = float(np.mean(top_direct == top_frozen))

    rows_replay = []
    for method in ("B0", "B3-H"):
        if method == "B0":
            protos = prototypes_single(direct_support, support_y, classes)
            sims = scores_from_prototypes(val_vectors, [p[None, :] for p in protos])
        else:
            protos, _ = prototypes_multi(direct_support, support_y, classes)
            p = scores_from_prototypes(val_vectors, protos)
            n = knn_scores(val_vectors, direct_support, support_y, classes, top_k=2)
            sims = 0.5 * p + 0.5 * n
        tau, _, info = select_threshold(sims, val_y, len(classes), classes)
        metrics, _, _ = evaluate(sims, val_y, classes, tau)
        rows_replay.append({"method": method, "tau": tau, "constraint_met": info["constraint_met"], **metrics})

    gate3 = json.loads((ROOT / "results" / "gate3_validation_metrics.json").read_text(encoding="utf-8")) if (ROOT / "results" / "gate3_validation_metrics.json").exists() else None
    # CSV is the stable frozen source.
    import pandas as pd

    g3 = pd.read_csv(ROOT / "results" / "gate3_validation_metrics.csv")
    g3 = g3[(g3.dataset == "banking77") & (g3.model == "qwen") & (g3.shot == 10)]
    gate3_rows = []
    for method in ("B0", "B3-H"):
        r = g3[g3.method == method].iloc[0]
        gate3_rows.append(
            {
                "method": method,
                "known_acceptance": float(r["val_known_acceptance"]),
                "known_correct_routing": float(r["val_known_correct_routing"]),
                "oos_recall": float(r["val_oos_recall"]),
                "auroc": float(r["val_unknown_auroc"]),
                "open_set_macro_f1": float(r["val_open_set_macro_f1"]),
            }
        )

    payload = {
        "environment": env,
        "load_seconds": round(load_s, 3),
        "probe_n": len(probe_rows),
        "vector_alignment": {
            "mean_cosine": float(cos.mean()),
            "min_cosine": float(cos.min()),
            "max_abs_diff": float(absdiff.max()),
            "mean_abs_diff": float(absdiff.mean()),
            "top1_intent_consistency": top1_consistency,
            "direct_dim": int(direct_probe.shape[1]),
            "norm_max_abs_error": probe_meta["norm_max_abs_error"],
            "prompt": qe.PROMPT,
            "pooling": "last_non_pad_token",
            "padding_side": "right",
            "max_length": qe.MAX_LEN,
        },
        "direct_support_cache_mean_cosine": float(np.mean(np.sum(direct_support * gate3_frozen_support, axis=1))),
        "direct_support_cache_max_abs": float(np.max(np.abs(direct_support - gate3_frozen_support))),
        "validation_replay": rows_replay,
        "gate3_validation": gate3_rows,
        "target_modules": {"n": len(modules), "examples": modules[:8]},
        "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "alignment_ok": bool(cos.mean() >= 0.999 and cos.min() >= 0.995 and top1_consistency >= 0.99),
    }
    (RESULTS / "gate6_qwen_alignment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("GATE6_QWEN_ALIGNMENT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
