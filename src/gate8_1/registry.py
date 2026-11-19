"""Gate 8.1 model/threshold registry and paired-configuration guard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gte() -> dict:
    p = RESULTS / "gate8_1_gte_recalibrated_threshold.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "name": "GTE-Frozen-RejectOnly",
        "encoder": "GTE-ModernBERT Frozen",
        "checkpoint": d["checkpoint"],
        "checkpoint_sha256": d["checkpoint_sha256"],
        "adapter_sha256": None,
        "tau": float(d["tau"]),
        "threshold_file": str(p.relative_to(ROOT)),
        "threshold_sha256": sha(p),
        "calibration_split": d["calibration_split"],
        "calibration_split_sha256": d["calibration_split_sha256"],
        "paired": True,
    }


def _qwen(name: str, tag: str, ckpt: str, threshold_csv: str) -> dict:
    import pandas as pd

    d = pd.read_csv(RESULTS / threshold_csv)
    tau = float(d[(d.method == "B4-RejectOnly")].tau.iloc[0])
    train = json.loads((RESULTS / ("gate6_qwen_plain_training.json" if tag == "plain42" else "gate6_qwen_frozen_training.json")).read_text(encoding="utf-8")) if tag == "plain42" else {}
    adapter = train.get("adapter_sha256") if tag == "plain42" else None
    return {
        "name": name,
        "encoder": "Qwen3-Embedding-0.6B Frozen" if tag == "frozen" else "Qwen3-Embedding-0.6B Plain LoRA seed42",
        "checkpoint": ckpt,
        "checkpoint_sha256": sha(RESULTS / Path(ckpt).name),
        "adapter_sha256": adapter,
        "tau": tau,
        "threshold_file": f"results/{threshold_csv}",
        "threshold_sha256": sha(RESULTS / threshold_csv),
        "calibration_split": "data/manifest/gate7_dev_threshold.csv",
        "calibration_split_sha256": sha(ROOT / "data" / "manifest" / "gate7_dev_threshold.csv"),
        "paired": True,
    }


def build() -> dict:
    entries = [_gte(), _qwen("Qwen-Frozen-RejectOnly", "frozen", "results/gate6_qwen_frozen_reject.pt", "gate7_dev_frozen.csv"), _qwen("Qwen-Plain42-RejectOnly", "plain42", "results/gate6_qwen_plain_reject.pt", "gate7_dev_plain42.csv")]
    payload = {"schema_version": "gate8.1-model-registry-v1", "entries": {e["name"]: e for e in entries}, "note": "GTE uses Gate5 checkpoint with its own Gate7-dev recalibration; Qwen thresholds are paired with their Gate6 checkpoints/adapters."}
    (RESULTS / "gate8_1_model_registry.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def validate(name: str) -> dict:
    payload = json.loads((RESULTS / "gate8_1_model_registry.json").read_text(encoding="utf-8"))
    e = payload["entries"].get(name)
    if not e:
        raise ValueError(name)
    if not e.get("paired"):
        raise RuntimeError(f"Unpaired router config: {name}")
    cp = ROOT / e["checkpoint"]
    if sha(cp) != e["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint SHA mismatch: {name}")
    if e["adapter_sha256"]:
        tp = RESULTS / ("gate6_qwen_plain_training.json")
        if json.loads(tp.read_text(encoding="utf-8"))["adapter_sha256"] != e["adapter_sha256"]:
            raise RuntimeError(f"Adapter SHA mismatch: {name}")
    tp = ROOT / e["threshold_file"]
    if sha(tp) != e["threshold_sha256"]:
        raise RuntimeError(f"Threshold file SHA mismatch: {name}")
    return e


def main() -> int:
    p = build()
    for name in p["entries"]:
        print(json.dumps(validate(name), ensure_ascii=False, indent=2))
    print("GATE8_1_REGISTRY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
