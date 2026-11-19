"""Actual model-switch pressure test: build, release, build the next router."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gate8_1.routers import build  # noqa: E402

OUT = ROOT / "results" / "gate8_1_switch_memory.json"


def mem():
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    torch.cuda.synchronize()
    return {"allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1), "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1)}


def main() -> int:
    rows = []
    for name in ["GTE-Frozen-RejectOnly", "Qwen-Frozen-RejectOnly", "Qwen-Plain42-RejectOnly", "GTE-Frozen-RejectOnly"]:
        router, meta = build(name)
        after_build = mem()
        rows.append({"step": f"build:{name}", "before_release": None, "after_build": after_build})
        del router, meta
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        rows.append({"step": f"release:{name}", "after_release": mem()})
    OUT.write_text(json.dumps({"rows": rows, "note": "Actual sequential loads in one process; no stub models."}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("GATE8_1_SWITCH_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
