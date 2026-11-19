"""One-shot preflight for the classroom demo.

Loads the fixed Qwen Plain42 router, validates its paired registry and caches,
runs one non-mutating CARD-1002 query, and prints READY only after the tool
result is verified.  It never calls the DeepSeek API and never prints secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.gate4 import b4_core  # noqa: E402
from src.gate5.kb import KBRetriever  # noqa: E402
from src.gate6 import qwen_data  # noqa: E402
from src.gate6 import qwen_encoder as qe  # noqa: E402
from src.gate8.agent import Gate8Agent  # noqa: E402
from src.gate8_1 import registry  # noqa: E402
from src.gate8_1.routers import build  # noqa: E402
from src.gate8_2.deepseek_client import DeepSeekClient  # noqa: E402

MODEL = "Qwen-Plain42-RejectOnly"
QUERY = "我的卡CARD-1002还没到"


def fail(message: str) -> int:
    print(f"NOT READY: {message}")
    return 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    t0 = time.perf_counter()
    checks: dict = {}
    try:
        import torch
        checks["python"] = sys.version.split()[0]
        checks["cuda_available"] = bool(torch.cuda.is_available())
        checks["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if not torch.cuda.is_available():
            return fail("CUDA unavailable; this demo expects the RTX 5070 local environment")
        checks["hf_home"] = os.environ.get("HF_HOME", "")
        checks["qwen_model_dir"] = str(qe.MODEL_CACHE / "models--Qwen--Qwen3-Embedding-0.6B")
        if not (qe.MODEL_CACHE / "models--Qwen--Qwen3-Embedding-0.6B").exists():
            return fail("Qwen base model cache missing")
        checks["gte_model_dir"] = str(qe.MODEL_CACHE / "models--Alibaba-NLP--gte-modernbert-base")
        if not (qe.MODEL_CACHE / "models--Alibaba-NLP--gte-modernbert-base").exists():
            return fail("GTE model cache missing (course package completeness)")
        kb = KBRetriever()
        checks["kb_docs"] = len(kb.docs)
        if len(kb.docs) != 30:
            return fail(f"expected 30 KB docs, got {len(kb.docs)}")
        checks["registry_entry"] = registry.validate(MODEL)
        checks["support_manifest"] = (ROOT / "data" / "manifest" / "gate3_banking77_support_10.csv").exists()
        if not checks["support_manifest"]:
            return fail("missing BANKING77 10-shot support manifest")
        b4_core.load_cache()
        qcache = qwen_data.load("plain")
        checks["qwen_cache"] = qcache["path"]
        checks["adapter_dir"] = str(ROOT / "results" / "gate6_qwen_plain_adapter")
        if not (ROOT / "results" / "gate6_qwen_plain_adapter").exists():
            return fail("Qwen Plain42 adapter directory missing")
        router, meta = build(MODEL)
        checks["model"] = meta["name"]
        checks["threshold"] = meta["threshold"]
        checks["checkpoint_sha256"] = meta["checkpoint_sha256"]
        with tempfile.TemporaryDirectory(prefix="openintent-preflight-") as td:
            agent = Gate8Agent(router, session_id="preflight", db_dir=Path(td))
            try:
                turn = agent.handle(QUERY)
                checks["intent"] = turn.route.intent if turn.route else None
                checks["accepted"] = turn.route.accepted if turn.route else None
                checks["score"] = turn.route.score if turn.route else None
                checks["tools"] = [x.get("tool") for x in turn.tools]
                if not turn.route or not turn.route.accepted or "get_card_delivery_status" not in checks["tools"]:
                    return fail(f"non-mutating CARD-1002 preflight did not reach the card tool: {checks}")
                result = next((x.get("result") or {} for x in turn.tools if x.get("tool") == "get_card_delivery_status"), {})
                checks["delivery_status"] = result.get("delivery_status")
                checks["card_status"] = result.get("status")
                if result.get("delivery_status") != "in_transit" or result.get("status") != "inactive":
                    return fail(f"unexpected CARD-1002 state: {result}")
            finally:
                agent.close()
        ds = DeepSeekClient().status()
        checks["deepseek"] = {k: ds[k] for k in ("available", "api_key_set", "key_format_ok", "config_error", "mode")}
        # Status never contains a key.  API unavailability is not a hard failure:
        # the classroom offline path remains available.
        checks["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
        checks["ready"] = True
        print(json.dumps(checks, ensure_ascii=False, indent=2) if args.json else f"READY in {checks['elapsed_seconds']:.2f}s | model={MODEL} | threshold={meta['threshold']:.6f} | intent={checks['intent']} | delivery={checks['delivery_status']} | card={checks['card_status']}")
        if not args.json:
            print(f"DEEPSEEK_API_AVAILABLE={ds['available']} (offline mode remains available)")
            print("PREFLIGHT_OK")
        return 0
    except Exception as exc:
        return fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
