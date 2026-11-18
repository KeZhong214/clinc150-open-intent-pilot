"""Freeze Gate-4 B4 thresholds and checkpoints before the one-shot exploratory test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FILES = [
    "gate4_b4_thresholds.json",
    "gate4_b4_fusion.pt",
    "gate4_b4_full.pt",
    "gate4_b4_training_config.json",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    payload = {
        "frozen": True,
        "test_evaluated": False,
        "dataset": "banking77",
        "model": "gte-modernbert-base",
        "shot": 10,
        "note": "BANKING77 test was already seen in Gate3. This is an exploratory Gate4 test, not a new blind test.",
        "files": {},
    }
    for name in FILES:
        path = RESULTS / name
        if not path.exists():
            raise FileNotFoundError(path)
        payload["files"][name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    (RESULTS / "gate4_b4_freeze.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("GATE4_FREEZE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
