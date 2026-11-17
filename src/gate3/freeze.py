"""Freeze the Gate-3 validation thresholds before the final test is opened."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
MANIFEST = PROJECT_ROOT / "data" / "manifest"
CONFIG = PROJECT_ROOT / "configs" / "gate3.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    threshold_path = RESULTS / "gate3_validation_thresholds.json"
    metric_path = RESULTS / "gate3_validation_metrics.csv"
    if not threshold_path.exists() or not metric_path.exists():
        raise FileNotFoundError("Run Gate-3 validation first")
    if (RESULTS / "gate3_test_metrics.csv").exists():
        raise RuntimeError("Gate-3 test metrics already exist; refusing to refreeze")
    freeze_path = RESULTS / "gate3_freeze.json"
    if freeze_path.exists():
        raise RuntimeError("Gate-3 freeze already exists")
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    if len(thresholds) != 72:
        raise RuntimeError(f"Expected 72 threshold configs, got {len(thresholds)}")
    payload = {
        "schema_version": "gate3-freeze-v1",
        "frozen": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": sha256_file(CONFIG),
        "validation_thresholds_sha256": sha256_file(threshold_path),
        "validation_metrics_sha256": sha256_file(metric_path),
        "dataset_manifest_sha256": {
            name: sha256_file(MANIFEST / f"gate3_{name}_dataset.json")
            for name in ("clinc150", "banking77", "massive")
        },
        "threshold_configs": len(thresholds),
        "test_policy": (
            "The 72 validation-selected thresholds are frozen. The final test "
            "is evaluated once; CLINC150 is marked historical because its test "
            "was seen in Gate 1/2. No test-driven method change is allowed."
        ),
        "test_evaluated": False,
    }
    freeze_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {freeze_path}")
    print("GATE3_FREEZE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
