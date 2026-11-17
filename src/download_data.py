"""Download the pinned official CLINC150 files used by this project.

Only the small official JSON/README/LICENSE files are downloaded.  Model
weights are handled separately; in particular this script never downloads
Qwen3-Embedding-0.6B.

Usage (from the project root):
    python -m src.download_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPOSITORY = "https://github.com/clinc/oos-eval"
REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
RAW_BASE = f"https://raw.githubusercontent.com/clinc/oos-eval/{REVISION}"

# Paths are relative to the official repository.
FILES = {
    "data/data_full.json": "data_full.json",
    "data/domains.json": "domains.json",
    "LICENSE": "LICENSE",
    "README.md": "README_clinc.md",
}

# Hashes observed for the pinned revision.  They make an accidental source
# change fail loudly instead of silently producing a different pilot dataset.
EXPECTED_SHA256 = {
    "data_full.json": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    "domains.json": "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a",
    "LICENSE": "e6bc9e9c474700b708f568bac9e5a8a9bcb2b1dad53442f5ba449fcb848b8e76",
    "README_clinc.md": "8a8df26c4de3d25b6c4cff385ca23602c30e4e56828b015206cbc202f6db363a",
}

EXPECTED_COUNTS = {
    "train": 15000,
    "val": 3000,
    "test": 4500,
    "oos_train": 100,
    "oos_val": 100,
    "oos_test": 1000,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "intent-open-set-gate0/1.0",
            "Accept": "application/octet-stream,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def validate_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if set(payload) != set(EXPECTED_COUNTS):
        raise RuntimeError(
            f"Unexpected top-level keys in {path.name}: {sorted(payload)}; "
            f"expected {sorted(EXPECTED_COUNTS)}"
        )

    actual = {key: len(value) for key, value in payload.items()}
    if actual != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Unexpected split counts in {path.name}: {actual}; "
            f"expected {EXPECTED_COUNTS}."
        )

    for split, rows in payload.items():
        for index, row in enumerate(rows):
            if not (isinstance(row, list) and len(row) == 2):
                raise RuntimeError(f"{split}[{index}] is not a [text, label] pair: {row!r}")
            text, label = row
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"{split}[{index}] has an empty/non-string text")
            if not isinstance(label, str) or not label.strip():
                raise RuntimeError(f"{split}[{index}] has an empty/non-string label")

    return {"counts": actual, "top_level_keys": sorted(payload)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Destination for data_full.json, domains.json and LICENSE.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when a file exists and its hash matches.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository : {REPOSITORY}")
    print(f"Revision   : {REVISION}")
    print(f"Output dir : {out_dir}")

    records: dict[str, dict] = {}
    for repo_path, filename in FILES.items():
        url = f"{RAW_BASE}/{repo_path}"
        destination = out_dir / filename
        expected_hash = EXPECTED_SHA256[filename]

        existing_hash = sha256_file(destination) if destination.exists() else None
        if existing_hash == expected_hash and not args.force:
            status = "cached"
            payload = destination.read_bytes()
            print(f"[{status}] {filename} sha256={expected_hash}")
        else:
            if destination.exists():
                print(
                    f"[replace] {filename} existing sha256={existing_hash} "
                    f"!= expected {expected_hash}"
                )
            print(f"[download] {url}")
            payload = fetch(url)
            actual_hash = sha256_bytes(payload)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Hash mismatch for {url}: got {actual_hash}, expected {expected_hash}. "
                    "The pinned upstream revision may have changed; stop and investigate."
                )
            destination.write_bytes(payload)
            status = "downloaded"
            print(f"[{status}] {filename} bytes={len(payload)} sha256={actual_hash}")

        actual_hash = sha256_file(destination)
        if actual_hash != expected_hash:
            raise RuntimeError(f"On-disk hash mismatch for {destination}")

        records[filename] = {
            "repo_path": repo_path,
            "url": url,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": actual_hash,
            "status": status,
        }

    data_path = out_dir / "data_full.json"
    validation = validate_payload(data_path)
    print(f"Validated schema and counts: {validation['counts']}")

    manifest = {
        "schema_version": "gate0-download-v1",
        "retrieved_at_utc": utc_now(),
        "dataset": "CLINC150 / OOS-Eval",
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": "CC BY 3.0 (see official LICENSE)",
        "raw_base_url": RAW_BASE,
        "files": records,
        "validation": validation,
        "note": (
            "Only official source files are stored here. No row was deleted, "
            "rewritten, translated, or relabeled."
        ),
    }
    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")
    print("SOURCE_DOWNLOAD_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI should print a clear failure
        print(f"SOURCE_DOWNLOAD_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
