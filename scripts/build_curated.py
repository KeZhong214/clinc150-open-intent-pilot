"""Build the two final, auditable release ZIPs.

The script copies an explicit allowlist, never the working tree as-is.  It does
not include .venv, API keys, session DBs, raw data, historical screenshots, or
the extraction directory.  Model base weights are intentionally not copied;
the selected heads/adapters/caches are.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
STAGE = ROOT / ".release_staging"

ROOT_FILES = [
    "app_demo.py",
    "app_gate8.py",
    "preflight_demo.py",
    "start_demo.ps1",
    "start_demo.cmd",
    "requirements.txt",
    "requirements_lock.txt",
    "configs/deepseek.env.example",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "src/__init__.py",
]

SRC_DIRS = ["src/gate3", "src/gate4", "src/gate5", "src/gate6", "src/gate7", "src/gate8", "src/gate8_1", "src/gate8_2"]
DATA_FILES = [
    "data/manifest/gate3_banking77_support_10.csv",
    "data/manifest/gate3_banking77_dataset.json",
    "data/manifest/gate7_dev_threshold.csv",
    "knowledge_base/fictional_bank/policies.jsonl",
]
WEIGHTS = [
    "results/gate5_reject_only_frozen.pt",
    "results/gate6_qwen_frozen_reject.pt",
    "results/gate6_qwen_plain_reject.pt",
    "results/gate6_qwen_plain_training.json",
    "results/gate8_1_gte_recalibrated_threshold.json",
    "results/gate7_dev_frozen.csv",
    "results/gate7_dev_plain42.csv",
    "results/gate8_1_model_registry.json",
]
ADAPTER_DIR = "results/gate6_qwen_plain_adapter"
CACHES = [
    "data/cache_embeddings/gate3/gte/banking77/e7f32e3c00f91d699e8c43b53106206bcc72bb22/L128_e3b0c44298fc1c14/embeddings.npz",
    "data/cache_embeddings/gate3/qwen/banking77/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/L128_4bbdf1c169061f64/embeddings.npz",
    "data/cache_embeddings/gate6/qwen_lora/banking77/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3_4efaee5c891e26058764b4a7ec999dec718436f57b724841650831ddc01c3818/L128/embeddings.npz",
]
RESULT_FILES = [
    "results/gate3_matrix.csv",
    "results/gate5_method_matrix_val.csv",
    "results/gate7_stability_b4_results.csv",
    "results/gate7_final_historical_test_v2.csv",
    "results/gate8_2_v1_rescore_summary.csv",
    "results/gate8_2_v1_rescore_per_case.csv",
    "results/gate8_2_faq_grounding_audit.csv",
    "results/gate8_1_agent_summary.csv",
]
REPORTS = [
    "reports/project_final_research_summary.md",
    "reports/gate8_1_final_report.md",
    "reports/gate8_1_benchmark_protocol.md",
    "reports/gate8_2_eval_finalization.md",
    "reports/gate8_2_chinese_support.md",
    "reports/gate8_2_deepseek_integration.md",
    "reports/gate8_2_deepseek_demo_guide.md",
    "docs/DEMO_OPERATIONS.md",
]
TESTS = [
    "tests/test_demo_release.py",
    "tests/test_gate8_2_chinese.py",
    "tests/test_gate8_2_deepseek.py",
    "tests/test_gate8_1.py",
]
FIGURES = [
    "figures/ppt_final_analysis/P04_baseline_heatmap_10shot_B0.png",
    "figures/ppt_final_analysis/P04_baseline_heatmap_10shot_B0.svg",
    "figures/ppt_final_analysis/P04_baseline_heatmap_10shot_B0_data.csv",
    "figures/ppt_final_analysis/P04_baseline_heatmap_10shot_B0_notes.md",
    "figures/ppt_final_analysis/P06_b4_module_ablation_dev.png",
    "figures/ppt_final_analysis/P06_b4_module_ablation_dev.svg",
    "figures/ppt_final_analysis/P06_b4_module_ablation_dev_data.csv",
    "figures/ppt_final_analysis/P06_b4_module_ablation_dev_notes.md",
    "figures/ppt_final_analysis/P07_qwen_lora_three_seeds_historical.png",
    "figures/ppt_final_analysis/P07_qwen_lora_three_seeds_historical.svg",
    "figures/ppt_final_analysis/P07_qwen_lora_three_seeds_historical_data.csv",
    "figures/ppt_final_analysis/P07_qwen_lora_three_seeds_historical_notes.md",
    "figures/ppt_final_analysis/P08_b4_paired_seed_deltas.png",
    "figures/ppt_final_analysis/P08_b4_paired_seed_deltas.svg",
    "figures/ppt_final_analysis/P08_b4_paired_seed_deltas_data.csv",
    "figures/ppt_final_analysis/P08_b4_paired_seed_deltas_notes.md",
    "figures/ppt_final_analysis/P10_agent_v1_corrected_tsr.png",
    "figures/ppt_final_analysis/P10_agent_v1_corrected_tsr.svg",
    "figures/ppt_final_analysis/P10_agent_v1_corrected_tsr_data.csv",
    "figures/ppt_final_analysis/P10_agent_v1_corrected_tsr_notes.md",
    "figures/gate8_2_demo_release.png",
]


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in paths:
        p = ROOT / s
        if p.is_dir():
            out.extend(p.rglob("*.py"))
        elif p.exists():
            out.append(p)
        else:
            print(f"WARN missing allowlisted file: {s}")
    return sorted(set(out), key=lambda x: str(x).lower())


def copy_tree(name: str, files: list[Path], extra: list[Path] | None = None) -> Path:
    dest = STAGE / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for p in files:
        rel = p.relative_to(ROOT)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    if extra:
        for p in extra:
            rel = p.relative_to(ROOT)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    return dest


def write_manifest(project: Path) -> None:
    rows = []
    for p in sorted(project.rglob("*")):
        if p.is_file() and p.name not in {"SHA256SUMS.txt", "RELEASE_MANIFEST.json"}:
            rows.append({"path": str(p.relative_to(project)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": digest(p)})
    (project / "RELEASE_MANIFEST.json").write_text(json.dumps({"files": rows, "count": len(rows)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (project / "SHA256SUMS.txt").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in rows), encoding="utf-8")


def zip_dir(src: Path, out: Path) -> dict:
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(src.parent))
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"zip integrity failure: {bad}")
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": digest(out), "testzip": "OK"}


def main() -> int:
    DIST.mkdir(exist_ok=True)
    common = collect(ROOT_FILES + DATA_FILES + WEIGHTS + CACHES + RESULT_FILES + REPORTS + FIGURES)
    # Adapter is a directory with safetensors + config; copy every regular file.
    adapter_files = [p for p in (ROOT / ADAPTER_DIR).rglob("*") if p.is_file()]
    course_files = collect(SRC_DIRS + TESTS)
    # Course keeps the full source + tests; demo keeps only the demo entry and runtime.
    course = copy_tree("course", common + adapter_files + course_files)
    readme = ROOT / "README_COURSE.md"
    (course / "README_RELEASE.md").write_text(readme.read_text(encoding="utf-8"), encoding="utf-8") if readme.exists() else None
    write_manifest(course)

    demo_files = [p for p in common if str(p.relative_to(ROOT)).replace("\\", "/").startswith(("app_demo.py", "preflight_demo.py", "start_demo.ps1", "start_demo.cmd", "src/", "knowledge_base/", "data/", "results/", "configs/", "requirements", "THIRD_PARTY"))]
    demo_files += collect(SRC_DIRS)
    demo = copy_tree("demo", demo_files + adapter_files)
    demo_readme = ROOT / "README_DEMO.md"
    (demo / "README_RELEASE.md").write_text(demo_readme.read_text(encoding="utf-8"), encoding="utf-8") if demo_readme.exists() else None
    # Demo gets a short release note instead of the full course report set.
    if (ROOT / "docs" / "DEMO_OPERATIONS.md").exists():
        target = demo / "docs" / "DEMO_OPERATIONS.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "docs" / "DEMO_OPERATIONS.md", target)
    write_manifest(demo)

    result = {
        "course": zip_dir(course, DIST / "OpenIntent_course_clean.zip"),
        "demo": zip_dir(demo, DIST / "OpenIntent_demo_clean.zip"),
        "stage": str(STAGE),
    }
    (DIST / "release_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



