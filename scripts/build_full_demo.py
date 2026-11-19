"""Build the full rich-UI Agent project ZIP, including the local API env.

The API env file is local-only and must be created by the owner. This builder
never prints the key and refuses to build when the file is absent or contains
the example placeholder.
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
STAGE = ROOT / ".full_release_staging"
sys.path.insert(0, str(ROOT / "scripts"))
from build_curated import collect, copy_tree, digest, write_manifest, zip_dir  # noqa: E402

ROOT_FILES = [
    "app_gate8.py",
    "app_demo.py",
    "preflight_demo.py",
    "start_full_demo.ps1",
    "start_demo.ps1",
    "start_demo.cmd",
    "README.md",
    "README_FULL.md",
    "requirements.txt",
    "requirements_lock.txt",
    "THIRD_PARTY_NOTICES.md",
    "configs/deepseek.env.example",
    "configs/demo_api.env.example",
]
SRC_DIRS = ["src/gate3", "src/gate4", "src/gate5", "src/gate6", "src/gate7", "src/gate8", "src/gate8_1", "src/gate8_2"]
DATA = [
    "knowledge_base/fictional_bank/policies.jsonl",
    "data/manifest/gate3_banking77_support_10.csv",
    "data/manifest/gate3_banking77_dataset.json",
    "data/manifest/gate7_dev_threshold.csv",
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
    "docs/DEMO_OPERATIONS_FULL.md",
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
    "figures/full_demo_api_success.png",
    "figures/full_demo_api_cancel.png",
]
LOGS = [
    "logs/full_demo_preflight.log",
    "logs/full_demo_api_smoke.log",
    "logs/full_demo_api_ui_probe.log",
    "logs/full_demo_api_cancel_probe.log",
]
TESTS = [
    "tests/test_demo_release.py",
    "tests/test_gate8_2_chinese.py",
    "tests/test_gate8_2_deepseek.py",
    "tests/test_gate8_1.py",
]


def main() -> int:
    api = ROOT / "configs" / "demo_api.env"
    if not api.exists():
        print("REFUSING TO BUILD: configs/demo_api.env is missing. Create it locally first.")
        return 2
    text = api.read_text(encoding="utf-8")
    if "sk-your-key-here" in text or "DEEPSEEK_API_KEY=" not in text:
        print("REFUSING TO BUILD: configs/demo_api.env still has the example/empty key.")
        return 2
    DIST.mkdir(exist_ok=True)
    files = collect(ROOT_FILES + SRC_DIRS + DATA + WEIGHTS + CACHES + RESULT_FILES + REPORTS + FIGURES + TESTS + LOGS)
    files.append(api)
    adapter = [p for p in (ROOT / "results" / "gate6_qwen_plain_adapter").rglob("*") if p.is_file()]
    full = copy_tree("full", files + adapter)
    (full / "README_RELEASE.md").write_text((ROOT / "README_FULL.md").read_text(encoding="utf-8"), encoding="utf-8")
    (full / "deliverables").mkdir(exist_ok=True)
    pptx = ROOT / "exports" / "少样本开放意图识别与智能客服_12页.pptx"
    if pptx.exists():
        shutil.copy2(pptx, full / "deliverables" / pptx.name)
    write_manifest(full)
    out = DIST / "OpenIntent_full_demo_project.zip"
    result = zip_dir(full, out)
    result["contains_api_env"] = True
    result["note"] = "The ZIP contains configs/demo_api.env; keep it private and do not upload it to GitHub."
    (DIST / "full_release_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
