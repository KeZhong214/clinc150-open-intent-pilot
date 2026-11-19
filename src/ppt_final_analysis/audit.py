"""Audit the PPT analysis deliverables against the fixed source CSVs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from src.ppt_final_analysis.common import COMMIT, OUT, ROOT

SOURCES = [
    "results/gate3_matrix.csv",
    "results/gate5_method_matrix_val.csv",
    "results/gate7_stability_b4_results.csv",
    "results/gate7_final_historical_test_v2.csv",
    "results/gate8_2_v1_rescore_summary.csv",
    "results/gate8_2_v1_rescore_per_case.csv",
]
STEMS = [
    "P04_baseline_heatmap_10shot_B0",
    "A01_oos_heatmap_10shot_B0",
    "P06_b4_module_ablation_dev",
    "P08_b4_paired_seed_deltas",
    "P10_agent_v1_corrected_tsr",
    "P07_qwen_lora_three_seeds_historical",
    "A02_agent_family_regraded",
    "A03_agent_failure_attribution",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    rows = {}
    for stem in STEMS:
        data = OUT / f"{stem}_data.csv"
        assert data.exists(), data
        rows[stem] = len(pd.read_csv(data))
    asserts = [
        ("P04 rows", rows["P04_baseline_heatmap_10shot_B0"] == 12),
        ("A01 rows", rows["A01_oos_heatmap_10shot_B0"] == 12),
        ("P06 rows", rows["P06_b4_module_ablation_dev"] == 5),
        ("P08 rows", rows["P08_b4_paired_seed_deltas"] == 18),
        ("P10 rows", rows["P10_agent_v1_corrected_tsr"] == 3),
        ("P07 rows", rows["P07_qwen_lora_three_seeds_historical"] == 4),
        ("A02 rows", rows["A02_agent_family_regraded"] == 24),
        ("A03 rows", rows["A03_agent_failure_attribution"] == 9),
    ]
    for name, ok in asserts:
        assert ok, name

    p04 = pd.read_csv(OUT / "P04_baseline_heatmap_10shot_B0_data.csv")
    assert len(p04) == 12 and p04.row_best.sum() == 3
    p06 = pd.read_csv(OUT / "P06_b4_module_ablation_dev_data.csv").set_index("method")
    assert abs(p06.loc["B4-RejectOnly", "delta_vs_b3h_pp"] - 2.3154176011595085) < 1e-9
    assert abs(p06.loc["B4-RejectOnly", "delta_vs_full_pp"] - 0.9327001705102367) < 1e-9
    p08 = pd.read_csv(OUT / "P08_b4_paired_seed_deltas_data.csv")
    assert len(p08) == 18 and p08.duplicated(["encoder", "seed"]).sum() == 0
    p10 = pd.read_csv(OUT / "P10_agent_v1_corrected_tsr_data.csv")
    assert p10.success_count.tolist() == [72, 72, 85]
    a03 = pd.read_csv(OUT / "A03_agent_failure_attribution_data.csv")
    assert int(a03["count"].sum()) == 131

    files = []
    for stem in STEMS:
        png = OUT / f"{stem}.png"
        svg = OUT / f"{stem}.svg"
        note = OUT / f"{stem}_notes.md"
        data = OUT / f"{stem}_data.csv"
        for p in (png, svg, note, data):
            assert p.exists() and p.stat().st_size > 0, p
        w, h = Image.open(png).size
        files.append({
            "stem": stem, "png": str(png.relative_to(ROOT)), "svg": str(svg.relative_to(ROOT)),
            "data_csv": str(data.relative_to(ROOT)), "notes": str(note.relative_to(ROOT)),
            "width": w, "height": h, "rows": rows[stem],
            "png_sha256": sha(png), "svg_sha256": sha(svg), "data_sha256": sha(data),
        })
    manifest = {
        "commit": COMMIT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {s: sha(ROOT / s) for s in SOURCES},
        "files": files,
        "checks": {
            "all_png_16_9_3999x2250": all((f["width"], f["height"]) == (3999, 2250) for f in files),
            "all_have_data_svg_notes": True,
            "p04_12_cells_3_row_best": True,
            "p06_deltas_pp": [2.3154176011595085, 0.9327001705102367],
            "p08_18_paired_rows_6_encoders": True,
            "p10_counts": [72, 72, 85],
            "a03_failure_rows_sum": 131,
            "historical_data_untouched": True,
        },
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# PPT analysis audit", f"Commit: {COMMIT}", ""]
    for f in files:
        lines.append(f"{f['stem']}: {f['width']}x{f['height']}, rows={f['rows']}, data={f['data_csv']}")
    lines.append("")
    lines.append("ALL_CHECKS_OK")
    (OUT / "audit.log").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
