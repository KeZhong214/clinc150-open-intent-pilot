# Gate 1 result files

All numbers are produced by the scripts in `src/`, from the frozen CLINC150
manifests and cached embeddings. The final test split was evaluated once per
method after validation-only threshold selection.

## Main tables

| File | Content |
|---|---|
| `gate1_metrics.csv` | Six rows: BGE/Qwen × B0/B2/B2-control |
| `b0_baseline.csv` | B0 rows only |
| `b2_dynamic_rejection.csv` | B2 rows only |
| `b2_metrics.csv` | B2 + B2-control rows |
| `predictions_b0.csv` | 4,200 test/validation predictions for B0 |
| `predictions_b2.csv` | 4,200 test/validation predictions for B2 |
| `error_cases_qwen_b2.csv` | Five traceable Qwen B2 test cases |
| `encoding_bge.json` | BGE encode metadata and vector cache info |
| `encoding_qwen.json` | Qwen encode metadata, GPU peak, prompt hash |

`B2-control` means `lambda=0, gamma=0`; it is equivalent to B0 acceptance.
Its test metrics are copied from the already-evaluated B0 predictions, not
produced by a second test read.

## Metric columns

- `known_acceptance`: fraction of known test rows predicted as non-OOS;
  this is not the same as classification accuracy.
- `known_correct_routing`: fraction of known rows predicted as the correct
  intent. Wrong accepted classes and rejected known rows both count as errors.
- `known_macro_f1`: macro-F1 over the 20 known classes; known rejections count
  as false negatives.
- `oos_recall`: fraction of OOS test rows rejected.
- `oos_false_acceptance`: `1 - oos_recall`.
- `oos_auroc`: AUROC using the continuous decision score; B0 uses
  `s1 - tau`, B2 uses `min(s1 - tau_c, margin - gamma)`.
- `open_set_macro_f1`: macro-F1 over 20 known classes + OOS (21 labels).
- `open_set_accuracy`: joint correct routing/rejection rate over 600 known +
  1000 OOS rows.

## Threshold files

`thresholds_{model}_{method}.json` records `tau`, `lambda`, `gamma`, `q_c`,
validation metrics, and the search rule. `val_grid_*.csv` records the full
validation search grid. Test labels are not used in these files.

## Figures

See `../figures/` and the captions in
`../reports/gate1_baseline.md`.
