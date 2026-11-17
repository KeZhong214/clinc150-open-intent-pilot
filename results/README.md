# Gate 1 and Gate 2 result files

All numbers are produced by the scripts in `src/`, from the frozen CLINC150
manifests and cached embeddings. Gate 1 and Gate 2 use separate OOS protocols;
their metrics must not be mixed into one claimed improvement.

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

---

# Gate 2 hard-OOS files

Gate 2 keeps the 20 known classes and support fixed. The remaining 130 classes
are split 65/65 into development unknown classes and held-out unknown classes.

| File | Content |
|---|---|
| `gate2_validation_metrics.csv` | B0/B2-original/B2-v2 on 400 known + 100 original OOS + 1,300 hard-OOS val |
| `gate2_validation_thresholds.json` | full validation thresholds and metrics |
| `gate2_validation_predictions.csv` | validation predictions for all three methods |
| `gate2_b0_validation_grid.csv` | B0 validation tau sweep |
| `gate2_b2_validation_grid.csv` | B2-original validation grid |
| `gate2_b2v2_validation_grid.csv` | B2-v2 validation grid |
| `gate2_b2v2_class_thresholds.csv` | q_c, u_c, n_c, alpha_c, target_c, tau_c per class |
| `gate2_freeze.json` | frozen thresholds; `hard_oos_test_evaluated` records one-shot status |
| `gate2_test_metrics.csv` | held-out hard-OOS test: 600 known + 1,950 hard-OOS |
| `gate2_test_predictions.csv` | held-out hard-OOS test predictions |
| `gate2_test_confusion_*.csv` | 21-label confusion matrices |
| `gate2_test_run.json` | test run metadata |
| `gate2_encoding_hard_val.json` | 1,300-row Qwen hard-OOS validation encode metadata |
| `gate2_encoding_hard_test.json` | 1,950-row Qwen hard-OOS test encode metadata |
| `gate2_validation_diagnostics.json` | Gate 1 thresholds on hard-OOS val and q_c comparison |
| `gate2_hard_class_stats.csv` | hard-OOS val counts/scores by predicted class |

Gate 2 test metrics use `hard_oos_recall` and `hard_open_set_macro_f1`; they do
not include the original `oos_test`. `orig_oos_recall` is blank in the test
table because the new protocol intentionally keeps the original OOS test
separate. See `../reports/gate2_method.md` for the final conclusion.
