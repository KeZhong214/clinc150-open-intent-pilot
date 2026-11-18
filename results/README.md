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

---

# Gate 3 multi-dataset files

Gate 3 contains 3 datasets × 4 encoders × 2 shots × B0/B3-P/B3-H = 72 test
configurations. CLINC150 is historical; BANKING77 and MASSIVE are new.

| File | Content |
|---|---|
| `gate3_matrix.csv` | 72 test configurations, full metrics |
| `gate3_test_metrics.csv` | Same 72-cell test matrix |
| `gate3_validation_metrics.csv` | 72 validation-selected configs |
| `gate3_validation_thresholds.json` | frozen tau per config; input to freeze |
| `gate3_freeze.json` | freeze record; test evaluated once |
| `gate3_summary_by_dataset_method.csv` | mean over models/shots |
| `gate3_summary_by_model.csv` | mean over datasets/shots |
| `gate3_b3h_vs_b0.csv` | per model/shot B3-H − B0 deltas |
| `gate3_best_configs.csv` | best config per dataset |
| `gate3_dataset_profile.csv` | supports, eval rows, classes, domains, imbalance |
| `gate3_validation_vs_test.csv` | validation→test OOS/F1 shift |
| `gate3_cost.csv` | encode seconds, texts/s, dim, peak VRAM |
| `gate3_predictions_{dataset}.csv` | test predictions with Top-3 and scores |
| `gate3_banking_confusions.csv` | class→class BANKING77 confusions + examples |
| `gate3_error_cases.csv` | 15 traceable error/correct cases |
| `gate3_setfit_metrics.csv` / `gate3_setfit_training.json` | SetFit-BGE training comparison |
| `gate3_adapter_metrics.csv` / `gate3_adapter_training.csv` | projection-adapter training comparison |
| `gate3_training_summary.csv` | compact training comparison |
| `gate3_encoding_{bge,qwen,gte,e5}.json` | per-model encoding metadata |

`gate3_test_metrics.csv` marks CLINC150 with `historical_clinc_test=1`.
SetFit and the adapter reuse the already-opened BANKING77 test; their notes say
this explicitly. See `../reports/gate3_final_report.md`.

---

# Gate 4 B4 and Agent files

Gate 4 uses BANKING77 GTE-ModernBERT 10-shot and the already-seen test. The
B4 test is exploratory, not a new blind test.

| File | Content |
|---|---|
| `gate4_b4_validation_metrics.csv` | B0/B3-H/B4-Fusion/B4-Full validation |
| `gate4_b4_thresholds.json` | frozen B4 thresholds |
| `gate4_b4_training_history.csv` | 60+60 epoch losses, gate stats |
| `gate4_b4_training_config.json` | episodes, hyperparameters, checkpoint paths |
| `gate4_b4_gate_stats.json` | gate mean/std, gradient, parameter deltas |
| `gate4_b4_gate_by_class.csv` / `gate4_b4_gate_by_difficulty.csv` | gate behavior |
| `gate4_b4_fusion.pt` / `gate4_b4_full.pt` | small checkpoints (~6 KB) |
| `gate4_b4_freeze.json` | pre-test freeze hashes; `test_evaluated=true` |
| `gate4_b4_test_metrics.csv` | one-shot exploratory test, 4 methods |
| `gate4_b4_test_predictions.csv` | B4-Full test predictions + gate values |
| `gate4_b4_test_cases.csv` | five traceable B4-Full cases |
| `gate4_agent_cases.csv` | 16 deterministic tool/state cases |
| `gate4_agent_metrics.json` | 16-case completion/tool/clarification/escalation |
| `gate4_agent_demos.json` | 6 real-router multi-turn demos + DB snapshots |
| `gate4_agent_demo_metrics.json` | 6-demo metrics |

`data/gate4/agent.db` is generated at runtime and is git-ignored. See
`../reports/gate4_final_report.md`.

---

# Gate 5 LoRA / RejectOnly / Agent benchmark files

Gate 5 starts from `gate4-train-agent@81e81a9`. BANKING77 test is historical
exploratory, not a new blind test.

| File | Content |
|---|---|
| `gate5_lora_training.json` | 2-epoch selected LoRA, params, adapter hash, vector checks |
| `gate5_lora_training_history.csv` | real epoch losses |
| `gate5_lora_hard_negatives.csv` | support-only hard-negative log |
| `gate5_lora_adapter/` | 6.78 MB adapter_config + adapter_model |
| `gate5_lora_encoding.json` | 4,280-row re-encode metadata and vector shift |
| `gate5_lora_validation.csv` | LoRA B0/B3-H validation |
| `gate5_lora_4ep/` | 4-epoch failed attempt, metrics only, not selected |
| `gate5_reject_only_{frozen,lora}.pt` | fixed-gate reject-head checkpoints |
| `gate5_reject_only_{frozen,lora}_validation.csv` | RejectOnly validation |
| `gate5_b4_full_lora.pt` | LoRA B4-Full checkpoint |
| `gate5_method_matrix_val.csv` | 9-row validation attribution matrix |
| `gate5_method_matrix_historical_test.csv` | one-shot historical exploratory test |
| `gate5_method_freeze.json` | pre-test hashes; `test_evaluated=true` |
| `gate5_agent_cases.csv` | Gate4 16 deterministic state/tool cases |
| `gate5_agent_demos.json` | 10 real-router demo traces |
| `data/gate5/benchmark_cases.jsonl` | 120 frozen natural-language tasks |
| `gate5_benchmark_freeze.json` | case SHA + family counts |
| `gate5_benchmark_results.csv` | 360 real-router runs |
| `gate5_benchmark_summary.csv` | TSR/tool/safe/routing/latency |
| `gate5_benchmark_by_family.csv` | family-level TSR |
| `gate5_benchmark_failure_summary.json` | routing/tool/rag/clarification failure counts |
| `gate5_llm_probe.json` / `gate5_llm_api_results.json` | API blocked status |

The 120-case benchmark is simulated fictional-bank data; it is not a real bank
test set. See `../reports/gate5_final_report.md`.

---

# Gate 6 Qwen3-Embedding files

| File | Content |
|---|---|
| `gate6_environment.json` | Python/torch/CUDA/Qwen/PEFT versions |
| `gate6_qwen_alignment.json` | 32-probe frozen alignment and B3-H replay |
| `gate6_qwen_modules.json` | real `q_proj/k_proj/v_proj/o_proj` modules |
| `gate6_qwen_hard_negatives.csv` | support-only hard negatives |
| `gate6_qwen_plain_adapter/` | Plain LoRA, 2 epochs, 9.20 MB |
| `gate6_qwen_anchor_adapter/` | Anchor λ=0.5 LoRA, 9.20 MB |
| `gate6_qwen_plain_training.json` / `anchor_training.json` | params/time/peak/hash/vector checks |
| `gate6_qwen_*_history.csv` | real epoch losses |
| `gate6_qwen_*_encoding.json` | 4,280-row encode metadata and drift |
| `gate6_qwen_lora_validation.csv` | Plain/Anchor B0/B3-H validation |
| `gate6_qwen_{frozen,plain}_{reject,full}.pt` | Qwen B4 checkpoints |
| `gate6_qwen_*_validation.csv` | B4 validation |
| `gate6_qwen_method_matrix_val.csv` | 9-row validation matrix |
| `gate6_qwen_method_matrix_historical_test.csv` | one-shot historical test |
| `gate6_qwen_method_freeze.json` | hashes and test_evaluated |
| `gate6_qwen_agent_demos.json` | 6 real Qwen Agent replays |
| `gate6_qwen_benchmark_*` | 120 tasks × 4 Qwen routers (480 rows) |
| `gate6_qwen_benchmark_summary.csv` | TSR/tool/safe/routing/latency |
| `gate6_qwen_benchmark_by_family.csv` | family-level TSR |
| `gate6_qwen_benchmark_failures.csv` | failure reasons |

Qwen test is historical exploratory; the 120 Agent tasks are old Gate5 tasks.
See `../reports/gate6_final_report.md`.
