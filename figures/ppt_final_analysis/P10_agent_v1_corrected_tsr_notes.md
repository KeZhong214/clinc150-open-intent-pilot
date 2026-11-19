# 图表说明与数据核验

```json
{
  "commit": "4c674cbc19ca3c623535033393878d20b5f04ad9",
  "input": "results/gate8_2_v1_rescore_summary.csv",
  "filter": "all 3 model rows; n=120",
  "metric": "v1_corrected_tsr",
  "values_pct": {
    "GTE-Frozen-RejectOnly": 60.0,
    "Qwen-Frozen-RejectOnly": 60.0,
    "Qwen-Plain42-RejectOnly": 70.83333333333334
  },
  "success_counts": {
    "GTE-Frozen-RejectOnly": 72,
    "Qwen-Frozen-RejectOnly": 72,
    "Qwen-Plain42-RejectOnly": 85
  },
  "original_v1_counts": {
    "GTE-Frozen-RejectOnly": 69,
    "Qwen-Frozen-RejectOnly": 69,
    "Qwen-Plain42-RejectOnly": 83
  },
  "delta_qwen_plain_vs_gte_pp": 10.833333,
  "partition": "same-author frozen simulated test; v1 original traces, post-hoc scoring correction; not DeepSeek API",
  "caption": "Gate 8.2 · 120 条模拟业务任务 · 首次冻结轨迹修正评分",
  "conclusion": "Qwen Plain42 在这批模拟任务上的 TSR 更高（85/120 vs 72/120），但样本为模拟测试，不宣称显著优越。"
}
```
