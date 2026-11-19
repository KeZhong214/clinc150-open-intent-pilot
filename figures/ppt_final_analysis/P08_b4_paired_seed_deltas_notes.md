# 图表说明与数据核验

```json
{
  "commit": "4c674cbc19ca3c623535033393878d20b5f04ad9",
  "input": "results/gate7_stability_b4_results.csv",
  "keys": "36 rows, 36 unique (encoder,method,seed); 6 encoders × 2 methods × 3 B4 seeds",
  "pairing": "same encoder × same B4 seed: delta_pp = (RejectOnly - Full) * 100",
  "summary": {
    "gte_frozen": {
      "mean_pp": 0.39382254830292823,
      "sample_std_pp": 0.33178987230665086,
      "n": 3
    },
    "gte_lora42": {
      "mean_pp": 0.6369963708246384,
      "sample_std_pp": 0.22650963737957172,
      "n": 3
    },
    "gte_lora43": {
      "mean_pp": 1.2555168357103723,
      "sample_std_pp": 0.2779691179547214,
      "n": 3
    },
    "gte_lora44": {
      "mean_pp": 2.3912871178469666,
      "sample_std_pp": 0.0931394621517101,
      "n": 3
    },
    "qwen_frozen": {
      "mean_pp": 1.9839386933693692,
      "sample_std_pp": 0.4785131425125051,
      "n": 3
    },
    "qwen_plain42": {
      "mean_pp": 1.04998545839932,
      "sample_std_pp": 0.6761224964141532,
      "n": 3
    }
  },
  "partition": "Gate7 dev_select; B4-head training seeds; support fixed",
  "caption": "B4-RejectOnly 相对 Full 的配对增益：6 组编码器均值为正",
  "conclusion": "6 组配置的 RejectOnly 均值为正，但效应大小随编码器和 seed 变化；不能夸大为跨支持集普遍稳定。"
}
```
