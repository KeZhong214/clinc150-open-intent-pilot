# 图表说明与数据核验

```json
{
  "commit": "4c674cbc19ca3c623535033393878d20b5f04ad9",
  "input": "results/gate7_final_historical_test_v2.csv",
  "filter": "method == 'B4-RejectOnly' and encoder in ['frozen','plain42','plain43','plain44']",
  "rows": 4,
  "frozen_f1": 0.8129976748317347,
  "seed_values": {
    "plain42": {
      "seed": 42,
      "f1": 0.8090621922971374
    },
    "plain43": {
      "seed": 43,
      "f1": 0.8355678763754396
    },
    "plain44": {
      "seed": 44,
      "f1": 0.8320672837688348
    }
  },
  "lora_mean": 0.8255657841471372,
  "lora_sample_std_ddof1": 0.014399303632915403,
  "caption": "Qwen LoRA 三个训练 seed 的 B4-RejectOnly 历史探索性结果",
  "conclusion": "LoRA 三种子均值 82.56%±1.44 pp，略高于 Frozen 81.30%；seed42 是负结果，保留展示，不能删。"
}
```
