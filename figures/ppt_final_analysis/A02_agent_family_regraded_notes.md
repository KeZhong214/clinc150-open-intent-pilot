# 图表说明与数据核验

```json
{
  "commit": "4c674cbc19ca3c623535033393878d20b5f04ad9",
  "input": "results/gate8_2_v1_rescore_per_case.csv",
  "filter": "8 families with n>=5",
  "families": [
    "card_status",
    "card_activation",
    "transfer_status",
    "cancel_transfer",
    "dispute_ticket",
    "permission",
    "policy_faq",
    "oos"
  ],
  "verification": "360 rows; each model 120 unique case_id, 18 families; case sets identical",
  "rates_pct": {
    "card_status": {
      "GTE-Frozen-RejectOnly": 80.0,
      "Qwen-Frozen-RejectOnly": 90.0,
      "Qwen-Plain42-RejectOnly": 90.0
    },
    "card_activation": {
      "GTE-Frozen-RejectOnly": 60.0,
      "Qwen-Frozen-RejectOnly": 70.0,
      "Qwen-Plain42-RejectOnly": 60.0
    },
    "transfer_status": {
      "GTE-Frozen-RejectOnly": 60.0,
      "Qwen-Frozen-RejectOnly": 30.0,
      "Qwen-Plain42-RejectOnly": 80.0
    },
    "cancel_transfer": {
      "GTE-Frozen-RejectOnly": 40.0,
      "Qwen-Frozen-RejectOnly": 60.0,
      "Qwen-Plain42-RejectOnly": 80.0
    },
    "dispute_ticket": {
      "GTE-Frozen-RejectOnly": 70.0,
      "Qwen-Frozen-RejectOnly": 20.0,
      "Qwen-Plain42-RejectOnly": 50.0
    },
    "permission": {
      "GTE-Frozen-RejectOnly": 40.0,
      "Qwen-Frozen-RejectOnly": 60.0,
      "Qwen-Plain42-RejectOnly": 60.0
    },
    "policy_faq": {
      "GTE-Frozen-RejectOnly": 50.0,
      "Qwen-Frozen-RejectOnly": 50.0,
      "Qwen-Plain42-RejectOnly": 41.66666666666667
    },
    "oos": {
      "GTE-Frozen-RejectOnly": 100.0,
      "Qwen-Frozen-RejectOnly": 100.0,
      "Qwen-Plain42-RejectOnly": 100.0
    }
  },
  "counts": {
    "card_status": 10,
    "card_activation": 10,
    "transfer_status": 10,
    "cancel_transfer": 10,
    "dispute_ticket": 10,
    "permission": 5,
    "policy_faq": 12,
    "oos": 10
  },
  "partition": "Gate8.2 v1 same-trace corrected; 120 simulated tasks; not v2; not API demo",
  "caption": "v1 校正后业务族成功率：差距集中在权限、政策 FAQ 与转账",
  "conclusion": "政策 FAQ 与权限族差距最大；小样本族（如 cancel_idempotent n=1）未进入本图。"
}
```
