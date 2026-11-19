# 图表说明与数据核验

```json
{
  "commit": "4c674cbc19ca3c623535033393878d20b5f04ad9",
  "input": "results/gate5_method_matrix_val.csv",
  "filter": "encoder == 'frozen'; 5 methods",
  "n": 5,
  "partition": "BANKING77 GTE Frozen development validation; not historical test",
  "values": {
    "B0": {
      "f1": 0.733685,
      "oos": 0.402778
    },
    "B3-H": {
      "f1": 0.767048,
      "oos": 0.547222
    },
    "B4-Fusion": {
      "f1": 0.740441,
      "oos": 0.470833
    },
    "B4-Full": {
      "f1": 0.780876,
      "oos": 0.604167
    },
    "B4-RejectOnly": {
      "f1": 0.790203,
      "oos": 0.606944
    }
  },
  "deltas_pp": {
    "RejectOnly_minus_B3H": 0.023154176011595085,
    "RejectOnly_minus_Full": 0.009327001705102367
  },
  "caption": "B4-RejectOnly 的主要收益来自拒识头，而非更复杂的可训练门控",
  "conclusion": "B3-H 带来基础提升；Fusion 单独训练 gate 不提升；RejectOnly 超过 Full，收益主要来自 reject head。"
}
```
