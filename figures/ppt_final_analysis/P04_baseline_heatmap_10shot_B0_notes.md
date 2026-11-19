# 图表说明与数据核验

```json
{
  "commit": "4c674cbc19ca3c623535033393878d20b5f04ad9",
  "input": "results/gate3_matrix.csv",
  "filter": "method == 'B0' and shot == 10",
  "rows": 12,
  "key": "dataset × model; values open_set_macro_f1",
  "row_best": {
    "clinc150": "Qwen3-Emb-0.6B",
    "banking77": "BGE-small",
    "massive": "Qwen3-Emb-0.6B"
  },
  "verification": "12 unique (dataset,model); B0/10-shot; same 12 cells as the matrix; no new samples",
  "partition": "CLINC150 historical exploratory; BANKING77/MASSIVE validation matrix; not a test leaderboard",
  "caption": "四个编码器在三个数据集上的开放集 F1（B0 / 10-shot）",
  "conclusion": "任务域差异显著，模型并非绝对通吃；BANKING77 的细粒度银行意图更适合客服主线。"
}
```
