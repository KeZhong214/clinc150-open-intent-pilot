# Gate 6 C：Qwen 联合矩阵

**分支**：`gate6-qwen-lora`  
**Qwen 版本**：`Qwen/Qwen3-Embedding-0.6B`，revision `97b0c614...`，last-token，1024d。

## Validation（BANKING77 10-shot，800 known + 720 dev-unknown）

| Encoder | Method | F1 | OOS | Routing |
|---|---|---:|---:|---:|
| Qwen Frozen | B0 | 0.7095 | 0.4708 | 0.8038 |
| Qwen Frozen | B3-H | 0.7404 | 0.5208 | 0.8213 |
| Qwen Frozen | B4-RejectOnly | 0.7501 | 0.5514 | 0.8250 |
| Qwen Frozen | B4-Full | 0.7431 | 0.5431 | 0.8188 |
| Qwen Plain LoRA | B0 | 0.7404 | 0.5625 | 0.8200 |
| Qwen Plain LoRA | B3-H | **0.7580** | 0.6139 | 0.8213 |
| Qwen Plain LoRA | B4-RejectOnly | **0.7690** | **0.6417** | 0.8288 |
| Qwen Plain LoRA | B4-Full | 0.7644 | 0.6292 | 0.8250 |
| Qwen Anchor LoRA | B3-H | 0.7573 | 0.6097 | 0.8200 |

## 历史探索性 test（一次，已看过，不是新盲测）

| Encoder | Method | F1 | OOS | Routing |
|---|---|---:|---:|---:|
| Qwen Frozen | B0 | 0.7821 | 0.5500 | 0.8219 |
| Qwen Frozen | B3-H | 0.8064 | 0.5855 | 0.8413 |
| **Qwen Frozen** | **B4-RejectOnly** | **0.8127** | **0.6000** | **0.8438** |
| Qwen Frozen | B4-Full | 0.8047 | 0.5803 | 0.8363 |
| Qwen Plain LoRA | B0 | 0.7837 | 0.5250 | 0.8213 |
| Qwen Plain LoRA | B3-H | 0.7972 | 0.5618 | 0.8281 |
| Qwen Plain LoRA | B4-RejectOnly | 0.8080 | 0.5750 | 0.8413 |
| Qwen Plain LoRA | B4-Full | 0.7973 | 0.5711 | 0.8294 |
| Qwen Anchor LoRA | B3-H | 0.7968 | 0.5355 | 0.8363 |

## 结论

- Qwen Plain/Anchor 在 **validation** 上确实高于 frozen（B3-H 0.758/0.757 vs 0.740），
  但 **historical test 没有复现**：Plain B3-H 0.797 < Frozen 0.806。
- Qwen 上 **RejectOnly 仍然优于 Full**：Frozen 0.8127 vs 0.8047；
  Plain 0.8080 vs 0.7973。Gate5 的模块归因在 Qwen 上成立。
- **Frozen Qwen RejectOnly 0.8127** 是 Qwen 最好结果，但仍低于 Gate5 Frozen GTE
  RejectOnly 0.8435；Qwen 0.6B 没有在 BANKING77 上超过 GTE。
- Anchor 把 support 漂移余弦从 Plain 0.881 提高到 0.945，但 test F1 0.7968，
  与 Plain 0.7972 基本持平：**减小漂移没有转化为 test 收益**。
- test 标签已在 Gate3/5 看过；本表只是历史探索性对照。

## 资源

- Qwen LoRA：2,293,760 可训练参数（0.38%），峰值 1,260 MiB；
- Plain 295 s / Anchor 290 s（2 epoch，50 steps）；
- B4 MLP 训练 60 epochs，秒级；Qwen B4 checkpoint 很小。
