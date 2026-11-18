# Gate 6 A/B 进度记录

**分支**：`gate6-qwen-lora`  
**基线**：`gate5-lora-agent@ba7c43f`

## A：Qwen 冻结对齐

- Qwen `Qwen/Qwen3-Embedding-0.6B` revision `97b0c614...`；
- 1024 维，last-token pooling，L2，max_length=128；
- prompt 与 Gate3 字节一致，SHA `4bbdf1c1...`；
- **右 padding** + last non-pad token 才与 Gate3 `SentenceTransformer` 对齐；
  左 padding 的 Qwen causal/RoPE 位置会产生错误向量（已记录，不采用）；
- 32 样本：mean cosine 0.999998，min 0.999995，top1 一致率 1.0；
- Gate3 B3-H validation 复跑 F1 0.74140（Gate3 0.74035），在数值容差内；
- 真实模块名 `self_attn.q_proj/k_proj/v_proj/o_proj`，112 个 Linear 目标；
- 2-step smoke：2,293,760 可训练参数（0.38%），峰值 1,213.6 MiB，
  save/reload max diff 0.0。

## B：Plain / Anchor LoRA

共同设置：400 support，seed 42，2 epochs，microbatch 2，grad accumulation 8，
eff. 16 triplets/step，50 optimizer steps，lr 1e-4。

| Run | Train s | Peak MiB | Adapter MB | Frozen cos |
|---|---:|---:|---:|---:|
| Plain | 295.2 | 1,259.9 | 8.78 | 0.8806 |
| Anchor (λ=0.5) | 290.2 | 1,259.9 | 8.78 | 0.9454 |

Anchor 明显减小 support 表征漂移。

Validation：

| Encoder | B0 F1 | B3-H F1 | B3-H OOS |
|---|---:|---:|---:|
| Qwen Frozen | 0.7095 | 0.7404 | 0.5208 |
| Qwen Plain | 0.7404 | **0.7580** | **0.6139** |
| Qwen Anchor | 0.7275 | 0.7573 | 0.6097 |

Plain 略优于 Anchor；Anchor 的漂移更小但指标没有更高。待 C 阶段在冻结 test
上比较。
