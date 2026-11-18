# Gate 6 PPT 大纲（12–14 页）

**题目**：Qwen3-Embedding LoRA、表征漂移与开放集路由联合优化

## Slide 1 — 结论先行

- Qwen3-Embedding 0.6B 真实 LoRA 训练成功；
- validation 提升、historical test 未复现；
- Anchor 减小漂移但不提高 test；
- Qwen RejectOnly 的业务 TSR 65.0%，四版本最高；
- 分类 F1 仍是 GTE-RejectOnly 最高（0.8435）；
- 一句话：**“表示微调有效”与“业务路由有效”不是同一件事。**

## Slide 2 — 研究问题与边界

- RQ1：Qwen LoRA 是否改善 known/OOS？
- RQ2：GTE-LoRA 退化是否与漂移有关，Anchor 是否有用？
- RQ3：Qwen 上 RejectOnly vs Full，是否转化为业务 TSR？
- BANKING77 test 已看过；120 任务旧回归；不是新盲测。

## Slide 3 — A：先对齐，再训练

- Qwen revision `97b0c614...`，1024d，last-token，L2，128 tokens；
- 第一次 left padding：mean cosine 0.412，top1 0.344 → 失败；
- 改成 Gate3 一致的 **right padding + last non-pad token**；
- 32 probe：mean 0.999998，min 0.999995，top1 1.000；
- Gate3 B3-H validation 复跑 0.74140（0.74035）。
- 视觉：`figures/gate6_qwen_alignment.png`。

## Slide 4 — A：LoRA smoke

- 真实模块：`self_attn.q_proj/k_proj/v_proj/o_proj`；
- 2,293,760 可训练参数 = 0.38%；
- 2-step smoke：梯度、save、reload 全部通过；
- 峰值 1,213.6 MiB，reload 向量差 0；
- 8GB 笔记本可训练。

## Slide 5 — B：Plain / Anchor

| Run | Support cosine | Val B3-H | Test B3-H |
|---|---:|---:|---:|
| Frozen | 1.000 | 0.7404 | 0.8064 |
| Plain | 0.892 | **0.7580** | 0.7972 |
| Anchor | **0.936** | 0.7573 | 0.7968 |

- Anchor 把漂移从 0.892 提到 0.936；
- validation 几乎不变，test 反而略低；
- **漂移更小 ≠ test 更好**。

## Slide 6 — B：训练损失与资源

- 400 support；2 epochs；50 steps；
- Plain 295.2 s；Anchor 290.2 s；峰值 1,259.9 MiB；
- adapter 9.20 MB；
- 4,280 条重编码：80–84 s，1,185.6 MiB；
- Qwen 比 GTE 慢约 4 倍。
- 视觉：`figures/gate6_qwen_lora_training.png`。

## Slide 7 — C：Validation 矩阵

| Encoder | B0 | B3-H | RejectOnly | Full |
|---|---:|---:|---:|---:|
| Qwen Frozen | 0.710 | 0.740 | 0.750 | 0.743 |
| Qwen Plain | 0.740 | 0.758 | **0.769** | 0.764 |

- Plain 整体最高；RejectOnly 最高；
- Gate5 的 RejectOnly > Full 在 Qwen 上成立。

## Slide 8 — C：Historical test

| Encoder | B0 | B3-H | RejectOnly | Full |
|---|---:|---:|---:|---:|
| Qwen Frozen | 0.782 | 0.806 | **0.813** | 0.805 |
| Qwen Plain | 0.784 | 0.797 | 0.808 | 0.797 |

- Qwen Plain B3-H **低于** frozen；
- Qwen Frozen RejectOnly 0.813 是 Qwen 最好；
- 但 GTE Frozen RejectOnly 0.8435 仍最高。
- 视觉：`figures/gate6_qwen_matrix.png`。

## Slide 9 — D：Agent 接入

- 真实 Qwen 在线编码，不是 CSV 查分；
- 三个 Qwen-Plain 路由：B3-H / RejectOnly / B4-Full；
- 复用 Gate5 的 30 条政策、SQLite、多轮、确认、所有权、审计；
- 6 条真实回放（5 条 Qwen-LoRA）；
- 视觉：`figures/gate6_qwen_agent_cancel.png`、`gate6_qwen_agent_rag.png`。

## Slide 10 — D：120 任务回归

| Router | TSR | Tool+arg | Safe | P50 |
|---|---:|---:|---:|---:|
| Qwen-Frozen-B3H | 0.625 | 0.700 | 1.000 | 0.138 s |
| Qwen-Plain-B3H | 0.592 | 0.708 | 1.000 | 0.220 s |
| Qwen-Plain-B4Full | 0.625 | 0.742 | 1.000 | 0.257 s |
| **Qwen-Plain-RejectOnly** | **0.650** | **0.767** | **1.000** | 0.244 s |

- RejectOnly 业务 TSR 最高，也高于 Gate5 GTE-B3H 0.625；
- 安全率 100%；
- 但这是旧 120 任务，不是盲测。

## Slide 11 — 分族与失败

| Family | Frozen | Plain | Full | Reject |
|---|---:|---:|---:|---:|
| card | 0.84 | 0.72 | 0.80 | **0.88** |
| transfer | 0.60 | 0.53 | 0.60 | **0.63** |
| dispute | **0.20** | 0.15 | 0.15 | 0.15 |
| policy | 0.90 | 0.90 | 0.90 | 0.90 |
| unknown | 0.67 | 0.67 | 0.67 | 0.67 |
| permission | 0.40 | 0.60 | 0.60 | 0.60 |

- 主要失败：routing 34–42/120；tool 2–11；clarification 0–5；
- 5 条应澄清任务 Qwen 全部直接接受；这是缺陷。

## Slide 12 — GTE vs Qwen

| | GTE | Qwen |
|---|---:|---:|
| 参数 | 0.15B | 0.6B |
| 4,280 编码 | 18.5 s | 80–84 s |
| Test B3-H | **0.831** | 0.806 |
| Test RejectOnly | **0.843** | 0.813 |
| Agent TSR (Reject/B3H) | 0.625（B3H） | **0.650（Reject）** |
| P50 Agent | 0.05 s | 0.14–0.26 s |

- Qwen 分类没有超过 GTE；
- Qwen RejectOnly 的业务 TSR 有正结果，但不能外推为全面更优。

## Slide 13 — 结论与限制

- 结论 1：Qwen LoRA validation 有效，test 不复现；
- 结论 2：Anchor 减小漂移但无 test 收益；
- 结论 3：RejectOnly > Full，跨 GTE/Qwen 一致；
- 结论 4：Qwen RejectOnly TSR 65%，业务最优；分类仍 GTE 最优；
- 限制：test 已看过、旧 120 任务、单 seed、无新盲测、API blocked。

## Slide 14 — 下一步

- 新独立 unknown-class 协议；
- 多 seed Qwen LoRA + Anchor λ 扫描；
- 修复澄清与 dispute 路由；
- 真实 API 凭据后跑 LLM 模式（与本轮 Qwen LoRA 独立）。
