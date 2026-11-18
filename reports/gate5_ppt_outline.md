# Gate 5 PPT 大纲（12–14 页）

**题目**：LoRA-GTE、B4 模块归因与真实客服 Agent：从模型指标到业务任务

## Slide 1 — 标题与一句话结论

- Gate 5 四项全做：LoRA、B4 归因、RAG Agent、120 任务 benchmark。
- 一句话：**RejectOnly 是唯一稳定正贡献；LoRA 没有改善路由；Frozen-B3-H 的业务成功率仍是最高。**
- 视觉：`figures/gate5_method_matrix.png` 缩略图。

## Slide 2 — 研究问题与边界

- 固定 10-shot support 下：LoRA 表征适配与学习式拒识是否互补？
- 模型改善能否转化为业务任务成功率？
- BANKING77 test 已看过：Gate 5 只做一次历史探索性复评。
- 120 条业务任务是模拟数据，不是真实银行线上数据。

## Slide 3 — A：LoRA 注入与训练

- GTE-ModernBERT，CLS pooling，L2，128 tokens。
- 检查实际模块名：`Wqkv / Wo / Wi`。
- r=8、alpha=16、dropout=0.05；1.69M 可训练参数 = 1.12%。
- SupCon + support-mined hard-negative triplet。
- 2 epoch，22.19 s，峰值 1,368 MiB，adapter 6.78 MB。
- 视觉：`figures/gate5_lora_training.png`。

## Slide 4 — LoRA 的真实效果

| Encoder | B0 F1 | B3-H F1 | B3-H routing |
|---|---:|---:|---:|
| Frozen | 0.734 | 0.767 | 0.851 |
| LoRA | 0.753 | 0.760 | 0.828 |

- LoRA-B0 的 OOS 更高，但 B3-H 反而下降；
- 4-epoch 第一次尝试更差（0.7397），按 validation 改成 2 epoch；
- 没有用 test 选择 epoch。
- 视觉：`figures/gate5_lora_validation.png`。

## Slide 5 — B：三种模块

- B3-H：固定 0.5/0.5；
- B4-RejectOnly：固定 g=0.5，只训练 7→16→1 拒识头；
- B4-Full：训练 gate + 拒识头；
- B4-Fusion：只训练 gate，保留消融。
- 强调：RejectOnly 的 gate 参数被冻结，不是偷偷训练。

## Slide 6 — 六组矩阵（validation）

| Frozen | F1 | LoRA | F1 |
|---|---:|---|---:|
| B0 | 0.734 | B0 | 0.753 |
| B3-H | 0.767 | B3-H | 0.760 |
| RejectOnly | **0.790** | RejectOnly | 0.767 |
| B4-Full | 0.781 | B4-Full | 0.753 |

- RejectOnly 在 frozen 上最好；LoRA 没有带来互补。
- 视觉：`figures/gate5_method_matrix.png`。

## Slide 7 — 历史探索性 test

| Frozen | F1 | OOS | LoRA | F1 | OOS |
|---|---:|---:|---|---:|---:|
| B3-H | 0.831 | 0.637 | B3-H | 0.803 | 0.584 |
| RejectOnly | **0.843** | 0.643 | RejectOnly | 0.814 | 0.574 |
| B4-Full | 0.837 | 0.658 | B4-Full | 0.802 | 0.500 |

- 只加 reject head 是正贡献；gate 和 LoRA 都不是。
- test 已看过，不能宣称新盲测。

## Slide 8 — C：RAG + 自然语言 Agent

- 30 条虚构银行政策，doc_id/version/source/citation。
- TF-IDF 检索；回答显示引用；检索不改数据库。
- 自然澄清只显示“checking whether a card has arrived”，不泄露内部标签。
- 保留 TXN 槽位；每个 Streamlit 会话独立 SQLite。
- 敏感动作二次确认；工具层所有权校验；审计表。
- 视觉：`figures/gate5_agent_rag.png`。

## Slide 9 — 真实业务链

- B4 查卡：CARD-1001 delivered。
- B4 查转账：TXN-2001 pending。
- B4 取消转账：pending → confirm → cancelled。
- B4 非本人卡：CARD-2001 → ownership denied → handoff。
- B4 拒绝激活：CARD-1002 remains inactive。
- 视觉：`figures/gate5_agent_cancel.png`。

## Slide 10 — D：120 条 benchmark

- 6 个任务族，120 条自然语言任务，先冻结文件哈希。
- 3 个真实路由：Frozen-B3H、Frozen-B4Full、LoRA-B4Full。
- 同输入、同初始 DB；Stub 不进入 benchmark。
- 视觉：`figures/gate5_benchmark_by_family.png`。

## Slide 11 — Benchmark 结果

| Router | TSR | Tool+arg | Safe | Routing |
|---|---:|---:|---:|---:|
| Frozen-B3H | **0.625** | 0.725 | 1.000 | 0.642 |
| Frozen-B4Full | 0.583 | 0.700 | 1.000 | 0.642 |
| LoRA-B4Full | 0.583 | 0.725 | 1.000 | 0.642 |

- F1 的小改善没有转化为业务成功率；
- 43/120 失败是 routing（每个 router 都一样）；
- 安全操作率 100%。
- 视觉：`figures/gate5_benchmark_overall.png` 和失败图。

## Slide 12 — 分任务族与失败原因

| Family | B3H | B4 | LoRA |
|---|---:|---:|---:|
| card | 0.84 | 0.84 | 0.68 |
| transfer | 0.43 | 0.43 | 0.47 |
| dispute | 0.25 | 0.25 | 0.35 |
| policy | 0.90 | 0.90 | 0.90 |
| unknown | 1.00 | 0.67 | 0.67 |
| safety | 0.30 | 0.30 | 0.40 |

- 主要瓶颈是自然语言路由，不是工具越权；
- RAG 对 policy 有效，但不能修复金融意图路由。

## Slide 13 — 限制与结论

- 单数据集、单编码器、单 seed；
- test 已看过，非盲测；
- 120 条为模拟任务；
- API 12 条因无凭据 **NOT RUN_API_BLOCKED**；
- 结论：RejectOnly 是当前最值得保留的模块；LoRA/B4-Full 未转化；
  B3-H 仍是业务默认路由。

## Slide 14 — 下一步

- 新独立 unknown-class 协议复现 RejectOnly；
- 多 seed LoRA，先修复 routing 退化；
- 扩充 transfer/dispute/permission 自然语言任务；
- 配好 API 后跑 12 条真实调用，单独报告 token/cost。
