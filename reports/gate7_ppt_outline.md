# Gate 7 PPT 大纲（12–14 页）

## Slide 1 — 结论先行

- Qwen LoRA 的单 seed42 test 负结果不是普遍规律；
- 3-seed Qwen Plain：B3-H test 0.8068、RejectOnly **0.8253**；
- RejectOnly 相对 frozen +1.25 pp，2/3 seed 胜出；
- B4-RejectOnly 在 GTE/Qwen、Frozen/LoRA 上仍稳定优于 Full；
- P1/P2 没超过 P0，P3 未完成：**没有优化成功**。

## Slide 2 — 研究问题

- 为什么 Qwen LoRA 验证提升、历史 test 下降？
- 微批/负例/漂移/unknown 迁移哪个是主因？
- 多种子下核心 B4 结论是否稳？
- 模型变化是否转化为 Agent 任务成功率？

## Slide 3 — A：双层开发集

- dev_threshold：40 类 ×10 + 9 unknown 类 ×40；
- dev_select：另 400 known + 另 9 unknown 类；
- sample/text/class overlap 全 0；
- test 从未参与 tau 或配方选择；
- 只是内部稳健性开发集，不是新盲测。

## Slide 4 — H1：6×6 不是 48×48

- microbatch=2：2 anchor + 2 positive + 2 hard negative = 6；
- SupCon 相似度矩阵 **6×6**；
- grad_accum=8 只是梯度累积，不是 48 向量共同对比；
- 400/2/8 恰好 25 step，余数 0；
- 图：`figures/gate7_toy_6x6.png`。

## Slide 5 — H2：hard negative 风险

| 指标 | 值 |
|---|---:|
| margin mean | −0.0126 |
| margin < 0 | 203/400 |
| 重复 hard negative | 243 |
| unique hard negative | 157 |

- 超过一半 anchor 与 hard negative 余弦高于 positive；
- 不能据此改官方标签，但说明少样本对比目标过强。

## Slide 6 — dev_select 结果

| Encoder | B3-H F1 | OOS | AUROC |
|---|---:|---:|---:|
| Frozen | 0.7904 | 0.6639 | 0.8749 |
| Plain42 | **0.7984** | **0.6917** | **0.8896** |
| Anchor | 0.7942 | 0.6694 | 0.8784 |

- Plain 在 dev_select 优于 frozen；
- 与 Gate6 test seed42 负结果方向相反 → seed/迁移问题。

## Slide 7 — B：P0–P3

| 配方 | dev B3H | dev Reject | 结论 |
|---|---:|---:|---|
| P0 | 0.7984 | **0.7995** | 保留 |
| P1 lr3e-5 | 0.7780 | 0.7978 | 更差 |
| P2 lr1e-5 | 0.7889 | 0.7981 | 更差 |
| P3 no triplet | INCOMPLETE | — | 40/50 step |

- 不把 P3 当完成；
- 不强行宣布优化成功。

## Slide 8 — 3-seed Plain

| Seed | Val B3H | Test B3H | Test Reject |
|---|---:|---:|---:|
| 42 | 0.7984 | 0.7969 | 0.8080 |
| 43 | 0.8051 | 0.8159 | **0.8361** |
| 44 | 0.8050 | 0.8077 | 0.8318 |
| mean±std | 0.8028±.0032 | 0.8068±.0095 | **0.8253±.0151** |

- RejectOnly +1.25 pp vs frozen，2/3 wins；
- 没有用 test 挑 seed。

## Slide 9 — C：冻结后一次 test

- `gate7_final_freeze.json`，test_evaluated=true；
- Frozen B3-H 0.8051、Reject 0.8127；
- Plain 3-seed B3-H 0.8068、Reject 0.8253；
- Optimized = NOT SELECTED（P1/P2/P3 未改善）；
- 历史 test，不是盲测。

## Slide 10 — 稳定性：B4 Reject vs Full

| Encoder | Reject | Full |
|---|---:|---:|
| GTE Frozen | **0.8144** | 0.8105 |
| GTE LoRA42 | **0.8047** | 0.7984 |
| GTE LoRA43 | **0.8277** | 0.8152 |
| GTE LoRA44 | **0.8266** | 0.8026 |
| Qwen Frozen | **0.7950** | 0.7752 |

- Reject 在全部对照中 > Full；
- 每个 seed 的 checkpoint SHA 不同；
- 结论不是单 seed 偶然。

## Slide 11 — Agent 公平对照

| Router | TSR | 成功/120 | Tool | Safe |
|---|---:|---:|---:|---:|
| GTE Frozen Reject | 0.608 | 73 | 0.708 | 1.0 |
| Qwen Frozen Reject | **0.675** | **81** | 0.750 | 1.0 |
| Qwen Plain42 Reject | **0.675** | **81** | 0.750 | 1.0 |

- 旧 120 任务探索性；
- 5 条澄清 bug 修复后 unknown 1.0；
- dispute 仍是弱族。

## Slide 12 — 机制结论

- 单 seed 负结果 ≠ 编码器普遍失败；
- hard negative 质量差、漂移不是唯一解释；
- RejectOnly 的收益跨 GTE/Qwen、跨 seed 稳定；
- F1 与业务 TSR 不单调：Qwen Reject F1 低但 TSR 高。

## Slide 13 — 限制

- test 已看过；
- P3 incomplete；
- 3 seeds 不是显著性；
- Qwen Plain B4-Full 稳定性仅 2 seed；
- 无新独立 unknown 协议；
- LLM API NOT RUN_API_BLOCKED。

## Slide 14 — 下一步

- 新独立 unknown-class 协议；
- 修复 hard negative 质量 / 真 batch 对比；
- 增加 seed 和 bootstrap；
- 修复 dispute 路由和澄清覆盖；
- 不用历史 test 继续调参。
