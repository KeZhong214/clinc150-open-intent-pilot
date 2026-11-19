# Gate 8.1 协议审计：权重、阈值、Adapter 与注册表

## 1. 结论

Gate 8 的 GTE 配置存在**权重—阈值不配套**：

- 权重：`results/gate5_reject_only_frozen.pt`
  SHA-256 `0e920d0fed6b13d30dd3e84c53cb9e49747760823b9f20c61df2cbdd7d89cf0f`
- 使用阈值：`0.4245788127`，来自 **Gate7 stability**
  `gte_frozen/B4-RejectOnly/seed42`，其 checkpoint 是
  `results/gate7_stability_gte_frozen_reject_seed42.pt`
  SHA-256 `04F0005A9AAC3AC291AC4464EF15046B25BA979B7D25A47BF12308C596D92B03`（不同权重）
- Gate5 权重原阈值：`0.4416714430`

因此原 Gate8 GTE 的“权重 + 阈值”不是同一模型，审计不通过。

## 2. 修复选择

选择规则在查看 Gate8.1 测试结果前固定：

1. 保留 **Gate5 GTE 权重**（不引入 Gate7 另一份 checkpoint）；
2. 在 **Gate7 dev_threshold**（400 known + 360 unknown）上重新校准 tau；
3. 约束 `known acceptance >= 0.90`，最大化 dev open-set macro-F1；
4. 保存校准文件、split SHA、checkpoint SHA；不允许用 test 选 tau。

结果：

| 项 | 值 |
|---|---|
| checkpoint | `results/gate5_reject_only_frozen.pt` |
| checkpoint SHA | `0e920d0f...` |
| calibration split | `data/manifest/gate7_dev_threshold.csv` |
| split SHA | `a6821544df31e95ba93a6f8b79ca4289ffd91f7a5e69d3a7a4315d71592f564a` |
| target acceptance | 0.90 |
| **new tau** | **0.43883611261844635** |
| dev known acceptance | 0.90 |
| dev OOS recall | 0.5444 |
| dev open-set F1 | 0.7961 |

旧 Gate5 tau 0.4416714430 不再用于 GTE Agent。

## 3. 三模型注册表

`results/gate8_1_model_registry.json`：

| Router | Encoder | Checkpoint SHA | Adapter SHA | tau | 校准来源 |
|---|---|---|---|---:|---|
| GTE-Frozen-RejectOnly | GTE-ModernBERT Frozen | `0e920d0f...` | — | 0.4388361126 | Gate7 dev_threshold recalb |
| Qwen-Frozen-RejectOnly | Qwen3-Embedding frozen | `d7512e52...` | — | 0.4424596250 | `gate7_dev_frozen.csv` |
| Qwen-Plain42-RejectOnly | Qwen Plain LoRA seed42 | `d222f968...` | `4efaee5c...` | 0.4168315291 | `gate7_dev_plain42.csv` |

`src/gate8_1/registry.py::validate` 会：

- 校验 checkpoint SHA；
- 校验 adapter SHA（Qwen Plain42）；
- 校验 threshold 文件 SHA；
- 拒绝 `paired=false` 或任何 SHA 不一致的配置。

Gate8.1 路由器只能通过该 registry 构建。

## 4. 特征/输入一致性

- GTE：Gate3 support embeddings + Gate5 RejectOnly checkpoint；
  `b4_scores` 使用 support prototypes、`compute_pn`、固定 gate 0.5、reject head；
- Qwen：frozen/plain adapter 的 last non-pad token、L2 归一化、1024 维；
- 所有 router 使用同一 support 10-shot 和 Gate7 dev_threshold 协议；
- GTE 和 Qwen 的 encoder 不同，不比较原始 score，只比较阈值后的接受/拒绝和业务结果；
- 没有把 GTE 768 维向量输入 Qwen 1024 维模型。

## 5. 原 Gate8 结果保留

- `results/gate8_agent_*` 与 `results/gate8_benchmark_freeze.json` 未覆盖；
- Gate8.1 新文件使用 `gate8_1_*` 前缀；
- 旧 Gate8 GTE 102 例回归（新 tau）：
  - old TSR **0.95098**
  - new TSR **0.97059**
  - 记录：`results/gate8_1_old102_comparison.json`；
  - 这是**历史协议纠错回归**，不是新盲测，也没有据此选模型。

## 6. 审计结论

- GTE 权重—阈值现在严格配套；
- Qwen 两套权重—阈值—adapter 可追溯；
- 原始 Gate8 结果完整保留；
- 新 GTE tau 只由 Gate7 dev 决定，未使用 Gate8/Gate8.1 test 选择。
