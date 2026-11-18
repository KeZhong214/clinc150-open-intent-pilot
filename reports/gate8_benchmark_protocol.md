# Gate 8 独立评测协议

## 1. 定位

- 这是**新冻结模拟测试**，不是外部盲测，也不是新盲测。
- 案例由同一项目执行者编写；已明确标注 `new_frozen_simulated_test_not_external_blind`。
- 旧 Gate5 120 条任务只用于开发回归，**不与本测试混合**。
- 本测试在最终三模型运行前冻结；运行后不据结果改案例、阈值或模型。

## 2. 案例集

| 项目 | 值 |
|---|---|
| 文件 | `data/gate8/benchmark_cases.jsonl` |
| 数量 | **102** |
| SHA-256 | `7ca2a276fc9b0be0e9c19113c4181359d31c013982d9615793df28c2079166e1` |
| 冻结记录 | `results/gate8_benchmark_freeze.json` |
| 冻结代码 Commit | `8495244` |
| 类别 | 16 个 family |

类别覆盖：

- 卡状态、激活确认；
- 转账状态、取消确认/拒绝；
- dispute（completed 合规 / pending 不合规）；
- FAQ 政策与引用；
- OOS；
- 所有权/无效对象；
- 多轮 slot、纠正、主题切换、reset、确认阶段改对象。

每例包含：`turns`、`expected_tools`、`expected_final`、`expected_terms`、
`expected_kb_docs`（只对需要精确验证的少数案例）、`sensitive`、`expected_escalation`。

## 3. 模型与阈值（正式协议）

三个路由器全部使用 **Gate7 dev_threshold**，不是 Gate5/6 旧阈值。

| Router | Encoder | tau | 来源 |
|---|---|---:|---|
| GTE-Frozen-RejectOnly | GTE-ModernBERT Frozen | 0.4245788127 | Gate7 stability gte_frozen/reject/seed42 |
| Qwen-Frozen-RejectOnly | Qwen3-Embedding-0.6B frozen | 0.4424596250 | `gate7_dev_frozen.csv` |
| Qwen-Plain42-RejectOnly | Qwen Plain LoRA seed42 | 0.4168315291 | `gate7_dev_plain42.csv` |

模型/权重：

- GTE checkpoint `gate5_reject_only_frozen.pt`；
- Qwen frozen checkpoint `gate6_qwen_frozen_reject.pt`；
- Qwen Plain42 checkpoint `gate6_qwen_plain_reject.pt`；
- adapter SHA `4efaee5c...`；
- 完整 SHA 在 `results/gate8_benchmark_freeze.json`。

## 4. 运行

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m src.gate8.benchmark --router GTE-Frozen-RejectOnly
.\.venv\Scripts\python.exe -m src.gate8.benchmark --router Qwen-Frozen-RejectOnly
.\.venv\Scripts\python.exe -m src.gate8.benchmark --router Qwen-Plain42-RejectOnly
.\.venv\Scripts\python.exe -m src.gate8.combine_benchmark
```

- 每个 case 用独立 session DB，初始 seed 固定；
- 不并发加载模型；每个 router 顺序运行；
- 延迟是 CPU/GPU 单进程真实 wall time；
- 所有工具执行、SQLite 状态、KB 引用均来自真实运行。

## 5. 指标

- **TSR**：final state + expected terms + tools/args + KB citation + escalation +
  clarification + route + safety 全部通过；
- **tool_accuracy**：工具名和显式参数完全匹配；
- **safe_action_rate**：敏感变更必须有“确认前工具调用 + confirmed=true”，
  且未确认/越权不能改变资产；
- **routing_accuracy**：首轮 route 是否落在允许集合；
- **clarification_success**：需要澄清的案例出现自然语言澄清；
- **citation_coverage**：至少一条真实 KB doc_id 的案例比例；
- **P50/P95 latency**：每例多轮总耗时。

## 6. 统计

- 每 router 102 例，非配对 bootstrap 2000 次，报告 TSR 95% 区间；
- 以 GTE 为 baseline，对同一 case 做配对 bootstrap 差值；
- 不做显著性“优胜”宣称；只报告区间和点估计。

## 7. 结果解释边界

- 这是模拟银行、英文为主的测试；
- 案例与 Agent 由同一作者构造；
- 102 例不足以代表真实银行分布；
- 结果不能与旧 120 条或分类 F1 直接配对；
- 旧 Agent 120 结果和 Gate7 分类结果是不同协议。
