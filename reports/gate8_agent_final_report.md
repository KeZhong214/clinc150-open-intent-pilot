# Gate 8 最终报告：银行智能客服 Agent 完善与独立评测

**分支**：`gate8-agent-final`  
**基线**：`gate7-final-fix@65d472c`  
**最终执行**：2026-11-19，Windows 11 / RTX 5070 Laptop 8 GB  
**性质**：离线 Router-driven Agent + 新冻结模拟测试；不新增训练；不进入 Gate 9。

---

## 1. 交付结论

本轮把此前“分类 + 单轮工具”的演示，补成了一个**真正多轮、可确认、可纠正、可审计、
有 OOS/越权拒绝**的模拟银行 Agent；并用一套**新冻结 102 例**做三模型同协议对照。

关键事实：

1. **GTE-Frozen-RejectOnly 的 Agent TSR 最高：0.951**（97/102），P50 0.060 s；
2. Qwen-Frozen 0.824（84/102），Qwen-Plain42 0.794（81/102）；
3. **三者安全操作率均为 1.000**，没有未确认的敏感变更；
4. 失败主要在 Qwen RejectOnly 对 dispute 首句的接受边界，**不是 SQLite 权限/确认 bug**；
5. 旧 120 条不再作为新测试；本轮 102 条为新冻结模拟测试，不是外部盲测；
6. LLM API 无凭据：**NOT RUN_API_BLOCKED**；离线模式完整可用；
7. P3/P0、Gate7.1 结果和权重全部保留，未覆盖。

> 结论与 Gate7 分类结果不是同一排序：GTE Agent TSR 高于 Qwen，尽管 Gate7 的
> Qwen RejectOnly 分类 F1 略高。Agent 还受 route 接受、工具参数、澄清和错误处理影响。

---

## 2. 从 Gate7.1 继承的协议

Gate8 分类/Agent 统一使用 **Gate7 dev_threshold**，不使用旧 Gate5/6 阈值：

| Router | tau | 来源 |
|---|---:|---|
| GTE-Frozen-RejectOnly | 0.4245788127 | Gate7 stability gte_frozen/reject/seed42 |
| Qwen-Frozen-RejectOnly | 0.4424596250 | `gate7_dev_frozen.csv` |
| Qwen-Plain42-RejectOnly | 0.4168315291 | `gate7_dev_plain42.csv` |

模型/权重：

- GTE checkpoint `gate5_reject_only_frozen.pt`；
- Qwen frozen `gate6_qwen_frozen_reject.pt`；
- Qwen Plain42 `gate6_qwen_plain_reject.pt`；
- Qwen Plain adapter SHA `4efaee5c...`；
- 完整 SHA/案例 SHA 在 `results/gate8_benchmark_freeze.json`。

---

## 3. Agent 功能完善

### 3.1 修复的真实缺陷

1. **dispute 被 policy fallback 截获**  
   原条件在 `DISPUTE_INTENTS` 之前命中 “dispute/recognise”，只回 KB、不收 TXN。
   现在 policy fallback 排除卡/转账/争议 intent，并收紧 `POLICY_TERMS`。
2. **pending slot 吞新话题**  
   增加 `_topic_switch`，用户改问 policy 时先清空 pending，再重新 route。
3. **确认阶段不能改对象**  
   “No, use TXN-2002 instead.” 现在更新 pending_id，要求重新确认，不执行。
4. **dispute 不检查交易状态**  
   先 `get_transfer_status`；只有 completed 可 dispute，pending 转人工；
   状态查询作为真实工具记录。
5. **reset/cancel 命令**  
   `reset/start over` 清空会话上下文；`cancel/never mind` 清 pending；
   不改变 DB 业务数据。
6. **OOS/无效对象安全**  
   越权、无效 CARD/TXN、未确认、拒绝确认均不改变资产；handoff 创建 ticket。
7. **自然语言澄清**  
   低 margin policy/费用歧义进入澄清；不直接执行工具。
8. **LLM-assisted 可选**  
   `LLMAssistant` 只做 schema 约束的槽位建议；allowlist、正则校验、fallback；
   不覆盖 route/OOS/权限/确认；无 API 时完全不调用。

### 3.2 会话与审计

- 每 case/每 UI session 独立 SQLite 文件；
- `Gate8Agent.history` 保存 route、tools、citations；
- `audit_events` 保存每次工具 args/result；
- 多轮状态：pending intent/slot/id、awaiting_confirmation、last route；
- 支持纠正、取消、reset、显式确认。

---

## 4. 统一 UI

`app_gate8.py`：

- 左侧：router 选择、模式、9 个 demo、reset；
- 中央：聊天、每轮 Top-3、accepted/score/threshold、工具时间线、KB 引用；
- 右侧：模型/adapter/checkpoint/threshold 来源、Cards/Transfers/Tickets/Disputes/Audit、
  policy 搜索；
- 默认 GTE（最佳 TSR）；可切 Qwen；
- 一次只加载一个编码器，避免 8 GB OOM；
- 9 个现场 demo：卡状态、激活确认、转账状态、取消确认、FAQ 引用、dispute、
  多轮纠正、OOS、越权。

真实截图：

- `figures/gate8_ui_card.png`
- `figures/gate8_ui_activation.png`
- `figures/gate8_ui_dispute.png`
- `figures/gate8_ui_ownership.png`
- `figures/gate8_ui_policy.png`
- `figures/gate8_ui_multiturn.png`
- `figures/gate8_ui_oos.png`

---

## 5. 新冻结评测

### 5.1 案例集

- `data/gate8/benchmark_cases.jsonl`；
- **102 例**，16 个 family；
- SHA-256 `7ca2a276...`；
- 与 Gate5 120 条 ID 不重叠；
- 冻结代码 Commit `8495244`，最终运行在该 freeze 之后；
- 标签：**new frozen simulated test**，不是严格外部盲测。

### 5.2 结果

| Router | TSR | Tool | Safe | Routing | P50 | P95 | TSR 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| GTE-Frozen-RejectOnly | **0.951** | 0.980 | 1.000 | 0.971 | 0.060 s | 0.101 s | 0.902–0.990 |
| Qwen-Frozen-RejectOnly | 0.824 | 0.824 | 1.000 | 0.843 | 0.140 s | 0.252 s | 0.755–0.892 |
| Qwen-Plain42-RejectOnly | 0.794 | 0.824 | 1.000 | 0.843 | 0.199 s | 0.374 s | 0.716–0.873 |

配对 bootstrap（同 102 case，vs GTE）：

| Router | Δ TSR | 95% CI |
|---|---:|---:|
| Qwen-Frozen | −12.75 pp | −20.59 ～ −4.90 |
| Qwen-Plain42 | −15.69 pp | −24.51 ～ −6.86 |

区间不跨越 0，说明在这套**新冻结模拟案例**上，GTE 的 TSR 优势不是单例偶然；
但由于案例由同一作者构造，仍不能外推为真实银行分布。

### 5.3 分族

GTE：除 OOS（10/12）和 permission（3/6）外，其余 14 个 family 全部 1.0。

Qwen Frozen：

- dispute_ticket 0/10；
- dispute_ineligible 0/6；
- 其余 family 1.0（OOS 10/12）。

Qwen Plain42：

- dispute_ticket 0/10、dispute_ineligible 0/6；
- multi_topic_switch 0/3（进入澄清，不是槽位串线）；
- 其余 1.0（OOS 10/12）。

### 5.4 安全

- 三个 router `safe_action_rate = 1.0`；
- 敏感操作全部满足“先 confirmed=false，再 confirmed=true”；
- 未确认/拒绝/改对象/越权/无效 ID 均无资产变更；
- 所有副作用工具有 audit；
- 没有 OOS 触发金融工具。

---

## 6. 关键负结果

1. **Qwen RejectOnly 对 dispute 首句过于保守**  
   “I do not recognise a card payment...” score 0.263 < 0.442，直接 OOS；
   但 GTE 同句 0.768 接受。结果：Qwen 的 16 个 dispute 案例全失败。
2. **policy fallback 仍可能误触发**  
   “cryptocurrency wallet” 命中 `currency`，返回不相关 KB；没有工具/DB 变更，
   但不是正确 handoff。
3. **Qwen Plain42 的 policy 主题切换进入澄清**  
   清空卡槽成功，但“transfer fee”被当成费用歧义，未直接给 KB；安全但任务未完成。
4. **GTE 对 CARD-9999 直接 OOS**  
   安全正确，但案例期望先进入 activate 再由工具层拒绝，route 不匹配。
5. **Qwen Agent TSR 低于 GTE**  
   与 Gate7 分类 F1 的轻微优势方向不同；Agent 的 route 接受边界是主因。

---

## 7. LLM 模式

- `LLMClient` 仅从环境变量读取凭据；
- `LLMAssistant` 使用 schema `{card_id, transfer_id, intent_hint}`，
  正则/allowlist 校验，失败 fallback；
- LLM 不能改变 route 接受、OOS、权限、确认或工具执行；
- 本机无 `OPENAI_*` 凭据，状态 `NOT RUN_API_BLOCKED`；
- 没有发送 API 请求，没有伪造 usage/延迟。

---

## 8. 复现与测试

```powershell
cd D:\code\intent-open-set
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m src.gate8.benchmark --router GTE-Frozen-RejectOnly
.\.venv\Scripts\python.exe -m src.gate8.benchmark --router Qwen-Frozen-RejectOnly
.\.venv\Scripts\python.exe -m src.gate8.benchmark --router Qwen-Plain42-RejectOnly
.\.venv\Scripts\python.exe -m src.gate8.combine_benchmark
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试：`logs/gate8_all_tests.log`，**78 tests OK**。

---

## 9. 限制

- 102 例是**新冻结模拟测试**，不是外部盲测；同作者构造；
- 英文为主；中文未标称完整支持；
- 只有 102 例，不能代表真实银行；
- Qwen 的 dispute 接受边界未在本版修复（避免按测试结果调阈值）；
- 旧 120 任务是开发回归，不与本结果混合；
- LLM API 未运行；
- 没有真实资金、真实客户、真实银行卡。

---

## 10. 文件

- `reports/gate8_audit.md`
- `reports/gate8_benchmark_protocol.md`
- `reports/gate8_failure_analysis.md`
- `reports/gate8_demo_script.md`
- `results/gate8_agent_results.csv`
- `results/gate8_agent_summary.csv`
- `results/gate8_agent_by_family.csv`
- `results/gate8_agent_failures.csv`
- `results/gate8_agent_paired_bootstrap.csv`
- `results/gate8_benchmark_freeze.json`
- `figures/gate8_ui_*.png`
