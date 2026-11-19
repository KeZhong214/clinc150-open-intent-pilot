# Gate 8.1 最终报告：质量修复与独立评测

**分支**：`gate8-agent-final-fix`  
**基线**：`gate8-agent-final@b3de96e`  
**性质**：权重—阈值修复、Agent 安全加固、120 条新模拟评测、最终 UI；不新增训练、不进入 Gate 9。

---

## 1. 结论

1. GTE 已改为**权重与阈值严格配套**：Gate5 权重
   `0e920d0f...` + Gate7 dev_threshold 重新校准 tau
   **0.4388361126**，不再使用另一份 Gate7 checkpoint 的 0.4245788。
2. 原 Gate8 的 102 条结果与 freeze 完整保留；旧 102 在新 GTE 配置下回归
   TSR **0.951→0.971**（历史协议回归，不是新盲测）。
3. 新 Gate8.1 集 **120 条 / 120 templates / 18 families**，
   与 Gate5/Gate8 无完全规范化重复，最高字符相似度 0.925。
4. Agent 修复了 OOS 子串误触发、dispute fallback、LLM 槽位无证据、
   模型切换会话引用等问题。
5. 最终 v2 结果：GTE **0.633**、Qwen Frozen **0.633**、
   Qwen Plain42 **0.742**；三者安全率 **1.000**。
6. Qwen Plain42 比 GTE +10.83 pp，但 family-cluster bootstrap 95% CI
   **−0.64 ～ +23.90 pp**，不能声称显著。
7. 真实 API 未运行：**NOT RUN_API_BLOCKED**；离线模式完整可用。

---

## 2. 阶段 A：权重/阈值协议修复

### 2.1 问题

原 Gate8 GTE：

- checkpoint：`gate5_reject_only_frozen.pt` SHA `0e920d0f...`
- tau：`0.4245788127`，来自 Gate7 stability `gte_frozen/seed42`，
  但该配置的 checkpoint 是 `gate7_stability_gte_frozen_reject_seed42.pt`
  SHA `04f0005a...`
- 因此权重与阈值不是同一模型。

### 2.2 修复

- 保留 Gate5 GTE 权重；
- 在 Gate7 `dev_threshold` 上重新校准，约束 known acceptance ≥ 0.90；
- 新 tau **0.43883611261844635**；
- 校准 split SHA `a6821544...`；
- 注册表：`results/gate8_1_model_registry.json`；
- `src/gate8_1/registry.py::validate` 检查 checkpoint/adapter/threshold SHA，
  拒绝 `paired=false` 或不匹配；
- Gate8.1 路由器只能通过 registry 构建。

三模型：

| Router | checkpoint SHA | adapter | tau |
|---|---|---|---:|
| GTE | `0e920d0f...` | — | 0.4388361126 |
| Qwen Frozen | `d7512e52...` | — | 0.4424596250 |
| Qwen Plain42 | `d222f968...` | `4efaee5c...` | 0.4168315291 |

### 2.3 旧 102 回归

| 配置 | TSR |
|---|---:|
| Gate8 原 GTE（错配 tau） | 0.95098 |
| Gate8.1 GTE（配套 tau） | **0.97059** |

记录：`results/gate8_1_old102_comparison.json`。  
没有根据该回归选择模型；原 Gate8 所有结果保留。

---

## 3. 阶段 B：Agent 安全/鲁棒性修复

1. **OOS 子串误触发**
   - `POLICY_TERMS` 改为 word-boundary 正则；
   - “cryptocurrency” 不再命中 `currency`；
   - 非政策 OOS 不再走 KB，直接 handoff。
2. **dispute fallback**
   - policy fallback 排除 CARD/TRANSFER/DISPUTE intent；
   - dispute 先查交易状态，completed 才建 dispute，pending 转人工；
   - 记录状态查询工具。
3. **LLM 槽位证据**
   - LLM 只能建议 ID；
   - ID 必须出现在原始用户文本、pending 原文或最近可信用户历史中；
   - allowlist + 正则 + schema；无证据的 CARD/TXN 不加入；
   - 模型不能改变 route/OOS/权限/确认。
4. **多轮**
   - reset/cancel 命令；
   - 主题切换清空 pending；
   - 确认阶段改对象要重新确认；
   - 敏感工具仍由工具层强制确认/所有权/幂等/审计。
5. **空输入/非法对象/重复请求**
   - 空输入安全回复；
   - 非法 CARD/TXN handoff；
   - 重复取消返回 already-cancelled，不重复执行。

LLM API 无凭据，所有真实 API 调用保持 `NOT RUN_API_BLOCKED`；只用 Mock 做安全单测。

---

## 4. 阶段 C：新独立评测集

### 4.1 新集

| 项 | 值 |
|---|---|
| 文件 | `data/gate8_1/benchmark_cases.jsonl` |
| 数量 | **120** |
| template_id | **120** |
| family | **18** |
| v2 SHA | `2eb718b454f4b42c317a4727f8dfc9a91f3641c1383b7cafbc68104791d2ceb2` |
| v1 SHA | `d2e934b1c4b03d7214903d824181186c6415d0504b981ece203f8c0e310d34c9` |

审计：

- 与 Gate5 + Gate8 完全规范化重复：**0**；
- 新集内部完全重复：**0**；
- 最高字符相似度：0.925（仅诊断）；
- 120 个不同模板；
- 覆盖卡、转账、取消、dispute、FAQ、OOS、权限、多轮、空输入、重复取消。

### 4.2 v1 与 v2

- **v1**：严格首次冻结运行，cases SHA `d2e934b...`；GTE .575、
  Qwen Frozen .575、Qwen Plain42 .692。
- v1 后发现 FAQ 的 `expected_terms` 误用了问题第一个词（如 “could”），
  导致回答正确仍失败；只修正期望词为 “policy”，**没有改 Agent/模型/阈值/
  案例文本**。
- **v2**：修复期望词后的回归，cases SHA `2eb718b...`；
  GTE .633、Qwen Frozen .633、Qwen Plain42 .742。
- v1 结果完整保留；v2 不是第二次独立盲测，报告两者。

### 4.3 v2 结果

| Router | TSR | Tool | Safe | Routing | P50 | P95 | cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| GTE-Frozen-RejectOnly | 0.633 | 0.725 | 1.000 | 0.667 | 0.074 s | 0.151 s | 0.492–0.754 |
| Qwen-Frozen-RejectOnly | 0.633 | 0.725 | 1.000 | 0.675 | 0.162 s | 0.372 s | 0.495–0.768 |
| Qwen-Plain42-RejectOnly | **0.742** | **0.817** | 1.000 | **0.775** | 0.218 s | 0.563 s | 0.651–0.835 |

Paired family-cluster bootstrap vs GTE：

- Qwen Frozen：0.00 pp，CI −15.08 ～ +14.43；
- Qwen Plain42：+10.83 pp，CI **−0.64 ～ +23.90**（不显著）。

---

## 5. 阶段 D：Streamlit Demo

`app_gate8.py` 继续使用，升级：

- 三模型：GTE、Qwen Frozen、Qwen Plain42；
- 每轮意图、score、accepted、threshold、Top-3；
- checkpoint/adapter/threshold 来源；
- 工具时间线、KB 引用、SQLite 业务状态、audit；
- 9 个 Demo；
- 模型切换前关闭旧 sessions、清空 router cache、GC、**synchronize + empty_cache**；
  显示实际 allocated/reserved；
- 默认离线；LLM-assisted 无凭据 `NOT RUN_API_BLOCKED`；
- 默认 GTE；可切换。

实际切换压力测试（`results/gate8_1_switch_memory.json`）：

| 模型 | build allocated | release |
|---|---:|---:|
| GTE | 294.1 MB | 0.0 MB |
| Qwen Frozen | 1136.4 MB | 0.0 MB |
| Qwen Plain42 | 1145.1 MB | 0.0 MB |
| GTE 再加载 | 294.1 MB | 0.0 MB |

不是只调用 `empty_cache()`；实际连续加载/释放通过。

截图：

- `figures/gate8_1_ui_overview.png`
- `figures/gate8_1_ui_card.png`
- `figures/gate8_1_ui_activation.png`
- `figures/gate8_1_ui_transfer.png`
- `figures/gate8_1_ui_cancel.png`
- `figures/gate8_1_ui_policy.png`
- `figures/gate8_1_ui_dispute.png`
- `figures/gate8_1_ui_multiturn.png`
- `figures/gate8_1_ui_oos.png`
- `figures/gate8_1_ui_ownership.png`

完整画廊：[`gate8_1_ui_gallery.md`](gate8_1_ui_gallery.md)

---

## 6. 安全与失败分析

- 三模型 `safe_action_rate=1.0`；
- OOS 10/10 handoff，没有金融工具；
- 未确认、拒绝确认、改对象、越权、无效 ID 均无资产变化；
- 主要失败是 route 边界：GTE/Qwen RejectOnly 拒绝自然措辞；
- Qwen 的 dispute 首句/自然措辞仍较弱；
- 没有因测试结果改 Agent 或阈值。

详见 `reports/gate8_1_failure_analysis.md`。

---

## 6.5 自动化测试

- **86 tests OK**：`logs/gate8_1_all_tests.log`
- 覆盖：注册表配对、OOS word-boundary、LLM 伪造 ID 拒绝、120 case 质量、
  旧 Gate8 保留、模型切换代码、原有 Gate0–8 全部回归。

## 7. 限制

- 120 条是同作者构造的新冻结模拟测试，**不是外部盲测**；
- 英文为主；
- 120 模板并非 120 个独立业务场景；统计使用 family-cluster bootstrap；
- 新集比 Gate8 102 更分散，TSR 不能与 Gate8 直接比较；
- v2 是测试期望修复后的回归，不是第二次独立盲测；
- Qwen Plain42 的优势不显著；
- 旧 120 只作历史回归；
- LLM API 未运行；
- 无真实资金/客户。

---

## 8. 文件

- `reports/gate8_1_protocol_audit.md`
- `reports/gate8_1_benchmark_protocol.md`
- `reports/gate8_1_failure_analysis.md`
- `reports/gate8_1_demo_guide.md`
- `results/gate8_1_model_registry.json`
- `results/gate8_1_gte_recalibrated_threshold.json`
- `results/gate8_1_benchmark_freeze_v1.json`
- `results/gate8_1_benchmark_freeze_v2.json`
- `results/gate8_1_agent_results.csv`
- `results/gate8_1_agent_summary.csv`
- `results/gate8_1_switch_memory.json`
- `figures/gate8_1_benchmark.png`
- `figures/gate8_1_failure_heatmap.png`
