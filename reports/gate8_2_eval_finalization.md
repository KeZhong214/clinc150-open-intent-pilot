# Gate 8.2 A：评测定稿与 FAQ 事实审计

**分支**：`gate8-2-eval-and-deepseek-demo`  
**基线**：`gate8-agent-final-fix@45c1318`  
**性质**：既有 v1 原始轨迹的离线复核；不重新推理、不训练、不覆盖旧结果。

---

## 1. v1 / v2 差异定稿

### 1.1 数据来源

按任务书，v1 案例从 **commit `7824bb7`** 读取，保存为：

`data/gate8_1/benchmark_cases_v1.jsonl`

- v1 cases SHA-256：`b653000c5c7ea4347053afc8d042c4358918a588f3cf897b13f0f75dbbf0959a`
- 原始 v1 三模型结果：
  - `results/gate8_1_agent_gte_frozen_rejectonly_v1_results.csv`
  - `results/gate8_1_agent_qwen_frozen_rejectonly_v1_results.csv`
  - `results/gate8_1_agent_qwen_plain42_rejectonly_v1_results.csv`

### 1.2 发现的 freeze 哈希不一致

`results/gate8_1_benchmark_freeze_v1.json` 记录的 cases SHA 是：

`d2e934b1c4b03d7214903d824181186c6415d0504b981ece203f8c0e310d34c9`

但从 commit `7824bb7` 实际取出的案例 SHA 是：

`b653000c5c7ea4347053afc8d042c4358918a588f3cf897b13f0f75dbbf0959a`

两者不一致。本定稿**不覆盖**任何旧 freeze，而是：

- 以任务书明确指定的 commit `7824bb7` 的案例作为 v1 输入；
- 在 `results/gate8_2_v1_source_manifest.json` 中同时记录实际 SHA、旧 freeze SHA 和差异；
- 报告明确说明该数据完整性问题。

### 1.3 真实差异

`results/gate8_2_v1_v2_diff.csv` 共 **13 行**：

| 类型 | 数量 | 内容 |
|---|---:|---|
| FAQ `expected_terms` | 12 | v1 是问题第一个词（could/how/what/...），v2 统一为 `policy` |
| 用户文本 | 1 | **G81-F009** 从 “What happens if a top-up fails?” 改为 “If a top-up does not go through, what should I do?” |

因此，旧报告“v2 只改评分字段、完全没有改案例文本”是**错误描述**。

**G81-F009 的 v1 输入必须保持 v1 原句；不得用 v2 换文后的回答补分。**

---

## 2. FAQ rubric

`configs/gate8_2_faq_rubric.json` 对 12 条 FAQ 定义了：

- 核心业务主题；
- 允许的真实 KB doc_id；
- 必须覆盖的政策事实；
- 无依据/错主题/编造时的失败规则；
- `needs_manual_review` 规则。

关键例子：

- F001/F002：KB-013，transfer fee 只是 fictional example，不计算真实费用；
- F003：KB-027，limits 不是个性化额度/不能自动提额；
- F004：30 篇 KB **没有**卡到期/续卡依据，正确行为是 handoff/不编造；
- F005：KB-019 只讲 illustrative FX rates，**没有**“支持币种列表”；
- F006：KB-009，pending transfer 不承诺到达时间；
- F007：KB-022，dispute 流程；
- F008：KB-014，top-up 仅虚构示例；
- F009：KB-016，失败 top-up 不改变 demo balance、不能声称已入账、可建 ticket；
- F010：30 篇 KB **没有** card acceptance 列表；
- F011：KB-019；
- F012：KB-028，不做真实身份核验。

`results/gate8_2_faq_grounding_audit.csv` 是 36 行（12 案例 × 3 模型）的离线审核。

### 审核结果

- F001/F008/F011：GTE 与 Qwen Frozen 的回答引用了正确 KB，只是 v1 的 `expected_terms` 写成了第一个词，属于**评分 bug**；
- Qwen Plain42 的 F001 没有正确引用，未修正；
- F003/F006：原系统没有给出政策答案；
- F004/F005/F010：引用了错误主题或没有依据；
- **F009：三模型 v1 都没有回答“失败后余额/入账/工单”事实；GTE/Qwen 只是澄清“top up failed vs reverted”，没有 KB-016 证据，不能得分**；
- 没有用 v2 换文后的回答给 F009 补分。

---

## 3. 同轨迹离线复评

`src/gate8_2/rescore_v1.py` 只读：

- v1 cases（commit 7824）；
- v1 三模型 results/traces；
- FAQ rubric。

不调用 encoder、router、DeepSeek、SQLite 工具，不生成新回答。

只允许的修正：**grounded FAQ + v1 失败原因仅为 `response_terms`**。

| 模型 | v1 原评分 | v1 同轨迹修正 | Δ | v2 换文/改规则（探索） |
|---|---:|---:|---:|---:|
| GTE-Frozen-RejectOnly | 0.5750 | **0.6000** | +2.50 pp | 0.6333 |
| Qwen-Frozen-RejectOnly | 0.5750 | **0.6000** | +2.50 pp | 0.6333 |
| Qwen-Plain42-RejectOnly | 0.6917 | **0.7083** | +1.67 pp | 0.7417 |

修正案例：

- GTE：F001、F008、F011（3 条）；
- Qwen Frozen：F001、F008、F011（3 条）；
- Qwen Plain42：F008、F011（2 条；F001 无正确引用）。

FAQ 子集：

| 模型 | 原通过 | 修正后 | grounded |
|---|---:|---:|---:|
| GTE | 3/12 | 6/12 | 6/12 |
| Qwen Frozen | 3/12 | 6/12 | 6/12 |
| Qwen Plain42 | 3/12 | 5/12 | 5/12 |

**复评没有假定提升；没有 F009 补分；没有把 v2 结果填进 v1。**

---

## 4. 定稿结论

1. **v1 首次冻结原评分**保留：0.575 / 0.575 / 0.692。
2. **v1 同轨迹修正评分**：0.600 / 0.600 / 0.708；这是 A 阶段科研定稿。
3. **v2 换文/改规则**：0.633 / 0.633 / 0.742；只能作为探索性回归，不是新盲测。
4. F009 文本变化和 12 条评分字段变化均已记录；旧报告的错误描述已修正。
5. 旧 v1/v2 文件、原始 trace、模型权重和阈值均未覆盖。

---

## 5. 输出

- `configs/gate8_2_faq_rubric.json`
- `data/gate8_1/benchmark_cases_v1.jsonl`
- `results/gate8_2_v1_rescore_per_case.csv`
- `results/gate8_2_v1_rescore_summary.csv`
- `results/gate8_2_faq_grounding_audit.csv`
- `results/gate8_2_v1_v2_diff.csv`
- `results/gate8_2_v1_source_manifest.json`
- `logs/gate8_2_tests_a.log`
