# Gate 8.1 独立评测协议与案例审计

## 1. 数据定位

- 新测试集：**Gate 8.1 new frozen simulated test**；
- 不是外部盲测，不是新盲测；
- 由同一项目执行者编写；
- 旧 Gate5 120 条、Gate8 102 条只作为历史回归，不进入 Gate8.1 主表；
- 原 Gate8 的 102 条结果与 freeze 完整保留。

## 2. 案例集

| 项 | 值 |
|---|---|
| 文件 | `data/gate8_1/benchmark_cases.jsonl` |
| 最终 v2 数量 | **120** |
| 不同 template_id | **120** |
| family | **18** |
| SHA-256（v2） | `2eb718b454f4b42c317a4727f8dfc9a91f3641c1383b7cafbc68104791d2ceb2` |
| v1 SHA | `d2e934b1c4b03d7214903d824181186c6415d0504b981ece203f8c0e310d34c9` |
| 原 Gate8 SHA | `7ca2a276...` |

### 去重和相似性

`results/gate8_1_case_audit.json`：

- 与 Gate5 + Gate8 的**完全规范化重复：0**；
- 新集内部完全重复：0；
- 最高字符相似度：0.925（G81-F008 vs B078），仅作诊断；
- 120 个 template_id 全部不同；
- 同一业务类型使用了不同措辞、实体和多轮组织方式；
- **不宣称外部独立**。

### 覆盖

- 卡片状态、激活确认；
- 转账状态、取消、拒绝确认；
- completed/pending dispute；
- 费用/政策/FAQ；
- OOS；
- 权限/无效对象；
- 多轮 slot、纠正、主题切换、reset、确认阶段改对象；
- 空输入、重复取消、非法 ID。

## 3. 冻结与版本

### v1（严格首次冻结运行）

- 冻结时间：最终运行前；
- cases SHA `d2e934b...`；
- code commit `7824bb7`；
- 结果：GTE 0.575、Qwen Frozen 0.575、Qwen Plain42 0.692；
- 结果文件 `*_v1_results.csv`，summary `gate8_1_agent_summary_v1.csv`。

### v2（评测元数据修复后的回归）

v1 运行后发现 FAQ 的 `expected_terms` 被错误地设成问题第一个词（例如
“could”），导致 8–9 个 FAQ 在回答正确时仍被记为 response_terms 失败。
只修改 FAQ 的期望词为 “policy”，**没有修改 Agent、模型、阈值、路由或案例
文本**。

- v2 cases SHA `2eb718b...`；
- v2 freeze：`gate8_1_benchmark_freeze_v2.json`；
- 结果：GTE 0.633、Qwen Frozen 0.633、Qwen Plain42 0.742；
- v1 结果保留，不覆盖。

v2 是**测试期望修复后的回归**，不是第二次独立盲测；报告同时给出 v1/v2。

## 4. 模型与阈值

严格使用 `gate8_1_model_registry.json`：

| Router | tau |
|---|---:|
| GTE-Frozen-RejectOnly | 0.4388361126（Gate5 权重 + Gate7 dev 重校准） |
| Qwen-Frozen-RejectOnly | 0.4424596250 |
| Qwen-Plain42-RejectOnly | 0.4168315291 |

三个 router 使用同一 120 个 case、同一 DB seed、同一评分规则；GTE 单独重新校准
不涉及 test 选择。

## 5. 评分

- TSR：final state + terms + tools/args + escalation + clarification + route + safety；
- Tool Call Accuracy：工具名和参数；
- Routing Accuracy：首轮 route 是否在允许集合；
- OOS/unsupported：handoff 或安全澄清；
- Clarification Success；
- Safe Action Rate：敏感操作必须 `confirmed=false` + `confirmed=true`；
- Citation Coverage：有真实 KB doc_id 的案例比例；
- P50/P95 latency。

## 6. 统计

- case-level bootstrap 2000 次；
- **family-cluster bootstrap**：18 个 family 作为 cluster；
- paired cluster bootstrap vs GTE；
- 不把 120 个模板当成 120 个完全独立业务场景；
- Qwen Plain42 vs GTE 的 +10.83 pp 区间为 −0.64 ～ +23.90 pp，
  **不能声称显著**。

## 7. 不可做

- 不根据 Gate8.1 test 调 Agent；
- 不用 test 选 GTE tau / model / seed；
- 不把 v2 称为新盲测；
- 不隐藏 v1 的低分或 Qwen 的 dispute 失败。
