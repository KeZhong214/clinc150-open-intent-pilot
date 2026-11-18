# Gate 4 PPT 大纲（10–12 页）

**题目**：从开放意图分类到真实客服 Agent：可训练 B4-AR 与 SQLite 工具体系

## Slide 1 — 标题与一句话结论

- 在 Gate 3 的 BANKING77 GTE 10-shot 上，训练 B4-AR 门控 + 拒识头。
- 一句话：**B4-Full 在探索性 test 上 open-set F1 0.8369，略高于 B3-H 0.8306；
  B4-Fusion 反而退化，收益主要来自拒识头。**
- 视觉：`figures/gate4_b4_comparison.png`。

## Slide 2 — 起点与问题

- Gate 3：B0 0.8081、B3-H 0.8306（同 test，已看过）。
- B3-H 固定 0.5/0.5 融合，不是可学习的。
- 两个问题：
  1. 能否用 support-only episodic 训练一个按类、按输入难度变化的融合门控？
  2. 能否把分类器变成真正有状态、可审计的客服 Agent？
- 强调：test 已暴露，本轮数字是探索性对照，不是新盲测。

## Slide 3 — B4-AR 结构

- `P_c` 多原型分数、`N_c` 近邻分数、`d_c` 类内分散度。
- 共享门控：`g_c = sigmoid(MLP([P_c,N_c,P_c-N_c,d_c]))`。
- `S_c = g_c P_c + (1-g_c) N_c`。
- 拒识头：`[s1, s1-s2, entropy, d_best, |P-N|, g, s1(1-s2)]` → known/OOS。
- 视觉：结构示意图（可用 PPT 形状）。

## Slide 4 — Episodic 训练

- 只用 400 条 10-shot support。
- 每类 5 条建原型、5 条 query；两折互换。
- 每 episode 屏蔽 6/40 类，其 query 作为 pseudo-OOS。
- 24 episodes = 2 folds × 12 masks。
- 170 known + 30 pseudo-OOS / episode。
- 训练 60 + 60 epoch，12.55 s；种子 42。
- 视觉：episode 构造示意。

## Slide 5 — 验证结果

| Method | Known routing | OOS recall | Open-set F1 |
|---|---:|---:|---:|
| B0 | 0.860 | 0.403 | 0.734 |
| B3-H | 0.851 | 0.547 | 0.767 |
| B4-Fusion | 0.844 | 0.471 | 0.740 |
| **B4-Full** | **0.853** | **0.604** | **0.781** |

- B4-Full 验证 F1 +1.39 pp，OOS +5.69 pp。
- B4-Fusion 单独退化：说明门控本身不是收益来源。

## Slide 6 — 探索性 test

| Method | Known routing | OOS recall | Open-set F1 |
|---|---:|---:|---:|
| B0 | 0.869 | 0.505 | 0.808 |
| B3-H | 0.863 | 0.637 | 0.831 |
| B4-Fusion | 0.865 | 0.555 | 0.820 |
| **B4-Full** | **0.863** | **0.658** | **0.837** |

- B4-Full vs B3-H：F1 +0.62 pp，OOS +2.11 pp。
- 提升不大；test 已看过，不能宣称新盲测。

## Slide 7 — 门控真的在变化

- 按 test margin 四分位：gate Top-1 = 0.560 / 0.746 / 0.862 / 0.919。
- 越难越偏向近邻分支 N，越容易越偏向原型分支 P。
- 梯度非零：fusion 0.00179→0.00117；full 0.00140→0.10773。
- gate max delta 1.296 / 0.641。
- 视觉：`figures/gate4_b4_gate_scatter.png` 或 `gate4_b4_gate_by_difficulty.png`。

## Slide 8 — Agent 架构

- `User → Router → slot filling → SQLite tool → response → audit`。
- 表：customers / cards / transfers / disputes / tickets / audit_events。
- 工具：查卡、查转账、取消转账、激活卡、争议、工单、handoff。
- 敏感动作先 confirm；所有权校验；幂等；审计。
- 视觉：`figures/gate4_agent_card.png` 或架构图。

## Slide 9 — 真实 Demo 链

- D1 查卡：CARD-1001 → delivered。
- D2 查转账：TXN-2001 → pending（B4-Full）。
- D3 取消转账：pending → confirmation → yes → cancelled。
- D4 未知问题 → handoff。
- D5 非本人卡 → ownership denied → handoff。
- D6 模糊 → clarify → 状态。
- 视觉：`figures/gate4_agent_b4_cancel.png`（Transfers cancelled + tool timeline）。

## Slide 10 — Agent 指标与安全

- 16 条确定性工具/状态测试：completion 16/16，tool correctness 16/16；
  clarification 1/16，confirmation 4/16，escalation 2/16。
- 6 条真实路由演示：completion 6/6。
- 每个工具调用写 audit_events；敏感动作未确认不写业务状态。
- 视觉：`figures/gate4_agent_metrics.png`。

## Slide 11 — 限制

- 单 seed、单数据集、单编码器。
- test 已看过，B4 提升只能探索性解释。
- 16 条 Agent 测试使用 StubRouter，不能冒充模型泛化。
- 无真实银行系统、无 LLM、无真实用户。
- B4-Fusion 负结果：门控本身没有带来收益。

## Slide 12 — 结论

1. B4-Full 小幅但真实地改善了开放集边界；收益主要来自拒识头。
2. 门控随输入难度改变融合，但单独使用反而退化。
3. Agent 的工具、状态、确认、所有权、审计链已真实闭环。
4. 下一步：新独立 test、多 seed、拒识头校准、更多 intent。

---

## 演讲节奏

- 1–2 页：1.5 分钟（问题与边界）。
- 3–4 页：2 分钟（结构、训练）。
- 5–7 页：3 分钟（B4 结果与门控）。
- 8–10 页：3 分钟（Agent 与 Demo）。
- 11–12 页：1.5 分钟（限制、结论）。
- 总计约 11 分钟，留 3 分钟提问。
