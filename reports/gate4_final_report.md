# Gate 4 报告：可训练 B4-AR 路由器与真实客服 Agent

**分支**：`gate4-train-agent`  
**基础**：`course-expansion@1d78dd9`（Gate 0–3）  
**日期**：2026-11-18  
**主实验**：BANKING77，GTE-ModernBERT-base，10-shot  
**状态**：B4-Fusion/B4-Full 训练、验证、冻结、探索性 test、SQLite Agent、Streamlit Demo 均已完成。

> **测试边界**：BANKING77 的 1,600 known + 760 final unknown 测试在 Gate 3 已经
> 被查看。Gate 4 的 test 是在 checkpoint/阈值冻结后只运行一次，但它是
> **探索性对照，不是新的盲测**。所有 B0/B3-H 数字来自 Gate 3 冻结结果，
> 没有在 Gate 4 中重新调参。

---

## 1. 本轮目标

1. 在已缓存的 GTE 768 维 BANKING77 10-shot 向量上，训练一个真正融入
   B3-H 的新模块：自适应原型—近邻门控 + 轻量拒识头；
2. 与 B0、B3-H、B4-Fusion、B4-Full 在同一验证预算下比较，冻结后只做一次
   test；
3. 把分类器升级为有状态、可审计、真实 SQLite 工具调用的多轮客服 Agent；
4. 完成真实 Demo、截图、16 条脚本化业务测试和 6 条真实路由器演示链。

---

## 2. B4-AR 结构

### 2.1 输入

对每个类 `c`：

- `P_c(x)`：B3-P 多原型最大余弦；
- `N_c(x)`：该类 support Top-2 余弦均值；
- `d_c`：该类 support 的类内分散度；
- 门控输入：`[P_c, N_c, P_c-N_c, d_c]`。

### 2.2 门控与融合

```text
g_c(x) = sigmoid(MLP_theta([P_c, N_c, P_c - N_c, d_c]))
S_c(x) = g_c(x) P_c(x) + (1 - g_c(x)) N_c(x)
```

- MLP 为 `4 → 32 → 1`，所有类共享；
- 最后一层初始化为 0，因此初始 `g=0.5`，与 B3-H 的固定 0.5 融合对齐；
- 门控不是常量：同一模型对不同类、不同输入的 `g_c` 不同。

### 2.3 拒识头

输入为类别置换不敏感统计量：

```text
[s1, s1-s2, entropy(top3), d_best, |P_best-N_best|, g_best, s1*(1-s2)]
```

- `MLP 7 → 16 → 1`，输出 known/OOS logit；
- 类别预测仍为 `argmax S_c`；
- 拒识阈值在 validation 的 calibration known + dev unknown 上校准。

### 2.4 Episodic 训练

- 仅使用 **400 条 10-shot support**；
- 每个类分成两折：5 条建原型、5 条做 query；两折互换；
- 每个 episode 随机屏蔽 6/40 类，屏蔽类的 query 作为 pseudo-OOS；
- 屏蔽类的 prototype 不参与该 episode；
- 24 个 episodes（2 folds × 12 masks）；
- 每个 episode：170 known query + 30 pseudo-OOS；
- B4-Fusion：仅 known CE；
- B4-Full：known CE + `0.5 ×` known/OOS BCE + weight decay；
- 不使用 calibration/dev 的标签做梯度训练；它们只用于验证、阈值和诊断。

---

## 3. B4 训练与门控证据

| 项目 | 值 |
|---|---:|
| 训练样本 | 400 support |
| episodes | 24 |
| known query / pseudo-OOS per episode | 170 / 30 |
| B4-Fusion epochs | 60 |
| B4-Full epochs | 60 |
| 训练耗时 | 12.55 s |
| gate hidden | 32 |
| reject hidden | 16 |
| seed | 42 |

### 3.1 门控不是常量

- B4-Fusion gate mean 0.110，std 0.245；
- B4-Full gate mean 0.023，std 0.121；
- B4-Full **Top-1 类上的 gate 均值 0.772**，std 0.281；
- 说明全类平均被大量低分/非 Top-1 类拉低，但实际决策类的门控值有明显变化。

按 test margin 四分位（1=最难）：

| Margin quartile | n | gate mean on Top-1 | mean margin |
|---|---:|---:|---:|
| 1 (hardest) | 400 | 0.560 | 0.032 |
| 2 | 400 | 0.746 | 0.081 |
| 3 | 400 | 0.862 | 0.127 |
| 4 (easiest) | 400 | 0.919 | 0.193 |

越难的样本门控越偏向近邻分支 `N`，越容易的样本越偏向原型分支 `P`。
这是 B4 确实在改变融合行为的直接证据。

### 3.2 梯度与参数更新

| 指标 | B4-Fusion | B4-Full |
|---|---:|---:|
| 首 epoch gate grad norm | 0.00179 | 0.00140 |
| 末 epoch gate grad norm | 0.00117 | 0.10773 |
| gate max abs delta | 1.296 | 0.641 |

梯度非零、参数确实更新；checkpoint 保存后在 `B4Router` 和 Streamlit 中真实加载
推理（不是只写报告）。

---

## 4. B4 验证与 test

### 4.1 同一验证口径

验证 = 800 calibration known + 720 dev unknown；阈值规则与 Gate 3 一致：
known acceptance ≥ 0.90，最大化 open-set macro-F1；B4-Fusion 用 fused S1，
B4-Full 用 reject-head known score。

| Method | Known accept | Known routing | OOS recall | AUROC | Open-set F1 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.920 | 0.860 | 0.403 | 0.843 | 0.734 |
| B3-H | 0.900 | 0.851 | 0.547 | 0.863 | 0.767 |
| B4-Fusion | 0.900 | 0.844 | 0.471 | 0.850 | 0.740 |
| **B4-Full** | **0.900** | **0.853** | **0.604** | **0.872** | **0.781** |

### 4.2 冻结后的一次 test（探索性）

| Method | Known accept | Known routing | OOS recall | AUROC | Open-set F1 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.934 | 0.869 | 0.505 | 0.892 | 0.808 |
| B3-H | 0.918 | 0.863 | 0.637 | 0.911 | 0.831 |
| B4-Fusion | 0.925 | 0.865 | 0.555 | 0.894 | 0.820 |
| **B4-Full** | **0.915** | **0.863** | **0.658** | **0.913** | **0.837** |

- B4-Full vs B3-H：open-set F1 **+0.62 pp**，OOS recall **+2.11 pp**，
  known routing **0.00 pp**，known acceptance **−0.31 pp**；
- B4-Full vs B4-Fusion：F1 **+1.67 pp**，OOS **+10.26 pp**；
- B4-Fusion 单独使用比 B3-H 差，说明**可学习门控本身没有带来提升**；
  改善主要来自拒识头，而不是把 0.5 融合“训练化”；
- 该提升幅度不大，不能包装成显著突破；test 已看过，不能作为新盲测。

### 4.3 可追踪案例

| Category | True | Pred | Reject score | Top-3 |
|---|---|---|---:|---|
| known correct | card_arrival | card_arrival | 0.930 | card_arrival / card_about_to_expire / card_not_working |
| known rejected | card_arrival | OOS | 0.179 | card_arrival 0.724 / verify_source_of_funds 0.621 |
| known misclassified | card_arrival | card_linking | 0.889 | card_linking 0.858 / card_arrival 0.834 |
| OOS correct | OOS | OOS | 0.203 | card_payment_fee_charged 0.753 / ... |
| OOS false accept | OOS | card_payment_not_recognised | 0.538 | card_payment_not_recognised 0.788 / ... |

完整字段见 `results/gate4_b4_test_cases.csv`。这些案例说明 B4-Full 的拒识
分数确实在改变边界，而不是只改类分数。

---

## 5. 客服 Agent

### 5.1 SQLite 业务库

虚构数据库 `data/gate4/agent.db`（运行时生成，不提交真实用户数据）：

- `customers`：CUST-1001/1002/1003；
- `cards`：CARD-1001/1002/2001，含状态、配送状态、last4；
- `transfers`：TXN-2001/2002/2003/2004，含金额、状态、收款方；
- `disputes`、`tickets`、`audit_events`。

工具：

- `get_card_delivery_status(card_id)`；
- `get_transfer_status(transfer_id)`；
- `cancel_transfer(transfer_id, confirmed)`；
- `activate_card(card_id, confirmed)`；
- `create_dispute(transaction_id, reason)`；
- `create_ticket(intent, issue, priority)`；
- `handoff_to_human(intent, reason)`。

每次工具调用写入 `audit_events`；敏感动作 `cancel_transfer` / `activate_card`
在 `confirmed=False` 时只返回 `confirmation_required`，不修改数据库；用户明确
`yes` 后才执行。所有权校验拒绝 CARD-2001 等他人虚构资产。

### 5.2 多轮状态机

`User → Router → intent → slot collection → tool → response → persist`：

- 卡/转账 ID 缺失时追问；
- `cancel_transfer` / `activate_card` 先确认，再执行；
- 模糊意图先澄清；
- OOS / 非本人 / 越权 → 创建高优先级工单并 handoff；
- 其他 known intent → 创建工单，不执行金融动作；
- 无 LLM，回复模板只引用工具真实结果。

### 5.3 固定演示链

| ID | Router | 对话 | 结果 |
|---|---|---|---|
| D1 | B3-H | “Where is my card? ... CARD-1001” | 查到 delivered |
| D2 | B4-Full | “My transfer TXN-2001 ...” | 查到 pending |
| D3 | B3-H | “cancel TXN-2002” → “yes” | DB 状态变为 cancelled |
| D4 | B3-H | “How do I buy Bitcoin?” | OOS → handoff |
| D5 | B3-H | “...CARD-2001” | ownership denied → handoff |
| D6 | B3-H | 转账缺失 → 澄清 → TXN-2001 | 查到 pending |

真实路由器（GTE、B3-H、B4-Full checkpoint）运行；结果在
`results/gate4_agent_demos.json`，含每轮 route、工具调用与最终 DB snapshot。

### 5.4 Agent 指标

**16 条确定性状态/工具测试**（`StubRouter`，不冒充模型效果）：

- Task Completion Rate：**16/16 = 1.00**
- Tool-call Correctness：**16/16 = 1.00**
- Clarification rate：1/16 = 0.0625
- Confirmation rate：4/16 = 0.25
- Escalation rate：2/16 = 0.125

**6 条真实路由器演示**：

- Task Completion Rate：**6/6 = 1.00**
- Tool-call Correctness：6/6
- Clarification：1/6（D6）
- Confirmation：1/6（D3）
- Escalation：2/6（D4/D5）

这两组数字含义不同：16 条是确定性工具/状态回归，6 条是真实模型路由的业务
链路；报告中分开写，不用 16/16 冒充模型泛化。

---

## 6. Demo

`app_gate4.py`：

- 选择 B3-H / B4-Full；
- 有状态聊天，多轮上下文；
- 显示 Router intent、score、threshold、Top-3、gate；
- 显示 Tool timeline；
- 右侧显示 Cards / Transfers / Tickets / Disputes / Audit；
- 可切换比较 B3-H 与 B4-Full 对同一句的决策；
- 6 个回放按钮；
- 全部本地、离线、虚构数据。

真实截图：

- `figures/gate4_agent_card.png`（B3-H 查卡）；
- `figures/gate4_agent_b4_cancel.png`（B4-Full 取消转账，Transfers 显示 cancelled，
  Tool timeline 展开 confirmation）。

---

## 7. 结论

1. **B4-Full 在 validation 和探索性 test 上都略优于 B3-H**：
   validation F1 +1.39 pp、OOS +5.69 pp；test F1 +0.62 pp、OOS +2.11 pp。
2. **B4-Fusion 单独并不优于 B3-H**：说明“可训练门控”本身不是收益来源；
   收益主要来自拒识头。
3. 门控确实随输入难度变化：难样本更偏向近邻分支，Top-1 gate 0.56→0.92。
4. Agent 的工具、状态、确认、所有权、审计链真实可运行；16 条确定性测试和
   6 条真实路由演示均完成。
5. 测试已在 Gate 3 看过，不能声称新盲测；B4 的提升是课程项目的探索性证据。

### 限制

- 单 seed、单数据集（BANKING77）、单编码器（GTE）；
- test 已暴露，B4 数字只能作为历史/探索性对照；
- B4-Full 提升幅度小，未做显著性检验；
- 16 条 Agent 测试使用 StubRouter，不是模型泛化测试；
- 未接真实银行系统、未接 LLM 生成、未做真实用户研究；
- 没有视频，只提供真实截图和可回放脚本。

### 下一步

- 在新的独立 unknown-class 协议上复现 B4-Full，避免 test 暴露；
- 多 seed 训练，报告均值/方差；
- 对拒识头做校准曲线和业务代价分析；
- 把 Agent 的 slot/confirmation 策略扩展到更多 BANKING77 intent；
- 在不改变 test 的前提下，对 B4 gate 的初始化/正则做一次系统消融。

---

## 8. 复现

```powershell
cd D:\code\intent-open-set
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# B4 训练 + validation
.\.venv\Scripts\python.exe -m src.gate4.train_b4

# 冻结
.\.venv\Scripts\python.exe -m src.gate4.freeze_b4

# 一次 test（探索性）
.\.venv\Scripts\python.exe -m src.gate4.run_b4_test

# Agent 16 条测试 + 6 条真实路由演示
.\.venv\Scripts\python.exe -m src.gate4.run_agent_cases

# 图表
.\.venv\Scripts\python.exe -m src.gate4.make_figures

# Demo
.\.venv\Scripts\python.exe -m streamlit run app_gate4.py
```

关键文件：

- `results/gate4_b4_validation_metrics.csv`
- `results/gate4_b4_test_metrics.csv`
- `results/gate4_b4_training_history.csv`
- `results/gate4_b4_gate_stats.json`
- `results/gate4_b4_freeze.json`
- `results/gate4_b4_fusion.pt` / `gate4_b4_full.pt`（小 checkpoint）
- `results/gate4_agent_cases.csv`
- `results/gate4_agent_metrics.json`
- `results/gate4_agent_demos.json`
- `figures/gate4_*.png`
