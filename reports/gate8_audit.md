# Gate 8 A｜现有 Agent 代码审查与最小变更方案

**基线**：`gate7-final-fix@65d472c`  
**新分支**：`gate8-agent-final`

## 1. 现有可复用组件

| 组件 | 现状 | Gate8 处理 |
|---|---|---|
| `src/gate5/agent_v2.py` | 规则/状态机编排，SQLite 工具，KB，RAG，确认，会话 DB | 复用；修 policy fallback 与 dispute 分支 |
| `src/gate4/db.py` | 虚构 customers/cards/transfers/disputes/tickets/audit；所有权、确认、幂等 | 复用，不改 seed |
| `src/gate4/router.py` | GTE B3-H/B4 路由器 | 复用 checkpoint |
| `src/gate6/router_qwen.py` | Qwen frozen/plain RejectOnly | 复用 checkpoint/adapter |
| `src/gate5/kb.py` | 30 条虚构政策，TF-IDF，真实 doc_id/citation | 复用 |
| `app_gate6.py` | Streamlit Qwen UI，聊天、工具、DB、RAG | 重写为 `app_gate8.py`，增加三模型、决策/failure 展示、9 demos |
| Gate7 结果 | 3-seed、B4 稳定性、阈值协议修复 | 作为 Gate8 阈值来源；不覆盖 |

## 2. 审查发现的实际缺陷

1. **dispute 被 policy fallback 截获**  
   `if not card and not txn and any(POLICY_TERMS)` 在 `DISPUTE_INTENTS` 之前，导致  
   “I do not recognise a card payment. Can you dispute it?” 只回 KB，不收集 TXN、不建 dispute。  
   Gate8 修复：排除 `CARD_INTENTS/TRANSFER_INTENTS/DISPUTE_INTENTS`，收紧 `POLICY_TERMS`，
   避免 OOS 问题被误当政策问答。

2. **pending slot 吞掉新话题**  
   `AgentV2.handle` 在 pending_intent/pending_slot 时先走 `_slot`，用户改问政策会被当作卡号/交易号。  
   Gate8 修复：新增 `_topic_switch`、`reset/start over`、`cancel/never mind` 命令，
   显式清空 pending 状态再路由。

3. **确认阶段无法改对象**  
   原实现只接受 yes/no；用户说 “No, use TXN-2002” 会被当作拒绝并清空，不能纠正。  
   Gate8 修复：在确认阶段识别新 ID，更新 pending_id 并要求重新确认，不执行工具。

4. **dispute 没有检查交易状态**  
   原 `create_dispute` 允许 pending/completed 任意交易。  
   Gate8 修复：先查 `get_transfer_status`；只有 completed 可 dispute，pending 转人工；
   状态查询作为真实工具记录保留。

5. **Agent 阈值协议与 Gate7.1 分类不一致**  
   Gate5/6 Agent 使用旧阈值；分类 C 使用 Gate7 dev_threshold。  
   Gate8 明确使用 **Gate7 dev_threshold** 作为正式 Agent 协议，版本固定：
   - GTE Frozen RejectOnly 0.4245788127（Gate7 stability seed42）；
   - Qwen Frozen RejectOnly 0.4424596250（gate7_dev_frozen.csv）；
   - Qwen Plain42 RejectOnly 0.4168315291（gate7_dev_plain42.csv）；
   - checkpoint/adapter SHA 写入 `results/gate8_benchmark_freeze.json`。
   旧 Agent 结果不再与本轮混合。

6. **“Qwen-Opt”命名不准确**  
   Gate7.1 已改为 `Qwen-Plain42-RejectOnly`；Gate8 沿用准确名称和 legacy 映射。

## 3. 最小变更方案

- 不重写数据库、KB、工具；
- `Gate8Agent` 继承 `BankAgentV2`，只覆盖上下文/命令/dispute 状态检查；
- 修复 `AgentV2` 的两个通用分支 bug，保持 Gate5–7 测试兼容；
- 新增 `src/gate8/routers.py` 统一三个路由器与 Gate7 dev_threshold；
- 新增 `app_gate8.py`，复用现有 SQLite/KB；
- 新增 102 条新冻结模拟案例，不复用 Gate5 的 120 条；
- 不新增训练，不覆盖 Gate0–7.1 权重。

## 4. 运行模式

- 主模式：**Offline Router-driven Agent**；规则/状态机编排，真实模型路由、真实工具、真实 DB；
- 可选：**constrained LLM JSON assist**（仅槽位/措辞建议，schema + allowlist + fallback）；
- 本机无 API 凭据：`NOT RUN_API_BLOCKED`，不生成伪造 API 成功记录；
- 初版主要支持英文 BANKING77 风格请求；中文未标称完整支持。
