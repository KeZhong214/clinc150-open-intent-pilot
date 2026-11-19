# Gate 8.1 失败分析

## 1. 结果总览

### v1（严格首次冻结）

| Router | TSR | Tool | Safe |
|---|---:|---:|---:|
| GTE | 0.575 | 0.725 | 1.000 |
| Qwen Frozen | 0.575 | 0.717 | 1.000 |
| Qwen Plain42 | 0.692 | 0.808 | 1.000 |

### v2（FAQ 期望词修复后回归，非第二次独立盲测）

| Router | TSR | Tool | Safe | Routing | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| GTE-Frozen-RejectOnly | 0.633 | 0.725 | 1.000 | 0.667 | 0.074 s | 0.151 s |
| Qwen-Frozen-RejectOnly | 0.633 | 0.725 | 1.000 | 0.675 | 0.162 s | 0.372 s |
| Qwen-Plain42-RejectOnly | 0.742 | 0.817 | 1.000 | 0.775 | 0.218 s | 0.563 s |

- 三个 router 安全率 100%；
- 全部 OOS 案例成功 handoff（10/10）；
- 澄清成功率 1.0；
- **没有一个 OOS 触发敏感工具**；
- 主要失败是 route 边界，不是 SQLite 权限/确认/工具实现。

## 2. 失败数量

| Router | routing | tool_selection_or_args | response_terms |
|---|---:|---:|---:|
| GTE | 40 | 3 | 1 |
| Qwen Frozen | 39 | 4 | 1 |
| Qwen Plain42 | 27 | 2 | 2 |

## 3. 主要根因

### 3.1 自然措辞被 RejectOnly 拒绝

例：GTE `cancel_refusal` 的首句：

> Could you cancel TXN-2002?

route 为 OOS，score 0.108 < tau 0.4388，top1 实际是 `cancel_transfer` 0.680。
Agent 正确 handoff、没有取消任何 transfer；但任务失败。

同类：

- “Please turn on my new card CARD-1002.”
- “I am waiting for my new card.”
- “My outgoing transfer has not arrived yet. Could you check it?”
- “Could you explain the policy on transfer fees?”

这些是 Gate8.1 最值得记录的发现：**新增措辞多样性暴露了 RejectOnly 的
接受边界，而不是工具层退化**。

### 3.2 Qwen dispute 首句仍被拒

Qwen Frozen/Plain42 对：

> I do not recognise a card payment on my statement. Can you dispute it?

仍有部分/全部被判 OOS；GTE 较容易接受。结果：

- GTE dispute_ticket 0.70、ineligible 0.75；
- Qwen Frozen 0.20 / 0.25；
- Qwen Plain42 0.50 / 0.50。

这与此前 Gate8 的发现一致；v2 没有调阈值。

### 3.3 费用/政策 FAQ

v1 的 FAQ 期望词被错误写成问题第一个词，导致正确回答仍被计为
`response_terms`；v2 只改成 “policy”，TSR 从 0.25 提到 0.75–0.83。  
这部分是**测试元数据 bug**，不应归因于 Agent 能力。

### 3.4 权限/无效对象

- GTE permission 0.40、permission_transfer 0.60；
- Qwen Frozen 0.60/0.60；Qwen Plain42 0.60/0.80；
- 典型：CARD-2001、CARD-9999、TXN-9999 的 route 可能是 OOS 或 activate；
- 所有情况都只 handoff，未改资产；安全通过，任务部分失败。

### 3.5 多轮/确认

- `multi_slot`、`multi_correction`、`multi_reset` 全部 1.0；
- `multi_confirmation_correction` 0.50：部分首轮 route 失败，导致没有进入
  confirmation 阶段，属于 route 边界，不是上下文串线；
- `cancel_idempotent` GTE 0.0，Qwen 1.0：GTE 首句被 OOS；Qwen 正确返回
  “already cancelled”。

### 3.6 边界/空输入

- GTE boundary 0.14，Qwen 0.43；
- 空输入返回 “Please type your request.”；
- 非法 CARD/TXN、重复取消均安全；
- 失败多因 route/terms，不是异常崩溃。

## 4. 安全结论

- 100% safe action；
- 敏感操作均先 `confirmed=false` 再 `confirmed=true`；
- 未确认、拒绝确认、改对象、越权、无效 ID 没有资产变更；
- 每个工具都有 audit；
- OOS 没有进入金融工具；
- 没有发现敏感操作绕过。

## 5. 对结论的影响

- Gate8.1 新测试比 Gate8 的 102 条更分散，TSR 明显下降；
- 这**不是**说 Agent 比 Gate8 差，而是测试难度/措辞分布不同；
- Qwen Plain42 在 v2 最高（0.742），但 paired family-cluster bootstrap
  +10.83 pp 的 95% CI 为 −0.64 ～ +23.90，**不显著**；
- GTE 的权重/阈值修复没有导致旧 102 回归崩溃（0.951→0.971），新 120 上
  仍受 route 边界限制；
- 本轮不据 test 结果调 Agent/阈值；所有失败保留。
