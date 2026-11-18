# Gate 8 课堂 Demo 脚本

## 0. 启动

```powershell
cd D:\code\intent-open-set
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m streamlit run app_gate8.py --server.port 8504
```

打开 `http://localhost:8504`。

- 默认 **GTE-Frozen-RejectOnly**（最佳 TSR，P50 约 0.064 s）；
- 可切换 Qwen frozen / Qwen Plain42；
- 同一时间只加载一个编码器，避免 8 GB 显存 OOM；
- 所有账号、卡、转账、工单均为虚构；没有真实银行数据。

## 1. 卡状态

侧栏 `1. Card delivery`。真实请求：

> I still have not received my card. The reference is CARD-1001.

应看到：

- intent `card_arrival`，accepted True，score ≈ 0.60，threshold 0.4246；
- Top-3 含 `card_arrival` / `card_not_working` / `card_linking`；
- 工具 `get_card_delivery_status`，结果 `delivered` / `active`；
- 右侧 Cards 表中 CARD-1001 状态。

## 2. 激活确认（敏感操作）

侧栏 `2. Activate card (confirm)`。两轮：

> Please activate my new card CARD-1002.  
> yes

应看到：

- 第一轮只调用 `activate_card(confirmed=False)`，返回 `confirmation_required`；
- 第二轮 `activate_card(confirmed=True)`，CARD-1002 变为 active；
- Tool timeline 两次调用，Audit 有记录；
- 没有确认前不会改状态。

## 3. 转账状态（槽位补全）

侧栏 `3. Transfer status (slot)`：

> My transfer has not arrived. Can you check it?  
> TXN-2001

应看到：

- 第一轮 route `transfer_not_received_by_recipient`，询问 TXN；
- 第二轮 `get_transfer_status`，返回 pending / 250 USD / ACME Supplies。

## 4. 取消确认

侧栏 `4. Cancel transfer (confirm)`：

> Please cancel my transfer TXN-2002.  
> yes

应看到：

- 第一轮 `cancel_transfer(confirmed=False)`，确认提示；
- 第二轮 `cancel_transfer(confirmed=True)`，TXN-2002 变 cancelled；
- TXN-2001 仍 pending（不会误改）。

## 5. 政策引用

侧栏 `5. Policy citation`：

> What is the policy on transfer fees?

应看到：

- 无金融工具；
- KB 引用 `KB-013` 等真实 doc_id + citation；
- 明确 “fictional demo policy”，不编造余额/费用。

## 6. 申诉/争议

侧栏 `6. Dispute completed txn`：

> I do not recognise a card payment. Can you dispute it?  
> TXN-2003

应看到：

- 第一轮 `card_payment_not_recognised`，询问 TXN；
- 第二轮 `get_transfer_status`（completed）+ `create_dispute`；
- Disputes 表新增 DSP-3001；无真实资金动作。

## 7. 多轮补充与纠正

侧栏 `7. Multi-turn correction`：

> Hi, I still have not received my card.  
> Actually use CARD-1002 instead.

应看到：

- 第一轮询问 CARD；
- 第二轮使用 CARD-1002，返回 `in_transit`；
- 不会把 “Actually use...” 当成新话题或错误卡号。

## 8. OOS 转人工

侧栏 `8. OOS handoff`：

> Can you book a flight to Paris for me?

应看到：

- route OOS，accepted False；
- `handoff_to_human`，创建 ticket；
- 不执行任何银行工具。

## 9. 越权 / 无效对象

侧栏 `9. Ownership / overreach`：

> Please activate CARD-2001 now.

应看到：

- route `activate_my_card`（accepted）；
- 工具层返回 ownership_denied；
- 只 handoff，不激活 CARD-2001；
- 右侧 Cards 中 CARD-2001 仍属于 CUST-1002、状态 active。

## 10. 切换模型

侧栏选择 `Qwen-Frozen-RejectOnly` 或 `Qwen-Plain42-RejectOnly`，再重复 demo。
说明：

- 模型/adapter、threshold 来源、checkpoint SHA 在 `Model decision provenance`；
- Qwen 的 dispute 首句可能被 reject head 拒绝，这是 Gate8 发现的真实边界；
- 不切换测试集、不修改阈值来“让 demo 成功”。

## 11. 截图

- `figures/gate8_ui_card.png`
- `figures/gate8_ui_activation.png`
- `figures/gate8_ui_dispute.png`
- `figures/gate8_ui_ownership.png`
- `figures/gate8_ui_policy.png`
- `figures/gate8_ui_multiturn.png`
- `figures/gate8_ui_oos.png`

## 12. 最低可用路径

- 无 GPU：GTE 仍可在 CPU 运行，但 Qwen 会更慢；
- 无 LLM API：离线模式完整可用；UI 显示 `NOT RUN_API_BLOCKED`；
- 不要同时加载 GTE 和 Qwen；切换时等待重新加载。
