# Gate 8.1 Demo Guide（Windows 11）

## 启动

```powershell
cd D:\code\intent-open-set
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m streamlit run app_gate8.py --server.port 8505
```

打开 `http://localhost:8505`。

- 默认 GTE-Frozen-RejectOnly；
- 阈值 0.4388，来源：Gate5 GTE 权重 + Gate7 dev 重校准；
- 可切换 Qwen-Frozen / Qwen-Plain42；
- 所有业务数据虚构；无真实银行/凭据；
- LLM API 未配置：`NOT RUN_API_BLOCKED`。

## 界面

- 左：模型、mode、9 个 Demo、reset；
- 中：聊天、intent、accepted、score、threshold、Top-3、tool timeline、KB 引用；
- 右：模型/checkpoint/threshold 来源、Cards/Transfers/Tickets/Disputes/Audit；
- 切换模型时：关闭旧 Agent、清空 router cache、GC、CUDA synchronize/empty_cache；
  侧栏显示 last switch 的 allocated/reserved。

## 课堂脚本

1. **卡状态**：`I still have not received my card. The reference is CARD-1001.`  
   期望 card_arrival、delivered、get_card_delivery_status。
2. **激活确认**：`Please activate my new card CARD-1002.` + `yes`  
   期望先 confirmation_required，再 activated。
3. **转账槽位**：`My transfer has not arrived. Can you check it?` + `TXN-2001`  
   期望 pending。
4. **取消确认**：`Please cancel my transfer TXN-2002.` + `yes`  
   期望 cancelled，TXN-2001 不变。
5. **政策引用**：`What is the policy on transfer fees?`  
   期望真实 KB doc_id/citation，无金融工具。
6. **dispute**：`I do not recognise a card payment. Can you dispute it?` + `TXN-2003`  
   期望 get_transfer_status + create_dispute。
7. **多轮纠正**：`Hi, I still have not received my card.` + `Actually use CARD-1002 instead.`  
   期望 CARD-1002 in_transit。
8. **OOS**：`Can you book a flight to Paris for me?`  
   期望 handoff，不执行银行工具。
9. **越权**：`Please activate CARD-2001 now.`  
   期望 ownership_denied + handoff，CARD-2001 不变。

## 模型切换

侧栏点 Qwen，等待加载；再点 GTE。  
期望：

- 会话清空/重置，不串线；
- 旧模型释放；`results/gate8_1_switch_memory.json` 显示 release 后 0 MB；
- 不出现 CUDA OOM。

## 注意

- Gate8.1 新集 120 例比 Gate8 102 例更分散，TSR 不是同一难度；
- Qwen 对部分自然措辞会在拒识阶段拒绝；这是真实边界；
- 不在课堂现场根据测试结果调阈值；
- 不使用真实 API 时不要填 `OPENAI_*` 变量。
