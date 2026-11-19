# 完整智能客服 Demo 操作卡（课堂 60–90 秒）

## 启动（API 优先）

```powershell
cd D:\code\intent-open-set
.\start_full_demo.ps1
```

看到 `PREFLIGHT_OK` 后打开 `http://127.0.0.1:8506`。  
页面默认：**DeepSeek 增强模式** + `Qwen-Plain42-RejectOnly`。

`preflight_demo.py` 必须真实加载模型并通过 CARD-1002 只读查询；
失败时不会启动，直接改用已保存截图。

## 现场 60–90 秒

### ① 卡片状态（20 秒）

输入（API 可用时完整中文）：

`我的银行卡 CARD-1002 还没收到，帮我查一下进度。`

看：

- `Reply source: DeepSeek 润色`
- 中文翻译适配可展开
- 意图 `card_arrival`；接受 `True`
- 工具 `get_card_delivery_status`
- 结果 `in_transit` / `inactive`
- 中文回复

### ② 取消转账 + 确认（40 秒）

输入：

`请取消转账 TXN-2002。`

看：

- 意图 `cancel_transfer`
- 仍需确认
- 工具返回 confirmation required
- 右侧/业务状态：`TXN-2002 = pending`（尚未取消）

输入：

`确认`

看：

- 工具 `cancel_transfer`
- `TXN-2002 = cancelled`
- 审计事件增加
- 中文回复“已确认……已取消”

### ③ OOS 备用（15 秒，可选）

输入：

`帮我订一张去巴黎的机票。`

预期：OOS / 拒绝或转人工；不执行卡片或转账资金工具。

## API 失效时

1. 页面显示 401/402/429/超时或 `NOT RUN_API_BLOCKED`；
2. 不反复重试，不展示 Key；
3. 切到 **离线模式**；
4. 中文卡片改用已验证短句：

`我的卡CARD-1002还没到`

取消流程仍用：

`请取消转账 TXN-2002。` → `确认`

5. 仍失败：用 PPT 第 11 页截图或 `figures/gate8_2_demo_release.png` 讲解。

## 重置

点击侧栏 **重置当前会话与虚构数据库**。  
只新建虚构会话 DB；CARD-1002 回到 `inactive / in_transit`，TXN-2002 回到 `pending`。

## 禁止

- 不在课堂输入、展示或朗读 API Key；
- 不把 `configs/demo_api.env` 上传 GitHub 或发聊天；
- 不现场切换多个模型；
- 不把 OOS 说成已执行资金操作；
- 不把 API 实验说成 120 条评测的提升。
