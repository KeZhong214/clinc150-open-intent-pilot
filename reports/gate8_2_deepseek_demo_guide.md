# Gate 8.2 C：Windows 11 / PowerShell DeepSeek Key 配置与 Demo 指南

> 这一阶段由你本机操作。不要把 Key 发到 DeepSeek 聊天、GitHub、日志或截图。
> 我只提供步骤和代码；真实 Key 只存在你的机器环境变量里。

## 1. 创建 Key

1. 打开 https://platform.deepseek.com/ 并登录；
2. 查看账户余额/计费，确认可承受少量测试费用；
3. 进入 API Keys / 创建 API Key；
4. 复制一次 Key（`sk-...`），不要粘贴到本项目文件；
5. 如果页面显示当前可用模型与示例，优先核对模型名。本项目按任务书默认：
   - `base_url = https://api.deepseek.com`
   - `model = deepseek-flash`
   - 如 `deepseek-flash` 在你的账户不可用，再按官方文档换成当时可用的模型，并只改本机 `DEEPSEEK_MODEL`。

官方文档：
- https://api-docs.deepseek.com/zh-cn/
- https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/
- https://api-docs.deepseek.com/zh-cn/guides/build_a_chatbot/

## 2. 只在当前 PowerShell 会话设置

```powershell
cd D:\code\intent-open-set

$env:DEEPSEEK_API_KEY  = "<仅在本机输入真实 Key>"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
$env:DEEPSEEK_MODEL    = "deepseek-flash"
$env:DEEPSEEK_TIMEOUT  = "12"
```

注意：

- 尖括号只是占位，不要原样执行；
- 这些变量只影响当前 PowerShell 窗口；
- 不要 `echo $env:DEEPSEEK_API_KEY`；
- 不要写进 `.env` 后提交；`.env` 已在 `.gitignore`；
- 本项目**不会自动读取** `.env`，请使用上面的当前会话变量。

## 3. 不泄露 Key 的检查

```powershell
if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
  "DEEPSEEK_API_KEY: NOT SET"
} else {
  "DEEPSEEK_API_KEY: SET (value hidden)"
}
"DEEPSEEK_BASE_URL = $env:DEEPSEEK_BASE_URL"
"DEEPSEEK_MODEL    = $env:DEEPSEEK_MODEL"
```

只应看到 `SET (value hidden)`，不能看到 `sk-...`。

## 4. 先做一次极短真实 smoke test

这一步会产生一次很小的 API 调用。`src/gate8_2/deepseek_smoke.py` 只发一句短请求，
`max_tokens=24`，不打印 Key，也不打印 Authorization。

```powershell
.\.venv\Scripts\python.exe -m src.gate8_2.deepseek_smoke
```

成功时类似：

```text
{'provider': 'deepseek', 'available': True, 'base_url': 'https://api.deepseek.com',
 'model': 'deepseek-flash', 'api_key_set': True, 'timeout_seconds': 12.0, 'mode': 'deepseek'}
{"status": "ok", "model": "...", "elapsed_seconds": 0.8, "usage": {...}, "text": "..."}
DEEPSEEK_SMOKE_OK
```

失败时：

- `NOT RUN_API_BLOCKED`：当前会话没有 Key；
- `DEEPSEEK_SMOKE_FAILED: HTTP 401`：Key 错误/无权限；
- `HTTP 402`：余额/计费问题（以实际返回为准）；
- `HTTP 429`：限流，停止重试，稍后再试；
- 超时：检查代理/网络，或临时增加 `DEEPSEEK_TIMEOUT`。

**不要在没有成功 smoke test 前打开 API Demo 并宣称已接入。**

## 5. 打开双模式 Demo

同一个 PowerShell 会话：

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

.\.venv\Scripts\python.exe -m streamlit run app_gate8.py --server.port 8506
```

打开 `http://localhost:8506`。

侧栏：

```text
Reply mode
  ○ Offline Router-driven
  ○ DeepSeek-assisted
```

选择 `DeepSeek-assisted`。每轮回复下方应显示：

```text
Reply source: deepseek · status: ok · elapsed: ...
{ usage: {...} }
```

如果显示 `offline_blocked` / `NOT RUN_API_BLOCKED`，说明当前 Streamlit 进程没有继承 Key；
重启 PowerShell 和 Streamlit。

## 6. 建议试的 5 个双模式场景

1. **卡查询 + 补卡号**  
   `I still have not received my card.` → `CARD-1001`  
   看工具结果与自然措辞是否一致。
2. **转账状态 + 追问**  
   `My transfer has not arrived. Can you check it?` → `TXN-2001`
3. **取消确认前后**  
   `Please cancel my transfer TXN-2002.` 先看确认提示；再输入 `yes`。  
   必须仍显示 `cancel_transfer` 先 `confirmed=false` 再 `true`。
4. **FAQ/RAG**  
   `What is the policy on transfer fees?`  
   必须保留 KB-013 等真实引用；DeepSeek 不能造新条款。
5. **OOS/越权**  
   `Can you book a flight to Paris?`  
   `Please activate CARD-2001 now.`  
   必须仍 handoff/拒绝，不允许 DeepSeek 绕过。

## 7. 费用控制

- smoke test 一次，`max_tokens=24`；
- 回复改写 `max_tokens=260`；
- 不自动重试、不批量跑测试集；
- 失败立即停止，不循环调用；
- 不在 API 模式跑 120 条 benchmark；
- A 阶段离线评分与 API 费用无关。

## 8. 故障排查

| 现象 | 处理 |
|---|---|
| `available: false` | 检查是否同一 PowerShell、三个变量是否设置 |
| 401 | 重新创建/确认 Key；不要打印 Key |
| 402/余额 | 充值/检查计费 |
| 404 | base URL 不要包含 `/chat/completions`；默认 `https://api.deepseek.com` |
| 429 | 停止重试，稍后再试 |
| timeout | 检查网络/代理；临时提高 `DEEPSEEK_TIMEOUT` |
| UI 显示 offline_blocked | Streamlit 未继承环境变量，重启 |
| UI 显示 fallback + error | 校验拒绝/上游错误；查看页面错误文案，不打印 Key |

## 9. 安全停止条件

- 没有 Key：保持 `NOT RUN_API_BLOCKED`；
- smoke test 失败：不要继续 Demo；
- 任何输出出现真实 Key：立即删除该截图/日志，并在 DeepSeek 控制台 revoke 该 Key；
- 不把 Key 写进 README、报告、CSV、截图、Git commit。

完成真实 smoke test 后，把**不含 Key**的：
- 模型名；
- 延迟；
- usage；
- 一条简短回复；
- 截图

发我，我再帮你判断是否进入 5–10 次体验。
