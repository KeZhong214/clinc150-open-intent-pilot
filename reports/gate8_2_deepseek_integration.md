# Gate 8.2 B：DeepSeek API 辅助客服集成

**分支**：`gate8-2-eval-and-deepseek-demo`  
**基线**：`gate8-agent-final-fix@45c1318`  
**状态**：代码、Mock 测试、双模式 UI 已完成；真实 API **NOT RUN_API_BLOCKED**（本机未配置 Key）。

---

## 1. 职责边界

| 组件 | 职责 | 不允许 |
|---|---|---|
| GTE/Qwen + B4-RejectOnly | 首轮意图、accepted/OOS、Top-3 | 不由 DeepSeek 改判 |
| Gate8Agent 状态机 | 追问、槽位、确认、取消、顺序 | 不由 DeepSeek 改状态 |
| SQLite 工具 | 所有权、确认、幂等、审计 | 不由 DeepSeek 执行/绕过 |
| KB Retriever | 30 篇虚构政策、doc_id/citation | 不由 DeepSeek 造政策 |
| DeepSeek | 受约束槽位建议、grounded 自然措辞 | 不改业务事实/工具/权限 |

---

## 2. DeepSeek 客户端

`src/gate8_2/deepseek_client.py`：

- 独立 `DEEPSEEK_*` 环境变量，**不回退到 OPENAI_*`**；
- 默认：
  - `DEEPSEEK_BASE_URL=https://api.deepseek.com`
  - `DEEPSEEK_MODEL=deepseek-flash`
  - `DEEPSEEK_TIMEOUT=12`
- `POST {base}/chat/completions`；
- `stream=false`，`max_tokens` 上限 512；
- `status()` 只返回 `api_key_set`，不返回 Key/Authorization；
- 401/429/网络/超时/空 response/无效 JSON 抛 `DeepSeekUnavailable`，由上层安全回退；
- 不记录 Key；错误信息只含状态码和截断 body，不含请求头。

`src/gate8_2/grounded_writer.py`：

输入只包含：

- 用户当前问题；
- 原 Agent 的权威 draft；
- intent/accepted；
- 真实工具结果；
- 允许的 KB doc_id；
- 是否 awaiting confirmation。

输出只作为**措辞替换**。校验：

- 不出现新的 CARD/TXN（必须来自用户/draft/工具结果）；
- 不出现未知 KB ID；
- 不声称工具未返回的 credited/refunded/successfully cancelled/activated；
- 确认阶段必须保留 `confirm` 和 pending ID；
- 空文本/过长/JSON/代码块 -> 回退；
- API 失败 -> 原离线模板。

---

## 3. Streamlit 双模式

`app_gate8.py`：

```text
Reply mode
  ○ Offline Router-driven
  ○ DeepSeek-assisted
```

- 离线：不创建 DeepSeek client 调用；完整可用；
- DeepSeek：有 Key 才调用；无 Key 显示
  `NOT RUN_API_BLOCKED`，回复 source=`offline_blocked`；
- 每轮显示：
  - Router/OOS、Top-3、score/threshold；
  - 真实工具/DB；
  - KB citation；
  - `Reply source: offline / deepseek / fallback / offline_blocked`；
  - `status`、耗时、服务端 usage（有则展示）；
  - 回退原因（如有）。
- 两种模式使用独立 session key 和独立虚构 DB；
- 切换模型继续 release 旧 encoder/session；
- 冻结的 Gate8.1 路由器/阈值不变。

截图：

- `figures/gate8_2_ui_offline.png`
- `figures/gate8_2_ui_deepseek_blocked.png`
- `figures/gate8_2_ui_assisted_blocked_demo.png`
- `figures/gate8_2_ui_chinese_offline.png`
- `figures/gate8_2_ui_chinese_blocked.png`

`gate8_2_ui_chinese_blocked.png` 使用本地合成 malformed key 演示 Key 格式拦截，
没有真实请求；真实 401 由 API 返回，不属于“翻译歧义”。

---

## 4. 安全测试（Mock）

`tests/test_gate8_2_deepseek.py`，9 项：

- 无 Key -> `NOT RUN_API_BLOCKED`，无请求；
- 独立 DeepSeek provider / endpoint；
- 正常 grounded rewrite + usage；
- 新 CARD/TXN -> fallback；
- 未执行却声称 successfully cancelled -> fallback；
- 未知 KB ID / 缺少确认约束 -> fallback；
- HTTP/网络错误 -> fallback；
- 401 错误不泄露 Key；
- App 双模式与 source/usage 展示。

中文专项 `tests/test_gate8_2_chinese.py` 7 项：20 案例、直接探针、ID 保真、
翻译拒绝新增/遗漏 ID、中文模板、中文 writer 拒绝虚假操作、中文 UI。
全量 **109 tests OK**。

日志：`logs/gate8_2_tests_b.log`、`logs/gate8_2_chinese_all_tests.log`。

---

## 4.5 401/Key 格式诊断

已修复两个误报来源：

1. **Key 形状检查**：`DEEPSEEK_API_KEY` 含尖括号 `< >`、引号或首尾空格时，
   `DeepSeekClient.available=False`、`mode=config_error`，在发出 HTTP 前拒绝；
2. **UI 错误分类**：401 显示“API 认证失败”，402 显示余额、429 显示限流、
   timeout 显示网络；只有真正的 `AMBIGUOUS`/ID mismatch 才显示“翻译歧义”。

用户截图中的 `****866>` 和 HTTP 401 属于 Key 被 API 拒绝，不是中文翻译歧义，
也未执行任何工具。处理：去掉尖括号/引号/空格，重新创建 Key，重启 PowerShell 与 Streamlit。

## 5. 真实 API 状态

本机当前**没有** `DEEPSEEK_API_KEY`，所以：

- 没有发送任何 DeepSeek 请求；
- 没有伪造成功记录/usage/延迟；
- `python -m src.gate8_2.deepseek_smoke` 返回
  `NOT RUN_API_BLOCKED`，exit 2；
- 真实连通测试等待用户在 Windows 11 本机配置 Key 后执行
  （见 `reports/gate8_2_deepseek_demo_guide.md`）。

---

## 6. 运行

```powershell
cd D:\code\intent-open-set
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m streamlit run app_gate8.py --server.port 8506
```

配置 DeepSeek 后，选择 `DeepSeek-assisted`；不要提交 Key。

---

## 7. 中文支持补充

- UI 已改为简体中文；
- 中文输入在 DeepSeek 增强模式下先翻译成英文，再交给原 GTE/Qwen RejectOnly；
- 翻译只保留原始 CARD/TXN，不决定 intent，不覆盖 OOS；
- 离线模式使用中文固定模板；
- 中文探针 20 条已运行直接路由：
  - GTE 0/20 accepted；Qwen Frozen 12/20；Qwen Plain42 12/20；
  - 这证明不能把英文模型直接当可靠中文开放集模型；
- 中文翻译后路由在无 Key 时为 `NOT RUN_API_BLOCKED`；
- 详细：`reports/gate8_2_chinese_support.md`。

## 8. 评估边界

- API Demo 与 Gate8.2 A 的科研评分**完全分开**；
- 不用 API Demo 改变 A 的 v1 复评/v2 结果；
- 不发起大规模 API 付费评测；
- 不进入 Gate 9。
