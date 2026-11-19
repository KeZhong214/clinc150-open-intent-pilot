# Gate 8.2 中文界面、中文输入与中文回复支持

## 1. 结论先行

- Streamlit 工作台已改为**简体中文界面**；
- 意图 ID、工具名、checkpoint、adapter、threshold 等技术标识保持原样；
- 离线模式提供**中文固定回复模板**；
- DeepSeek 增强模式提供中文翻译适配 + 中文自然回复；
- **不修改任何 Gate8.1 算法、阈值、checkpoint 或冻结结果**；
- 中文实验是新增探索性测试，不是新盲测。

本机没有 DeepSeek Key，因此：

- 中文**直接路由探针**已真实运行；
- DeepSeek **翻译后路由**为 `NOT RUN_API_BLOCKED`；
- 没有伪造翻译结果，没有用 Mock 结果生成真实成绩。

## 2. 中文界面

`app_gate8.py` 已中文化：

- 标题：`Gate 8.2 · 银行智能客服工作台`；
- 模型与运行模式、意图路由模型、回复模式；
- 客服对话、中文输入提示；
- 意图/是否接受/分数/阈值；
- 工具调用记录、知识库引用；
- 业务状态：银行卡/转账/工单/争议/审计；
- 虚构政策知识库、错误提示、演示按钮；
- 回复来源、中文翻译适配、usage、回退原因。

内部 intent/工具/checkpoint 原样保留，例如 `card_arrival`、
`get_card_delivery_status`、`gate6_qwen_plain_reject.pt`。

截图：

- `figures/gate8_2_ui_chinese_offline.png`
- `figures/gate8_2_ui_chinese_blocked.png`

## 3. 中文离线模板

`src/gate8_2/chinese_templates.py`：

- 卡状态：卡片 ID、delivered/in_transit、active/inactive；
- 激活确认：明确“敏感操作、请回复确认、目前尚未激活”；
- 转账状态：ID、金额、币种、收款方、状态；
- 取消确认/成功/已取消；
- dispute：争议编号、交易号、人工复核；
- OOS/越权：转人工、没有执行资金操作；
- FAQ：真实 KB doc_id + 标题/引用，注明原文为英文。

模板只使用 Agent 真实工具结果和 KB 引用，不编造事实。

## 4. 中文输入探针

数据：`data/gate8_2/chinese_cases.json`，20 条：

- 卡状态/激活；
- 转账查询/取消；
- 政策/到账时间；
- 争议；
- OOS；
- 模糊；
- 多轮补充/纠正/reset；
- 越权；
- 敏感操作拒绝确认。

当前**只运行中文原文直接路由**（英文模型未改）：

| Router | 直接 accepted | 首轮 intent 命中 | OOS 安全 |
|---|---:|---:|---:|
| GTE-Frozen-RejectOnly | 0/20 | 2/20 | 20/20 |
| Qwen-Frozen-RejectOnly | 12/20 | 13/20 | 20/20 |
| Qwen-Plain42-RejectOnly | 12/20 | 14/20 | 20/20 |

原始逐例：

- `results/gate8_2_chinese_results.csv`（60 行）
- `results/gate8_2_chinese_summary.csv`
- 各 router `gate8_2_chinese_direct_*_results.csv`

结论：

- GTE 直接中文几乎全部 OOS，符合“英文 BANKING77 模型不能假定中文能力”；
- Qwen 对部分中文短句有偶然接受，但不能当作可靠中文开放集能力；
- 没有修改阈值来“提高中文成绩”。

## 5. 中文翻译适配

`src/gate8_2/chinese_adapter.py`：

```text
中文原文
  → DeepSeek 只翻译成英文
  → 保留 CARD/TXN，禁止新增/遗漏
  → 原 GTE/Qwen + RejectOnly 路由
  → 业务工具
  → 中文模板/DeepSeek 中文润色
```

安全规则：

- 翻译结果必须包含原文全部 CARD/TXN，且不能多出新 ID；
- `AMBIGUOUS`/空文本/非英文/ID 不一致 → 不路由，提示澄清；
- 翻译 API 失败 → 不执行工具，安全回退；
- 翻译后的文本仍经过原模型 accepted/OOS；
- DeepSeek 不能指定 intent、不能覆盖 OOS、不能执行工具。

无 Key 时：

- 中文翻译状态 `NOT RUN_API_BLOCKED`；
- 不发送外部请求；
- 不回填“翻译后路由”成绩。

## 6. DeepSeek 中文回复

`GroundedReplyWriter(language="zh")`：

- 输入中文原文、英文路由文本、真实工具结果、KB doc_id、确认状态；
- 只输出简体中文纯文本；
- 新 CARD/TXN/未知 KB → 拒绝；
- 中文“已取消/已激活/已入账/已退款/余额”等不在草稿中 → 拒绝；
- 确认阶段必须保留“确认”和 pending ID；
- 失败/超时/无 Key → 回退离线中文模板。

## 7. 限制

- 中文翻译后路由**尚未真实运行**（无 Key）；
- 20 条是新增同作者探索性探针，不是正式盲测；
- 离线中文模板不是中文模型推理，只是基于真实结构化事实的中文渲染；
- Qwen 中文直接接受有偶然性，不改变 Gate8.1 英文结论；
- 不用 API Demo 覆盖 A 的离线复评。

## 8. 测试

- 中文专项 7 项：`tests/test_gate8_2_chinese.py`；
- 全量 **108 tests OK**：`logs/gate8_2_chinese_all_tests.log`；
- 覆盖 20 案例、直接探针 60 行、ID 保真、翻译拒绝新增/遗漏 ID、中文模板、
  中文 writer 拒绝虚假“已取消”、中文 UI 标签。

## 9. 下一步

用户本机配置 `DEEPSEEK_API_KEY` 后：

1. 跑 `src.gate8_2.deepseek_smoke` 一次；
2. 在 Streamlit 选择 `DeepSeek 增强模式`；
3. 用中文 20 条探针复跑翻译后路由；
4. 保存真实调用状态/usage/延迟/截图；
5. 对比 direct vs translated，但不改 A 的冻结结果。
