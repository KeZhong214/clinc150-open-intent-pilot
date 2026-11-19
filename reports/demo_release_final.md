# 课堂演示与最终发行封装

## 1. 版本与范围

- 研究底座：`ppt-final-analysis@e8b97b3727d516d3e2a71eff5009f89c5aa46449`
- 发行分支：`demo-release-final`
- 完整发行代码 Commit：`d9e033fa25829dbe6c4bda34b25f8d37ab4f14b8`
- 未改动 Gate3–Gate8.2 的冻结 CSV、阈值、checkpoint 或测试集；
  未新增训练、未做 120 条 API 重评、未进入 Gate9。

本机真实环境：

| 项 | 实测值 |
|---|---|
| OS | Windows 11 |
| Python | 3.14.4 |
| PyTorch | 2.11.0+cu128 |
| CUDA | True |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Demo 模型 | `Qwen-Plain42-RejectOnly` |
| 阈值 | `0.4168315291404724` |

## 2. 完整工作台入口

完整页面是原有的：

```text
app_gate8.py
```

它包含模型/模式选择、9 个 demo、意图/Top-3、DeepSeek 翻译适配、
中文回复、工具时间线、KB 引用、SQLite 业务状态、审计和会话重置。

新的 API 优先启动器：

```powershell
cd D:\code\intent-open-set
.\start_full_demo.ps1
```

- 读取 `configs/demo_api.env`（本地、Git 忽略、不打印 Key）
- 运行 `preflight_demo.py`，通过后才启动
- 默认 `DeepSeek-assisted` + `Qwen-Plain42-RejectOnly`
- 地址：`http://127.0.0.1:8506`

`app_demo.py` 只是早期 60–90 秒简化入口，不是本次主交付。

## 3. 真实预检

本机：

```text
READY in 9.10s | model=Qwen-Plain42-RejectOnly | threshold=0.416832
intent=card_arrival | delivery=in_transit | card=inactive
DEEPSEEK_API_AVAILABLE=True (offline mode remains available)
PREFLIGHT_OK
```

从完整项目 ZIP 解包后（基础模型缓存用 junction 指向本机缓存）：

```text
READY in 10.13s | model=Qwen-Plain42-RejectOnly | threshold=0.416832
intent=card_arrival | delivery=in_transit | card=inactive
DEEPSEEK_API_AVAILABLE=True (offline mode remains available)
PREFLIGHT_OK
```

基础 Qwen/GTE 权重在 `data/model_cache` 中，约 1.94 GB，不进入 ZIP。
ZIP 包含所选 heads、Plain42 adapter、支持集和三个 embedding 缓存，
仍是 **source-only**：需要在保留基础模型缓存的机器上运行。

## 4. 真实 DeepSeek API 测试

真实一次 smoke（只调用一次，不打印 Key）：

```text
DEEPSEEK_SMOKE_OK
model=deepseek-flash
elapsed_seconds=1.875
usage={prompt_tokens:50, completion_tokens:89, total_tokens:139,
       reasoning_tokens:71}
```

注意：`deepseek-flash` 实际是带 reasoning token 的模型，
过小的 `max_tokens=24` 会被 reasoning 消耗完而返回空文本；
本轮已将 smoke 调到 256、翻译调到 320、中文润色调到 420。

完整 UI 的真实 API 卡片场景：

- 中文原文：`我的银行卡 CARD-1002 还没收到，帮我查一下进度。`
- 中文翻译适配：`ok`
- 路由：`card_arrival` / accepted `True` / score `0.8652` / threshold `0.4168`
- 工具：`get_card_delivery_status`
- 工具结果：`in_transit` / `inactive`
- 回复来源：`DeepSeek 润色` / status `ok` / 约 1.842 秒
- 截图：`figures/full_demo_api_success.png`

完整 UI 的真实 API 取消场景：

- `请取消转账 TXN-2002。`
- 路由 `cancel_transfer` / score `0.9458`
- 确认前工具返回 `confirmation_required`，DB 仍 `pending`
- 输入 `确认` 后，翻译仍返回 ok，工具返回 `cancelled`，DB 变为 `cancelled`
- 回复来源：`DeepSeek 润色`，确认步骤约 2.362 秒
- 截图：`figures/full_demo_api_cancel.png`

真实 API 调用记录：

- `logs/full_demo_api_smoke.log`
- `logs/full_demo_api_ui_probe.log`
- `logs/full_demo_api_cancel_probe.log`

## 5. 中文回归修复

- `CARD_RE` / `TXN_RE` / `CARD_TXN` 改为 Unicode 安全边界：
  `(?<![A-Za-z0-9])...(?![A-Za-z0-9])`
  修复 `我的卡CARD-1002还没到` 紧贴汉字时的漏匹配。
- 仅当 `awaiting_confirmation` 且存在 pending action 时，规范化中文/英文控制词：
  - `确认`、`确定`、`确认取消`、`是的`、`是`、`好的`、`继续`、`Confirm.` → yes
  - `取消`、`不用了`、`不要`、`否`、`不`、`停止`、`no` → no
- `取消转账 TXN-2002。` 不会被误映射为 no；未确认不修改数据库。
- 确认分支不把中文词塞进全局 `YES_WORDS`，避免普通中文请求被误判。

## 6. 测试

```text
Ran 122 tests in 5.179s
OK
```

其中新增 13 项 demo 回归：

1. 汉字紧贴 CARD/TXN；
2. 卡片查询只读工具；
3. 确认前 DB 不变；
4. 中文确认后 pending→cancelled；
5. 重复取消幂等；
6. 越权不执行；
7. OOS 不执行资金工具；
8. 话题切换；
9. 新会话 DB 独立；
10. 中文确认词与标点；
11. 无 Key 离线状态；
12. 模型/阈值配对；
13. 超时显式失败。

日志：`logs/full_demo_all_tests.log`。

## 7. API 边界与备用

- Key 只在本机 `configs/demo_api.env`，已被 Git 忽略；启动器/预检/报告均不打印 Key。
- 完整 ZIP **包含这个本地 API 配置文件**，因此 ZIP 本身是敏感的：
  - 不上传 GitHub；
  - 不发送到群聊；
  - 截图不打开 env 文件。
- API 失败时页面显示真实原因（401/402/429/超时/Mock 测试），可切回离线模式；
  离线模式使用已验证中文短句、英文可控输入 + 中文模板，不声称 API 提升 TSR。
- 本次 API 实验是 demo 可用性验证，没有在 120 条模拟任务上重新评分。

## 8. 完整项目 ZIP

主交付：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `dist/OpenIntent_full_demo_project.zip` | 55,225,519 B | `7a69489bb590b03be3eeaa6176990c80feb84dc295b885b4f176a12e5de6f05d` |

`zipfile.testzip()`：`OK`。  
包含：

- 完整 `app_gate8.py` 工作台；
- `start_full_demo.ps1`、`preflight_demo.py`；
- 完整源码 `src/`；
- 3 个所选 head/checkpoint；
- Plain42 adapter；
- 3 个 embedding 缓存；
- 30 条虚构政策；
- 核心结果 CSV、5 张最终分析图；
- 12 页最终 PPTX；
- 完整 API 环境配置 `configs/demo_api.env`（本地敏感）；
- 完整 API 卡片/取消成功截图；
- 两页操作卡和 README。

排除：`.venv`、`data/raw`、旧界面、旧截图、历史全量预测 CSV、
session DB、临时日志和 `.release_staging`。Git 历史未改动。

## 9. 操作卡

`docs/DEMO_OPERATIONS_FULL.md`

最短流程：

1. `.\start_full_demo.ps1`
2. 打开 `http://127.0.0.1:8506`
3. 输入 `我的银行卡 CARD-1002 还没收到，帮我查一下进度。`
4. 输入 `请取消转账 TXN-2002。`
5. 输入 `确认`
6. OOS 备用：`帮我订一张去巴黎的机票。`

## 10. 停止条件

本轮已完成真实 API smoke、完整 UI 卡片/取消验证和完整项目 ZIP 构建。
已停止本机演示服务，等待你本机验收。未继续添加 Agent 功能，未改写冻结实验结果。
