# 课堂演示与最终发行封装

## 1. 版本与范围

- 研究底座：`ppt-final-analysis@e8b97b3727d516d3e2a71eff5009f89c5aa46449`
- 发行分支：`demo-release-final`
- 发行代码 Commit：`4eec5583e4f408fc04f9a20f93b7f84b730dd8c5`
- 未改动 Gate3–Gate8.2 的冻结 CSV、阈值、checkpoint 或测试集；
  未新增训练、未做 API 重评、未进入 Gate9。

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

## 2. 现场入口

```powershell
cd D:\code\intent-open-set
.\start_demo.ps1
```

或双击 `start_demo.cmd`。

- 新入口：`app_demo.py`
- 固定模型：Qwen Plain42 RejectOnly（从 `results/gate8_1_model_registry.json` 加载并校验 SHA）
- 默认模式：离线模式（无 API）
- 端口：`8509`
- 启动前自动执行 `preflight_demo.py`；失败不会启动 Streamlit。
- 重置按钮：`重置演示（新会话 DB）`，只创建新的虚构会话数据库。

## 3. 真实预检

```text
READY in 7.30s | model=Qwen-Plain42-RejectOnly | threshold=0.416832 |
intent=card_arrival | delivery=in_transit | card=inactive
DEEPSEEK_API_AVAILABLE=False (offline mode remains available)
PREFLIGHT_OK
```

从最终 demo ZIP 解包后（用 junction 链接本机模型缓存）再次执行 preflight：

```text
READY in 7.55s | model=Qwen-Plain42-RejectOnly | threshold=0.416832 |
intent=card_arrival | delivery=in_transit | card=inactive
PREFLIGHT_OK
```

基础 Qwen/GTE 权重在 `data/model_cache` 中仅存本机，不进入 ZIP。
Demo ZIP 是 **source-only**：代码、所选 heads/adapter、支持集和缓存已包含，
但仍依赖本机基础模型缓存；README 已明确标注，不把它宣称为完全独立 portable。

## 4. 真实课堂剧情

证据日志：`logs/release_demo_ui_probe.log`  
截图：`figures/gate8_2_demo_release.png`

### 4.1 卡片查询

输入：`我的卡CARD-1002还没到`

实测：

- intent `card_arrival`
- accepted `True`
- score `0.7766`
- threshold `0.4168`
- tool `get_card_delivery_status`
- delivery `in_transit`
- card `inactive`
- 中文回复：运输中、未激活

该短句故意让英文 ID 紧贴汉字，验证了新的 ID 边界正则；
完整句“我的银行卡 CARD-1002 还没收到，帮我查一下进度”在离线 Qwen 直路由
下曾被判 OOS，因此课堂固定使用已验证短句；需要完整中文句子时使用
DeepSeek 增强模式（需要有效 Key）。

### 4.2 取消转账与确认

输入：`请取消转账 TXN-2002。`

确认前实测：

- intent `cancel_transfer`
- accepted `True`
- score `0.8177`
- tool `cancel_transfer` 返回 confirmation required
- DB：`TXN-2002 = pending`（未改变）

输入：`确认`

确认后实测：

- 工具返回 `cancelled`
- DB：`TXN-2002 = cancelled`
- 已触发确认与审计路径

### 4.3 OOS 备用

输入：`帮我订一张去巴黎的机票。`

实测：OOS / accepted `False`，仅 `handoff_to_human`，不调用卡片或转账资金工具。

## 5. 中文回归修复

- `CARD_RE` / `TXN_RE` / `CARD_TXN` 从 `\b...\b` 改为 Unicode 安全的
  `(?<![A-Za-z0-9])...(?![A-Za-z0-9])`，修复 `我的卡CARD-1002还没到`
  这类紧贴汉字的编号漏匹配。
- `Gate8Agent` 增加有限控制命令规范化：
  - 确认：`确认`、`确定`、`确认取消`、`是的`、`是`、`好的`、`继续`、
    `yes`、`Confirm.` 等；
  - 拒绝：`取消`、`不用了`、`不要`、`否`、`不`、`停止`、`no` 等。
- 只有 `awaiting_confirmation` 且已有 pending action 时才进行映射；
  不把中文词直接塞进全局 `YES_WORDS`，未确认不会修改交易状态。
- `取消转账 TXN-2002。` 不会被误映射为拒绝；必须精确控制词。

## 6. 测试

```text
Ran 122 tests
OK
```

新增 `tests/test_demo_release.py` 的 13 项回归覆盖：

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

另外保留原有 109 项 Gate0–8.2 测试，共 **122 项通过**。

## 7. API / 断网边界

- 本机没有有效 `DEEPSEEK_API_KEY`；preflight 明确输出
  `DEEPSEEK_API_AVAILABLE=False`。
- 在 DeepSeek 增强模式输入中文时，真实 UI 显示：
  `NOT RUN_API_BLOCKED` / `DEEPSEEK_API_KEY not configured`，
  不执行任何业务工具。
- 401/402/429/超时由现有 Mock 测试覆盖；**没有伪造真实 API 成功记录**。
- 课堂备用采用离线模式 + 已验证中文短句 + 中文模板；
  不声称 API 带来了 TSR 提升。

## 8. ZIP 交付

限制：基础 Qwen/GTE 模型权重和 `.venv` 不进入 ZIP；课程包与演示包均为
source-only，模型基础权重使用本机缓存。两个 ZIP 均通过 `zipfile.testzip()`。

| 包 | 文件 | 大小 | SHA256 |
|---|---|---:|---|
| 课程源码 | `dist/OpenIntent_course_clean.zip` | 53,437,738 B | `344214718386ce831fd5eec16e660bcf039e5ccbebec311420db153c73ac7410` |
| 现场演示 | `dist/OpenIntent_demo_clean.zip` | 51,530,483 B | `899cde45e53c0c95a4d0081491150aa92e4e8ebbf057891b04fa677149cd29f8` |

课程包包含：Agent/B3-H/B4/Embedding/LoRA 载入代码、三组所选 head/checkpoint、
Plain42 adapter、支持集、三个所选 embedding 缓存、30 条 KB、核心结果 CSV、
5 张最终分析图、关键单测、报告与 SHA256 清单。

演示包包含：`app_demo.py`、`preflight_demo.py`、`start_demo.ps1/.cmd`、
运行模块、固定模型配置、KB、所选缓存、README 与两页操作卡。

均排除：`.venv`、`data/raw`、旧 Streamlit 界面（app_gate4/5/6）、旧截图、
历史全量预测 CSV、session DB、API Key、临时日志、`.release_staging`。
GitHub 历史与原始实验结果未删除。

## 9. 文件清单与校验

ZIP 内附：

- `RELEASE_MANIFEST.json`
- `SHA256SUMS.txt`

ZIP 外部构建记录：`dist/release_manifest.json`。

## 10. 明日使用

1. 按 `docs/DEMO_OPERATIONS.md` 启动；
2. 点击“重置演示”；
3. 按 ① → ② → `确认` 操作；
4. OOS 仅作为备用；
5. 失败时切 PPT 截图，不在现场重装或调阈值。

## 11. 停止条件

本轮已完成并停止，等待本机验收。没有继续添加 Agent 功能，也没有改写任何
冻结实验指标。
