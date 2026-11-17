# Gate 0 Vibe Coding / 修改记录

> 本文件记录 Gate 0 当时的状态；Gate 1 的修改见
> [`reports/gate1_vibe_coding.md`](gate1_vibe_coding.md)。

本文件记录 Gate 0 期间由 AI 协作产生的内容、关键人工可核查决策、真实报错
与修复。它不是“AI 已完成实验”的证明；实验结论只来自实际运行日志。

## 1. 本轮用户请求

- 完整阅读《少样本智能客服意图识别_DeepSeek实验任务书.md》；
- 当前只执行 Gate 0：环境与数据准备；
- 复述研究问题与 B0/B1/B2/B3；
- 检查 CLINC150 官方结构和划分；
- 建立最小项目目录与独立 Python 环境；
- 编写数据获取、检查、20 类选择、K-shot 抽样的完整可运行代码；
- 固定 seed 与样本划分，保存可复现 manifest；
- 检查 BGE/Qwen 依赖与计算资源；
- 给出第一条执行命令、成功输出和需反馈日志；
- 不进入 Gate 1，不下载 Qwen 权重，不编造指标。

## 2. 生成/修改的文件

### 代码

| 文件 | 作用 |
|---|---|
| `src/download_data.py` | 固定 commit 下载官方 4 个文件，校验 SHA-256 与计数 |
| `src/prepare_data.py` | 数据校验、20 类选择、support/eval manifest、泄漏审计 |
| `src/check_environment.py` | 采集系统/GPU/内存/磁盘/包版本，写 JSON |
| `src/check_models.py` | 查询 BGE/Qwen 元数据，CPU BGE 冒烟；禁止下载 Qwen |
| `tests/test_protocol.py` | 计数、隔离、support、eval、泄漏测试 |
| `tests/test_config.py` | YAML 与冻结 manifest、Qwen prompt hash 一致性 |
| `tests/test_repeatability.py` | 重新生成 manifest，比较字节级/去时间戳一致性 |

### 配置与说明

| 文件 | 作用 |
|---|---|
| `configs/pilot.yaml` | 冻结数据、支持集、评测、模型、prompt、方法口径 |
| `README.md` | 环境、运行命令、协议、禁止事项 |
| `requirements.txt` / `requirements_lock.txt` / `pip_freeze.txt` | 依赖记录 |
| `reports/gate0_environment.md` | Gate 0 检查报告 |

### 生成的数据产物

- `data/raw/`：官方数据、domains、LICENSE、README、下载 manifest；
- `data/manifest/selected_intents.json`、`split_counts.csv`、
  `source_manifest.json`、`support_manifest.csv`、`eval_manifest.csv`、
  `leakage_report.json`、`support_leakage_exclusions.json`；
- `data/manifest/environment.json`、`model_resource_manifest.json`；
- `logs/gate0_*.log`。

## 3. 关键决策与理由

1. **项目放 D 盘**：`D:\code\intent-open-set`。C 盘只剩约 11 GiB；D 盘约
   62 GiB，模型缓存也指向 D 盘。
2. **venv 复用基环境 CUDA torch**：本机已有 `torch 2.11.0+cu128`，且
   RTX 5070 Laptop（sm_120）可以用它做 CUDA 矩阵乘。重新装 CUDA torch 会
   多占数 GB，因此用 `python -m venv --system-site-packages`。这不是完全
   hermetic 环境，已在报告和 README 披露。
3. **类别选择不用随机模块，改用 SHA-256 排序**：避免 Python 版本/随机实现
   差异；seed 42 参与哈希输入，规则可审计。每个官方域选 2 个类，保证 10 域
   覆盖，共 20 类。
4. **K-shot 用同一排序的前 K 条**：K=5 自动成为 K=10 的子集，后续比较
   5-shot/10-shot 时支持集嵌套，减少额外变量。
5. **官方跨分区重复不删除**：按任务书要求先报告。support 候选与 eval 文本
   重复才过滤，并记录；本轮过滤 0 条。
6. **Qwen 只查元数据**：Gate 0 明确不下载 Qwen。BGE 只下载约 128 MiB 并用
   CPU 编码 3 条文本。
7. **固定 Qwen prompt**：依据模型卡“任务 instruction 用英文、query 侧通常
   有收益”的建议，固定一个客服意图匹配 instruction；support 与 query 使用
   同一前缀。没有用 test 选择 prompt。
8. **不生成任何 B0–B3 指标**：Gate 0 没有编码全量数据、没有原型、没有阈值、
   没有 test 预测，因此没有可报的准确率。

## 4. 真实报错与修复

### 4.1 GitHub API 未认证限流

准备阶段查询 `api.github.com` 时出现：

```text
API rate limit exceeded for ...
```

处理：

- 不依赖 GitHub API 动态解析 commit；
- 改为固定 commit 的 raw URL 下载；
- 固定 SHA-256，若上游内容变化会直接失败；
- 数据下载成功，计数与哈希已记录。

### 4.2 早期数据 URL 404

第一次尝试用未成功取得的 commit 变量拼 raw URL，导致 404。

处理：

- 改为显式常量
  `828f8093932c8fe6ca7936c3d2e52903b1c523de`；
- 重新下载并核对 `data_full.json` 为 2,495,390 bytes。

### 4.3 domain 文件哈希手抄错误

在把实测哈希写入 `src/download_data.py` 时，`domains.json` 的哈希曾多抄了
一个字符。运行前对照 `Get-FileHash` 发现并修正为：

```text
b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a
```

之后下载脚本校验通过。

### 4.4 venv 的 `system_site_packages` 误报

最初用 `sys.flags` 推测 venv 是否继承系统包，输出 `system_site=False`，
与实际 `pyvenv.cfg` 不符。

处理：

- 改为读取 `.venv\pyvenv.cfg` 的
  `include-system-site-packages = true`；
- 重新生成 `environment.json`，现在为 `system_site=True`。

### 4.5 PowerShell 把 Python stderr 显示成 NativeCommandError

BGE 加载时 torch 输出：

```text
NOTE: Redirects are currently not supported in Windows or MacOs.
```

PowerShell 将其包装成 `NativeCommandError`，但 Python 退出码为 0，模型实际
加载成功，输出 `MODEL_CHECK_OK`。这不是模型失败；日志保留原始警告，报告中
没有把它伪装成成功之外的指标。

## 5. 实际验证

- `SOURCE_DOWNLOAD_OK`：官方 4 文件 + 哈希 + 计数；
- `PREPARE_DATA_OK`：20 类、900 条 support、2,100 条 eval；
- `Ran 15 tests ... OK`：协议、配置、可复现性；
- `ENVIRONMENT_CHECK_OK`：Windows/RTX 5070/torch CUDA；
- `MODEL_CHECK_OK`：BGE `(3, 384)`、finite、norm=1、CPU；
- CUDA 1024×1024 FP16 矩阵乘：finite、峰值 allocated 17.1 MiB；
- 项目 Qwen cache 不存在；Qwen 未加载。

## 6. 没有做的事（避免误报）

- 没有下载 Qwen 权重；
- 没有 Qwen 前向/显存测试；
- 没有 B0/B1/B2/B3；
- 没有阈值搜索、AUROC、F1、混淆矩阵；
- 没有生成图表或 Streamlit；
- 没有把论文数字、占位数字或随机数写进结果；
- 没有删除官方重复样本，也没有修改官方划分。

## 7. 人工复核建议

1. 打开 `data/manifest/selected_intents.json`，确认 20 类符合预先约定；
2. 打开 `data/manifest/leakage_report.json`，确认 5 处官方重复已知情；
3. 运行 `python -m unittest discover -s tests -v`，确认 15 个测试通过；
4. 确认 `data/model_cache` 中只有 BGE，没有 Qwen；
5. 再决定是否批准 Gate 1 的 BGE B0；Qwen 需单独批准。
