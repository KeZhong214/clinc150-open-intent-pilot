# 少样本智能客服开放意图识别（CLINC150 Pilot）

课程项目：**面向少样本智能客服的开放意图识别与自适应拒识方法研究**

当前状态：**Gate 0、Gate 1、Gate 2 均已完成真实运行。** 已完成 seed=42、
5-shot 的 BGE/Qwen B0/B2，以及 Qwen 的 hard-OOS B2-v2 实验。
B1/B3、更多 shot、其他 seed 尚未展开。

Gate 1 核心结果（fixed test：600 known + 1000 original OOS）：

| Model | Method | Known accept | Known routing | OOS recall | Open-set macro-F1 |
|---|---|---:|---:|---:|---:|
| BGE-small | B0 | 0.9117 | 0.8933 | 0.9420 | 0.8929 |
| BGE-small | B2 | 0.9117 | 0.8950 | **0.9540** | **0.9041** |
| Qwen3-Emb-0.6B | B0 | **0.9667** | **0.9600** | **0.9740** | **0.9570** |
| Qwen3-Emb-0.6B | B2 | 0.9617 | 0.9550 | 0.9680 | 0.9499 |

Gate 2 使用 65/65 disjoint hard-OOS 类别划分。Validation 上 B2-v2 三项
都优于 B0；但 held-out hard-OOS test 上收益没有稳健迁移：

| Qwen | Known routing | Hard-OOS recall | Hard open-set F1 |
|---|---:|---:|---:|
| B0 | **0.9033** | 0.9426 | **0.8796** |
| B2-original | 0.8867 | 0.9364 | 0.8694 |
| B2-v2 | 0.8917 | **0.9482** | 0.8795 |

结论：**Qwen B0 仍是当前默认方案；B2-v2 在 validation 上有效，但在
held-out hard-OOS test 上没有超过 B0 的 known routing 和 open-set F1。**
B2-v2 比原始 B2 好，但不足以推翻 B0。

Gate 1 明细：[results/gate1_metrics.csv](results/gate1_metrics.csv) ·
[Gate 1 报告](reports/gate1_baseline.md)  
Gate 2 明细：[results/gate2_test_metrics.csv](results/gate2_test_metrics.csv) ·
[Gate 2 报告](reports/gate2_method.md) ·
[核心对比图](figures/gate2_test_comparison.png)

---

## 1. 当前已完成的内容

- 从 CLINC150 官方仓库固定 commit 下载 `data_full.json`、`domains.json`、
  `LICENSE`、`README.md`，记录 SHA-256 和逐分区计数。
- 校验官方 JSON 结构、150 个已知意图、10 个业务域、OOS 标签和
  train/val/test 数量。
- 用固定规则选出 20 个已知意图（10 域 × 2 类），写入
  `data/manifest/selected_intents.json`。
- 生成 3 个 support seed（42/43/44）× 2 个 shot（5/10）的 K-shot
  manifest；K=5 是 K=10 的嵌套子集。
- 生成 calibration / final-test 索引，测试集只建立索引，未参与任何选择。
- 完成文本重复与切分重叠审计；报告官方数据中 5 处跨分区重复文本，
  没有删除或改写官方样本。
- 建立 D 盘项目专用 venv，完成 NumPy/Pandas/Scikit-learn/BGE 最小
  CPU 编码冒烟测试。
- Gate 0 只记录 BGE/Qwen 的 Hub revision、许可证和下载体积，未下载 Qwen；
  Gate 1 才下载并加载 Qwen3-Embedding-0.6B，完成 2200 条文本的 GPU 编码：
  batch=8，峰值 allocated 1,155.5 MiB，编码 27.7 s，无 OOM。
- 完成 BGE/Qwen × B0/B2 的 validation 阈值选择和 test 一次评测。
- Gate 2 完成 65/65 hard-OOS 类别划分；1,300 条 hard-OOS val、
  1,950 条 held-out hard-OOS test；hard-OOS 文本隔离重叠为 0。
- Gate 2 实现 B2-v2：corrected `q_c`、hard-OOS predicted-class 上尾 `u_c`、
  向全局阈值收缩，并且类别阈值不低于全局 tau。
- Validation 上冻结 B2-v2 后才编码/评测 held-out test；test 只读一次。
- 25 个协议、配置、可复现性和 Gate 1/2 逻辑测试通过。
- 生成 Gate 1/2 结果表、预测明细、混淆矩阵和核心图表。

详细报告：[Gate 0 环境与数据](reports/gate0_environment.md)  
Gate 1 报告：[B0/B2 Baseline 与动态拒识](reports/gate1_baseline.md)  
Gate 2 报告：[Hard-OOS 与 B2-v2 类别边界校准](reports/gate2_method.md)  
修改记录：[Gate 0](reports/gate0_vibe_coding.md)、
[Gate 1](reports/gate1_vibe_coding.md)、
[Gate 2](reports/gate2_vibe_coding.md)

---

## 2. 项目目录

```text
D:\code\intent-open-set\
├── README.md
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── requirements_lock.txt
├── pip_freeze.txt
├── configs\
│   ├── pilot.yaml
│   └── gate2.yaml
├── data\
│   ├── raw\
│   │   ├── data_full.json
│   │   ├── domains.json
│   │   ├── LICENSE
│   │   ├── README_clinc.md
│   │   ├── download_manifest.json
│   │   └── model_cards\
│   ├── manifest\
│   │   ├── selected_intents.json
│   │   ├── split_counts.csv
│   │   ├── source_manifest.json
│   │   ├── support_manifest.csv
│   │   ├── eval_manifest.csv
│   │   ├── leakage_report.json
│   │   ├── support_leakage_exclusions.json
│   │   ├── environment.json
│   │   ├── model_resource_manifest.json
│   │   ├── gate2_hard_oos_manifest.csv
│   │   ├── gate2_hard_oos_split.json
│   │   └── gate2_hard_oos_checks.json
│   ├── model_cache\                 # BGE + Qwen weights (git-ignored)
│   └── cache_embeddings\            # Gate 1/2 vectors (git-ignored)
├── src\
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── check_environment.py
│   ├── check_models.py
│   ├── encode_embeddings.py
│   ├── run_baseline.py
│   ├── aggregate_results.py
│   ├── make_figures.py
│   ├── prepare_gate2.py
│   ├── encode_gate2_hard_oos.py
│   ├── analyze_gate2_validation.py
│   ├── gate2_hyperparameter_check.py
│   ├── run_gate2.py
│   └── make_gate2_figures.py
├── tests\
│   ├── test_protocol.py
│   ├── test_config.py
│   ├── test_repeatability.py
│   ├── test_gate1_logic.py
│   ├── test_gate2_logic.py
│   └── test_gate2_protocol.py
├── logs\
├── reports\
│   ├── gate0_environment.md
│   ├── gate0_vibe_coding.md
│   ├── gate1_baseline.md
│   ├── gate1_vibe_coding.md
│   ├── gate2_method.md
│   └── gate2_vibe_coding.md
├── results\                         # Gate 1/2 metrics, predictions, thresholds
└── figures\                         # Gate 1/2 PNGs
```

---

## 3. 环境说明（重要）

本机已有可用的 CUDA PyTorch：

```text
Python 3.14.4
torch 2.11.0+cu128
CUDA available=True
NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, compute capability 12.0
```

因此项目 venv 是这样创建的：

```powershell
cd D:\code\intent-open-set
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install `
  "sentence-transformers==6.3.0" `
  "scikit-learn==1.9.1" `
  "PyYAML>=6.0"
```

这是一个项目专用 venv，新增包进入 `.venv`；但 **torch 是从基环境
`D:\Environment` 继承的，不是完全 hermetic 的隔离环境**。这样做是因为
RTX 5070 Laptop 的 sm_120 已经由现有 torch 2.11.0+cu128 验证，重复安装
一套 CUDA torch 会多占数 GB。`pip check` 显示无依赖冲突。

重建到另一台机器时，应先安装与显卡匹配的 PyTorch，再安装
`requirements.txt`。完整版本见 `requirements_lock.txt` / `pip_freeze.txt`。

### 固定缓存目录

为避免写入 C 盘，所有模型和嵌入缓存都在 D 盘：

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TOKENIZERS_PARALLELISM = "false"
```

`src/check_models.py` 也会默认把 `HF_HOME` 指向该目录。

---

## 4. Gate 0 运行顺序

所有命令都在 `D:\code\intent-open-set` 下执行。

### 4.1 数据下载（小文件，可重复运行）

```powershell
.\.venv\Scripts\python.exe -m src.download_data
```

成功输出末尾：

```text
Validated schema and counts: {'oos_val': 100, 'val': 3000, ...}
SOURCE_DOWNLOAD_OK
```

### 4.2 固定类别、K-shot 与 manifest

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data `
  --seed 42 --known-classes 20 --per-domain 2
```

成功输出包含：

```text
Selected 20 intents (2 per domain):
...
PREPARE_DATA_OK
```

### 4.3 协议测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

成功输出末尾：

```text
Ran 25 tests in ...s
OK
```

### 4.4 环境与 BGE 最小冒烟测试

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
.\.venv\Scripts\python.exe -m src.check_environment
.\.venv\Scripts\python.exe -m src.check_models --bge-smoke
```

成功输出末尾分别为 `ENVIRONMENT_CHECK_OK` 和 `MODEL_CHECK_OK`。

`--bge-smoke` **只**下载/加载 BGE-small-en-v1.5（约 128 MiB），不会下载
Qwen；Qwen 是在 Gate 1 单独下载和编码的。BGE 当前结果：3 条文本编码为
`(3, 384)`，向量有限且 L2 norm=1.0；缓存后加载约 2.95 s，3 条文本编码
约 0.13 s（CPU，不能外推为全量耗时）。

### 4.5 第一个验收命令

如果只想快速确认当前 Gate 0 产物，先执行：

```powershell
cd D:\code\intent-open-set
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

预期最后一行是 `OK`，共 25 个测试。请反馈
`logs/gate2_all_tests.log`；如需环境信息，再反馈
`data/manifest/environment.json`。

### 4.6 Gate 1 复现命令（实际已运行）

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# 编码缓存
.\.venv\Scripts\python.exe -m src.encode_embeddings --model bge --device cpu --batch-size 32
.\.venv\Scripts\python.exe -m src.encode_embeddings --model qwen --device cuda --batch-size 8

# seed=42 / 5-shot；B0 先跑，B2 再跑
.\.venv\Scripts\python.exe -m src.run_baseline --model bge  --method B0 --seed 42 --shot 5
.\.venv\Scripts\python.exe -m src.run_baseline --model qwen --method B0 --seed 42 --shot 5
.\.venv\Scripts\python.exe -m src.run_baseline --model bge  --method B2 --seed 42 --shot 5
.\.venv\Scripts\python.exe -m src.run_baseline --model qwen --method B2 --seed 42 --shot 5

# 汇总与图表
.\.venv\Scripts\python.exe -m src.aggregate_results
.\.venv\Scripts\python.exe -m src.make_figures
```

`run_baseline.py` 默认拒绝覆盖已有结果，避免无意中第二次读取 test。
只有明确写 `--force` 才会重新评测；本轮正式结果没有使用 `--force`。

### 4.7 Gate 2 hard-OOS 复现命令（实际已运行）

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# 65/65 hard-OOS 类别划分和 manifest
.\.venv\Scripts\python.exe -m src.prepare_gate2

# 只编码 hard-OOS validation
.\.venv\Scripts\python.exe -m src.encode_gate2_hard_oos --split hard_oos_val --batch-size 8

# validation 校准；检查后再加 --freeze
.\.venv\Scripts\python.exe -m src.run_gate2 --phase validation
.\.venv\Scripts\python.exe -m src.run_gate2 --phase validation --freeze

# 冻结后才编码 held-out hard-OOS test
.\.venv\Scripts\python.exe -m src.encode_gate2_hard_oos --split hard_oos_test --batch-size 8

# 一次性 test
.\.venv\Scripts\python.exe -m src.run_gate2 --phase test

# 图表
.\.venv\Scripts\python.exe -m src.make_gate2_figures
```

`run_gate2.py` 在 freeze 前不会编码或评测 hard-OOS test；test 结果文件
存在后默认拒绝第二次读取。不要用 `--force` 重跑正式 test。

---

## 5. 数据协议（冻结）

| 项目 | 设置 |
|---|---|
| 数据集 | CLINC150 / OOS-Eval，官方 `data_full.json` |
| 官方 revision | `828f8093932c8fe6ca7936c3d2e52903b1c523de` |
| 已知类别 | 20 个，官方 10 个 domain 各 2 个 |
| 选择 seed | 42（SHA-256 排序，非 test 调参） |
| Support | 官方 `train`，每类 K=5/10；seed 42/43/44 |
| Calibration known | 20 类官方 `val` 全部 400 条 |
| Calibration OOS | 官方 `oos_val` 100 条 |
| Final known test | 20 类官方 `test` 全部 600 条 |
| Final OOS test | 官方 `oos_test` 1000 条 |
| `oos_train` | 本 pilot 不使用 |

**测试集不允许用于**：方法选择、prompt 选择、阈值调优、挑 seed、看错误后
反向改方法。Gate 1 只能读取固定参数并在最后跑一次。

### Gate 2 的独立 hard-OOS 协议

Gate 2 不覆盖 Gate 1，也不混用原始 `oos_test`：

- 剩余 130 个未支持类按 `SHA-256("hard-oos|42|<intent>")` 分成 65/65；
- dev 65 类的 1,300 条官方 val = hard-OOS validation；
- held-out 65 类的 1,950 条官方 test = hard-OOS final test；
- hard-OOS val/test 与 known support/val/test、原始 OOS 的文本重叠均为 0；
- B0/B2/B2-v2 在同一 validation 上重新校准；
- B2-v2 的 `kappa`、`u_quantile` 只在 validation 选，test 冻结后只评测一次。

详见 `reports/gate2_method.md`。

### 已知的官方数据重复

官方数据内部没有分区内重复，但规范化文本有 **5 处跨分区碰撞**：

- `train` ↔ `val`：3 处；
- `train` ↔ `test`：2 处。

完整样本 ID、标签和文本见
[data/manifest/leakage_report.json](data/manifest/leakage_report.json)。
没有删除或重标官方样本。构造出的 support 与 calibration/test 之间
ID 和规范化文本交集均为 0；但 `user_name` 的一个 test 文本在 train 中
存在另一个标签的重复，这是上游数据的残余风险，报告中已注明。

---

## 6. 不要做的事

- 不要用 test 结果倒推阈值、prompt、K、seed 或方法。
- 不要把 BGE smoke test 当成 B0 准确率。
- 不要改写或删除官方 CLINC150 样本来“修复”重复文本。
- 不要无意中用 `--force` 重新评估 test；如果必须重跑，要在报告中审计。
- 不要把 `Qwen3-0.6B`（生成模型）与
  `Qwen3-Embedding-0.6B`（嵌入模型）混淆。

---

## 7. 当前边界与后续扩展

已完成：Gate 0、Gate 1、Gate 2；seed=42、5-shot；BGE/Qwen B0/B2；
Qwen hard-OOS B2-v2；validation-only 阈值选择；两次 test 各自只评测一次；
结果表和 Gate 2 的 5 张图。

Gate 2 的结论是：B2-v2 在 hard-OOS validation 上三项优于 B0，但在
65 个 disjoint held-out unknown classes 的 test 上，hard recall +0.56 pp，
known routing −1.17 pp，hard open-set F1 基本持平。**B0 仍是默认方案。**

尚未做：

- 10-shot、seed 43/44；
- B1 多原型和 B3 组合；
- 新的独立 held-out unknown classes；
- 更多 prompt / 模型 / 训练策略；
- Streamlit Demo 和最终报告。

不建议在当前 hard-OOS test 上继续调 B2-v2 参数；若继续，需要一组新的
独立 unknown classes，否则会变成 test 调参。B1/B3 和更多 seed 仍作为
后续扩展，不在本轮自动展开。
