# 少样本智能客服开放意图识别（CLINC150 Pilot）

课程项目：**面向少样本智能客服的开放意图识别与自适应拒识方法研究**

当前状态：**Gate 0 已完成真实运行；Gate 1 尚未开始。** 仓库中没有任何
B0/B1/B2/B3 准确率、阈值结果或 Qwen 实测显存数据。

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
- 记录 BGE/Qwen 的 Hub revision、许可证、下载体积；**没有下载 Qwen 权重**。
- 15 个协议/配置/可复现性测试全部通过。

详细报告：[reports/gate0_environment.md](reports/gate0_environment.md)  
修改记录：[reports/gate0_vibe_coding.md](reports/gate0_vibe_coding.md)

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
│   └── pilot.yaml
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
│   │   └── model_resource_manifest.json
│   ├── model_cache\                 # BGE only; no Qwen weights
│   └── cache_embeddings\            # empty until Gate 1
├── src\
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── check_environment.py
│   └── check_models.py
├── tests\
│   ├── test_protocol.py
│   ├── test_config.py
│   └── test_repeatability.py
├── logs\
├── reports\
└── results\ / figures\              # empty until Gate 1
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
Ran 15 tests in ...s
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
Qwen。当前结果：3 条文本编码为 `(3, 384)`，向量有限且 L2 norm=1.0；
缓存后加载约 2.95 s，3 条文本编码约 0.13 s（CPU，不能外推为全量耗时）。

### 4.5 第一个验收命令

如果只想快速确认当前 Gate 0 产物，先执行：

```powershell
cd D:\code\intent-open-set
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

预期最后一行是 `OK`，共 15 个测试。请反馈
`logs/gate0_tests.log`；如需环境信息，再反馈
`data/manifest/environment.json`。

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

- 不要在未确认前下载 Qwen3-Embedding-0.6B 权重。
- 不要用 test 结果倒推阈值、prompt、K、seed 或方法。
- 不要把 BGE smoke test 当成 B0 准确率。
- 不要改写或删除官方 CLINC150 样本来“修复”重复文本。
- 不要把 `Qwen3-0.6B`（生成模型）与
  `Qwen3-Embedding-0.6B`（嵌入模型）混淆。

---

## 7. 下一步（尚未执行）

Gate 1 的建议顺序：

1. 先用 BGE CPU 完成 B0 的 support 原型、validation 阈值选择、test 一次评测；
2. 再在用户明确批准后，对 Qwen 做 batch=1/2/4 的小样本 GPU 冒烟测试；
3. 只有冒烟测试记录峰值显存和输出维度后，才提取全量选定样本向量。

当前没有执行以上任何一步，也没有任何 B0/B2 指标。
