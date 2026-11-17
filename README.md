# 少样本开放意图识别：多数据集、跨模型与原型—近邻实验

课程项目：**面向智能客服的少样本开放意图识别：多模型跨场景评测与原型—近邻协同优化**

当前状态：**Gate 0–3 均已完成真实运行。** 3 个数据集、4 个冻结编码器、
5/10-shot、B0/B3-P/B3-H 的 72 格矩阵已完成；另完成 BANKING77 SetFit-BGE
训练实验和 Streamlit Demo。

Gate 3 的核心结果（best config per dataset；CLINC 为历史测试）：

| Dataset | Best config | Known routing | OOS recall | Open-set macro-F1 |
|---|---|---:|---:|---:|
| CLINC150（历史） | Qwen 10-shot B0 | 0.9817 | 0.9010 | 0.9226 |
| BANKING77 | GTE 10-shot B3-H | 0.8625 | 0.6368 | **0.8306** |
| MASSIVE en-US | Qwen 10-shot B0 | 0.8272 | 0.5298 | 0.7982 |

B3-H 的场景差异：

- BANKING77：相对 B0 平均 OOS recall +5.21 pp，open-set F1 +0.99 pp；
- CLINC150（历史）：OOS recall −1.11 pp，F1 −0.92 pp；
- MASSIVE：OOS recall −0.43 pp，F1 −1.46 pp。
- **B3-H 在 BANKING77 有用，在 MASSIVE/CLINC 没有稳定收益。**

训练式对照（BANKING77 5-shot，同一 test）：BGE-B0 open-set F1 0.7505；
SetFit-BGE 0.6696；投影适配器 0.7394。两者都未超过冻结 BGE-B0。

Gate 1/2 历史结果保持冻结：

| Scope | Model/Method | Key result |
|---|---|---:|
| Gate 1 CLINC | Qwen B0 | open-set F1 0.9570 |
| Gate 1 CLINC | Qwen B2 | open-set F1 0.9499 |
| Gate 2 hard-OOS | Qwen B0 | hard OOS recall 0.9426 |
| Gate 2 hard-OOS | Qwen B2-v2 | hard OOS recall 0.9482, known routing −1.17 pp |

详细报告：[Gate 0 环境与数据](reports/gate0_environment.md) ·
[Gate 1 Baseline](reports/gate1_baseline.md) ·
[Gate 2 Hard-OOS](reports/gate2_method.md) ·
[Gate 3 多数据集](reports/gate3_final_report.md)  
结果索引：[results/gate3_matrix.csv](results/gate3_matrix.csv) ·
[results/gate3_summary_by_dataset_method.csv](results/gate3_summary_by_dataset_method.csv)  
图表：[figures/gate3_heatmap_macro_f1.png](figures/gate3_heatmap_macro_f1.png) ·
[figures/gate3_ablation.png](figures/gate3_ablation.png)  
Demo：[app.py](app.py) · [真实运行截图](figures/gate3_streamlit_demo.png)

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
- Gate 3 新建 CLINC150/BANKING77/MASSIVE en-US 统一 manifest；BANKING77
  40/18/19、MASSIVE 35/12/13 类划分，support 5/10 严格嵌套。
- 完成 4 个冻结编码器（BGE、Qwen、GTE-ModernBERT、E5）在 3 数据集上的
  向量提取；总 11,011 条文本/模型。
- 完成 3 数据集 × 4 模型 × 2 shot × B0/B3-P/B3-H 的 72 格矩阵；validation
  选阈值、freeze 后 test 只评一次。
- 完成 BANKING77 SetFit-BGE 训练和投影适配器对照，记录训练曲线；两者均
  未超过 BGE-B0。
- 完成数据集画像、模型×数据集热图、shot 曲线、B3 消融、阈值权衡、
  BANKING77 混淆、成本散点和 4 类可追踪错误案例。
- 完成 Streamlit Demo 并在真实浏览器中运行截图。
- 29 个协议、配置、可复现性和 Gate 1/2/3 逻辑测试通过。

详细报告：[Gate 0 环境与数据](reports/gate0_environment.md)  
Gate 1 报告：[B0/B2 Baseline 与动态拒识](reports/gate1_baseline.md)  
Gate 2 报告：[Hard-OOS 与 B2-v2 类别边界校准](reports/gate2_method.md)  
Gate 3 报告：[多数据集、跨模型与 B3 实验](reports/gate3_final_report.md)  
修改记录：[Gate 0](reports/gate0_vibe_coding.md)、
[Gate 1](reports/gate1_vibe_coding.md)、
[Gate 2](reports/gate2_vibe_coding.md)、
[Gate 3](reports/gate3_vibe_coding.md)

---

## 2. 项目目录

```text
D:\code\intent-open-set\
├── README.md
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── requirements_lock.txt
├── pip_freeze.txt
├── app.py                           # Streamlit demo
├── configs\
│   ├── pilot.yaml
│   ├── gate2.yaml
│   └── gate3.yaml
├── data\
│   ├── raw\
│   │   ├── data_full.json
│   │   ├── domains.json
│   │   ├── LICENSE
│   │   ├── README_clinc.md
│   │   ├── download_manifest.json
│   │   └── model_cards\
│   ├── gate3\
│   │   └── raw\                     # BANKING77 parquet + MASSIVE en-US JSONL
│   ├── manifest\
│   │   ├── selected_intents.json
│   │   ├── gate2_*                  # hard-OOS split/manifests
│   │   ├── gate3_*                  # 3 datasets, classes, support/cal/test
│   │   └── ...
│   ├── model_cache\                 # BGE/Qwen/GTE/E5 weights (git-ignored)
│   └── cache_embeddings\            # Gate 1/2/3 vectors (git-ignored)
├── src\
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── encode_embeddings.py
│   ├── run_baseline.py
│   ├── gate3\
│   │   ├── prepare.py
│   │   ├── encode.py
│   │   ├── methods.py
│   │   ├── run_matrix.py
│   │   ├── freeze.py
│   │   ├── train_adapter.py
│   │   ├── train_setfit.py
│   │   ├── aggregate.py
│   │   └── make_figures.py
│   └── ...                          # Gate 1/2 scripts retained
├── tests\
│   ├── test_gate1_logic.py
│   ├── test_gate2_protocol.py
│   ├── test_gate3_protocol.py
│   └── ...
├── logs\
├── reports\
│   ├── gate0_*.md
│   ├── gate1_*.md
│   ├── gate2_*.md
│   ├── gate3_final_report.md
│   ├── gate3_ppt_outline.md
│   ├── gate3_speech.md
│   └── gate3_vibe_coding.md
├── results\                         # Gate 1/2/3 metrics, predictions, thresholds
└── figures\                         # Gate 1/2/3 PNGs, Streamlit screenshot
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
Ran 29 tests in ...s
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

预期最后一行是 `OK`，共 29 个测试。请反馈
`logs/gate3_all_tests.log`；如需环境信息，再反馈
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

### 4.8 Gate 3 复现命令（实际已运行）

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# 1. 3 数据集统一 manifest
.\.venv\Scripts\python.exe -m src.gate3.prepare

# 2. 四个冻结编码器，每个模型一次性编码 3 个数据集
.\.venv\Scripts\python.exe -m src.gate3.encode --model bge
.\.venv\Scripts\python.exe -m src.gate3.encode --model qwen --device cuda --batch-size 8
.\.venv\Scripts\python.exe -m src.gate3.encode --model gte --device cuda --batch-size 16
.\.venv\Scripts\python.exe -m src.gate3.encode --model e5 --device cuda --batch-size 32

# 3. validation-only 的 72 格阈值选择
.\.venv\Scripts\python.exe -m src.gate3.run_matrix --phase validation

# 4. 冻结后才跑 final test；不要重复
.\.venv\Scripts\python.exe -m src.gate3.freeze
.\.venv\Scripts\python.exe -m src.gate3.run_matrix --phase test

# 5. 汇总与图
.\.venv\Scripts\python.exe -m src.gate3.aggregate
.\.venv\Scripts\python.exe -m src.gate3.make_figures

# 6. 训练式对照（可选，但本轮已运行）
.\.venv\Scripts\python.exe -m src.gate3.train_adapter
.\.venv\Scripts\python.exe -m src.gate3.train_setfit

# 7. Demo
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Gate 3 的 test 已在 `results/gate3_freeze.json` 后评测一次；再次运行
`--phase test` 会被拒绝。SetFit/Adapter 的 BANKING77 test 与矩阵 test 相同，
因此报告中明确标注“非盲测复用”。

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

### Gate 3 的 3 数据集协议

| Dataset | Known / dev-unknown / final-unknown | Support | Calibration known | Known test | Dev unknown | Final unknown |
|---|---:|---:|---:|---:|---:|---:|
| CLINC150（历史） | 20 / — / — | 100 / 200 | 400 | 600 | 100 | 1,000 |
| BANKING77 | 40 / 18 / 19 | 200 / 400 | 800 | 1,600 | 720 | 760 |
| MASSIVE en-US | 35 / 12 / 13 | 175 / 350 | 1,323 | 1,915 | 407 | 436 |

- 类别选择：固定 SHA-256 seed=42；MASSIVE known 额外要求官方 train ≥10；
- BANKING77 的 dev/final unknown 来自官方 test 中互不重叠的类别；
- MASSIVE 的 dev unknown 用官方 `dev`，final unknown 用官方 `test`；
- 支持集/校准/测试的规范化文本交集为 0；
- CLINC150 原始 test 已查看，标为 historical，仅作追加分析。

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

已完成：Gate 0–3；3 数据集 × 4 编码器 × 2 shot × B0/B3-P/B3-H 的 72 格
test 矩阵；BANKING77 SetFit/Adapter 训练对照；数据集画像、热图、shot 曲线、
消融、阈值权衡、混淆、成本与错误案例；Streamlit Demo 真实截图。

Gate 3 的主要判断：

- B3-H 在 BANKING77 有正收益：OOS recall +5.21 pp，open-set F1 +0.99 pp；
- B3-H 在 MASSIVE 和 CLINC150 没有稳定收益；
- Qwen 平均最强，GTE 在 BANKING77 最强且编码成本低；
- SetFit 和投影适配器都没有超过冻结 BGE-B0；
- MASSIVE 的 F1-first 阈值选择在 BGE/GTE 上把 known acceptance 推到
  0.99+，几乎不拒 OOS，说明阈值目标本身是主要限制；本轮没有用 test
  重新调阈值。

尚未做：

- 3 个数据集之外的新独立 test / 多 seed 方差；
- B1/B3-P 的更多原型数、融合权重的系统消融；
- 近域/远域未知的可靠统一定义；
- 最终课程报告的精排、PPT 实际排版与演讲录像。

不会在当前 Gate 3 test 上继续调 B3 参数；若继续，需要新的独立 unknown
classes 或明确的新协议。
