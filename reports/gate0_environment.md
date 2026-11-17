# Gate 0 环境与数据检查报告

**项目**：面向少样本智能客服的开放意图识别与自适应拒识方法研究  
**数据**：CLINC150 / OOS-Eval  
**执行地点**：本机 Windows 11，项目目录 `D:\code\intent-open-set`  
**执行日期**：2026-11-17（Asia/Shanghai）  
**状态**：Gate 0 的真实运行项已完成；**Gate 1 尚未开始**。  
**结论**：Gate 0 的环境、数据、20 类选择和 K-shot manifest 可以验收；
建议进入 Gate 1 时先只做 **BGE CPU B0**，Qwen 单独批准后先做小 batch 冒烟，
不要直接下载全量权重并跑实验。

> 本报告没有 B0/B1/B2/B3 的任何准确率、F1、AUROC 或阈值效果数字。
> 唯一的模型运行是 **BGE-small-en-v1.5 的 3 条文本 CPU 冒烟测试**，
> 它只证明环境能加载模型并产生合法向量，不是 baseline 结果。
> Qwen3-Embedding-0.6B **没有下载、没有加载、没有测显存**。

---

## 1. 对研究问题和四种方法的理解

### 1.1 问题

智能客服只支持一组已知业务意图；用户还会提出不属于这些业务的问题（OOS /
unknown）。少样本条件下，每类只有 K 条带标签支持样本。系统需要同时做到：

1. 把已知意图分到正确类别；
2. 对未知意图拒绝，而不是强行分到某个已知类；
3. 在“已知意图被误拒”和“未知意图被误接收”之间做可解释的权衡。

四个研究问题：

- **RQ1**：Qwen3-Embedding-0.6B 与轻量 BGE-small-en-v1.5 在 K-shot 已知分类
  和 OOS 拒识上的差异；
- **RQ2**：单原型是否因类内多样性失效；自适应多原型是否对特定类别有帮助；
- **RQ3**：统一阈值是否适合所有类别；类别感知阈值 + Top1/Top2 间隔门控能否
  在已知接受率约束下提高 OOS 拒识；
- **RQ4**：组合策略在什么数据条件下有效、无效或退化，能否用类别分散度、
  相似度间隔和错误案例解释。

### 1.2 四种方法（严格按任务书）

记 L2 归一化文本向量为 `z(x)`，类别 `c` 的 support 向量均值为
`p_c = norm(mean_i z(x_i))`，相似度 `s_c(x) = cos(z(x), p_c)`，
`s1` / `s2` 为最高/次高类别分数。

- **B0：单原型 + 全局阈值**
  - 每类一个原型 `p_c`；
  - 取 `s1`；
  - 若 `s1 < tau` 输出 OOS，否则输出 Top-1 类别；
  - `tau` 只在固定 validation 上选择。

- **B1：自适应多原型 + 全局阈值**
  - 先用 support 计算类内分散度
    `d_c = mean_i [1 - cos(z(x_i), p_c)]`；
  - 在已知类内按 `d_c` 排序，首轮只对最高 25% 的类做
    `KMeans(n_clusters=2, random_state=seed, n_init=10)`，两个中心再 L2 归一化；
  - 这些类 `s_c = max_j cos(z(x), p_cj)`；其余类仍用单原型；
  - KMeans 退化、K<2、重复样本时回退单原型并记录；
  - 阈值仍为全局 `tau`。

- **B2：单原型 + 类别感知阈值 + 间隔门控**
  - 对每个已知类 `c`，用 validation 中真实标签为 `c` 的样本，计算该类别
    自身分数的 10% 分位数 `q_c`；这需要 **K-shot support 之外的额外带标签
    validation 数据**，报告中必须说明额外监督成本；
  - `tau_c = (1 - lambda) * tau + lambda * q_c`，`lambda ∈ {0, 0.5, 1}`；
  - `m = s1 - s2`；
  - 接受条件：`s1 >= tau_predicted_class` **且** `m >= gamma`，否则拒绝；
  - `tau`、`lambda`、`gamma` 只在 validation 上选，不能看 test；
  - 必须同时报告 `lambda=0, gamma=0` 的退化基础情况。

- **B3：B1 + B2**
  - 用 B1 的自适应多原型分数替换 B2 的单原型分数；
  - 仍用类别感知阈值和 margin 门控；
  - 与 B0/B1/B2 在相同 validation 上公平校准。

### 1.3 防泄露协议

- **support**：只来自官方 `train`，每类 K 条，K 只计这些原型支持样本。
- **calibration**：官方 `val` 的 20×20=400 条已知样本 + 官方 `oos_val` 100 条；
  B2 的 `q_c` 使用其中的真实标签，属于额外监督。
- **final test**：官方 `test` 的 20×30=600 条已知样本 + 官方 `oos_test`
  1000 条未知样本；固定配置后只读一次。
- **`oos_train` 不使用**于原型训练。
- 测试集不能用于方法选择、prompt 选择、阈值调优、挑 seed，也不能看了
  错误样本后反向修改方法再宣称 test 结果。
- `Known Acceptance Rate` 是“已知样本中预测非 OOS 的比例”，不是分类正确率；
  `Known Macro-F1` 要把已知误拒计为错误；`Open-set Macro-F1` 是 20 已知类 +
  OOS 共 21 类，必须同时列出分类/拒识分项。

---

## 2. 环境检查

### 2.1 硬件与系统

| 项目 | 实测值 |
|---|---|
| 操作系统 | Windows 11，`10.0.26200` |
| CPU | AMD Ryzen 9 8940HX，16 核 / 32 逻辑处理器 |
| 内存 | 15,583.3 MiB 总物理内存；检查时可用 1,586.9–1,830.0 MiB；负载 88–89% |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU，8,151 MiB |
| GPU 计算能力 | 12.0（sm_120） |
| 驱动 | 596.13，nvidia-smi 报告 CUDA 13.2 |
| 检查时空闲显存 | 7,012 MiB |
| C 盘 | 191.24 GiB 总，10.91 GiB 空闲 |
| D 盘 | 439.45 GiB 总，61.77 GiB 空闲 |
| E 盘 | 292.97 GiB 总，21.62 GiB 空闲 |

**D 盘足够放当前数据、BGE 和 Qwen 权重；C 盘只剩约 11 GiB，项目全部落在
D 盘。** 当前最大的环境风险不是磁盘，而是系统内存只剩约 1.6–1.8 GiB，
且内存负载已接近 90%。

### 2.2 Python 环境

| 项目 | 实测值 |
|---|---|
| Python | 3.14.4 |
| 项目解释器 | `D:\code\intent-open-set\.venv\Scripts\python.exe` |
| 项目 venv | 已创建；`include-system-site-packages = true` |
| 基环境 | `D:\Environment` |
| torch | 2.11.0+cu128 |
| torch CUDA | `torch.cuda.is_available() == True`，build CUDA 12.8 |
| cuDNN | 91900 |
| NumPy | 2.4.4 |
| Pandas | 3.0.2 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.1 |
| sentence-transformers | 6.3.0 |
| transformers | 5.23.0 |
| huggingface-hub | 2.4.0 |
| tokenizers | 0.23.3 |
| safetensors | 0.8.0 |
| PyYAML | 6.0.3 |
| psutil | 7.2.2 |

`pip check` 结果：

```text
No broken requirements found.
```

**环境隔离的如实说明**：这是项目专用 venv，新增包安装在 `.venv`；但 torch
和部分数值包是从 `D:\Environment` 继承的，不是完全 hermetic 的环境。
原因是本机已有通过 CUDA 验证的 torch 2.11.0+cu128，重新安装一套 CUDA torch
会多占数 GB，而且任务书要求在 Gate 0 避免不必要的大体积安装。完整继承包
见 `pip_freeze.txt`；项目直接依赖的精确版本见 `requirements_lock.txt`。
如果基环境的 torch 被改动，本项目环境可能失效，因此后续不要升级基环境。

### 2.3 CUDA 可用性冒烟

为确认不是只“导入 torch 成功”，额外跑了一个 1024×1024 FP16 矩阵乘：

| 项目 | 实测 |
|---|---|
| 设备 | RTX 5070 Laptop |
| shape | 1024×1024 |
| 有限值 | true |
| 用时 | 0.6545 s |
| 峰值 allocated | 17.1 MiB |
| 峰值 reserved | 24.0 MiB |

这只是 CUDA kernel 冒烟，**不是 Qwen 显存测试**。

### 2.4 环境日志

- `logs/gate0_venv_create.log`
- `logs/gate0_pip_install.log`
- `logs/gate0_pip_check.log`
- `logs/gate0_pip_freeze.log`
- `logs/gate0_environment.log`
- `logs/gate0_cuda_smoke.log`
- JSON：`data/manifest/environment.json`

---

## 3. 数据来源与结构

### 3.1 固定来源

| 项目 | 值 |
|---|---|
| 官方仓库 | <https://github.com/clinc/oos-eval> |
| 分支 | `master` |
| 固定 commit | `828f8093932c8fe6ca7936c3d2e52903b1c523de` |
| commit 日期 | 2021-06-01 |
| 数据文件 | `data/data_full.json` |
| 大小 | 2,495,390 bytes |
| SHA-256 | `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0` |
| domains 文件 | `data/domains.json`，SHA-256 `b947b579...e1211b3a` |
| 许可证 | 官方 `LICENSE` 为 **CC BY 3.0** |
| 下载方式 | 固定 commit 的 raw URL，见 `src/download_data.py` |

**没有使用**缩小版、重采样版、UCI 二次加工版或 `data_small.json`。

### 3.2 实测 JSON 结构

顶层是对象，实际键为：

```text
oos_val, val, train, oos_test, test, oos_train
```

每个值是 `list`，每条记录是 `[text: str, label: str]`。实测计数：

| 分区 | 条数 | 已知类标签 | OOS 标签 |
|---|---:|---|---|
| `train` | 15,000 | 150 类，每类 100 | 无 |
| `val` | 3,000 | 150 类，每类 20 | 无 |
| `test` | 4,500 | 150 类，每类 30 | 无 |
| `oos_train` | 100 | 无 | 全部 `oos` |
| `oos_val` | 100 | 无 | 全部 `oos` |
| `oos_test` | 1,000 | 无 | 全部 `oos` |

`domains.json` 把 150 个已知意图完整划分为 10 个域，每域 15 类，无遗漏、
无重复、与 train/val/test 的已知标签集合一致。10 个域为：

```text
auto_and_commute, banking, credit_cards, home, kitchen_and_dining,
meta, small_talk, travel, utility, work
```

数据文件、domains、LICENSE 的字节数和 SHA-256 同时写入：

- `data/raw/download_manifest.json`
- `data/manifest/source_manifest.json`

---

## 4. 20 类选择与 K-shot 冻结

### 4.1 选择规则

- 10 个官方域，每域选 2 个类，共 20 类；
- 在每个域内，对类名计算
  `SHA-256("class-selection|42|<domain>|<intent>")`，按十六进制升序，
  取前 2 个；
- 这是**确定性哈希排序**，不依赖 Python `random` 实现，也不使用任何
  validation/test 标签或指标；
- 选完后写死在 `selected_intents.json`，所有方法共用，不能因为 test
  表现重新挑选。

### 4.2 冻结的 20 类

| 域 | 类别 1 | 类别 2 |
|---|---|---|
| auto_and_commute | `uber` | `directions` |
| banking | `balance` | `routing` |
| credit_cards | `report_lost_card` | `credit_score` |
| home | `reminder` | `reminder_update` |
| kitchen_and_dining | `meal_suggestion` | `restaurant_reviews` |
| meta | `user_name` | `change_ai_name` |
| small_talk | `who_made_you` | `are_you_a_bot` |
| travel | `plug_type` | `exchange_rate` |
| utility | `measurement_conversion` | `spelling` |
| work | `insurance` | `insurance_change` |

需要预先说明：`reminder` / `reminder_update` 和 `insurance` /
`insurance_change` 语义接近，可能天然难分。这是 **按固定规则选出的结果**，
不能因为 test 表现不好而替换。后续分析应把这类近义类别作为误差来源之一。

### 4.3 Support 抽样

- 只从官方 `train` 中选；
- 每个 support seed、每个类，按
  `SHA-256("support|<seed>|<intent>|<sample_id>")` 升序排；
- 取前 10 条，K=5 就是前 5 条，因此 **K=5 是 K=10 的嵌套子集**；
- support seeds：42、43、44；shots：5、10；
- 每类每个 seed/shot 恰好 K 条，不含 OOS，不含 val/test；
- 如果某条 train 文本的规范化形式也出现在官方 eval 分区，会从 support
  候选中排除并记录。**本轮 20 类的 support 候选中没有需要排除的样本**
  （`n_excluded_candidates = 0`），但过滤逻辑保留。

实测 support manifest：

| seed | shot | 行数 | 每类行数 |
|---:|---:|---:|---:|
| 42 | 5 | 100 | 5 |
| 42 | 10 | 200 | 10 |
| 43 | 5 | 100 | 5 |
| 43 | 10 | 200 | 10 |
| 44 | 5 | 100 | 5 |
| 44 | 10 | 200 | 10 |

总计 900 行；所有行 `source_split=train`。

### 4.4 Calibration / test 索引

| 视图 | 行数 | 用途 |
|---|---:|---|
| known val（20 类全部） | 400 | 阈值、lambda、gamma 选择；B2 的 `q_c` |
| OOS val | 100 | 阈值校准 |
| known test（20 类全部） | 600 | 最终测试，只读一次 |
| OOS test | 1,000 | 最终测试，只读一次 |
| **eval manifest 总计** | **2,100** | 400 + 600 + 100 + 1000 |

`oos_train` 的 100 条没有被放入任何 support 或 calibration 文件。

### 4.5 Manifest 文件

| 文件 | 内容 |
|---|---|
| `data/manifest/selected_intents.json` | 20 类、域映射、冻结规则、每类 100/20/30 |
| `data/manifest/split_counts.csv` | 官方各分区总数与选中数 |
| `data/manifest/source_manifest.json` | 来源、commit、哈希、计数、选择/支持/测试政策 |
| `data/manifest/support_manifest.csv` | 3 seeds × 2 shots 的 900 条 support 明细 |
| `data/manifest/eval_manifest.csv` | 2,100 条 calibration/test 索引 |
| `data/manifest/leakage_report.json` | 分区内/跨分区重复、support/eval 重叠 |
| `data/manifest/support_leakage_exclusions.json` | 被排除的 support 候选（本轮为空） |

关键哈希（用于后续确认文件未被误改）：

| 文件 | SHA-256 |
|---|---|
| `selected_intents.json` | `bc525eb6da4fc25673f75b3fc8579d9ea3ec12f69409b285d4249e6bc7239681` |
| `support_manifest.csv` | `48984302fc690aa3d5bde80ea8277a78c7013665154999a706f138ae5725196e` |
| `eval_manifest.csv` | `87051244de72f8914d8dfa8311b0ed4c3b51fdf398205d0be4f812880eec5a54` |
| `split_counts.csv` | `3904adaafc53f8e9a10034670baab6f5cf864e6a02acbe88e58d2cb46d21e080` |
| `source_manifest.json` | `a5ca08981ecf2c9084b475843a5408b028e6ce1ace68f03a44f298237a8d6674` |
| `leakage_report.json` | `d6247471df57cd8d3b2a1a093887643a4b8ceb36d036a0ff3c71d907539de947` |

---

## 5. 数据质量与泄漏检查

### 5.1 分区内

6 个官方分区内部均为：

- 精确重复组：0；
- 规范化重复组：0；
- 空文本：0（读取时校验）；
- 非 `[text, label]` 记录：0。

### 5.2 跨分区：官方数据确实有 5 处重复

规范化规则是 NFKC + 去首尾空白 + 小写 + 连续空白折叠。实测：

| 分区对 | 共享规范化文本数 | 例子 |
|---|---:|---|
| `train` ↔ `val` | 3 | `what is on my to do list`（train `todo_list` / val `reminder`） |
| `train` ↔ `test` | 2 | `what's your designation`（train `what_is_your_name` / test `user_name`） |

全部 5 处：

1. `clinc150:train:07424` `todo_list` ↔ `clinc150:val:01011` `reminder`
   — `what is on my to do list`
2. `clinc150:train:11130` `whisper_mode` ↔ `clinc150:val:01794` `change_volume`
   — `turn up your volume`
3. `clinc150:train:11896` `greeting` ↔ `clinc150:val:02369` `greeting`
   — `hey what's up`
4. `clinc150:train:12066` `what_is_your_name` ↔ `clinc150:test:00938` `user_name`
   — `what's your designation`
5. `clinc150:train:14018` `how_old_are_you` ↔ `clinc150:test:00599`
   `where_are_you_from` — `where did you grow up`

其中第 1 处涉及选中的 `reminder`（val 侧），第 4 处涉及选中的 `user_name`
（test 侧）。但它们的 train 重复样本分别标为 `todo_list` 和
`what_is_your_name`，**不在 20 类 support 中**。

### 5.3 处理决定

- **没有删除、改写、翻译或重标任何官方样本**；
- 没有修改官方 train/val/test 划分；
- support 候选若与官方 eval 文本重复，会被排除并写入
  `support_leakage_exclusions.json`；本轮排除 0 条；
- 构造出的 support 与 calibration/test 之间：
  - sample ID 交集：0；
  - 规范化文本交集：0；
  - 6 个 seed/shot 组合全部为 0。

### 5.4 残余风险

1. 官方 `user_name` 的一个 test 文本在 train 中存在另一个标签的精确重复。
   我们不在 full train 上训练，只用 20 类 support，因此该 train 行不会进入
   本项目原型；但这是上游数据的残余缺陷，最终报告必须披露。
2. 规范化文本匹配不能发现“换述/同义改写”级别的泄漏。
3. 预训练编码器是否在预训练阶段见过 CLINC150 无法从本地验证；BGE/Qwen
   都不是本项目训练的，但“预训练污染”只能作为限制说明，不能声称绝对没有。
4. 因此不能写“官方 CLINC150 完全没有 train/test 重复”；只能写
   “本项目构造的 support/eval 集合没有 ID 或规范化文本重叠，官方数据存在
   5 处跨分区碰撞并已记录”。

---

## 6. BGE 最小冒烟测试（真实运行）

### 6.1 模型

| 项目 | 值 |
|---|---|
| 模型 | `BAAI/bge-small-en-v1.5` |
| revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| 许可证 | MIT |
| 参数量 | 33,360,000 |
| 向量维度 | 384 |
| max_seq_length | 128 |
| 设备 | CPU |
| dtype | float32 |
| 缓存目录 | `D:\code\intent-open-set\data\model_cache` |
| 实际缓存 | 134,505,940 bytes（约 128.27 MiB） |

### 6.2 输入与输出

输入 3 条英文短句：

```text
I need to reset my card PIN.
My card was swallowed by the ATM.
What is the weather in Paris?
```

实测：

| 项目 | 值 |
|---|---|
| 输出 shape | `(3, 384)` |
| 有限值 | `True` |
| 三个向量 L2 norm | `[1.0, 1.0, 1.0]` |
| norm 与 1 的最大绝对误差 | `0.0` |
| 首次加载（含下载） | 67.898 s |
| 缓存后加载 | 2.947 s |
| 首次 3 条编码 | 0.171 s |
| 缓存后 3 条编码 | 0.126 s |
| 进程 RSS（缓存后） | 752.3 → 934.7 MiB |
| 是否使用 CUDA | 否 |
| Qwen 缓存 | 不存在 |

余弦矩阵（保留 6 位）：

```text
[[1.000000, 0.685914, 0.325725],
 [0.685914, 1.000000, 0.304475],
 [0.325725, 0.304475, 1.000000]]
```

**解释边界**：这 3 条文本只用于验证加载、维度、有限值、归一化和基本语义
相似度；不能据此报分类准确率，也不能把 0.171 s / 0.126 s 外推为全量
support + val + test 的编码时间。全量编码必须在 Gate 1 实测。

日志：`logs/gate0_models.log`；JSON：
`data/manifest/model_resource_manifest.json`。

---

## 7. Qwen3-Embedding-0.6B 资源与依赖检查（未下载）

### 7.1 元数据

| 项目 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3-Embedding-0.6B` |
| revision | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| 许可证 | Apache-2.0 |
| pipeline | feature-extraction |
| 仓库总文件 | 1,151.55 MiB（1.207 GB decimal） |
| `model.safetensors` | 1,191,586,416 bytes = 1,136.39 MiB = 1.192 GB decimal |
| 权重是否下载 | **否** |
| 是否加载/前向 | **否** |
| 是否测过显存 | **否** |
| 本项目 Qwen 缓存 | 不存在 |

### 7.2 固定 Qwen prompt

模型卡说明 query 侧加 instruction 通常有 1%–5% 收益，并建议按任务定制。
Gate 0 固定了以下 prompt（不是模型卡自带的 web-search prompt）：

```text
Instruct: Given a customer service message, identify the supported intent it expresses.
Query: 
```

- SHA-256：`4bbdf1c169061f64ba742f64bca5e422d05e94ccc7261d29495c233ed9a90005`
- 范围：support / query / validation / test **全部使用同一前缀**，避免
  support 与 query 采用不兼容的文本处理；
- BGE 不使用这个 prompt，也不强加 Qwen instruction；
- prompt 已写在 `configs/pilot.yaml`，后续编码缓存要记录 prompt hash；
- 没有用 test 表现选择 prompt。

### 7.3 依赖兼容性判断

Qwen 模型卡要求：

- `transformers >= 4.51.0`
- `sentence-transformers >= 2.7.0`

本机：

- `transformers 5.23.0`（满足最低版本，但**高于**模型卡写作时的版本，
  属于未实测风险）；
- `sentence-transformers 6.3.0`（满足）；
- `torch 2.11.0+cu128` 在 sm_120 上 CUDA 可用（已用矩阵乘验证）。

因此“依赖版本下限”满足，但 **Qwen 架构 + transformers 5.23 + ST 6.3
的实际加载与输出尚未验证**。Gate 1 必须先做单条/小 batch 冒烟。

### 7.4 显存与内存风险

规划值（**不是实测**）：

- FP16 权重约 1.14 GiB；
- batch 4/8、max_seq_length=128；
- 预留 3–6 GiB 显存作为规划起点。

本机现实：

- 8,151 MiB 显存，检查时空闲 7,012 MiB；
- 系统内存 15.58 GiB，检查时只有约 1.6–1.8 GiB 可用，负载 88–89%；
- Windows WDDM 可能把显存压力转到共享内存，峰值不能靠“8GB 一定够”判断。

结论：**8 GiB 显存名义上够 0.6B FP16 小 batch，但系统内存余量很低，
Qwen 可能在加载/分词/CPU 侧 OOM 或严重换页。** Gate 1 应先关闭不必要的
前台程序，用 batch=1/2/4 做冒烟，记录 `torch.cuda.max_memory_allocated()`、
`max_memory_reserved()`、耗时和输出维度；失败则优先实验室单张
3090/4090，或先完成 BGE 的 B0/B2，不能假报 Qwen 已跑通。

---

## 8. 实际运行命令与验收

### 8.1 已真实执行的命令

```powershell
cd D:\code\intent-open-set

.\.venv\Scripts\python.exe -m src.download_data
.\.venv\Scripts\python.exe -m src.prepare_data --seed 42 --known-classes 20 --per-domain 2
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
.\.venv\Scripts\python.exe -m src.check_environment
.\.venv\Scripts\python.exe -m src.check_models --bge-smoke
```

实际结果：

- `SOURCE_DOWNLOAD_OK`
- `PREPARE_DATA_OK`
- `Ran 15 tests ... OK`
- `ENVIRONMENT_CHECK_OK`
- `MODEL_CHECK_OK`

### 8.2 第一条验收命令

如果只复核当前 Gate 0 产物，先运行：

```powershell
cd D:\code\intent-open-set
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

成功时应看到：

```text
Ran 15 tests in ...s

OK
```

测试覆盖：

- 20 类、10 域、每域 2 类；
- 官方计数与数据哈希；
- support 每类恰好 K 条、只来自 train；
- K=5 是 K=10 子集；
- support 与 eval 的 ID/文本交集为 0；
- 400/600/100/1000 的 eval 计数；
- 21 个 open-set 标签；
- 官方 5 处跨分区碰撞被报告；
- 从 domains + seed 重新选择能得到同一 20 类；
- 重新生成 support/eval/split 文件字节级一致；
- `configs/pilot.yaml` 与冻结 manifest 一致；
- Qwen prompt hash 与配置一致。

### 8.3 需要反馈给助手的日志

优先贴回：

1. `logs/gate0_tests.log`（最后 30 行即可）；
2. `data/manifest/environment.json`；
3. 如重跑模型：`logs/gate0_models.log` 最后 20 行；
4. 如重跑数据：`logs/gate0_prepare.log` 中 `PREPARE_DATA_OK` 和选择列表。

如果要反馈错误，请同时给出完整 traceback，不要只写“报错了”。

---

## 9. 验收清单

| Gate 0 要求 | 状态 | 证据 |
|---|---|---|
| 独立 Python 环境 | 已完成（venv；复用基环境 torch，已披露） | `.venv`、`environment.json` |
| Windows/GPU/RAM/磁盘检查 | 已完成 | `environment.json` |
| 可用 torch / CUDA / ST | 已完成 | CUDA 冒烟、BGE 冒烟 |
| 获取官方 `data_full.json` | 已完成 | `data/raw/data_full.json`，哈希 |
| 打印真实分区结构和计数 | 已完成 | `gate0_download.log`、`source_manifest.json` |
| 固定 20 类 | 已完成 | `selected_intents.json` |
| K-shot 抽样与 manifest | 已完成 | `support_manifest.csv`，900 行 |
| 固定随机种子 | 已完成（哈希排序，seed 42/43/44） | 规则写入 manifest |
| 检查 train/val/test 重叠 | 已完成，发现 5 处官方碰撞并报告 | `leakage_report.json` |
| 检查模型缓存/下载体积 | 已完成 BGE；Qwen 仅元数据 | `model_resource_manifest.json` |
| BGE 最小编码例子 | 已完成（CPU，3 条） | `bge_smoke_test` |
| `requirements_lock.txt` / `pip_freeze.txt` | 已完成 | 项目根目录 |
| 说明 Qwen 下载与显存风险 | 已完成 | 本报告第 7 节 |
| 不下载 Qwen 权重 | 已遵守 | 本项目 Qwen cache 不存在 |
| 不报虚构准确率 | 已遵守 | 本报告无任何 B0–B3 指标 |

### Gate 0 结论

- **数据来源、结构、计数、20 类、K-shot、support/eval 隔离**：已建立并可复现；
- **环境**：NumPy/Pandas/BGE 最小 CPU 闭环已实际跑通；
- **Qwen**：依赖下限满足、显存/内存风险已说明，但**未下载、未加载、未实测**；
- **官方数据重复**：5 处跨分区碰撞已如实报告，未删除；构造的
  support/eval 无交集；
- **Gate 1 建议**：可以进入 Gate 1 的 BGE 部分；Qwen 部分建议单独立项、
  先做小 batch 冒烟，且必须重新获得用户批准。

---

## 10. 尚未真实运行的内容

以下均为 **Gate 1 或以后**，当前没有任何结果：

- B0 的单原型、全局阈值、validation 选择、test 评测；
- B1 的类内分散度、KMeans 多原型、回退逻辑；
- B2 的类别分位数 `q_c`、`lambda`/`gamma` 搜索、margin 门控；
- B3 的组合；
- 任何准确率、Macro-F1、OOS Recall、AUROC、混淆矩阵、错误案例；
- Qwen 权重下载、加载、GPU 前向、显存峰值；
- 全量 BGE/Qwen 嵌入缓存；
- 图表、Streamlit、最终报告和 PPT。

**不要**把本报告的 BGE 冒烟向量或耗时当作 B0/B2 结果。

---

## 11. 问题与限制汇总

1. **系统内存余量低**（约 1.6–1.8 GiB 可用，负载 88–89%），Qwen 的主要风险
   在主机内存而不是 8 GiB 显存。
2. **venv 非完全 hermetic**：复用基环境 torch 2.11.0+cu128；基环境变更会
   影响项目。当前 `pip check` 无冲突。
3. **官方 CLINC150 有 5 处跨分区规范化文本碰撞**；未删除。构造的
   support/eval 无交集，但 `user_name` 的 test 样本文本在 train 中有另一标签
   的重复，属于上游残余风险。
4. **Qwen 未实测**：transformers 5.23 高于模型卡最低版本，存在 API/加载
   兼容性风险。
5. **BGE CPU 首次加载 67.9 s 含下载**；缓存后 2.95 s。全量编码时间未知。
6. **近义类别**（`reminder`/`reminder_update`、`insurance`/`insurance_change`）
   可能造成混淆；这是固定选择的结果，不能看 test 后替换。
7. **预训练污染不可验证**：BGE/Qwen 的预训练语料是否含 CLINC150 无法从本地
   确认，最终报告应作为限制说明。
8. **GitHub API 在准备阶段触发过未认证限流**；数据下载改为固定 commit 的
   raw URL，不影响数据哈希和复现。

---

## 12. 交付物索引

- 环境报告：`reports/gate0_environment.md`（本文件）
- 代码修改记录：`reports/gate0_vibe_coding.md`
- 环境 JSON：`data/manifest/environment.json`
- 数据来源：`data/manifest/source_manifest.json`
- 20 类：`data/manifest/selected_intents.json`
- 分区计数：`data/manifest/split_counts.csv`
- Support：`data/manifest/support_manifest.csv`
- Eval：`data/manifest/eval_manifest.csv`
- 泄漏：`data/manifest/leakage_report.json`
- 模型资源：`data/manifest/model_resource_manifest.json`
- 依赖：`requirements_lock.txt`、`pip_freeze.txt`
- 测试：`tests/`
- 日志：`logs/gate0_*.log`
