# Gate 6 最终报告：Qwen3-Embedding LoRA、表征漂移与联合路由

**分支**：`gate6-qwen-lora`  
**基线**：`gate5-lora-agent@ba7c43f`  
**日期**：2026-11-18  
**模型**：`Qwen/Qwen3-Embedding-0.6B`，revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`

> **测试边界**：BANKING77 的 1,600 known + 760 final-unknown 已在 Gate 3/4/5 查看。
> 本报告的 test 表是**冻结 adapter/阈值后的历史探索性对照**，不是新盲测。
> 120 条 Agent 任务已在 Gate 5 用于系统修复，本轮只是**探索性回归**。

---

## 0. 一句话结论

**Qwen3-Embedding 的 LoRA 在 validation 上确实超过 frozen（B3-H 0.7580 vs 0.7404），但 historical test 没有复现（0.7972 vs 0.8064）；Anchor 把表征漂移从 0.892 提高到 0.936，却没有带来 test 收益。Qwen 上 B4-RejectOnly 仍然优于 B4-Full；Qwen-LoRA-RejectOnly 的业务 120 任务成功率 65.0%，高于 Qwen frozen/Plain-B3-H 的 62.5%/59.2%，但分类 test F1 仍低于 Frozen GTE RejectOnly 0.8435。**

---

## 1. A：Qwen 冻结对齐与可行性

### 1.1 环境

| 项目 | 值 |
|---|---|
| Python | 3.14.4 |
| torch | 2.11.0+cu128 |
| CUDA | 可用 |
| GPU | RTX 5070 Laptop，8,150.6 MiB |
| 起始空闲显存 | 7,012 MiB |
| transformers / PEFT | 5.23.0 / 0.21.3 |
| sentence-transformers / tokenizers | 6.3.0 / 0.23.3 |

模型本地加载 3.0 s；未上传任何 token 或私有缓存。

### 1.2 关键修正：right padding + last non-pad token

第一次用 **left padding** 直接编码，32 样本与 Gate3 cache 的 mean cosine 只有
0.412。原因不是模型坏了，而是 Qwen causal/RoPE 的位置处理与 Gate3
`SentenceTransformer` 不一致：

- Gate3 Qwen tokenizer 使用 **right padding**；
- `Pooling(lasttoken)` 使用最后一个非 pad token；
- prompt 不添加额外 `add_special_tokens` 差异。

改成 right padding + 最后一个非 pad token 后：

| 指标 | 值 |
|---|---:|
| 32 probe mean cosine | **0.999998** |
| min cosine | **0.999995** |
| top-1 intent consistency | **1.000** |
| max abs diff | 0.00043 |
| dim | 1024 |
| Gate3 B3-H validation replay F1 | **0.74140**（Gate3 0.74035） |
| Gate3 B0 validation replay F1 | 0.70970（Gate3 0.70953） |

**对齐通过后才开始训练。** 左 padding 的错误尝试保留在
`logs/gate6_qwen_alignment.log` / `gate6_qwen_pool_debug.log`。

### 1.3 LoRA 模块与 smoke

真实打印 `named_modules()`：`layers.*.self_attn.q_proj/k_proj/v_proj/o_proj`，
共 112 个 Linear 目标。LoRA r=8、alpha=16、dropout=0.05：

- 可训练参数 **2,293,760 / 598,070,272 = 0.38%**；
- 2-step smoke：前向、反向、optimizer、save/reload 全通过；
- 峰值 **1,213.6 MiB**；adapter 9,203,168 bytes；
- reload 向量最大差 **0.0**。

---

## 2. B：Plain / Anchor LoRA 与表征漂移

### 2.1 训练配方

400 条 10-shot support；seed 42；2 epochs；microbatch 2；
gradient accumulation 8；每个 optimizer step 有效 16 triplets；
lr 1e-4；SupCon + support-only hard negative triplet；Anchor 额外加
`0.5 × (1 - cos(z, frozen_qwen))`，teacher detached/只读。

| Run | Time | Peak | Adapter | Support cosine vs frozen | Test B3-H F1 |
|---|---:|---:|---:|---:|---:|
| Frozen Qwen | — | — | — | 1.000 | 0.8064 |
| Plain | 295.2 s | 1,259.9 MiB | 8.78 MB | 0.892 | 0.7972 |
| Anchor | 290.2 s | 1,259.9 MiB | 8.78 MB | **0.936** | 0.7968 |

### 2.2 Validation

| Encoder | B0 F1 | B3-H F1 | B3-H OOS | B3-H routing |
|---|---:|---:|---:|---:|
| Qwen Frozen | 0.7095 | 0.7404 | 0.5208 | 0.8213 |
| Qwen Plain | 0.7404 | **0.7580** | **0.6139** | 0.8213 |
| Qwen Anchor | 0.7275 | 0.7573 | 0.6097 | 0.8200 |

Plain/Anchor 的 validation 确实提升；Anchor 把漂移明显压低，但
validation 只比 Plain 低 0.07 pp，说明“更小漂移”本身没有带来更高指标。

### 2.3 Historical test

| Encoder | B0 F1 | B3-H F1 | B3-H OOS |
|---|---:|---:|---:|
| Qwen Frozen | 0.7821 | 0.8064 | 0.5855 |
| Qwen Plain | 0.7837 | 0.7972 | 0.5618 |
| Qwen Anchor | — | 0.7968 | 0.5355 |

**Qwen LoRA 的 validation 提升没有复现到 test。** 这是本轮最重要的负结果。
Anchor 的 test OOS 甚至低于 Plain；不能把“漂移更小”解释成“泛化更好”。

---

## 3. C：Qwen Frozen/LoRA × B4 联合矩阵

### 3.1 Validation

| Encoder | Method | F1 | OOS | Routing |
|---|---|---:|---:|---:|
| Qwen Frozen | B0 | 0.7095 | 0.4708 | 0.8038 |
| Qwen Frozen | B3-H | 0.7404 | 0.5208 | 0.8213 |
| Qwen Frozen | B4-RejectOnly | **0.7501** | 0.5514 | 0.8250 |
| Qwen Frozen | B4-Full | 0.7431 | 0.5431 | 0.8188 |
| Qwen Plain | B0 | 0.7404 | 0.5625 | 0.8200 |
| Qwen Plain | B3-H | 0.7580 | 0.6139 | 0.8213 |
| Qwen Plain | B4-RejectOnly | **0.7690** | **0.6417** | 0.8288 |
| Qwen Plain | B4-Full | 0.7644 | 0.6292 | 0.8250 |
| Qwen Anchor | B3-H | 0.7573 | 0.6097 | 0.8200 |

### 3.2 Historical test（冻结后一次）

| Encoder | Method | F1 | OOS | Routing |
|---|---|---:|---:|---:|
| Qwen Frozen | B0 | 0.7821 | 0.5500 | 0.8219 |
| Qwen Frozen | B3-H | 0.8064 | 0.5855 | 0.8413 |
| **Qwen Frozen** | **B4-RejectOnly** | **0.8127** | **0.6000** | **0.8438** |
| Qwen Frozen | B4-Full | 0.8047 | 0.5803 | 0.8363 |
| Qwen Plain | B0 | 0.7837 | 0.5250 | 0.8213 |
| Qwen Plain | B3-H | 0.7972 | 0.5618 | 0.8281 |
| Qwen Plain | B4-RejectOnly | 0.8080 | 0.5750 | 0.8413 |
| Qwen Plain | B4-Full | 0.7973 | 0.5711 | 0.8294 |
| Qwen Anchor | B3-H | 0.7968 | 0.5355 | 0.8363 |

### 3.3 回答 RQ2 / RQ3

- **RQ2：GTE-LoRA 退化是否与漂移有关？** Qwen 上出现了同样的模式：
  validation 提升、test 不提升；Anchor 把 support cosine 从 0.892 提到
  0.936，test 仍 0.7968（与 Plain 0.7972 基本持平）。**漂移不是唯一原因，
  也不是“漂移越小 test 越好”。**
- **RQ3：Qwen 上 RejectOnly 与 Full 谁更好？** RejectOnly 在 frozen 和 plain
  上都优于 Full；Qwen frozen 0.8127 vs 0.8047，plain 0.8080 vs 0.7973。
  **Gate5 的模块归因在 Qwen 上成立。**
- Qwen 最好 Qwen 内部结果是 Frozen-RejectOnly 0.8127，但仍低于 GTE
  Frozen-RejectOnly 0.8435；Qwen 0.6B 没有在 BANKING77 上超过 GTE。

---

## 4. D：Qwen Agent 与 120 任务回归

### 4.1 接入方式

- 新建 `app_gate6.py`，通过真实 Qwen encoder 在线编码，不是从 CSV 查分；
- 三个 Qwen-Plain LoRA 路由：B3-H、RejectOnly、B4-Full；
- 复用 Gate5 的 30 条虚构政策 KB、SQLite 工具、多轮状态机、所有权校验、
  确认和审计；未重写客服系统；
- 6 条真实模型回放（Q1–Q6，其中 5 条 Qwen-LoRA），真实数据库状态变化；
- LLM API 仍是 `NOT RUN_API_BLOCKED`（无凭据），与 Qwen LoRA 无关。

### 4.2 120 条旧任务探索性回归（4 个真实 Qwen 路由）

| Router | TSR | Tool+arg | Safe | Routing | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen-Frozen-B3H | 0.625 | 0.700 | 1.000 | 0.717 | 0.138 s | 0.257 s |
| Qwen-Plain-B3H | 0.592 | 0.708 | 1.000 | 0.650 | 0.220 s | 0.415 s |
| Qwen-Plain-B4Full | 0.625 | 0.742 | 1.000 | 0.692 | 0.257 s | 0.400 s |
| **Qwen-Plain-RejectOnly** | **0.650** | **0.767** | **1.000** | 0.708 | 0.244 s | 0.363 s |

**Qwen-LoRA-RejectOnly 的 65.0% 是四者中最高，也高于 Gate5 GTE-B3H 的
62.5%。** 同一任务、同一工具、同一 KB；但这是旧 120 条任务，不能称新盲测。

### 4.3 分任务族

| Family | Frozen-B3H | Plain-B3H | Plain-B4Full | Plain-RejectOnly |
|---|---:|---:|---:|---:|
| card status/activation | 0.840 | 0.720 | 0.800 | **0.880** |
| transfer query/cancel | 0.600 | 0.533 | 0.600 | **0.633** |
| dispute/ticket | **0.200** | 0.150 | 0.150 | 0.150 |
| policy FAQ | 0.900 | 0.900 | 0.900 | 0.900 |
| unknown/ambiguous | 0.667 | 0.667 | 0.667 | 0.667 |
| permission/safety | 0.400 | 0.600 | 0.600 | 0.600 |

### 4.4 失败原因

| Router | routing | tool/args | clarification |
|---|---:|---:|---:|
| Qwen-Frozen-B3H | 34 | 11 | 0 |
| Qwen-Plain-B3H | 42 | 2 | 5 |
| Qwen-Plain-B4Full | 37 | 3 | 5 |
| Qwen-Plain-RejectOnly | 35 | 2 | 5 |

- 安全操作率四个都是 **100%**；
- Qwen RejectOnly 的工具+参数正确率 **76.7%**，是四个中最高；
- 但 **5 条“应该澄清”的任务全部没有澄清**（clarification=0），这是
  Qwen 路由阈值偏高/直接接受造成的真实缺陷；Qwen Frozen-B3H 没有这 5 条
  失败，但工具失败 11 次。
- `dispute_ticket` 仍是所有版本最弱族（0.15–0.20）。

### 4.5 业务结论

- **分类 test F1 与业务 TSR 不是单调对应。** Qwen Plain-B3H 的 test F1
  0.797，低于 Frozen 0.806，TSR 也从 0.625 降到 0.592；但 Qwen
  Plain-RejectOnly test F1 0.808，TSR 升到 0.650。
- RejectOnly 带来的显式拒识减少了错误的金融动作，并提高了 card/transfer
  任务的成功率；但它没有解决 dispute 路由和澄清失败。
- 因而 Qwen-LoRA-RejectOnly 是**当前 Gate6 最好的业务配置**，不是
  “Qwen 全面优于 GTE”；分类 F1 仍然是 GTE-RejectOnly 最高。

---

## 5. GTE vs Qwen 横向对比

| 配置 | Val B3-H F1 | Test B3-H F1 | Test RejectOnly F1 | Agent TSR |
|---|---:|---:|---:|---:|
| GTE Frozen B3-H | 0.767 | 0.831 | — | 0.625 |
| GTE Frozen RejectOnly | 0.790 | — | **0.843** | NOT RUN |
| Qwen Frozen B3-H | 0.740 | 0.806 | — | 0.625 |
| Qwen Frozen RejectOnly | 0.750 | — | 0.813 | NOT RUN |
| Qwen Plain B3-H | **0.758** | 0.797 | — | 0.592 |
| Qwen Plain RejectOnly | 0.769 | — | 0.808 | **0.650** |

- Qwen 参数 0.6B，比 GTE 大；编码 4,280 条 **80–84 s / 1,186 MiB**，
  GTE 为 18.5 s / 618 MiB；Qwen Agent P50 0.14–0.26 s，GTE 0.05–0.10 s。
- Qwen 分类 F1 没有超过 GTE；Qwen LoRA 的业务 TSR 在 RejectOnly 上超过
  GTE-B3H，但不能推广为所有任务更好。
- 在 8GB 笔记本上 Qwen LoRA 可训练（峰值 1.26GB），但吞吐/延迟明显更差。

---

## 6. 资源与可复现

- Qwen LoRA 训练：2 epochs、50 steps、295 s、峰值 1,259.9 MiB；
- Qwen 4,280 条重编码：Plain 84.2 s / 50.8 texts/s；Anchor 80.5 s / 53.2 texts/s；
  峰值 1,185.6 MiB；
- adapter 9.20 MB，SHA：
  - Plain `4efaee5c891e26058764b4a7ec999dec718436f57b724841650831ddc01c3818`
  - Anchor `5c9c9a17b7bfb2604238391a476bcb6367b404fe6ad6bf0a3b92d84c463dd499`
- 所有阈值/checkpoint/数据 SHA 先写入
  `results/gate6_qwen_method_freeze.json`，test 一次后置 `test_evaluated=true`；
- 55 个测试通过。

---

## 7. 限制

- BANKING77 test 已看过；本表是历史探索性；
- 120 条 Agent 任务是旧任务，且已被 Gate5 用于一次系统修复；本轮没有改规则，
  但仍不是新盲测；
- Qwen LoRA 只做 2 epochs / 单 seed；没有做多 seed 方差和显著性检验；
- Anchor 只在一个 λ=0.5 下训练，没有系统调 λ；
- 5 条 clarif 任务 Qwen 全部直接接受，未澄清，是明确的 Agent 缺陷；
- LLM API 仍无凭据，`NOT RUN_API_BLOCKED`；
- 没有新的独立 unknown-class 协议；因此不能宣称 Qwen LoRA 的 test 退化
  是普遍规律。

---

## 8. 复现

```powershell
cd D:\code\intent-open-set
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
$env:HF_HUB_OFFLINE = "1"

# A
.\.venv\Scripts\python.exe -m src.gate6.inspect_qwen

# B
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_lora --run plain --epochs 2 --microbatch 2 --grad-accum 8
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_lora --run anchor --epochs 2 --microbatch 2 --grad-accum 8 --lambda-anchor 0.5
.\.venv\Scripts\python.exe -m src.gate6.encode_qwen_lora --run plain
.\.venv\Scripts\python.exe -m src.gate6.encode_qwen_lora --run anchor
.\.venv\Scripts\python.exe -m src.gate6.evaluate_qwen_lora

# C
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_b4 --encoder frozen --method reject
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_b4 --encoder frozen --method full
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_b4 --encoder plain --method reject
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_b4 --encoder plain --method full
.\.venv\Scripts\python.exe -m src.gate6.run_qwen_matrix

# D
.\.venv\Scripts\python.exe -m src.gate6.run_qwen_demos
.\.venv\Scripts\python.exe -m streamlit run app_gate6.py
# 4 router benchmark; each process isolated to avoid Windows page-file pressure
foreach ($n in @('Qwen-Frozen-B3H','Qwen-Plain-B3H','Qwen-Plain-RejectOnly','Qwen-Plain-B4Full')) {
  .\.venv\Scripts\python.exe -m src.gate6.benchmark_qwen --router $n
}
.\.venv\Scripts\python.exe -m src.gate6.combine_qwen_benchmark
.\.venv\Scripts\python.exe -m src.gate6.make_figures
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

关键结果：

- `results/gate6_qwen_alignment.json`
- `results/gate6_qwen_plain_training.json`
- `results/gate6_qwen_anchor_training.json`
- `results/gate6_qwen_lora_validation.csv`
- `results/gate6_qwen_method_matrix_val.csv`
- `results/gate6_qwen_method_matrix_historical_test.csv`
- `results/gate6_qwen_method_freeze.json`
- `results/gate6_qwen_benchmark_summary.csv`
- `results/gate6_qwen_benchmark_by_family.csv`
- `results/gate6_qwen_agent_demos.json`
