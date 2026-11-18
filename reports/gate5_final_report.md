# Gate 5 最终报告：LoRA-GTE、B4 模块归因、RAG Agent 与 120 任务 Benchmark

**分支**：`gate5-lora-agent`  
**起点**：`gate4-train-agent@81e81a9`  
**日期**：2026-11-18  
**主实验**：BANKING77，40 known / 18 dev-unknown / 19 final-unknown，GTE-ModernBERT-base，10-shot

> **测试边界**：BANKING77 的 1,600 known + 760 final unknown 已在 Gate 3/4 查看。
> Gate 5 只在所有 adapter/阈值/方法冻结后做一次 `historical_exploratory` 对照；
> 它不是新的盲测。120 条 Agent 业务任务是在 Gate 5 新建并冻结的模拟任务集，
> 不是真实银行线上数据。

---

## 1. 一句话结论

**只训练拒识头（B4-RejectOnly）是本轮唯一真正改善开放集指标的新模块；LoRA-GTE 没有带来收益，B4-Full 的门控 + 拒识头也不如 RejectOnly；在 120 条业务 benchmark 上，Frozen-B3-H 的任务成功率最高（62.5%），模型侧的微小 F1 改善没有转化为业务成功。**

---

## 2. A：真实 LoRA-GTE 训练

### 2.1 结构与环境

- 基座：`Alibaba-NLP/gte-modernbert-base`，revision `e7f32e3...`；
- 通过 `named_modules()` 检查实际层名，而不是照抄 q_proj/v_proj；
- 注入模块：`Wqkv`（attention QKV）、`Wo`（attention/MLP output）、`Wi`（MLP input）；
- `r=8`、`alpha=16`、`dropout=0.05`、CLS pooling、L2 归一化、max_length=128；
- 可训练参数 **1,689,600 / 150,703,872 = 1.12%**；
- 首次 1-batch 冒烟验证了 forward、backward、optimizer step、save/reload；reload 向量差 0。

### 2.2 训练

- 仅使用 400 条 10-shot support；
- 主损失：SupCon（temperature 0.07）；
- 困难负样本：在 support 内按冻结向量余弦挖掘不同标签的最近样本，保存到
  `results/gate5_lora_hard_negatives.csv`；
- 附加 `0.5 ×` hard-negative triplet loss（margin 0.10）；
- 2 epochs、8 triplets/batch、AdamW lr 2e-4、seed 42；
- 训练 **22.19 s**，峰值 **1,368 MiB**；adapter **6.78 MB**；
- 2-epoch adapter SHA：`f15f68beaeb9cf3db431ba07e801c2f17442353930e733e9e2917c93c44ed113`。

**4-epoch 第一次尝试**：B3-H validation F1 0.7397，明显低于 2-epoch 的 0.7605；
4-epoch adapter 未进入最终矩阵，指标保留在 `results/gate5_lora_4ep/`，并在
Vibe 记录中说明这次针对性调整。没有用 test 做选择。

### 2.3 编码与向量检查

- LoRA 重编码 BANKING77 五个角色：4,280 条，18.53 s，峰值 618 MiB；
- 与冻结 GTE 的同一批 support 向量平均余弦 **0.716**（2 epoch），说明权重确实
  改变了表征；不是把旧向量换个文件名；
- 同一 adapter 重新加载后，8 条 probe 向量最大差 **0.0**；
- 缓存路径按 `base_revision + adapter_hash + prompt_hash` 区分，绝不覆盖 Gate 3。

### 2.4 LoRA validation

| Encoder | Method | Known acceptance | Known routing | OOS recall | Open-set F1 |
|---|---|---:|---:|---:|---:|
| Frozen | B0 | 0.920 | 0.860 | 0.403 | 0.734 |
| Frozen | B3-H | 0.900 | 0.851 | 0.547 | **0.767** |
| LoRA | B0 | 0.900 | 0.828 | 0.574 | 0.753 |
| LoRA | B3-H | 0.900 | 0.828 | 0.607 | 0.760 |

LoRA-B0 的 validation F1 比 Frozen-B0 高 1.94 pp，但 **LoRA-B3-H 比 Frozen-B3-H
低 0.66 pp**，known routing 从 0.851 降到 0.828。LoRA 改善了部分 OOS 拒识，
却损伤了已知类路由；这不是“全面变好”。

---

## 3. B：RejectOnly 与六组矩阵

### 3.1 三种模块

- `B3-H`：固定 `S=0.5P+0.5N`，无训练式拒识头；
- `B4-RejectOnly`：**固定 `g=0.5`**，只训练 `7→16→1` 拒识头；gate 参数冻结；
- `B4-Full`：训练 gate + 拒识头；
- `B4-Fusion`：只训练 gate，保留作消融。

### 3.2 Validation（9 行，含 B4-Fusion）

| Encoder | Method | Known routing | OOS recall | Open-set F1 |
|---|---|---:|---:|---:|
| Frozen | B0 | 0.860 | 0.403 | 0.734 |
| Frozen | B3-H | 0.851 | 0.547 | 0.767 |
| Frozen | B4-Fusion | 0.844 | 0.471 | 0.740 |
| Frozen | B4-RejectOnly | **0.861** | **0.607** | **0.790** |
| Frozen | B4-Full | 0.853 | 0.604 | 0.781 |
| LoRA | B0 | 0.828 | 0.574 | 0.753 |
| LoRA | B3-H | 0.828 | 0.607 | 0.760 |
| LoRA | B4-RejectOnly | 0.834 | 0.608 | 0.767 |
| LoRA | B4-Full | 0.829 | 0.549 | 0.753 |

### 3.3 冻结后的历史探索性 test

| Encoder | Method | Known routing | OOS recall | Open-set F1 |
|---|---|---:|---:|---:|
| Frozen | B0 | 0.869 | 0.505 | 0.808 |
| Frozen | B3-H | 0.863 | 0.637 | 0.831 |
| Frozen | B4-Fusion | 0.865 | 0.555 | 0.820 |
| **Frozen** | **B4-RejectOnly** | **0.876** | **0.643** | **0.843** |
| Frozen | B4-Full | 0.863 | 0.658 | 0.837 |
| LoRA | B0 | 0.836 | 0.557 | 0.799 |
| LoRA | B3-H | 0.838 | 0.584 | 0.803 |
| LoRA | B4-RejectOnly | 0.854 | 0.574 | 0.814 |
| LoRA | B4-Full | 0.852 | 0.500 | 0.802 |

### 3.4 回答四个问题

1. **只加 reject head 是否更好？** 是。Frozen-RejectOnly 的 test F1 0.843，
   比 B3-H 0.831 高 1.28 pp，OOS 高 0.6 pp，routing 也高 1.3 pp。
2. **LoRA 是否改善表示？** 没有稳定改善。LoRA-B0 的 OOS 更高，但 B3-H/RejectOnly
   的 F1 都低于对应 frozen；routing 从 0.85 降到 0.83。
3. **两者组合是否互补？** 否。LoRA-B4-RejectOnly 0.814，低于 Frozen-B4-RejectOnly
   0.843；LoRA-B4-Full 0.802，也低于 frozen 版本。
4. **OOS 提升是否伤害正常路由？** RejectOnly 的 routing 反而略升；B4-Full
   把 OOS 提到 0.658，但 F1 低于 RejectOnly；LoRA 则明显伤 routing。

**结论：本轮真正的模块贡献是“固定融合 + 只学拒识头”，不是 gate，也不是 LoRA。**

---

## 4. C：RAG + 自然语言 Agent

### 4.1 知识库

- 30 条明确标记为虚构银行演示政策的文档：
  `knowledge_base/fictional_bank/policies.jsonl`；
- 每条有 `doc_id / title / version / source_id / citation / text / tags`；
- TF-IDF（1–2 gram）+ cosine 检索，返回 top-k 证据与 `doc_id`；
- 不访问外部网页，不声称真实银行政策；检索不修改数据库。

### 4.2 Agent 升级

- 路由层仍是 B3-H/B4/LoRA，不让 LLM 替代意图路由；
- 自然语言澄清：只展示“checking whether a card has arrived”这类描述，
  不展示 `card_arrival` 等内部标签；
- 保留前文 ID 槽位：澄清回复不会丢掉原始 `TXN-2001`；
- 会话隔离：每个 Streamlit 会话独立 SQLite 文件；切换方法不清空其他会话；
- 敏感动作先展示对象、金额/卡号，再要求明确 yes；工具层再次校验所有权；
- 知识库问答显示引用；路由拒识但问题是政策/FAQ 时，允许仅走本地知识库，
  不发生任何金融动作；
- LLM 接口：`LLMClient` 支持 OpenAI 兼容端点，密钥只读环境变量；当前环境
  无凭据，状态为 **NOT RUN_API_BLOCKED**，没有伪造 API 调用。

### 4.3 演示

`app_gate5.py` 提供：

- B3-H / Frozen-B4Full / LoRA-B4Full 切换；
- 10 个真实模型回放按钮（其中 7 个 B4-Full）；
- 聊天、Top-3、gate、工具时间线、RAG 引用、业务状态、审计；
- 同句三路由比较；
- 显式 reset 和独立 session DB。

真实截图：

- `figures/gate5_agent_card_b4.png`：B4 查卡；
- `figures/gate5_agent_rag.png`：FAQ + KB-013 引用；
- `figures/gate5_agent_cancel.png`：B4 取消转账，Transfers 显示 TXN-2002 cancelled。

10 条 demo 轨迹：`results/gate5_agent_demos.json`。

---

## 5. D：120 条 Agent Benchmark

### 5.1 任务集

- 120 条自然语言模拟业务任务，冻结于 `data/gate5/benchmark_cases.jsonl`；
- 文件 SHA-256：`2d3f157bf6ebc1f711430523b8b99510f7558c96db799a093d92262ac5c57f43`；
- 分层：card 25、transfer 30、dispute/ticket 20、policy FAQ 20、
  unknown/ambiguous 15、permission/safety 10；
- 每条有独立 ID、初始 DB 状态/预置动作、自然语言 turns、期望工具、预期终态；
- 三个真实路由版本：Frozen-B3H、Frozen-B4Full、LoRA-B4Full；
- 同一输入、同一预置 DB；StubRouter 不进入真实 benchmark。

> 生成 benchmark 时曾发现 7 条 `turns` 被写成字符串，导致字符级输入；该错误
> 在正式评测前修复并重新冻结。正式结果来自修复版 120 条任务；旧的错误运行
> 不作为结果保留。

### 5.2 总结果

| Router | TSR | Tool+arg | Safe action | Routing acc. | OOS escalation | Clarification* | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Frozen-B3H** | **0.625** | **0.725** | **1.000** | 0.642 | 1.000 | 1.000 | 0.050 s | 0.094 s |
| Frozen-B4Full | 0.583 | 0.700 | 1.000 | 0.642 | 1.000 | 0.000 | 0.056 s | 0.118 s |
| LoRA-B4Full | 0.583 | 0.725 | 1.000 | 0.642 | 1.000 | 0.000 | 0.097 s | 0.213 s |

\*Clarification success is computed only on the five cases that explicitly expect
clarification; B4/LoRA answered those directly instead of asking, so the strict
success rate is 0.0, not an average over all 120 cases.

**Frozen-B3H 仍是业务 benchmark 最好的路由器。** B4-Full 和 LoRA-B4Full 都没有
把模型侧变化转化为更高任务成功率。

### 5.3 分任务族

| Family | Frozen-B3H | Frozen-B4Full | LoRA-B4Full |
|---|---:|---:|---:|
| card status/activation | 0.84 | 0.84 | 0.68 |
| transfer query/cancel | 0.43 | 0.43 | 0.47 |
| dispute/ticket | 0.25 | 0.25 | 0.35 |
| policy FAQ | 0.90 | 0.90 | 0.90 |
| unknown/ambiguous | 1.00 | 0.67 | 0.67 |
| permission/safety | 0.30 | 0.30 | 0.40 |

- 安全操作率三个路由器都是 **1.00**：没有未确认的敏感变更；
- 主要失败不是工具越权，而是 **路由错误**：Frozen-B3H 43 次 routing failure，
  Frozen-B4Full 43 次，LoRA 43 次；工具失败分别 2/7/2；
- 业务失败集中在 transfer/dispute/permission，说明“模型 F1 小改善”不足以
  解决真实自然语言表达的槽位/路由问题；
- 政策 FAQ 达到 90%，KB 引用正常；未知请求 B3H 100% escalation，B4/LoRA 因
  澄清行为差异为 66.7%。

### 5.4 API 模式

- 12 条代表性 API 会话：**NOT RUN_API_BLOCKED**；
- 缺少 `OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL`；
- 接口与测试已实现，离线 Agent 和 120 条真实路由 benchmark 不受影响；
- 没有把模板输出冒充 LLM 输出。

---

## 6. 图表

- `figures/gate5_lora_training.png`
- `figures/gate5_lora_validation.png`
- `figures/gate5_method_matrix.png`
- `figures/gate5_module_attribution.png`
- `figures/gate5_benchmark_overall.png`
- `figures/gate5_benchmark_by_family.png`
- `figures/gate5_benchmark_failures.png`
- `figures/gate5_agent_card_b4.png`
- `figures/gate5_agent_rag.png`
- `figures/gate5_agent_cancel.png`

---

## 7. 结论与限制

### 结论

1. **RejectOnly 是唯一稳定正贡献的新模块**：Frozen-RejectOnly test F1 0.843，
   高于 B3-H 0.831；不需要 gate。
2. **LoRA 没有转化为更好路由**：validation B3-H 下降，test B3-H/RejectOnly/Full
   都低于 frozen；不能把训练成功等同于模型提升。
3. **B4-Full 的 gate 不是收益来源**：validation/test 都低于 RejectOnly。
4. **Agent 工具/安全链真实可用**，但业务 benchmark 的成功率主要由路由决定；
   Frozen-B3H 62.5% 仍高于 B4/LoRA 58.3%。
5. **RAG 对政策问题有效**：policy FAQ 90%，引用可追踪；但 RAG 不能修复错误的
   金融意图路由。

### 限制

- BANKING77 test 已看过；Gate 5 的数字是历史探索性对照；
- LoRA 只做 2 个有效 epoch（第一次 4 epoch 经 validation 后放弃），未做多种子；
- 单数据集、单编码器；
- 120 条任务为模拟数据，不是真实银行线上分布；
- 测试用 Stub 的 16 条 Gate4 状态测试不计入 Gate5 120 条真实路由 benchmark；
- API 模式因凭据缺失未运行；
- 在首次 benchmark 错误分析后，只加入了一次**通用 FAQ fallback**（路由拒识时允许
  本地知识库回答政策问题），任务内容未修改，随后重跑 benchmark；因此最终 Agent
  结果不是完全盲测的“第一次运行”，但任务集哈希保持不变；
- 没有做显著性检验；提升幅度小。

### 下一步

- 在新的独立 unknown-class 协议上复现 Frozen-RejectOnly；
- 多 seed 训练 LoRA，先解决 routing 退化，再谈组合；
- 用真实用户表达扩充 transfer/dispute/permission 任务族；
- 配好 API 凭据后跑 12 条真实 API 会话，并单独报告 token/cost；
- 不基于当前 benchmark 继续调 Agent 规则；若继续，需要新的独立任务集。

---

## 8. 复现

```powershell
cd D:\code\intent-open-set
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# A. LoRA
.\.venv\Scripts\python.exe -m src.gate5.train_lora_gte --epochs 2
.\.venv\Scripts\python.exe -m src.gate5.encode_lora
.\.venv\Scripts\python.exe -m src.gate5.evaluate_lora

# B. RejectOnly / LoRA B4 / matrix
.\.venv\Scripts\python.exe -m src.gate5.train_reject_only --encoder frozen
.\.venv\Scripts\python.exe -m src.gate5.train_reject_only --encoder lora
.\.venv\Scripts\python.exe -m src.gate5.train_b4_lora
.\.venv\Scripts\python.exe -m src.gate5.run_method_matrix --phase validation
.\.venv\Scripts\python.exe -m src.gate5.run_method_matrix --phase test

# C. KB / Agent / Demo
.\.venv\Scripts\python.exe -m src.gate5.make_kb
.\.venv\Scripts\python.exe -m src.gate5.run_demos_gate5
.\.venv\Scripts\python.exe -m streamlit run app_gate5.py

# D. Benchmark
.\.venv\Scripts\python.exe -m src.gate5.make_benchmark
.\.venv\Scripts\python.exe -c "from src.gate5.benchmark import load_cases, freeze; freeze(load_cases())"
.\.venv\Scripts\python.exe -m src.gate5.benchmark
.\.venv\Scripts\python.exe -m src.gate5.run_llm_api_cases

# Figures / tests
.\.venv\Scripts\python.exe -m src.gate5.make_figures
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

关键结果：

- `results/gate5_lora_training.json`
- `results/gate5_lora_validation.csv`
- `results/gate5_lora_encoding.json`
- `results/gate5_method_matrix_val.csv`
- `results/gate5_method_matrix_historical_test.csv`
- `results/gate5_method_freeze.json`
- `results/gate5_benchmark_summary.csv`
- `results/gate5_benchmark_by_family.csv`
- `results/gate5_benchmark_failure_summary.json`
- `results/gate5_llm_api_results.json`
