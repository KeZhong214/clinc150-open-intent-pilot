# 项目最终研究总结：少样本开放意图识别与智能客服（Gate 0–7）

**项目**：面向智能客服的少样本开放意图识别  
**主线**：冻结编码器评测 → 可训练路由器 → 真实客服 Agent → 编码器微调与多种子诊断  
**最终分支**：`gate7-final-fix`  
**最终冻结基线**：`gate7-qwen-diagnosis@d1b2a7a`

> 所有 BANKING77 历史 test 结果均是**已查看过的历史探索性复核**，不是新盲测。
> 120 条 Agent 任务是模拟虚构银行任务，且已被 Gate5 用于系统修复；只用于
> 探索性回归。LLM API 无凭据，未运行。

---

## 1. 研究问题

1. 少样本、冻结编码器条件下，原型/近邻/拒识模块如何影响已知意图和未知意图？
2. B3-H、B4-RejectOnly、B4-Full 的收益分别来自哪里？
3. 把 Qwen3-Embedding 从冻结模型升级为 LoRA 可训练模型，是否能改善开放集路由？
4. 微调的 validation 提升为什么不一定转化为 historical test？
5. 模型指标变化是否能转化为真实客服任务成功率？
6. 这些结论在多 seed 下是否稳定？

---

## 2. 数据与协议

### BANKING77

| Split | 数量 | 用途 |
|---|---:|---|
| known classes | 40 | 已知意图 |
| dev-unknown classes | 18 | 开发期未知 |
| final-unknown classes | 19 | 历史 test |
| support | 400（40×10） | 训练/原型，固定 |
| calibration-known | 800 | 阈值/校准 |
| dev-unknown | 720 | 开发期 OOS |
| known test | 1,600 | 历史 test |
| final unknown | 760 | 历史 test |

- support 固定 10-shot；5-shot 是其前五条；
- support/calibration 用于训练与阈值；final test 只在冻结后评估；
- Gate7 进一步冻结 `dev_threshold`（400 known + 9 unknown 类×40）和
  `dev_select`（另 400 known + 另 9 unknown 类×40），互不重叠；
- Gate7 test 是历史探索性，不是新盲测。

### 其他数据集

- CLINC150：Gate0/1/2 历史协议，20 个 known intent、65/65 hard-OOS 类划分；
- MASSIVE en-US：35/12/13 类，用于跨数据集矩阵；
- 二者不再重跑，保留 Gate0–3 冻结结果。

---

## 3. 模型与算力

| 模型 | 维度 | 角色 |
|---|---:|---|
| BGE | 768 | Gate3 编码器对照 |
| Qwen3-Embedding-0.6B | 1024 | Gate3 冻结、Gate6/7 LoRA |
| GTE-ModernBERT | 768 | Gate3 最强 BANKING77；Gate4/5/7 主模型 |
| E5 | 768 | Gate3 对照 |

- 机器：Windows 11，RTX 5070 Laptop 8GB；
- torch 2.11.0+cu128，transformers 5.23.0，PEFT 0.21.3；
- Qwen 固定 revision `97b0c614...`，prompt SHA `4bbdf1c1...`，right padding、
  last non-pad token、max_length 128；
- GTE LoRA：`Wqkv/Wo/Wi`，r=8，alpha=16；
- Qwen LoRA：`q_proj/k_proj/v_proj/o_proj`，r=8，alpha=16。

---

## 4. Gate 0–2：协议、Baseline、Hard-OOS

### Gate 0

- 下载并校验 CLINC150 官方数据，记录 SHA-256；
- 固定 20 个 known intent（10 域×2 类）；
- 生成 support/calibration/test manifest，K=5 嵌套 K=10；
- 发现 5 处官方跨分区重复文本，只记录、不改写。

### Gate 1

| Scope | Method | Open-set F1 |
|---|---|---:|
| CLINC150 | Qwen B0 | 0.9570 |
| CLINC150 | Qwen B2 | 0.9499 |

- B2 动态拒识没有超过 B0。

### Gate 2

- 65/65 hard-OOS 类划分，1,300 val / 1,950 held-out test；
- B2-v2：corrected `q_c`、hard-OOS predicted-class 上尾 `u_c`、向全局阈值收缩；
- Hard-OOS recall 0.9426 → 0.9482（+0.56 pp），known routing −1.17 pp；
- 结论：类别阈值校准只能小幅移动，不能解决全部 OOS。

---

## 5. Gate 3：72 格跨数据集矩阵

- 3 数据集 × 4 编码器 × 2 shot × B0/B3-P/B3-H = 72；
- validation 选阈值、freeze、test 只评一次；
- 4 个编码器共 11,011 条文本/模型。

最强配置：

| Dataset | Best config | OOS | F1 |
|---|---|---:|---:|
| CLINC150（历史） | Qwen 10-shot B0 | 0.9010 | 0.9226 |
| BANKING77 | GTE 10-shot B3-H | 0.6368 | **0.8306** |
| MASSIVE en-US | Qwen 10-shot B0 | 0.5298 | 0.7982 |

关键判断：

- B3-H 在 BANKING77 有正收益（OOS +5.21 pp，F1 +0.99 pp）；
- 在 MASSIVE/CLINC150 没有稳定收益；
- Qwen 平均最强；GTE 在 BANKING77 最强且编码便宜；
- SetFit-BGE 0.6696、投影适配器 0.7394，均低于 BGE-B0 0.7505（5-shot）。

---

## 6. Gate 4：B4 可训练路由器与 Agent

- GTE BANKING77 400 support 训练 B4-AR：4→32→1 gate + 7→16→1 reject head；
- 24 个 episodic，60+60 epoch，12.55 s；
- B4-Fusion、B4-Full checkpoint、validation、freeze、历史探索性 test。

| Method | Val F1 | Test F1 | Test OOS |
|---|---:|---:|---:|
| B0 | 0.734 | 0.808 | 0.505 |
| B3-H | 0.767 | 0.831 | 0.637 |
| B4-Fusion | 0.740 | 0.820 | 0.555 |
| B4-Full | 0.781 | 0.837 | 0.658 |

- B4-Fusion 单独退化；收益主要来自 reject head；
- 门控随难度变化：最难门控 0.56，最易 0.92；
- Agent：SQLite 虚构银行、7 工具、多轮状态、确认、所有权、审计。

---

## 7. Gate 5：GTE-LoRA 与 RejectOnly

### GTE-LoRA

- `Wqkv/Wo/Wi`，1.69M 可训练参数（1.12%），2 epoch，22.19 s，
  adapter 6.78 MB；
- B0：0.734→0.753；B3-H：0.767→0.760；LoRA 没有改善 B3-H；
- 4-epoch 首次更差（0.7397），按 validation 选 2 epoch。

### RejectOnly

| Method | Val F1 | Test F1 |
|---|---:|---:|
| Frozen B3-H | 0.767 | 0.831 |
| **Frozen RejectOnly** | **0.790** | **0.843** |
| Frozen B4-Full | 0.781 | 0.837 |
| GTE-LoRA B3-H | 0.760 | 0.803 |
| GTE-LoRA RejectOnly | 0.767 | 0.814 |

- **固定 gate 0.5 + 只训练 reject head** 是 Gate5 唯一稳定正贡献；
- LoRA 与 reject head 不互补。

### Agent

- 30 条虚构政策 + TF-IDF RAG + 引用；
- 10 条真实模型 demo；
- 120 条任务，3 路由；
- Frozen-B3H TSR 62.5%，B4/LoRA 58.3%，安全率 100%。

---

## 8. Gate 6：Qwen LoRA

- Qwen 冻结对齐：right padding + last non-pad token；
  32 probe mean cosine 0.999998，top1 1.0；
- Plain/Anchor：2.29M 可训练参数（0.38%），2 epoch，adapter 9.20 MB；
- Plain support cosine 0.892；Anchor 0.936；
- Qwen 上 RejectOnly > Full；但 Qwen test F1 仍低于 GTE；
- Qwen RejectOnly Agent TSR 65.0%，安全率 100%。

---

## 9. Gate 7 / 7.1：机制诊断、多种子与协议修复

### 9.1 诊断

- microbatch=2 的真实 SupCon 矩阵 **6×6**；`grad_accum=8` 不产生 48×48；
- hard negative：203/400 负 margin，243 重复，157 唯一；
- dev_select 上 Plain B3-H 0.7984 > Frozen 0.7904；
- 结论：Gate6 的 seed42 test 负结果不是随机噪声，但不是普遍失败。

### 9.2 P0–P3

| 配方 | dev B3-H | dev Reject | 选中 |
|---|---:|---:|---|
| P0 | **0.7984** | **0.7995** | 是 |
| P1 lr3e-5 | 0.7780 | 0.7978 | 否 |
| P2 lr1e-5 | 0.7889 | 0.7981 | 否 |
| P3 no triplet | 0.7945 | 0.7995 | 否（完成） |

- P1/P2/P3 都没有超过 P0；没有 optimized 3-seed；
- 这是负结果，不是遗漏。

### 9.3 3-seed Qwen Plain（protocol_fix_v2）

| Seed | Test B3-H | Test RejectOnly |
|---|---:|---:|
| 42 | 0.7969 | 0.8091 |
| 43 | 0.8159 | 0.8356 |
| 44 | 0.8077 | 0.8321 |
| mean ± std | **0.8068 ± .0095** | **0.8256 ± .0144** |

- Frozen：0.8051 / 0.8130；
- B3-H +0.18 pp；RejectOnly **+1.26 pp**；2/3 seed 胜出；
- 所有 seed 报告，没有挑最佳 seed；
- 历史 test 是探索性，不是新盲测。

### 9.4 阈值协议修复

- 原 B0/B3-H 用 Gate7 dev_threshold，B4-RejectOnly 错用 Gate6 阈值；
- 修复后三者统一使用 Gate7 dev_threshold；
- 原 test/freeze/summary 保留 `*_pre_fix.*`，v2 单独写；
- 3-seed mean RejectOnly 0.82528→0.82557（+0.03 pp），结论不变；
- 逐行差异 `gate7_protocol_fix_diff.csv`。

### 9.5 B4 多种子稳定性（36 组）

| Encoder | RejectOnly dev | Full dev |
|---|---:|---:|
| GTE Frozen | **0.8144 ± .0015** | 0.8105 ± .0022 |
| GTE-LoRA42 | **0.8047 ± .0008** | 0.7984 ± .0027 |
| GTE-LoRA43 | **0.8277 ± .0005** | 0.8152 ± .0023 |
| GTE-LoRA44 | **0.8266 ± .0009** | 0.8026 ± .0014 |
| Qwen Frozen | **0.7950 ± .0008** | 0.7752 ± .0047 |
| Qwen Plain42 | **0.7997 ± .0002** | 0.7892 ± .0066 |

- 36 组 `(encoder,method,seed)` 无重复；
- Qwen Plain43/44 没有完整 B4 交叉矩阵，只完成 LoRA 与 final test；
- B4 seed 与 LoRA seed 分开定义，不能混称。

### 9.6 Agent

| Router | TSR | 成功/120 | Tool | Safe |
|---|---:|---:|---:|---:|
| GTE-Frozen-RejectOnly | 0.608 | 73 | 0.708 | 1.0 |
| Qwen-Frozen-RejectOnly | **0.675** | **81** | 0.750 | 1.0 |
| Qwen-Plain42-RejectOnly | **0.675** | **81** | 0.750 | 1.0 |

- `Qwen-Plain42-RejectOnly` 旧名 `Qwen-Opt-RejectOnly`，实际不是新优化；
- 5 条澄清 bug 修复，Qwen unknown 族 1.0；
- Agent 仍使用 Gate5/6 旧阈值，与分类 C 的 Gate7 dev_threshold 是不同协议；
- 120 条旧任务，探索性。

---

## 9.7 Gate 8：可用 Agent 与独立模拟评测

Gate8 把分类/拒识模块接成可操作的多轮银行 Agent：

- 统一 Gate7 `dev_threshold`；三个 router：GTE Frozen、Qwen Frozen、Qwen Plain42；
- 修复 dispute fallback、pending slot 吞话题、确认阶段改对象、dispute 状态检查；
- 9 个真实 Demo；SQLite 业务状态、audit、KB 引用、每轮 Top-3；
- 102 条新冻结模拟案例（不是外部盲测），同案例三模型公平对照。

| Router | TSR | Tool | Safe | P50 |
|---|---:|---:|---:|---:|
| GTE-Frozen-RejectOnly | **0.951** | 0.980 | 1.000 | 0.060 s |
| Qwen-Frozen-RejectOnly | 0.824 | 0.824 | 1.000 | 0.140 s |
| Qwen-Plain42-RejectOnly | 0.794 | 0.824 | 1.000 | 0.199 s |

- GTE 在本模拟测试上 TSR 最高，不能与分类 F1 直接配对；
- Qwen 的 dispute 首句被 RejectOnly 拒绝，是真实负结果；
- 三个 router 安全率均 100%；
- LLM API 无凭据 `NOT RUN_API_BLOCKED`；
- 详细：`reports/gate8_agent_final_report.md`。

## 10. 最终研究结论

### 可以支持的结论

1. **B4-RejectOnly 是稳定有效的模块**：GTE/Qwen、Frozen/LoRA、多个 seed 上
   均优于 B4-Full；GTE Frozen RejectOnly 仍是整体分类最高（历史 test 0.8435）。
2. **B3-H 的场景收益有限**：BANKING77 有效，MASSIVE/CLINC150 不稳定。
3. **Qwen LoRA 三种子均值略优于 Frozen**：B3-H +0.18 pp、
   RejectOnly +1.26 pp；波动约 1–1.4 pp；不能称为大幅提升。
4. **Qwen 的 seed42 test 负结果是真实负结果**，但不是普遍失败；不能挑 seed。
5. **分类 F1 与业务 TSR 不单调**：Qwen Reject F1 低于 GTE Reject，但 Agent
   TSR 更高。
6. **微批对比、hard negative 质量是机制风险**；6×6 矩阵、203/400 负 margin
   是直接证据。

### 只能作为探索性发现

- 120 条 Agent 任务的成功率；
- Qwen vs GTE 的业务优势；
- Anchor 的漂移-性能关系；
- 任何单 seed 的 test 差异；
- 没有新独立 unknown 协议下的泛化结论。

### 负结果

- GTE-LoRA 没有改善 B3-H；
- Qwen LoRA validation 提升、test 不稳定；
- Anchor 降低漂移但不提高 test；
- P1/P2/P3 没有超过 P0；
- B4-Full 没有超过 RejectOnly；
- 5 条澄清 bug 曾长期存在，修复后 Qwen unknown 才到 1.0。

---

## 11. 应用成果

- `app_gate6.py`：Qwen 客服 Agent，RAG、SQLite 工具、会话隔离、确认、审计；
- 30 条虚构银行政策；
- 120 条业务任务；
- 真实截图：查卡、RAG 引用、取消转账、澄清；
- Agent 安全操作率 100%（所有对照）；
- LLM API 接口已实现，但无凭据 `NOT RUN_API_BLOCKED`。

---

## 12. 限制与后续

- BANKING77 test 已看，所有 test 是探索性；
- 120 任务旧、模拟；
- 只有 3 个 seed，无 bootstrap/显著性；
- Qwen Plain43/44 无完整 B4 矩阵；
- 没有新的独立 unknown-class 盲测；
- LLM API 未运行。

后续优先级：

1. 新的独立 unknown-class 协议；
2. 真 batch/负例质量修复；
3. 更多 seed + bootstrap；
4. dispute 路由与澄清覆盖；
5. 不用当前历史 test 继续调参。

---

## 13. 关键文件

- `reports/gate7_1_fix_report.md`
- `reports/gate7_final_report.md`
- `results/gate7_final_historical_test_v2.csv`
- `results/gate7_final_summary.csv`
- `results/gate7_protocol_fix_diff.csv`
- `results/gate7_stability_b4_results.csv`
- `results/gate7_stability_summary.csv`
- `results/gate7_stability_test_summary.csv`
- `results/gate7_agent_summary.csv`
- `figures/gate7_final_test.png`
- `figures/gate7_stability.png`
