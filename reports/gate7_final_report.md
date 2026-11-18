# Gate 7 最终报告：Qwen LoRA 多种子诊断、机制归因与联合优化

**分支**：`gate7-qwen-diagnosis`  
**起点**：`gate6-qwen-lora@a0e01f3`  
**日期**：2026-11-18

> **边界**：BANKING77 的 `known_test/final_unknown` 已在 Gate 3–6 查看，本轮只在
> 全部 adapter/阈值/配方冻结后做一次历史探索性复核。120 条 Agent 任务同样是
> Gate5 旧任务的探索性回归，不是新盲测。

---

## 0. 一句话结论

**Gate6 的“验证提升、test 下降”主要是 seed 42 的偶然/迁移问题，而不是 Qwen LoRA 的普遍失败：3 个 seed 的 Qwen Plain 在 dev_select 上 B3-H 均值 0.8028、RejectOnly 0.8130；历史 test 均值 B3-H 0.8068、RejectOnly 0.8253，RejectOnly 相对 frozen 0.8127 提升 +1.25 pp，3 个 seed 中 2 个胜出。B4-RejectOnly 在 GTE/Qwen、Frozen/LoRA 上仍稳定优于 B4-Full。**

同时保留负结果：P1/P2 没有超过 P0；P3 因 Windows 内存压力停在 40/50 step，未保存 adapter；B4 Full 没有超过 RejectOnly；LLM API 仍无凭据。

---

## 1. A：双层开发集与机制诊断

### 1.1 双层开发集

从 calibration_known 800 + dev_unknown 720 冻结：

| Split | Known | Unknown | 用途 |
|---|---:|---:|---|
| dev_threshold | 40 类 × 10 = 400 | 9 类 × 40 = 360 | 只选 tau |
| dev_select | 40 类 × 10 = 400 | 另 9 类 × 40 = 360 | 只选配方/报告 |

- sample_id overlap 0，text SHA overlap 0，unknown class overlap 0；
- dev_threshold SHA `a6821544...`，dev_select SHA `07ba9a1f...`；
- final-test 从未参与任何选择。

### 1.2 H1：microbatch 的阴性规模

当前实现每个 microbatch 是 **2 anchor + 2 positive + 2 hard negative = 6 条**；
SupCon 相似度矩阵是 **6×6**。`grad_accum=8` 只是累积 8 个 microbatch 的梯度，
**不会**形成 48×48 的对比矩阵。

- 400/2 = 200 microbatches/epoch；
- 200/8 = 25 optimizer steps/epoch，余数 0；
- 若余数非 0，旧循环会丢掉最后一个不足 accum 的 microbatch；本配置未触发。

证据：`results/gate7_toy_batch.json`、`figures/gate7_toy_6x6.png`。

### 1.3 H2：hard negative 过强/重复

400 条 support 的 hard negative：

| 指标 | 值 |
|---|---:|
| hard negative cosine mean | 0.807 |
| positive − negative margin mean | **−0.0126** |
| margin < 0 | **203 / 400** |
| margin p10 | −0.182 |
| 重复 hard negative | **243** |
| unique hard negative | **157 / 400** |

超过一半 anchor 与 hard negative 的余弦高于其 positive，且 400 个 hard negative
只来自 157 个唯一样本。这会放大少样本过拟合风险。

### 1.4 dev_select 上的 Frozen / Plain / Anchor

| Encoder | B3-H F1 | OOS recall | Routing | AUROC |
|---|---:|---:|---:|---:|
| Qwen Frozen | 0.7904 | 0.6639 | 0.8250 | 0.8749 |
| Qwen Plain42 | **0.7984** | **0.6917** | 0.8300 | **0.8896** |
| Qwen Anchor | 0.7942 | 0.6694 | 0.8225 | 0.8784 |

Plain 在 dev_select 上确实优于 frozen；这与 Gate6 historical test 的 seed42
负结果**方向相反**，说明不是简单的训练失败，而是 seed/unknown 迁移。

---

## 2. B：3-seed Plain 与 P0–P3

### 2.1 P0–P3 单 seed（seed 42）

| 配方 | 改变 | dev_select B3-H F1 | dev_select Reject F1 | 结论 |
|---|---|---:|---:|---|
| P0 | 原 Plain，lr 1e-4，SupCon+0.5 triplet | **0.7984** | **0.7995** | 保留 |
| P1 | lr 3e-5 | 0.7780 | 0.7978 | 更差 |
| P2 | lr 1e-5 | 0.7889 | 0.7981 | 更差 |
| P3 | 移除 triplet | **INCOMPLETE** | — | 40/50 step 后 Windows 压力停顿，无 adapter |

P3 日志：`logs/gate7_qwen_p3_train.log`；记录：
`results/gate7_p3_incomplete.json`。**没有把 P3 当成完成。**

### 2.2 3-seed Plain（P0）

| Seed | Val B3-H | Val Reject | Test B3-H | Test Reject | Adapter SHA |
|---|---:|---:|---:|---:|---|
| 42 | 0.79837 | 0.79947 | 0.79694 | 0.80798 | `4efaee5c...` |
| 43 | **0.80508** | **0.82216** | **0.81589** | **0.83611** | `8ca9d985...` |
| 44 | 0.80504 | 0.81746 | 0.80768 | 0.83175 | `1711bcc9...` |
| **mean ± std** | **0.8028 ± 0.0032** | **0.8130 ± 0.0100** | **0.8068 ± 0.0095** | **0.8253 ± 0.0151** | — |

相对 frozen：

| Method | Frozen test F1 | 3-seed mean | Δ mean | Wins |
|---|---:|---:|---:|---:|
| B3-H | 0.80509 | 0.80684 | +0.18 pp | 2/3 |
| B4-RejectOnly | 0.81274 | **0.82528** | **+1.25 pp** | **2/3** |

OOS recall：
- B3-H mean 0.6009 vs frozen 0.5724（+2.85 pp）；
- RejectOnly mean 0.6443 vs frozen 0.6000（+4.43 pp）。

Known correct routing：
- B3-H mean 0.8323 vs frozen 0.8425（−1.02 pp）；
- RejectOnly mean 0.8458 vs frozen 0.8438（+0.21 pp）。

**结论：seed42 的 test 负结果是真实的，但不是三种子的普遍结论；3-seed 均值是
小幅正结果，且 RejectOnly 比 B3-H 更稳。**

### 2.3 优化是否成功？

**没有。** P1/P2 在 dev_select 均低于 P0；P3 未完成。按任务书保留 P0/
Frozen-RejectOnly 为主结果，没有强行把 P1/P2 包装成“优化成功”，也没有
为了得到一个正的 optimized 结果去扫 test。

因此 C 表中的 **Optimized = NOT SELECTED**；Plain 3-seed 是最终 Qwen LoRA
代表。

---

## 3. C：冻结与历史探索性 test

所有 seed、adapter、B4 checkpoint、dev 阈值在
`results/gate7_final_freeze.json` 中冻结，`test_evaluated=true` 后只评估一次。

| Encoder | Method | Test F1 | OOS | Routing | AUROC |
|---|---|---:|---:|---:|---:|
| Qwen Frozen | B3-H | 0.8051 | 0.5724 | 0.8425 | 0.8872 |
| Qwen Frozen | B4-RejectOnly | 0.8127 | 0.6000 | 0.8438 | 0.8815 |
| Qwen Plain 42 | B3-H | 0.7969 | 0.5579 | 0.8288 | 0.8677 |
| Qwen Plain 42 | B4-RejectOnly | 0.8080 | 0.5750 | 0.8413 | 0.8661 |
| Qwen Plain 43 | B3-H | 0.8159 | 0.6197 | 0.8388 | 0.8834 |
| Qwen Plain 43 | B4-RejectOnly | **0.8361** | **0.6789** | 0.8494 | 0.8864 |
| Qwen Plain 44 | B3-H | 0.8077 | 0.6250 | 0.8294 | 0.8976 |
| Qwen Plain 44 | B4-RejectOnly | 0.8318 | 0.6789 | 0.8469 | 0.8910 |

- **没有用 test 选 seed/配方；42/43/44 全部报告。**
- seed42 的 B3-H 负结果保留；
- 3-seed 均值不是“最优 seed”挑选；
- 历史 test 仍不是新盲测。

---

## 4. D：Agent 与工程修正

### 4.1 修复 5 条应澄清任务

Gate6 的 5 条 unknown/ambiguous 任务（B106–B110）第一次均被直接接受。
Gate7 加入一个**通用语义歧义守卫**：当 accepted intent 属于 transfer-not-arrived
语义族且文本出现 “not arrived / not received / not shown” 时，先自然语言澄清；
不改阈值、不对旧 120 规则做逐条特判。

新增回归测试：`tests/test_gate7.py::test_transfer_ambiguity_guard`。
真实截图：`figures/gate7_agent_clarification.png`。

### 4.2 同一 120 条旧任务的公平对照

| Router | TSR | 成功/120 | Tool+arg | Safe | Routing | P50 |
|---|---:|---:|---:|---:|---:|---:|
| GTE-Frozen-RejectOnly | 0.6083 | 73 | 0.708 | 1.000 | 0.667 | 0.087 s |
| Qwen-Frozen-RejectOnly | **0.6750** | **81** | 0.750 | 1.000 | **0.725** | 0.138 s |
| Qwen-Opt-RejectOnly | **0.6750** | **81** | 0.750 | 1.000 | 0.708 | 0.189 s |

`Qwen-Opt-RejectOnly` 实际是 **Qwen Plain42 + RejectOnly**；因为优化配方未
被选中，没有另一个 optimized adapter。

分族：

| Family | GTE Reject | Qwen Frozen Reject | Qwen Plain42 Reject |
|---|---:|---:|---:|
| card | 0.92 | 0.84 | 0.88 |
| transfer | 0.467 | 0.60 | 0.633 |
| dispute | 0.25 | 0.25 | 0.15 |
| policy | 0.90 | 0.90 | 0.90 |
| unknown/ambiguous | 0.667 | **1.00** | **1.00** |
| permission | 0.30 | 0.40 | 0.40 |

- Qwen 两个 Reject 版本都把 unknown/ambiguous 从 0.667 提到 1.0，说明 5 条
  澄清 bug 修复真实生效；
- Qwen transfer 也高于 GTE；GTE card 仍最高；
- dispute 仍是所有路由器最弱族；
- 安全率全部 100%。

对比 Gate6 的 F1 结论：Qwen RejectOnly 的 test F1 0.825（3-seed mean）低于
GTE RejectOnly 0.843，但 Agent TSR 0.675 > GTE RejectOnly 0.608。**分类 F1
和业务 TSR 不是同一排序**，与 Gate6 的结论一致。

### 4.3 GTE-LoRA / Qwen-LoRA 多种子稳定性

B4-RejectOnly / Full 的 3-seed dev_select 均值±std：

| Encoder | RejectOnly | Full |
|---|---:|---:|
| GTE Frozen | **0.8144 ± 0.0015** | 0.8105 ± 0.0022 |
| GTE LoRA seed42 | **0.8047 ± 0.0008** | 0.7984 ± 0.0027 |
| GTE LoRA seed43 | **0.8277 ± 0.0005** | 0.8152 ± 0.0023 |
| GTE LoRA seed44 | **0.8266 ± 0.0009** | 0.8026 ± 0.0014 |
| Qwen Frozen | **0.7950 ± 0.0008** | 0.7752 ± 0.0047 |
| Qwen Plain42 | **0.7997 ± 0.0002** | 0.7879 ± 0.0088（仅 2 seed 完成） |

- 每个 seed 的 B4 checkpoint SHA 独立（`gate7_stability_checkpoint_hashes.csv`）；
- RejectOnly 在 GTE Frozen、GTE-LoRA 42/43/44、Qwen Frozen 上都优于 Full；
- 训练/初始化/episode 的 seed 确实改变了输出，不是“复制同一模型”；
- GTE-LoRA 的 seed43/44 高于 seed42，说明 adapter 本身也有明显 seed 波动；
- Qwen Plain 的 3-seed test RejectOnly mean 0.8253，已在上节报告。

### 4.4 LLM API

仍无凭据，`NOT RUN_API_BLOCKED`；没有模拟真实 LLM 调用。

---

## 5. 研究问题的回答

1. **为什么 GTE-LoRA 在 400 条 support 上没改善 B3-H？**
   400 条 support 的 hard negative 有 203/400 负 margin、157 个唯一负例，
   对比信号质量差；GTE-LoRA 的 seed42 也退化。Qwen 上不是必然退化：
   3-seed 均值在 dev/test 都略正，但单 seed42 test 为负。**不能把单 seed
   负结果当成编码器能力结论。**
2. **Qwen 的漂移与指标关系？**
   Anchor 把 support cosine 从 0.892 提到 0.936，但 dev_select 从 0.7984
   降到 0.7942，test 0.7968 vs Plain 0.7972；漂移更小不等于泛化更好。
3. **Anchor 是否减小漂移？**
   是；但代价是 OOS/分类没有提升，说明只约束 anchor 子集不够。
4. **Qwen 上 RejectOnly 还是 Full？**
   RejectOnly 在 Frozen 与 Plain 上都优于 Full；GTE 的稳定性实验也复现同一
   方向。
5. **GTE vs Qwen？**
   GTE 分类 F1 更高、编码快约 4 倍；Qwen 0.6B 在 Agent TSR 上更好
   （Reject 81 vs GTE 73），但延迟高。**没有单一赢家。**
6. **F1 变化是否对应业务成功率？**
   不完全。Qwen 3-seed test Reject F1 0.825 < GTE 0.843，但 TSR 0.675 >
   GTE 0.608；agent 修复、拒识分布和工具执行共同决定业务结果。

---

## 6. 资源与失败

- Qwen Plain 训练：42/43/44 分别 295/321/306 s，峰值 1,260 MiB；
- adapter 9.20 MB/seed；
- Qwen 4,280 条重编码：78–84 s，峰值 1,186 MiB；
- GTE LoRA seed43/44：约 30 s 训练，24 s 重编码，618 MiB；
- P3：40/50 step 后停顿，**未保存 adapter**；
- 曾出现 Windows page file / CUDA unknown error（并发稳定性 + Agent 时）；
  通过分进程、释放模型和串行运行修复；日志保留；
- 64 个测试全部通过。

---

## 7. 限制

- BANKING77 test 已看过；历史探索性；
- 120 条 Agent 任务是旧任务；Agent 修复后重跑也只能算探索性回归；
- Qwen 只有 3 个 seed，不能声称显著性；
- P3 未完成，优化结论只能保留 P0；
- Qwen Plain 的 B4-Full 稳定性只完成 2 个 seed（Qwen Frozen 完成 3 个）；
- 没有 bootstrap 置信区间；
- 没有新的独立 unknown-class 协议；
- LLM API 无凭据。

---

## 8. 复现

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# A
.\.venv\Scripts\python.exe -m src.gate7.make_dev_split
.\.venv\Scripts\python.exe -m src.gate7.diagnose
.\.venv\Scripts\python.exe -m src.gate7.make_figures

# B (P1/P2; P3 was stopped and is recorded incomplete)
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_lora --run p1 --out-prefix gate7_qwen --lr 3e-5 --epochs 2 --microbatch 2 --grad-accum 8
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_lora --run p2 --out-prefix gate7_qwen --lr 1e-5 --epochs 2 --microbatch 2 --grad-accum 8
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_lora --run plain43 --out-prefix gate7_qwen --seed 43 --epochs 2 --microbatch 2 --grad-accum 8
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_lora --run plain44 --out-prefix gate7_qwen --seed 44 --epochs 2 --microbatch 2 --grad-accum 8

# C / stability
.\.venv\Scripts\python.exe -m src.gate7.final_test
.\.venv\Scripts\python.exe -m src.gate7.stability_b4 --encoders gte_frozen,gte_lora42,gte_lora43,gte_lora44,qwen_frozen --seeds 42,43,44

# D
.\.venv\Scripts\python.exe -m src.gate7.agent_benchmark --router GTE-Frozen-RejectOnly
.\.venv\Scripts\python.exe -m src.gate7.agent_benchmark --router Qwen-Frozen-RejectOnly
.\.venv\Scripts\python.exe -m src.gate7.agent_benchmark --router Qwen-Opt-RejectOnly --opt-tag plain42
.\.venv\Scripts\python.exe -m src.gate7.combine_agent
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

关键结果：

- `results/gate7_dev_split.json`
- `results/gate7_dev_diagnosis.csv`
- `results/gate7_toy_batch.json`
- `results/gate7_hard_negative_stats.json`
- `results/gate7_final_historical_test.csv`
- `results/gate7_final_summary.csv`
- `results/gate7_stability_summary.csv`
- `results/gate7_agent_summary.csv`
- `figures/gate7_stability.png`
- `figures/gate7_final_test.png`
- `figures/gate7_agent_clarification.png`
