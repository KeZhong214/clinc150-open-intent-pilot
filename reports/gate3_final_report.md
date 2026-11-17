# Gate 3 最终实验报告：多数据集、跨模型与原型—近邻协同

**项目**：面向智能客服的少样本开放意图识别：多模型跨场景评测与原型—近邻协同优化  
**日期**：2026-11-18  
**分支**：`course-expansion`  
**状态**：3 数据集 × 4 编码器 × 2 shot × B0/B3-P/B3-H 的 72 格 test 矩阵
已完成；BANKING77 训练式对照与 Streamlit Demo 已运行。

> 本报告中的所有数字均来自 `results/gate3_*` 下的 CSV/JSON。没有使用模拟数字。
> Gate 1/2 的 CLINC150 结果保持冻结；CLINC150 test 已在本轮之前查看，标为
> historical，不作为 B3 的盲测证据。

---

## 1. 研究问题

1. 四个冻结编码器（BGE-small、Qwen3-Embedding-0.6B、GTE-ModernBERT-base、
   E5-base-v2）在 3 个不同客服/助手场景中的少样本已知分类与未知拒识差异；
2. 自适应多原型 B3-P 是否改善类内多样性场景；
3. 原型—近邻融合 B3-H 是否在 B0/B3-P 之上改善 open-set macro-F1 与 OOS recall；
4. 训练式少样本方法（SetFit-BGE / 投影适配器）是否优于冻结编码器 B0；
5. 不同数据集、shot、模型下，收益/退化能否用类别结构、阈值行为和分数间隔解释。

---

## 2. 数据

### 2.1 三个数据集

| Dataset | 场景 | Known / dev-unknown / final-unknown | Official rows used |
|---|---|---|---:|
| CLINC150 | 多领域客服，历史 Pilot | 20 / — / original OOS | 2,300 |
| BANKING77 | 单领域细粒度银行客服 | 40 / 18 / 19 | 4,280 |
| MASSIVE en-US | 多领域语音助手 | 35 / 12 / 13 | 4,431 |

`CLINC150` 使用 Gate 0 的 20 类、seed=42、5/10-shot；原始 `oos_val` 作为
dev unknown，原始 `oos_test` 作为 final unknown。其 test 已查看，标为
historical。

`BANKING77` 从 77 类中用固定 SHA-256 排序选 40 known、18 dev、19 final；
support 来自官方 train 每类 10 条（5-shot 为前 5），calibration 为 train 中
不重叠的 20 条/类；known test 和 unknown test 均来自官方 test 中互不重叠的类。

`MASSIVE en-US` 从 60 类中选 35 known、12 dev、13 final；known 类要求官方
train ≥10，dev/final 按实际 dev/test 可用性分配；support 来自 train，calibration
known 与 dev unknown 来自官方 dev，final unknown 来自官方 test。

### 2.2 计数与隔离

| Dataset | Support 5/10 | Calibration | Known test | Dev unknown | Final unknown |
|---|---:|---:|---:|---:|---:|
| CLINC150 | 100 / 200 | 400 | 600 | 100 | 1,000 |
| BANKING77 | 200 / 400 | 800 | 1,600 | 720 | 760 |
| MASSIVE | 175 / 350 | 1,323 | 1,915 | 407 | 436 |

Support/calibration/test 的规范化文本交集全部为 0；K=5 严格嵌套于 K=10。
MASSIVE 的已知类 test 每类从 1 到 209 条不等，未做人为平衡；BANKING77
每类 40 条；CLINC150 每类 30 条。MASSIVE 35 个 known 类覆盖 17/18 个 domain，
剩余 domain 只出现在 unknown 或未被 known 覆盖，报告中已标注。

### 2.3 类别划分规则

- 每个数据集独立按 `SHA-256("gate3|42|<dataset>|<intent>")` 排序；
- BANKING77 取前 40/18/19；
- MASSIVE known 从满足 train≥10 的类中取 35，剩余按 dev/test 可用性
  强制分配 12/13；
- 不读取任何模型分数；类别选择在编码前冻结。

---

## 3. 编码器

| Model | Revision | Dim | Device | Total encode time (11,011 texts) | Peak VRAM |
|---|---|---:|---|---:|---:|
| BGE-small-en-v1.5 | `5c38ec7...` | 384 | CPU | 19.24 s | — |
| Qwen3-Embedding-0.6B | `97b0c61...` | 1,024 | RTX 5070, FP16 | 160.65 s | 1,168 MiB |
| GTE-ModernBERT-base | `e7f32e3...` | 768 | RTX 5070, FP16 | 12.69 s | 336 MiB |
| E5-base-v2 | `f52bf8e...` | 768 | RTX 5070, FP16 | 6.03 s | 277 MiB |

输入前缀按模型卡固定：

- BGE / GTE：无前缀；
- E5：`query: `；
- Qwen：Gate 1/2 冻结的客服意图 instruction。

所有向量在 float32 中显式 L2 归一化；每个数据集单独缓存，未把 npz 提交到
GitHub。

---

## 4. 方法

### B0

```text
p_c = normalize(mean(support_c))
s_c = cos(z(x), p_c)
accept iff s1 >= tau
```

### B3-P

- 从 support 计算类内分散度 `d_c = mean_i(1-cos(z_i,p_c))`；
- 分散度最高 25% 的类用 `KMeans(k=2, random_state=42, n_init=10)`；
- 其余类单原型；样本不足/空簇回退单原型；
- 类分数为最大原型余弦。

### B3-H

- `P_c(x)` 为 B3-P 多原型分数；
- `N_c(x)` 为到该类 support 的 top-2 cosine 均值；
- 固定融合：`S_c = 0.5 P_c + 0.5 N_c`；
- 全局阈值 `tau`，`s1 >= tau` 接受，否则 OOS；
- 本轮没有使用 margin gate；B3-H 的差异来自原型表示 + 局部近邻融合。

### 阈值选择

对每个 dataset/model/shot/method：

- 候选 `tau`：validation known top-1 分数的 101 个分位点；
- 约束：validation known acceptance ≥ 0.90；
- 目标：最大化 validation open-set macro-F1；并列时比较 OOS recall、
  known acceptance、tau；
- 72 个配置全部在 validation 上选好，写入 `gate3_freeze.json` 后才跑 test。

**重要限制**：该 F1-first 目标在 MASSIVE 的弱分离模型上会选出非常高的
known acceptance（BGE 约 1.00），几乎不拒 OOS；这是阈值目标的真实问题，
不是伪造结果。见第 7 节。

---

## 5. 72 格主结果

### 5.1 B0 的 5-shot 模型×数据集

| Model | CLINC F1 | BANKING F1 | MASSIVE F1 | Mean F1 |
|---|---:|---:|---:|---:|
| BGE | 0.848 | 0.750 | 0.663 | 0.754 |
| Qwen | 0.880 | 0.722 | 0.776 | 0.793 |
| GTE | 0.859 | 0.764 | 0.671 | 0.765 |
| E5 | 0.853 | 0.693 | 0.656 | 0.734 |

Qwen 的 5-shot 平均值最高；GTE 在 BANKING77 5-shot 最好（0.764）。
10-shot 后 overall B0 从 0.7614 提升到 0.8042（4 模型 3 数据集平均）。

### 5.2 5-shot vs 10-shot

| Shot | Method | Known routing | OOS recall | Open-set F1 |
|---:|---|---:|---:|---:|
| 5 | B0 | 0.8451 | 0.4771 | 0.7614 |
| 5 | B3-H | 0.8372 | 0.4767 | 0.7533 |
| 10 | B0 | 0.8780 | 0.5212 | **0.8042** |
| 10 | B3-H | 0.8769 | **0.5461** | 0.8030 |

10-shot 对 B0 的提升明显；B3-H 在 10-shot 上 OOS recall 更高（+2.49 pp），
但 open-set F1 略低（−0.12 pp），known routing 基本不变（−0.11 pp）。

### 5.3 B3 消融（4 模型 × 2 shot 平均）

| Dataset | Method | Known routing | OOS recall | Open-set F1 |
|---|---|---:|---:|---:|
| CLINC150 | B0 | 0.9715 | 0.8319 | **0.8768** |
| CLINC150 | B3-P | 0.9694 | 0.8274 | 0.8716 |
| CLINC150 | B3-H | 0.9683 | 0.8208 | 0.8676 |
| BANKING77 | B0 | 0.8147 | 0.4939 | 0.7625 |
| BANKING77 | B3-P | 0.8083 | 0.5016 | 0.7591 |
| BANKING77 | B3-H | **0.8172** | **0.5461** | **0.7725** |
| MASSIVE | B0 | **0.7985** | 0.1717 | **0.7091** |
| MASSIVE | B3-P | 0.7901 | **0.1760** | 0.7012 |
| MASSIVE | B3-H | 0.7857 | 0.1674 | 0.6945 |

B3-H 相对 B0 的平均变化：

- CLINC150：OOS −1.11 pp，F1 −0.92 pp；
- BANKING77：OOS **+5.21 pp**，F1 **+0.99 pp**，routing +0.25 pp；
- MASSIVE：OOS −0.43 pp，F1 −1.46 pp，routing −1.28 pp。

**B3-H 只在 BANKING77 显示稳定正收益；B3-P 在三个数据集都没有稳定收益。**

### 5.4 最佳配置

| Dataset | Best config | Known routing | OOS recall | Open-set F1 |
|---|---|---:|---:|---:|
| CLINC150（历史） | Qwen 10-shot B0 | 0.9817 | 0.9010 | 0.9226 |
| BANKING77 | GTE 10-shot B3-H | 0.8625 | 0.6368 | **0.8306** |
| MASSIVE | Qwen 10-shot B0 | 0.8272 | 0.5298 | 0.7982 |

模型平均（4 模型 3 数据集平均）：

- Qwen B3-H F1 0.8178，B0 0.8135；Qwen 是最强平均模型；
- GTE B3-H F1 0.7840，B0 0.7832；OOS +3.37 pp，routing −0.85 pp；
- BGE、E5 的 B3-H 平均下降。

---

## 6. 训练式对照（BANKING77，5-shot）

### SetFit-BGE

- 训练数据：40 类 × 5 条 = 200 条 support；
- 官方 SetFit 1.2，BGE-small body fine-tune，1 epoch，head 16 epochs；
- 训练 283.6 s，2438 个 embedding steps；
- 与 B0 使用同一 validation 阈值选择、同一 BANKING test。

| Method | Known acceptance | Known routing | OOS recall | Open-set F1 |
|---|---:|---:|---:|---:|
| BGE-B0 5-shot | 0.9044 | 0.7925 | 0.5342 | **0.7505** |
| SetFit-BGE 5-shot | 0.8769 | 0.7256 | 0.3618 | 0.6696 |
| BGE 投影适配器 5-shot | 0.8806 | 0.7813 | 0.5197 | 0.7394 |

SetFit 在训练集上很快达到 100% support accuracy，但 test 明显过拟合：
known routing 下降 6.69 pp，OOS recall 下降 17.24 pp，open-set F1 下降
8.09 pp。投影适配器（residual 384→256→384，监督对比 + identity）也没有
超过 BGE-B0，但比 SetFit 更接近。

结论：**训练式少样本适配在本数据/预算下没有带来开放集收益；不能把
support 上的高准确率当作 test 提升。**

---

## 7. 阈值行为与 MASSIVE 的失败模式

### 7.1 Validation vs test

| Dataset | B0 val OOS | B0 test OOS | B0 val F1 | B0 test F1 |
|---|---:|---:|---:|---:|
| CLINC150 | 0.925 | 0.832 | 0.971 | 0.877 |
| BANKING77 | 0.399 | 0.494 | 0.696 | 0.763 |
| MASSIVE | 0.096 | 0.172 | 0.710 | 0.709 |

CLINC 和 BANKING 的 test OOS 与 validation 有差异，但仍能比较方法。

### 7.2 MASSIVE 的 F1-first 退化

MASSIVE B0 的具体阈值行为：

| Model | Shot | Known acceptance | OOS recall | Unknown AUROC | tau |
|---|---:|---:|---:|---:|---:|
| BGE | 5 | 0.999 | 0.000 | 0.765 | 0.527 |
| BGE | 10 | 0.9995 | 0.000 | 0.770 | 0.542 |
| GTE | 5 | 1.000 | 0.000 | 0.767 | 0.478 |
| GTE | 10 | 0.991 | 0.025 | 0.782 | 0.566 |
| Qwen | 5 | 0.920 | 0.477 | 0.859 | 0.679 |
| Qwen | 10 | 0.922 | 0.530 | 0.863 | 0.702 |
| E5 | 5 | 0.910 | 0.177 | 0.748 | 0.824 |
| E5 | 10 | 0.943 | 0.165 | 0.772 | 0.829 |

在 36 类的 open-set macro-F1 中，OOS 只占 1/36 的权重；validation 上
F1-first 规则允许 BGE/GTE 选到 known acceptance≈1.0 的极低阈值。AUROC 仍
约 0.77，说明分数并非完全不可分，但所选阈值几乎不拒 OOS。**这不是
B3-H 的失败，而是阈值选择目标在 MASSIVE 上的失败。** 本轮没有用 test 重新
选择阈值；若要在 MASSIVE 上得到业务可用结果，必须在新的独立协议上改用
更合理的 recall/oracle 约束，而不是继续调这个 test。

---

## 8. 错误与混淆分析

### 8.1 BANKING77 最易混淆的类（GTE 10-shot B3-H）

| True | Predicted | Count | Example |
|---|---|---:|---|
| `pending_top_up` | `top_up_failed` | 7 | Is there a reason my top up hasn't... |
| `transfer_not_received_by_recipient` | `balance_not_updated_after_bank_transfer` | 6 | The transfer I did hasn't arrived |
| `transfer_timing` | `balance_not_updated_after_bank_transfer` | 6 | ... |
| `transfer_fee_charged` | `top_up_by_bank_transfer_charge` | 6 | ... |
| `wrong_exchange_rate_for_cash_withdrawal` | `card_payment_wrong_exchange_rate` | 6 | ... |

完整 Top-15 与示例在 `results/gate3_banking_confusions.csv`。

### 8.2 四个可追踪案例

| Dataset | Type | True | Pred | Top-3 / decision |
|---|---|---|---|---|
| CLINC150 | known correct | insurance_change | insurance_change | 0.936 / 0.700 / 0.569, accepted |
| CLINC150 | known rejected | restaurant_reviews | OOS | 0.644 / 0.641 / 0.620, below tau |
| BANKING77 | known misclassified | card_arrival | card_linking | 0.837 / 0.832 / 0.819 |
| BANKING77 | OOS false accept | OOS | card_payment_not_recognised | 0.793 / 0.757 / 0.720 |
| MASSIVE | OOS correct reject | OOS | OOS | 0.682 / 0.640 / 0.635, below tau |
| MASSIVE | known rejected | audio_volume_up | OOS | top-1 audio_volume_mute 0.715 |

完整字段（文本、真实类、Top-3、分数、tau）见 `results/gate3_error_cases.csv`。

---

## 9. 图表索引

| Figure | Content |
|---|---|
| `figures/gate3_dataset_overview.png` | 长度、test 不平衡、role 计数、domain 覆盖 |
| `figures/gate3_domain_distribution.png` | 各数据集 known 类的 domain 分布 |
| `figures/gate3_heatmap_macro_f1.png` | 模型 × 数据集 B0 5-shot F1 |
| `figures/gate3_heatmap_oos_recall.png` | 模型 × 数据集 B0 5-shot OOS recall |
| `figures/gate3_shot_curves.png` | 5 vs 10-shot，四模型三数据集 |
| `figures/gate3_ablation.png` | B0/B3-P/B3-H 消融 |
| `figures/gate3_validation_tradeoff.png` | Qwen 10-shot validation 阈值权衡 |
| `figures/gate3_banking_confusions.png` | BANKING77 类混淆 |
| `figures/gate3_cost_scatter.png` | 模型效果 vs 编码成本 |
| `figures/gate3_training.png` | SetFit/Adapter 训练曲线与测试对比 |
| `figures/gate3_streamlit_demo.png` | 真实 Streamlit 运行截图 |

---

## 10. 结论

### 10.1 研究问题回答

- **RQ1**：Qwen 平均最强；GTE 在 BANKING77 最强且编码成本低；BGE/E5 较弱。
- **RQ2**：B3-P 没有稳定收益，不能用“多原型”作为普适改进。
- **RQ3**：B3-H 在 BANKING77 有稳定正收益（OOS +5.21 pp，F1 +0.99 pp），
  在 MASSIVE/CLINC150 没有；收益取决于场景和类内结构。
- **RQ4**：MASSIVE 的主要限制是阈值目标/分数分离，而不是 B3 表示本身；
  训练式方法在 5-shot 上明显过拟合。

### 10.2 最有价值的发现

**开放集方法的收益是场景条件的，不是模型榜单上的单调改进。** B3-H 在
细粒度、单领域的 BANKING77 上通过局部近邻补足原型边界；在 MASSIVE 上，
F1-first 阈值先把已知接受率推到 0.99+，OOS 几乎不拒，任何分类器改进都被
阈值目标吞掉。后续工作必须把“阈值目标”作为一等方法问题，而不是在固定
test 上继续调分类器。

### 10.3 限制

1. 单 seed；没有方差估计。
2. CLINC150 test 已查看，只能作历史/追加分析。
3. BANKING77/MASSIVE test 已在 Gate 3 使用一次；SetFit/Adapter 复用同一
   BANKING test，不是盲测。
4. MASSIVE known 类有 17/18 domain 覆盖，类别 test 数极不平衡。
5. 阈值选择采用 F1-first；MASSIVE 的 OOS 结果受此严重限制。
6. 没有做可靠的 near-domain/far-domain 统一定义；该分析标记为 NOT RUN。
7. SetFit 仅 5-shot 200 条，未调超参数；其负结果不能外推为“SetFit 永远差”。

### 10.4 建议

- 课程展示中把 **B3-H 的 BANKING77 正收益**作为主要正面结果；
- 把 **MASSIVE 阈值失败**作为方法学发现，而不是隐藏；
- 若继续研究，先在新的独立 unknown-class 协议上设计 recall-constrained
  threshold，再比较 B3-H/B1；不要在当前 test 上调参。
- 训练式方向若要继续，应使用更多 support / 更强正则或完整 SetFit 调参，
  并重新划独立 test。

---

## 11. 复现

```powershell
cd D:\code\intent-open-set
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

.\.venv\Scripts\python.exe -m src.gate3.prepare
.\.venv\Scripts\python.exe -m src.gate3.encode --model bge
.\.venv\Scripts\python.exe -m src.gate3.encode --model qwen --device cuda --batch-size 8
.\.venv\Scripts\python.exe -m src.gate3.encode --model gte --device cuda --batch-size 16
.\.venv\Scripts\python.exe -m src.gate3.encode --model e5 --device cuda --batch-size 32
.\.venv\Scripts\python.exe -m src.gate3.run_matrix --phase validation
.\.venv\Scripts\python.exe -m src.gate3.freeze
.\.venv\Scripts\python.exe -m src.gate3.run_matrix --phase test
.\.venv\Scripts\python.exe -m src.gate3.aggregate
.\.venv\Scripts\python.exe -m src.gate3.make_figures
.\.venv\Scripts\python.exe -m src.gate3.train_adapter
.\.venv\Scripts\python.exe -m src.gate3.train_setfit
.\.venv\Scripts\python.exe -m streamlit run app.py
```

关键结果：

- `results/gate3_matrix.csv`
- `results/gate3_test_metrics.csv`
- `results/gate3_validation_metrics.csv`
- `results/gate3_summary_by_dataset_method.csv`
- `results/gate3_b3h_vs_b0.csv`
- `results/gate3_best_configs.csv`
- `results/gate3_freeze.json`
- `results/gate3_setfit_metrics.csv`
- `results/gate3_adapter_metrics.csv`
- `results/gate3_error_cases.csv`

日志：`logs/gate3_*.log`。
