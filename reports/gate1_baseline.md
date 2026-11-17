# Gate 1：B0 / B2 Baseline 与动态拒识对照

**状态**：已完成真实运行。seed=42，5-shot，20 个已知意图，BGE 与 Qwen 两个
编码器，B0 与 B2 两种拒识策略；test 只在阈值冻结后评测一次。

**一句话结论**：Qwen3-Embedding-0.6B 的 B0 是本轮最强配置（open-set
macro-F1 0.9570）。B2 在 BGE 上把 OOS recall 从 0.9420 提到 0.9540，
open-set macro-F1 从 0.8929 提到 0.9041；但在 Qwen 上没有复现验证集收益，
open-set macro-F1 降到 0.9499。B2 的 margin gate 在本轮被验证集选为
`gamma=0`，实际没有启用。

> Gate 2 后续使用独立的 hard-OOS 协议继续尝试 B2 改进，见
> [`reports/gate2_method.md`](gate2_method.md)。本报告 Gate 1 结果保持冻结。

本报告中的每个数字都来自 `results/` 下的 CSV；没有使用论文数字、占位数字
或 test 调参。

---

## 1. 实验口径

| 项目 | 设置 |
|---|---|
| 已知意图 | 20 个，官方 10 个域各 2 个 |
| Support | 官方 train，seed=42，K=5，每类 5 条，共 100 条 |
| Calibration known | 20 类官方 val 全部 400 条 |
| Calibration OOS | 官方 oos_val 100 条 |
| Final known test | 20 类官方 test 全部 600 条 |
| Final OOS test | 官方 oos_test 1000 条 |
| 编码文本 | 100 support + 2100 eval = 2200 条 |
| B0 | 单原型 + 全局阈值 `tau` |
| B2 | 单原型 + 类别阈值 `tau_c` + margin gate `gamma` |
| 阈值选择 | 只用 validation；test 冻结后只评测一次 |
| 额外监督 | B2 的 `q_c` 使用每类 20 条带标签 validation 数据 |

B0：

```text
p_c = normalize(mean(support_c))
s_c = cos(z(x), p_c)
accept iff s1 >= tau
```

B2：

```text
q_c = 10th percentile of s_c on validation rows whose true label is c
tau_c = (1 - lambda) * tau + lambda * q_c
m = s1 - s2
accept iff s1 >= tau_c[predicted class] and m >= gamma
lambda in {0, 0.5, 1}
```

阈值搜索：

- `tau`：validation known top-1 分数的 101 个分位点；
- `gamma`：0 加上 validation known margin 的 19 个 0–90% 分位点；
- 目标：validation known acceptance ≥ 0.90 时最大化 OOS recall；
- `lambda`、`tau`、`gamma` 全部只在 validation 上选。

---

## 2. 编码器实际运行

### BGE-small-en-v1.5

| 项目 | 实际值 |
|---|---|
| Revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| 设备 | CPU |
| 维度 | 384 |
| 文本数 | 2200 |
| Batch | 32 |
| 编码耗时 | 6.617 s |
| 吞吐 | 332.5 texts/s |
| 峰值进程 RSS | 964.6 MiB |
| 向量归一化误差 | 1.19e-7 |

### Qwen3-Embedding-0.6B

| 项目 | 实际值 |
|---|---|
| Revision | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| 设备 | CUDA，RTX 5070 Laptop |
| 精度 | float16 |
| 维度 | 1024 |
| 文本数 | 2200 |
| Batch | 8 |
| 编码耗时 | 27.741 s |
| 吞吐 | 79.3 texts/s |
| 峰值 allocated VRAM | 1,155.5 MiB |
| 峰值 reserved VRAM | 1,170.0 MiB |
| 峰值进程 RSS | 2,920.5 MiB |
| 模型加载 | 6.8 s |
| OOM 重试 | 无 |

Qwen prompt 固定为：

```text
Instruct: Given a customer service message, identify the supported intent it expresses.
Query: 
```

SHA-256：`4bbdf1c169061f64ba742f64bca5e422d05e94ccc7261d29495c233ed9a90005`。
support / calibration / test 使用同一前缀。BGE 不加 instruction。

实现细节：Qwen FP16 输出直接转 float32 时 norm 误差约 `1.96e-4`，因此在
float32 中显式再做一次 L2 归一化，最终误差 `1.19e-7`。这不是改模型输出，
只是消除半精度归一化的数值误差。

---

## 3. 阈值选择结果

| Model | Method | lambda | tau | gamma | Val known accept | Val OOS recall |
|---|---|---:|---:|---:|---:|---:|
| BGE | B0 | — | 0.75210 | — | 0.9100 | 0.9800 |
| BGE | B2 | 0.5 | 0.73320 | 0.00000 | 0.9350 | 0.9900 |
| Qwen | B0 | — | 0.75216 | — | 0.9400 | 1.0000 |
| Qwen | B2 | 0.5 | 0.70735 | 0.00000 | 0.9525 | 1.0000 |

B2 的 `tau_c` 实际范围：

- BGE：`q_c` 0.7012–0.8803，`tau_c` 0.7172–0.8068；
- Qwen：`q_c` 0.6733–0.8726，`tau_c` 0.6903–0.7900。

`gamma=0` 不是手填的。validation 上存在正的、几乎同分的候选
（BGE `gamma≈0.000545`，Qwen `gamma≈0.005499`），与 `gamma=0` 的
known acceptance / OOS recall 指标相同；按预注册的 tie-break 规则选择了
更小的 `gamma=0`。因此本轮 **不能声称 margin gate 带来了收益**。

---

## 4. 最终 test 结果

固定 test：600 条 known + 1000 条 OOS。Test 不参与任何阈值选择。

| Model | Method | Known accept | Known routing | Known macro-F1 | OOS recall | OOS FAR | OOS AUROC | Open-set macro-F1 | Open-set acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE | B0 | 0.9117 | 0.8933 | 0.9310 | 0.9420 | 0.0580 | 0.9773 | 0.8929 | 0.9237 |
| BGE | B2 | 0.9117 | 0.8950 | 0.9333 | **0.9540** | **0.0460** | 0.9776 | **0.9041** | **0.9319** |
| BGE | B2-control* | 0.9117 | 0.8933 | 0.9310 | 0.9420 | 0.0580 | 0.9773 | 0.8929 | 0.9237 |
| Qwen | B0 | **0.9667** | **0.9600** | **0.9758** | **0.9740** | **0.0260** | **0.9950** | **0.9570** | **0.9688** |
| Qwen | B2 | 0.9617 | 0.9550 | 0.9728 | 0.9680 | 0.0320 | 0.9930 | 0.9499 | 0.9631 |
| Qwen | B2-control* | 0.9667 | 0.9600 | 0.9758 | 0.9740 | 0.0260 | 0.9950 | 0.9570 | 0.9688 |

`*` B2-control 是 `lambda=0, gamma=0`，等价于 B0 接受条件。它直接复用 B0
已经评测的 test 指标，没有第二次读取 test。

### 4.1 Qwen B0 是当前主结果

Qwen B0 相对 BGE B0：

- Known acceptance：0.9667 vs 0.9117（+5.50 pp）
- Known correct routing：0.9600 vs 0.8933（+6.67 pp）
- OOS recall：0.9740 vs 0.9420（+3.20 pp）
- Open-set macro-F1：0.9570 vs 0.8929（+6.41 pp）
- OOS AUROC：0.9950 vs 0.9773（+1.77 pp）

Qwen 的 closed-set Top-1 为 0.9917，B0 的 known routing 为 0.9600：
主要损失来自 20 条 known 被拒识，而不是类别混淆。

### 4.2 BGE：B2 有小幅正收益

B2 相对 B0：

- Known acceptance：0.9117 → 0.9117（不变）
- Known correct routing：0.8933 → 0.8950（+0.17 pp）
- OOS recall：0.9420 → 0.9540（**+1.20 pp**）
- OOS FAR：0.0580 → 0.0460（−1.20 pp）
- Open-set macro-F1：0.8929 → 0.9041（**+1.11 pp**）
- Open-set accuracy：0.9237 → 0.9319（+0.81 pp）

混淆矩阵对应的计数变化：

- known rejected：53 → 53（不变）
- OOS accepted：58 → 46（少接收 12 条 OOS）

也就是说，在 BGE 上，B2 的收益主要来自**多拒绝 12 条未知样本，而没有
增加已知样本误拒**。这个结果支持“类别阈值在部分类别上更合适”的假设。

### 4.3 Qwen：B2 没有复现验证集收益

B2 相对 B0：

- Known acceptance：0.9667 → 0.9617（−0.50 pp）
- Known correct routing：0.9600 → 0.9550（−0.50 pp）
- OOS recall：0.9740 → 0.9680（**−0.60 pp**）
- OOS FAR：0.0260 → 0.0320（+0.60 pp）
- Open-set macro-F1：0.9570 → 0.9499（**−0.72 pp**）
- Open-set accuracy：0.9688 → 0.9631（−0.56 pp）
- Known rejected：20 → 23（多拒 3 条）
- OOS accepted：26 → 32（多接收 6 条）

这不是实现错误：B2 的 validation known acceptance 从 0.9400 升到 0.9525，
OOS recall 保持 1.0000；按预注册目标它确实是 validation 上的更优配置。
问题是 validation 的 OOS recall 已经到天花板 1.0，没有剩余区分度，
选择过程实际偏向“更高 known acceptance”，而 `q_c` 只来自每类 20 条
validation 样本，迁移到 test 后不稳定。

---

## 5. 为什么 B2 在 Qwen 上退化

### 5.1 高 `q_c` 类别被推高阈值，低 `q_c` 类别被放宽

Qwen B2 的 `lambda=0.5`，所以 `tau_c` 是全局 `tau=0.7074` 与类别
`q_c` 的中点。`q_c` 高的类阈值变严，`q_c` 低的类阈值变松。

Qwen test 上 known routing 变化（30 条/类）：

| 变化 | 类别 |
|---|---|
| 变差 | `report_lost_card` −10.0 pp，`meal_suggestion` −6.7 pp，`user_name` −3.3 pp，`change_ai_name` −3.3 pp，`insurance_change` −3.3 pp |
| 变好 | `balance`、`who_made_you`、`restaurant_reviews`、`plug_type`、`spelling` 各 +3.3 pp |
| 不变 | 其余 10 类 |

`report_lost_card` 的 `q_c=0.8520`，`tau_c` 被推到约 0.780，
多拒了 3 条本来正确的 known；`meal_suggestion` 的 `q_c=0.8318`，
也多拒了 2 条。反过来，`balance` 的 `q_c=0.6733`，阈值降低后多接收了
known，但也更容易接收 OOS。净效果是负的。

BGE 上也有同样方向，但低 `q_c` 类（`spelling` 0.7012、
`measurement_conversion` 0.7153）的收益更大，足以抵消高 `q_c` 类的损失，
所以总体 B2 略好。

### 5.2 Margin gate 没有进入最终配置

最终 B2 的 `gamma=0`，所以“相似度间隔门控”在本轮没有实际作用。
validation 上最好的正 `gamma` 候选与 0 同分（见第 3 节），没有证据表明
margin gate 单独带来了提升。

这是一个需要保留的负结果：在当前 20 类、5-shot、单 seed 设置下，
类别阈值是 B2 的主要变量，margin gate 没有可测收益。

---

## 6. 错误案例（Qwen B2，test）

| 类型 | 真实标签 | 预测 | s1 | margin | 文本 |
|---|---|---|---:|---:|---|
| 正确路由 | insurance_change | insurance_change | 0.913 | 0.218 | i would like to change my insurance policy |
| 已知错分 | user_name | who_made_you | 0.743 | 0.010 | what do i go by |
| 已知误拒 | insurance_change | OOS | 0.759 | 0.120 | i need to sign up for a new allstate plan |
| 正确拒识 | OOS | OOS | 0.678 | 0.051 | how much has the dow changed today |
| 未知误接收 | OOS | balance | 0.764 | 0.087 | can you tell me what credit cards i have open |

最后一条说明：OOS 文本包含 “credit cards” 词汇，被嵌入模型拉向
`balance`，且分数超过该类阈值；这是语义近邻导致的误接收，单靠 margin
不一定能修复。

---

## 7. 图表

| 图 | 文件 | 说明 |
|---|---|---|
| B0/B2 四项指标 | `figures/gate1_b0_b2_comparison.png` | 两模型 × 两方法 |
| 分数分布 | `figures/score_distribution.png` | known/OOS test top-1 分布与 B0 tau |
| validation trade-off | `figures/gate1_validation_tradeoff.png` | B0 扫描曲线与 B2 选中点 |
| Qwen B0 混淆矩阵 | `figures/baseline_confusion.png` | baseline confusion |
| Qwen B2 混淆矩阵 | `figures/gate1_qwen_b2_confusion.png` | B2 confusion |
| 每类 B2−B0 变化 | `figures/gate1_per_class_delta.png` | known routing delta |
| q_c 与 B2 效果 | `figures/gate1_qc_vs_delta.png` | 阈值强度与类别收益/损失 |

`figures/gate1_score_distribution.png` 与 `score_distribution.png` 内容相同。

---

## 8. 结果解释与限制

1. **Qwen 明显强于 BGE**：四个主要指标都更好，符合主模型选择。
2. **BGE B2 有小幅正收益**：OOS recall +1.2 pp，open-set F1 +1.1 pp，
   known acceptance 不变。
3. **Qwen B2 退化**：validation OOS recall 已饱和，`q_c` 仅基于每类
   20 条 validation，选择偏向 known acceptance；test 上 known 和 OOS
   同时略差。
4. **Margin gate 未被选中**：`gamma=0`；正 gamma 候选与 0 同分。不能据此
   断言 margin gate 在其他数据/模型上无效。
5. **单 seed、单 shot**：seed=42、K=5，没有方差估计；结论只适用于当前
   固定划分。
6. **20 类 pilot，不是完整 150 类 CLINC150**：known test 只有 600 条，
   OOS test 1000 条；不能把结果外推为完整 CLINC150 的 SOTA。
7. **test 只读一次**：没有用 test 重新选阈值、gamma、lambda 或类别。
8. **B1/B3 未运行**：本轮按用户要求只做 B0/B2 主线。
9. **官方数据残余重复**：Gate 0 已记录 5 处跨分区文本碰撞；本轮
   support 与 eval 无交集，但最终报告仍需保留该限制。

---

## 9. 复现与日志

```powershell
cd D:\code\intent-open-set
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

.\.venv\Scripts\python.exe -m src.encode_embeddings --model bge --device cpu --batch-size 32
.\.venv\Scripts\python.exe -m src.encode_embeddings --model qwen --device cuda --batch-size 8

.\.venv\Scripts\python.exe -m src.run_baseline --model bge  --method B0 --seed 42 --shot 5
.\.venv\Scripts\python.exe -m src.run_baseline --model qwen --method B0 --seed 42 --shot 5
.\.venv\Scripts\python.exe -m src.run_baseline --model bge  --method B2 --seed 42 --shot 5
.\.venv\Scripts\python.exe -m src.run_baseline --model qwen --method B2 --seed 42 --shot 5

.\.venv\Scripts\python.exe -m src.aggregate_results
.\.venv\Scripts\python.exe -m src.make_figures
```

关键日志：

- `logs/gate1_encode_bge.log`
- `logs/gate1_qwen_smoke.log`
- `logs/gate1_encode_qwen.log`
- `logs/gate1_bge_b0.log`
- `logs/gate1_qwen_b0.log`
- `logs/gate1_bge_b2.log`
- `logs/gate1_qwen_b2.log`
- `logs/gate1_aggregate.log`
- `logs/gate1_figures.log`
- `logs/gate1_all_tests.log`

关键结果：

- `results/gate1_metrics.csv`
- `results/b0_baseline.csv`
- `results/b2_dynamic_rejection.csv`
- `results/predictions_b0.csv`
- `results/predictions_b2.csv`
- `results/thresholds_{bge,qwen}_{b0,b2}.json`
- `results/encoding_bge.json`
- `results/encoding_qwen.json`
- `results/error_cases_qwen_b2.csv`

---

## 10. 结论

本轮主线已经得到可复现的真实对照：

- **主结果：Qwen B0**，open-set macro-F1 0.9570，OOS recall 0.9740；
- **BGE B2** 在同等 known acceptance 下将 OOS recall 提高 1.2 pp，
  open-set F1 提高 1.1 pp；
- **Qwen B2** 没有复现 validation 收益，open-set F1 下降 0.72 pp；
- **margin gate** 本轮未启用（`gamma=0`），不能宣称有效。

下一步若继续，优先级应是 10-shot / seed 复核和针对 Qwen 的类别阈值稳定性，
而不是直接加 B1/B3。
