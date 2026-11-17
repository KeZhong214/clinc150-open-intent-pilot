# Gate 2：困难未知意图验证与 B2-v2 类别边界校准

**状态**：Gate 2 已完成真实运行、冻结和一次性 hard-OOS test 评测。  
**结论**：B2-v2 在 validation 上明显优于 B0，但收益没有稳健迁移到 65 个
disjoint held-out unknown classes 的 test；主目标“Qwen 上 B2 改进成功”**尚未达成**。
最终建议仍是 Qwen B0 作为当前默认方案。

本报告使用新协议，**不覆盖 Gate 1 的原始 OOS 结果**。Gate 1 的
`results/gate1_metrics.csv` 和原始 `oos_test` 指标保持不变。

---

## 1. 目标与协议

### 1.1 目标

针对 Gate 1 中 Qwen B2 在原始 OOS 上不如 B0 的问题，尝试：

1. 用更有区分度的 hard-OOS validation 取代/补充原始 100 条 OOS val；
2. 修正 `q_c` 的定义，明确使用真实类别分数 `s_c(x)`，而不是 top-1 `s1`；
3. 只提出一个 B2-v2：利用 hard-OOS validation 中“被预测为某类”的未知
   样本上尾，估计类别拒识边界，并向全局阈值收缩；
4. 在冻结后，对 65 个 disjoint held-out 未知类的 test 只评测一次。

### 1.2 固定内容

- 已知类：Gate 0 的 20 类，不变；
- support：seed=42，K=5，每类 5 条，不变；
- 编码器：Qwen3-Embedding-0.6B，revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，不变；
- prompt：Gate 1 的固定客服意图 instruction，不变；
- 原始 Gate 1 test 结果：不改写、不重训、不用于选 B2-v2。

### 1.3 Hard-OOS 类别划分

CLINC150 剩余 130 个未选类别按固定哈希排序：

```text
SHA-256("hard-oos|42|<intent>")
```

前 65 类为 development unknown classes，后 65 类为 held-out final unknown
classes。没有使用任何模型分数或 test 表现来划分类别。

| 集合 | 类别数 | 样本 | 用途 |
|---|---:|---:|---|
| hard-OOS dev | 65 | 1,300（官方 val，20/类） | hard-OOS validation |
| hard-OOS held-out | 65 | 1,950（官方 test，30/类） | 冻结后一次性 final test |

文本隔离检查：hard-OOS val/test 与 known support、known val/test、原始
`oos_val`/`oos_test`、以及彼此之间的规范化文本交集全部为 0。

原始 `oos_val` 仍保留在 calibration 中；原始 `oos_test` 不进入新协议。
新协议的 hard-OOS final test 是 600 known test + 1,950 hard-OOS test。

---

## 2. q_c 口径修正

Gate 1 的 `run_baseline.py` 中，B2 的 `q_c` 是：

```text
q_c = 10th percentile of top-1 s1 among validation rows with true label c
```

严格定义应为：

```text
q_c = 10th percentile of s_c(x) among validation rows with true label c
```

其中 `s_c(x)` 是真实类 `c` 的原型分数，不是 top-1 分数。两者不同仅当
validation 样本被 top-1 分错；本数据 400 条 validation 中只有 2 条分错，
所以多数 `q_c` 相同，最大绝对差 0.0419，平均绝对差 0.0021。

处理：

- **Gate 1 历史代码和结果不修改**；
- Gate 2 的 B2-original 保留 Gate 1 的旧口径，作为历史对照；
- B2-v2 使用修正后的 `s_c(x)` 口径；
- 这个修正对最终 B2-v2 的影响有限，但口径已明确。

---

## 3. 方法

### 3.1 B0

单原型 + 全局阈值：

```text
p_c = normalize(mean(support_c))
s_c = cos(z(x), p_c)
accept iff s1 >= tau
```

`tau` 在合并 validation（400 known + 100 original OOS + 1,300 hard-OOS）
上选 101 个分位点，目标为 known acceptance ≥ 0.90 时最大化 OOS recall。

### 3.2 B2-original

保留 Gate 1 公式，但在新 validation 上重新校准：

```text
q_c_old = q10(s1 on known val rows with true label c)
tau_c = (1 - lambda) * tau + lambda * q_c_old
accept iff s1 >= tau_c and margin >= gamma
lambda ∈ {0, 0.5, 1}
```

`tau`、`lambda`、`gamma` 都在新 validation 上重新选，保证与新协议的其他
方法使用同一验证集。

### 3.3 B2-v2（本轮唯一改进版）

对每个已知类 `c`：

```text
q_c = q10({s_c(x) | x in known val, y(x)=c})
u_c = q99({s1(x) | x in hard-oos val, pred(x)=c})
n_c = |{x in hard-oos val | pred(x)=c}|
alpha_c = n_c / (n_c + kappa)
target_c = max(tau, min(u_c, max(tau, q_c)))
tau_c = (1 - alpha_c) * tau + alpha_c * target_c
accept iff s1 >= tau_c and margin >= gamma
```

固定：

```text
kappa = 10
u_quantile = 0.99
q_c_quantile = 0.10
```

关键性质：

- **不会把类别阈值降到全局 `tau` 以下**，避免 Gate 1 B2 在低 `q_c` 类上
  无依据放宽阈值；
- `u_c` 只看 hard-OOS validation 中**预测到该类**的未知样本，不需要未知
  类的真实标签；
- `n_c` 小的时候 `alpha_c` 小，向全局阈值收缩；
- `gamma` 仍从 validation margin 分位点中选；只有验证集真实选中且有效时
  才声称 margin 有贡献。

`kappa=10` 与 `u_quantile=0.99` 是在 validation-only 的小网格中固定的，
没有使用 hard-OOS final test。完整探索日志见
`logs/gate2_v2_search.log`，可运行分析见
`src/gate2_hyperparameter_check.py`。

### 3.4 额外监督预算

新协议中所有方法都使用：

- 100 条 K-shot support（已知标签）；
- 400 条 known validation（已知标签）；
- 100 条原始 OOS validation（只需 OOS 身份）；
- 1,300 条 hard-OOS validation（只需 OOS 身份，不需要 65 个未知类的类名）。

B2-v2 额外使用的是 hard-OOS 样本的 **predicted class** 和分数，不要求未知
类标签。原 Gate 1 的原始 OOS test 与 hard-OOS final test 都没有参与训练或
阈值选择。

---

## 4. Validation 结果

Validation = 400 known + 100 original OOS + 1,300 hard-OOS。

| Method | Known acceptance | Known routing | Hard-OOS recall | Hard open-set F1 | Hard AUROC | Open-set acc |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.900 | 0.895 | 0.9731 | 0.9048 | 0.9822 | 0.9547 |
| B2-original | 0.905 | 0.900 | 0.9769 | 0.9153 | 0.9848 | 0.9588 |
| **B2-v2** | **0.910** | **0.905** | **0.9862** | **0.9270** | **0.9852** | **0.9671** |

B2-v2 相对 B0：

- Known acceptance：+1.0 pp
- Known routing：+1.0 pp
- Hard-OOS recall：+1.31 pp
- Hard open-set F1：+2.22 pp
- Open-set accuracy：+1.24 pp

Validation 上 B2-v2 是三个方法中最优的。特别是 hard-OOS accepted 从
35 条降到 18 条，同时 known rejected 从 40 条降到 36 条。

### 4.1 阈值行为

B2-v2 的 `tau=0.78594`，20 个类中只有 5 个类的阈值实际高于全局值：

| intent | q_c | u_c | n_c | tau_c | 提高 |
|---|---:|---:|---:|---:|---:|
| directions | 0.8051 | 0.8190 | 121 | 0.8036 | +0.0177 |
| report_lost_card | 0.8520 | 0.8106 | 98 | 0.8084 | +0.0224 |
| credit_score | 0.8710 | 0.8378 | 95 | 0.8329 | +0.0470 |
| reminder_update | 0.8501 | 0.8509 | 54 | 0.8401 | +0.0541 |
| meal_suggestion | 0.8318 | 0.8183 | 72 | 0.8144 | +0.0284 |

其余 15 类保持 `tau`，没有被降低。`gamma` 被选为 0，因此本轮 **margin
gate 没有实际贡献**，不能把收益归因于 margin。

---

## 5. 冻结与 hard-OOS test

Validation 配置写入：

```text
results/gate2_freeze.json
```

冻结内容包括：

- B0/B2/B2-v2 的 `tau`、`lambda`、`gamma`、`q_c`、`u_c`、`n_c`、`alpha_c`、
  `tau_c`；
- validation 指标；
- 目标函数、`kappa`、`u_quantile`；
- hard-OOS split hash；
- `hard_oos_test_evaluated: false`。

冻结后，才编码 1,950 条 held-out hard-OOS test 文本：

- Batch 8；
- 56.28 s；
- 34.6 texts/s；
- 峰值 allocated VRAM 1,156.4 MiB；
- 无 OOM。

随后 `run_gate2.py --phase test` 只运行一次，评测 600 known + 1,950 hard-OOS。
脚本会拒绝第二次读取 test；`gate2_freeze.json` 现在记录
`hard_oos_test_evaluated: true`。

### 5.1 Test 结果

| Method | Known acceptance | Known routing | Hard-OOS recall | Hard open-set F1 | Hard AUROC | Open-set acc |
|---|---:|---:|---:|---:|---:|---:|
| **B0** | **0.9050** | **0.9033** | 0.9426 | **0.8796** | 0.9796 | 0.9333 |
| B2-original | 0.8883 | 0.8867 | 0.9364 | 0.8694 | 0.9743 | 0.9247 |
| B2-v2 | 0.8933 | 0.8917 | **0.9482** | 0.8795 | **0.9800** | **0.9349** |

B2-v2 相对 B0：

- Hard-OOS recall：+0.56 pp（多拒 11 条 hard-OOS）
- Known routing：**−1.17 pp**（多拒 7 条 known）
- Hard open-set F1：−0.01 pp（基本持平）
- Open-set accuracy：+0.16 pp
- Known rejected：57 → 64
- Hard-OOS accepted：112 → 101

B2-v2 相对 B2-original：

- Hard-OOS recall：+1.18 pp
- Known routing：+0.50 pp
- Hard open-set F1：+1.01 pp
- Open-set accuracy：+1.02 pp

因此 B2-v2 修正了原始 B2 的方向性错误，但**没有超过 B0** 的 known routing
和 hard open-set F1。另一个需要明确报告的事实是：test known acceptance
为 B0 0.9050、B2 0.8883、B2-v2 0.8933；B2 和 B2-v2 在 test 上都没有
达到 0.90。0.90 只约束 validation，不能保证 test 也满足。

---

## 6. 为什么 validation 收益没有迁移

1. **held-out unknown classes 更难/不同**：B0 的 hard-OOS recall 从
   validation 的 0.9731 降到 test 的 0.9426，下降 3.05 pp。final 65 类
   与 dev 65 类完全不重合。
2. **B2-v2 只修改了 5 个类的阈值**，收益和损失都集中在这几个类：
   - hard-OOS accepted 的下降主要来自 `report_lost_card`（14 → 4）和
     `meal_suggestion`（20 → 14），但被 `change_ai_name`（25 → 28）、
     `uber`（0 → 2）、`balance`（13 → 14）等未提高阈值的类部分抵消；
   - known routing 的损失主要来自 `meal_suggestion`（−10.0 pp）、
     `reminder_update`（−6.7 pp）、`report_lost_card`（−6.7 pp）、
     `credit_score`（−3.3 pp）、`directions`（−3.3 pp）；
   - 只有 `balance` 和 `change_ai_name` 各 +3.3 pp。
3. B2-v2 选到的全局 `tau=0.7859` 低于 B0 的 `0.7904`；未提高阈值的 15 个类
   因此比 B0 略宽松，会增加 known 和 hard-OOS 的接受。5 个被提高的类才是
   净减少 hard-OOS 误接收的主要来源。这个“全局略松 + 少数类收紧”的结构
   在 test 上没有产生稳定的净收益。
4. 类别阈值来自 dev 65 类的 predicted-class 分布；这组分布不能代表 held-out
   65 类的组合，说明 **class-level 的 hard-OOS 上尾估计对未知类组成敏感**。
5. 15 个类的阈值完全等于全局 `tau`，因此 B2-v2 的改善空间本来就有限；
   test 上 11 条 hard-OOS 的收益不足以抵消 7 条 known 的损失。

### 6.1 最有价值的发现

**在 65 个 dev unknown classes 上获得的 hard-OOS 收益，不能可靠迁移到
65 个 disjoint unknown classes。** 类条件阈值的效果高度依赖于未知类组成；
validation 上的 1.31 pp hard recall / 2.22 pp F1 提升，在 test 上缩到
0.56 pp hard recall / 基本持平的 F1，并伴随 known routing 下降。

这不是“把参数再调一下”就能在同一个 test 上解决的问题；继续在当前
hard-OOS test 上试参数会构成 test 调参。

---

## 7. 与 Gate 1 原始 OOS 结果的关系

- Gate 1 原始 `oos_test` 结果没有被覆盖；
- 新协议不把原始 `oos_test` 混入 hard-OOS test；
- 新协议中所有方法都在新 validation 上重新校准，因此不能把本报告的 test
  数字与 Gate 1 的 `gate1_metrics.csv` 直接逐项比较；
- 原始 Gate 1 的结论仍是：Qwen B0 最强，B2 没有成功。

---

## 8. 限制

1. 单 seed=42、5-shot、20 类；没有多 seed 方差。
2. hard-OOS dev/final 各 65 类，类别组成不同；这是本报告的核心发现，也是
   最大限制。
3. B2-v2 的 `kappa=10`、`u_quantile=0.99` 是在 validation-only 小网格中
   固定的；有 validation 过拟合风险。
4. 每类 known validation 只有 20 条，`q_c` 和 `u_c` 都是小样本分位数。
5. `gamma=0` 被选中，margin gate 没有贡献；不能宣称 margin 有效。
6. known test 600 条在 Gate 1 已用于原始 OOS 评测；本轮没有用它选 B2-v2，
   但它不是“从未见过”的 known 样本。
7. 只使用 Qwen3-Embedding-0.6B；没有 B1/B3、其他 seed、10-shot 或新编码器。

---

## 9. 复现命令

```powershell
cd D:\code\intent-open-set
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# 1. 冻结类别划分与 hard-OOS manifest
.\.venv\Scripts\python.exe -m src.prepare_gate2

# 2. 只编码 validation hard-OOS
.\.venv\Scripts\python.exe -m src.encode_gate2_hard_oos --split hard_oos_val --batch-size 8

# 3. validation-only 校准
.\.venv\Scripts\python.exe -m src.run_gate2 --phase validation

# 4. 检查 validation 后冻结
.\.venv\Scripts\python.exe -m src.run_gate2 --phase validation --freeze

# 5. 冻结后才编码 held-out hard-OOS test
.\.venv\Scripts\python.exe -m src.encode_gate2_hard_oos --split hard_oos_test --batch-size 8

# 6. 一次性 test
.\.venv\Scripts\python.exe -m src.run_gate2 --phase test

# 7. 图表与全量测试
.\.venv\Scripts\python.exe -m src.make_gate2_figures
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

注意：一旦 `results/gate2_freeze.json` 和
`results/gate2_test_metrics.csv` 已存在，不要重跑 test；脚本会拒绝。

---

## 10. 结论

- B2-v2 的 validation 方法设计成立：+1.0 pp known routing，+1.31 pp
  hard-OOS recall，+2.22 pp hard open-set F1；
- 冻结后的一次性 test 只显示 +0.56 pp hard-OOS recall，known routing
  下降 1.17 pp，hard open-set F1 基本持平；
- **Qwen 上 B2/B2-v2 的改进尚未成功，B0 仍是当前默认方案**；
- 原 B2 在 test 上更差（hard recall −0.62 pp，known routing −1.67 pp），
  B2-v2 至少修正了方向，但不足以推翻 B0；
- 不建议在当前 hard-OOS test 上继续调 B2-v2。若继续，需要新的独立
  held-out unknown classes 或更强的跨类别收缩/稳定性设计。
