# Gate 2 Vibe Coding 记录

## 1. 本轮目标

用户要求针对 Qwen B2 在 Gate 1 上不如 B0 的问题：

- 固定 20 个已知类和 seed=42 / 5-shot；
- 构造 65/65 的 hard-OOS dev/final 类别划分；
- 只做一个 B2-v2；
- 修正 `q_c` 口径；
- validation 冻结后，对 held-out hard-OOS test 只评测一次；
- 如实报告，不为了让 B2 好看而用 test 调参。

## 2. 新增/修改文件

| 文件 | 作用 |
|---|---|
| `src/prepare_gate2.py` | 65/65 hard-OOS 划分、1,300/1,950 条 manifest、隔离检查 |
| `src/encode_gate2_hard_oos.py` | 只补算 hard-OOS 文本；test 必须有 freeze 才能编码 |
| `src/analyze_gate2_validation.py` | 复核 Gate 1 阈值在 hard-OOS val 上的瓶颈 |
| `src/gate2_hyperparameter_check.py` | 小网格选择 B2-v2 的 `kappa` / `u_quantile`，只看 validation |
| `src/run_gate2.py` | 统一 validation、B0/B2/B2-v2 校准、freeze、一次性 test |
| `src/make_gate2_figures.py` | 5 张 Gate 2 图 |
| `tests/test_gate2_logic.py` | B2-v2 双条件、空 orig OOS 指标测试 |
| `tests/test_gate2_protocol.py` | 65/65、计数、文本隔离、阈值单调性、test 一次性 |
| `configs/gate2.yaml` | 冻结的 κ、u 分位点、目标、test 政策 |

新增结果/日志/报告：

- `data/manifest/gate2_hard_oos_*`
- `results/gate2_*`
- `figures/gate2_*.png`
- `logs/gate2_*.log`
- `reports/gate2_method.md`
- 本文件

## 3. 关键决策

### 3.1 q_c 口径

审查 Gate 1 代码发现：

```python
class_scores = scores.s1[val_known_mask][val_true_idx == c]
q_c[c] = np.quantile(class_scores, 0.10)
```

这用的是 true-class 行的 **top-1 s1**，而不是 true class 自身的 `s_c(x)`。
400 条 validation 中只有 2 条 top-1 错分，所以差异小，但定义确实不严谨。

处理：

- Gate1 结果不覆盖；
- Gate2 的 B2-original 保留旧口径；
- B2-v2 用正确的 `s_c(x)`；
- 报告中写出差异（max 0.0419，mean 0.0021）。

### 3.2 Hard-OOS 划分

对剩余 130 类计算：

```text
SHA-256("hard-oos|42|<intent>")
```

排序取前 65 类做 dev，后 65 类做 final。没有读取任何模型分数。
dev 的 val = 1,300 条 hard-OOS validation；final 的 test = 1,950 条
held-out hard-OOS test。文本重叠检查全部为 0。

### 3.3 B2-v2

最终公式：

```text
q_c = q10(s_c(x) | known val, true label c)
u_c = q99(s1(x) | hard-OOS val, pred(x)=c)
n_c = count(pred(x)=c)
alpha_c = n_c / (n_c + 10)
target_c = max(tau, min(u_c, max(tau, q_c)))
tau_c = (1-alpha_c) * tau + alpha_c * target_c
accept iff s1 >= tau_c and margin >= gamma
```

设计意图：

- 只用 hard-OOS 的 **predicted class**，不需要未知类真实标签；
- 不把阈值降到全局 tau 以下；
- 稀疏类别通过 `alpha_c` 向全局 tau 收缩；
- `gamma` 仍从 validation 选，不强行声称 margin 有效。

`kappa=10`、`u_quantile=0.99` 是用 `src/gate2_hyperparameter_check.py`
在 validation 上做的小网格检查后固定的；`hard_oos_test` 完全没有参与。

## 4. Validation 与冻结

Validation 结果：

| Method | Known routing | Hard-OOS recall | Hard open-set F1 |
|---|---:|---:|---:|
| B0 | 0.895 | 0.9731 | 0.9048 |
| B2-original | 0.900 | 0.9769 | 0.9153 |
| B2-v2 | **0.905** | **0.9862** | **0.9270** |

B2-v2 在 validation 上三项都优于 B0，因此写入
`results/gate2_freeze.json`，并将 `hard_oos_test_evaluated` 初始设为 false。

冻结前没有编码或读取 hard-OOS test。冻结后才运行：

```text
python -m src.encode_gate2_hard_oos --split hard_oos_test
python -m src.run_gate2 --phase test
```

## 5. 真实测试结果与结论

Held-out test（600 known + 1,950 hard-OOS）：

| Method | Known routing | Hard-OOS recall | Hard open-set F1 |
|---|---:|---:|---:|
| B0 | **0.9033** | 0.9426 | **0.8796** |
| B2-original | 0.8867 | 0.9364 | 0.8694 |
| B2-v2 | 0.8917 | **0.9482** | 0.8795 |

B2-v2 比 B0 多拒 11 条 hard-OOS，但多拒 7 条 known；hard open-set F1
基本持平，known routing 下降 1.17 pp。B2-v2 比 B2-original 好，但没有
超过 B0。

**结论：主目标未稳健达成。** B2-v2 的 validation 收益没有迁移到
disjoint unknown classes。最有价值的发现是：class-level hard-OOS 上尾
估计对未知类组成敏感，65 类 dev 的收益不能可靠外推到另一组 65 类。

## 6. 真实报错与修复

1. `select_b2_original` 中最初有一行写错的 `orig_recall` 布尔表达式；
   在正式 validation 前改为显式计算 `orig_accept` / `hard_accept`。
2. test 阶段没有原始 OOS test，`metric_payload` 对空数组求 mean 会得到
   NaN；改为 `orig_oos_recall=None`，并保持 new-protocol 指标只使用
   known test + hard-OOS test。
3. 最初保留了一个未使用的 `select_b2_v2` stub，正式运行前删除。
4. 协议测试第一次把“`tau_c <= q_c`”写错；实际约束是
   `tau_c <= max(tau, q_c)`，因为 q_c < tau 时阈值停在 tau。修正后 25 项
   测试全部通过。
5. hard test 编码脚本增加了 freeze guard：没有
   `results/gate2_freeze.json` 不能编码 test，降低误用风险。

## 7. 没有做的事

- 没有修改 Gate 1 的 `run_baseline.py`、`gate1_metrics.csv` 或原始 test 结果；
- 没有用 hard-OOS test 调 `kappa`、`u_quantile`、`tau` 或 `gamma`；
- 没有在 test 结果出来后重跑或 `--force`；
- 没有额外跑 B1/B3、多 seed、10-shot、其他编码器；
- 没有上传模型权重、embedding npz、`.venv` 或 token；
- 没有把 validation 收益写成 test 收益。
