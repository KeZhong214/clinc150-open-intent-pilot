# Gate 1 Vibe Coding 记录

本文件记录本轮从 Gate 0 转向 Gate 1 主线实验的实际修改与真实报错。实验结果
只以 `results/` 和 `logs/` 为准。

## 1. 本轮范围调整

用户明确要求：

- 不再扩展防御性实验和备用路线；
- 固定 seed=42、5-shot；
- 顺序：BGE B0 → Qwen B0 → B2 与 B0 对照 → 真实结果分析/图表；
- 不提前展开 B1/B3、其他 seed、10-shot；
- 完成后提交并推送到原仓库。

因此本轮只新增 Gate 1 最小闭环，没有实现 B1/B3、Streamlit 或多 seed 框架。

## 2. 新增/修改文件

### 代码

| 文件 | 作用 |
|---|---|
| `src/encode_embeddings.py` | BGE/Qwen 编码 2200 条文本，缓存向量和 metadata |
| `src/run_baseline.py` | B0/B2 原型、validation 阈值选择、test 一次评测 |
| `src/aggregate_results.py` | 汇总指标、预测、control 行、错误案例 |
| `src/make_figures.py` | 生成 7 张核心图 |
| `tests/test_gate1_logic.py` | B0 阈值、toy metrics、B2 双条件测试 |

### 结果与报告

- `results/gate1_metrics.csv`
- `results/b0_baseline.csv`
- `results/b2_dynamic_rejection.csv`
- `results/predictions_b0.csv`
- `results/predictions_b2.csv`
- `results/thresholds_*.json`
- `results/val_grid_*.csv`
- `results/error_cases_qwen_b2.csv`
- `figures/*.png`
- `reports/gate1_baseline.md`

### 日志

- `logs/gate1_encode_bge.log`
- `logs/gate1_qwen_smoke.log`
- `logs/gate1_encode_qwen.log`
- `logs/gate1_bge_b0.log`
- `logs/gate1_qwen_b0.log`
- `logs/gate1_bge_b2.log`
- `logs/gate1_qwen_b2.log`
- `logs/gate1_aggregate.log`
- `logs/gate1_figures.log`
- `logs/gate1_logic_tests.log`
- `logs/gate1_all_tests.log`

## 3. 关键实现决策

1. **BGE 先跑通**：CPU，2200 条文本，6.617 s，确认原型/阈值/test 链路。
2. **Qwen 作为主模型**：CUDA FP16，batch=8，2200 条，27.741 s，
   峰值 allocated 1,155.5 MiB，无 OOM。
3. **固定 prompt**：使用 `configs/pilot.yaml` 中的客服意图 instruction，
   support/query/val/test 同前缀；prompt hash 写入 embedding metadata。
4. **阈值网格很小**：`tau` 101 个分位点；`gamma` 0 + 19 个分位点；
   `lambda ∈ {0, 0.5, 1}`；只优化 validation 上
   “known acceptance ≥ 0.90 时最大化 OOS recall”。
5. **B2 的 `q_c` 只用 validation 真实标签**：每类 20 条，10% 分位点；
   不碰 test。
6. **test 保护**：`run_baseline.py` 发现同方法结果文件已存在时默认拒绝运行，
   防止无意中第二次评测 test。没有用 `--force` 重跑正式结果。
7. **B2-control**：`lambda=0, gamma=0` 直接复用 B0 的 test 指标，
   不重新读取 test。

## 4. 真实报错与修复

### 4.1 Qwen FP16 归一化误差

第一次 Qwen smoke 报：

```text
RuntimeError: Embeddings are not unit-normalized: max error=0.00019598007202148438
```

原因：模型在半精度下做了 normalize，转 float32 后仍有约 2e-4 的 norm 误差。
处理：

- 在 float32 中显式做一次 L2 归一化；
- 重新检查 `norm_max_abs_error`，成功 smoke 为 `5.96e-8`；
- 全量编码后为 `1.19e-7`；
- 这不是改模型输出，只是数值归一化。

### 4.2 ST 参数弃用

`sentence-transformers 6.3` 提示：

```text
The `tokenizer_kwargs` argument was renamed ... use `processor_kwargs`
```

处理：改为 `processor_kwargs={"padding_side": "left"}`，Qwen smoke 通过。

### 4.3 Windows allocator 提示

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在 Windows 上提示
不支持。处理：移除该设置；batch=8 实际未 OOM。

### 4.4 B2 结果不理想

Qwen B2 在 test 上比 B0 略差；这不是代码报错，而是真实结果。没有为了让
B2 好看而修改阈值、换 seed、换 test 或重新调 gamma。报告中保留负结果。

## 5. 结果摘要

| Model | Method | OOS recall | Open-set macro-F1 |
|---|---|---:|---:|
| BGE | B0 | 0.9420 | 0.8929 |
| BGE | B2 | 0.9540 | 0.9041 |
| Qwen | B0 | 0.9740 | 0.9570 |
| Qwen | B2 | 0.9680 | 0.9499 |

Qwen B0 最强；BGE B2 小幅提升；Qwen B2 未复现验证集收益；
B2 的 `gamma` 被验证集选为 0，margin gate 本轮未实际生效。

## 6. 没有做的事

- 没有 B1/B3；
- 没有 10-shot、seed 43/44；
- 没有 test 调参或重复评测；
- 没有因为 B2 结果不理想而挑选有利 seed；
- 没有把 smoke 或 validation 数字写成 test 结果。
