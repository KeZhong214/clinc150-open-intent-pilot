# Gate 7.1 修复报告：阈值协议、补齐、命名与成果冻结

**分支**：`gate7-final-fix`  
**基线**：`gate7-qwen-diagnosis@d1b2a7a`  
**日期**：2026-11-19  
**性质**：协议修复与成果整理，不新增模型、不重新调参、不进入 Gate 8。

---

## 1. 修复的问题

### 1.1 B4-RejectOnly 历史 test 阈值来源错误

原 `src/gate7/final_test.py`：

- B0/B3-H：使用 `gate7_dev_{tag}.csv` 的 Gate7 `dev_threshold` tau；
- B4-RejectOnly：读取 Gate6 的 `gate6_qwen_*_reject_threshold.json`；
- 因此同一张 test 表里方法使用了不同校准协议。

修复前实际阈值：

| Encoder | Old B4 tau (Gate6) | Gate7 dev_threshold tau |
|---|---:|---:|
| frozen | 0.445477 | **0.442460** |
| plain42 | 0.413429 | **0.416832** |
| plain43 | 0.442301 | **0.418433** |
| plain44 | 0.433705 | **0.451184** |

B0/B3-H 未受影响。

### 1.2 修复方式

- `final_test.py` 增加 `--protocol-fix-v2`；
- v2 中 B0/B3-H/B4-RejectOnly 全部从 `gate7_dev_{tag}.csv` 读取 tau；
- 对 12 行逐项断言 `tau_used == dev_threshold_tau`；
- v2 写入独立文件：
  - `results/gate7_final_historical_test_v2.csv`
  - `results/gate7_final_freeze_v2.json`
- 原 `gate7_final_freeze.json` 的 `test_evaluated=true` 保留，不重置；
- 原 test/summary/freeze 复制为 `*_pre_fix.*`，并记录 SHA-256：
  `results/gate7_protocol_fix_manifest.json`。

### 1.3 v2 修复前后差异

| Encoder | Δ tau | Old F1 | New F1 | Δ F1 | Δ OOS |
|---|---:|---:|---:|---:|---:|
| frozen | −0.003018 | 0.812741 | 0.812998 | +0.000257 | −0.002632 |
| plain42 | +0.003403 | 0.807982 | 0.809062 | +0.001080 | +0.006579 |
| plain43 | −0.023869 | 0.836114 | 0.835568 | −0.000546 | −0.027632 |
| plain44 | +0.017478 | 0.831755 | 0.832067 | +0.000313 | +0.015789 |

- AUROC 不变（阈值不改变排序）；
- RejectOnly 3-seed mean：**0.82528 → 0.82557（+0.03 pp）**；
- frozen：0.81274 → 0.81300；
- **原结论没有改变**：Qwen RejectOnly 略高于 frozen，2/3 seed 胜出；
- 逐行结果：`results/gate7_protocol_fix_diff.csv`。

---

## 2. 补齐 B4 稳定性缺失配置

补跑唯一缺失键：

- Encoder：`qwen_plain42`
- Method：`B4-Full`
- Seed：`44`

安全合并方式：

- 先保存原 35 行为
  `results/gate7_stability_b4_results_pre_supplement.csv`，SHA-256
  `3c7a922f...`；
- `stability_supplement.py` 追加一行，不覆盖旧行；
- 断言最终 **36 行**、`(encoder,method,seed)` 唯一；
- 保存：
  - `results/gate7_stability_qwen_plain42_full_seed44.pt`
  - `results/gate7_stability_qwen_plain42_full_seed44_history.csv`
  - `results/gate7_stability_supplement.json`

补跑结果：

| Encoder | Method | Seed | dev F1 | OOS | Routing | AUROC |
|---|---|---:|---:|---:|---:|---:|
| qwen_plain42 | B4-Full | 44 | 0.791814 | 0.677778 | 0.8250 | 0.880208 |

补齐后的 Qwen Plain42 稳定性：

| Method | dev F1 mean ± std | test F1 mean ± std |
|---|---:|---:|
| B4-RejectOnly | **0.79968 ± 0.00018** | **0.80877 ± 0.00092** |
| B4-Full | 0.78918 ± 0.00664 | 0.79560 ± 0.00430 |

- 全部 36 组 checkpoint/history 完整；
- `gate7_stability_checkpoint_hashes.csv` 重新生成，36 个 SHA 不同；
- **Qwen Plain43/44 没有完成完整 B4 三种子矩阵**，只完成了 LoRA 编码器与最终
  test；这一点在 `configs/gate7_stability.yaml` 和报告中明确标注，不误称全部
  Qwen LoRA Adapter 都完成了交叉多种子实验。

---

## 3. 命名与记录修正

### 3.1 Qwen-Opt-RejectOnly → Qwen-Plain42-RejectOnly

旧结果标签 `Qwen-Opt-RejectOnly` 实际加载的是 **Gate6 Qwen Plain LoRA seed42**
+ RejectOnly，不是新优化配方。已：

- `gate7_agent_summary.csv`、`gate7_agent_results.csv`、by-family、traces 的主
  `router` 改为 `Qwen-Plain42-RejectOnly`；
- 增加 `router_legacy` 列，保留 `Qwen-Opt-RejectOnly` 兼容映射；
- 原文件复制为 `*_pre_rename.csv`；
- `configs/gate7_agent.yaml` 和 `combine_agent.py` 更新；
- Agent 代码保留旧名 alias。

### 3.2 P3 状态

`gate7_final_freeze.json` 中旧的 `P3 incomplete` 已过时。P3 实际完成
2 epochs / 50 steps，adapter SHA `84926b52...`，dev B3-H 0.7945 < P0 0.7984，
因此**完成但未选中**。

处理：

- 修改前 freeze 复制为 `gate7_final_freeze_pre_p3_update.json`；
- 原 test SHA、test_evaluated、selection 的模型范围不变；
- 增加 `p3_status=COMPLETED_NOT_SELECTED`、adapter SHA、50 steps、dev 值；
- `amendments` 记录修改前 SHA 和保留文件；
- 不把 P3 追加到 final test。

### 3.3 三种 seed 的区分

| 名称 | 取值 | 说明 |
|---|---|---|
| 支持集抽样 seed | 42 | Gate3 support 固定；本轮不重采样 |
| LoRA 训练 seed | 42/43/44 | 初始化、样本顺序、正/负样本选择、CUDA 非确定性 |
| B4 训练 seed | 42/43/44 | B4 初始化、episode mask、伪 OOS 采样、训练顺序 |

B4 seed 不改变 LoRA adapter；LoRA seed 不改变 B4 的 seed 定义。  
`gate7_stability` 的 `encoder=qwen_plain42` 只表示固定 adapter seed42，
`seed=42/43/44` 是 B4 模块训练 seed。

### 3.4 Agent 阈值来源

Agent 120 任务仍使用 **Gate5/Gate6 原阈值**：

- GTE RejectOnly 0.4417；
- Qwen Frozen RejectOnly 0.4455；
- Qwen Plain42 RejectOnly 0.4168。

分类 C 使用 Gate7 `dev_threshold`。两者是不同协议，本轮没有把 Agent 结果与
C 的分类指标混为同一校准，也没有为统一阈值重跑 Agent；旧 Agent 结果保留。

---

## 4. 最终统一结果

### 4.1 Qwen Plain 3-seed（protocol_fix_v2）

| Seed | Test B3-H | Test RejectOnly |
|---|---:|---:|
| 42 | 0.7969 | 0.8091 |
| 43 | 0.8159 | 0.8356 |
| 44 | 0.8077 | 0.8321 |
| mean ± std | **0.8068 ± 0.0095** | **0.8256 ± 0.0144** |

Frozen：B3-H **0.8051**、RejectOnly **0.8130**。

- B3-H 均值 +0.18 pp；
- **RejectOnly 均值 +1.26 pp，2/3 seed 胜出**；
- 仍低于 GTE Frozen-RejectOnly 0.8435。

### 4.2 稳定性

| Encoder | RejectOnly dev | Full dev |
|---|---:|---:|
| GTE Frozen | **0.8144 ± .0015** | 0.8105 ± .0022 |
| GTE LoRA42 | **0.8047 ± .0008** | 0.7984 ± .0027 |
| GTE LoRA43 | **0.8277 ± .0005** | 0.8152 ± .0023 |
| GTE LoRA44 | **0.8266 ± .0009** | 0.8026 ± .0014 |
| Qwen Frozen | **0.7950 ± .0008** | 0.7752 ± .0047 |
| Qwen Plain42 | **0.7997 ± .0002** | 0.7892 ± .0066 |

历史 test 3-seed：

| Encoder | RejectOnly | Full |
|---|---:|---:|
| GTE Frozen | **0.8415 ± .0007** | 0.8404 ± .0006 |
| GTE LoRA42 | **0.8122 ± .0008** | 0.8014 ± .0012 |
| GTE LoRA43 | **0.8249 ± .0002** | 0.8130 ± .0005 |
| GTE LoRA44 | **0.8181 ± .0005** | 0.7998 ± .0017 |
| Qwen Frozen | **0.8125 ± .0005** | 0.8002 ± .0052 |
| Qwen Plain42 | **0.8088 ± .0009** | 0.7956 ± .0043 |

**B4-RejectOnly > B4-Full 在全部 6 个 encoder 上成立**。

---

## 5. 自动化检查

新增/更新 `tests/test_gate7.py`：

- B4 test 阈值与 Gate7 dev_threshold 完全一致；
- 原 test/freeze 保留且 SHA 可核对；
- 36 组稳定性配置、无重复键、缺失配置已补齐；
- 历史结果均 `historical_exploratory=True`；
- 汇总与原始 CSV 的均值一致；
- Agent 名称 `Qwen-Plain42-RejectOnly` 与 legacy mapping 一致；
- P3 完成但未选中。

最终测试结果：**69 tests OK**，日志 `logs/gate7_1_all_tests.log`。

---

## 6. 结论

1. **阈值协议修复不改变原结论**：Qwen Plain RejectOnly 3-seed mean 从
   0.82528 变为 0.82557，仍仅小幅高于 frozen 0.81300；B3-H 不变。
2. **B4-RejectOnly 的模块优势稳定成立**，且不是某一个 seed 的偶然。
3. **P3 完成但没有超过 P0**；P1/P2 也没有；优化未成功。
4. **Qwen-Opt 命名是历史误称**，实际是 Qwen Plain42；已改为
   `Qwen-Plain42-RejectOnly`，保留 legacy alias。
5. Gate7 的历史 test 仍是探索性，不是新盲测；没有用 test 选模型/seed/阈值。

---

## 7. 限制

- Qwen Plain43/44 没有完整 B4 三种子矩阵；
- 无 bootstrap 置信区间；
- 无新的独立 unknown-class 协议；
- 120 条 Agent 任务是旧任务，且使用 Gate5/6 旧阈值；
- LLM API 无凭据，`NOT RUN_API_BLOCKED`。
