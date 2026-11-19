# PPT 主讲分析图（固定 Commit）

固定数据版本：`4c674cbc19ca3c623535033393878d20b5f04ad9`

本目录只放**新图**，不覆盖历史 `figures/` 下的 87 张图。所有图由
`src/ppt_final_analysis/plot_*.py` 从 `results/*.csv` 生成；每张图均有：

```text
*_data.csv    # 数据整理表
*.png         # 3999×2250（16:9），300 dpi
*.svg         # 可编辑矢量
*_notes.md    # 输入、筛选、核验、口径、图注、结论
```

全量校验：

- `audit.log`
- `audit_manifest.json`（含源 CSV 与交付文件 SHA256）

---

## 1. 主讲图（P0-1 至 P0-4）

| 图 | 文件 | 建议页 | 一句话结论 |
|---|---|---|---|
| P0-1 基线热力矩阵 | `P04_baseline_heatmap_10shot_B0.png/.svg` | 第 4 页 | 任务域差异显著，模型不存在绝对通吃；B0/10-shot 只是 72 格中的 12 格。 |
| P0-2 B4 模块归因 | `P06_b4_module_ablation_dev.png/.svg` | 第 6 页 | 收益主要来自拒识头，而不是更复杂的可训练门控。 |
| P0-3 配对 seed 差值 | `P08_b4_paired_seed_deltas.png/.svg` | 第 8 页 | 6 组编码器的 RejectOnly−Full 均值均为正，但效应随编码器/seed 变化。 |
| P0-4 Agent v1 TSR | `P10_agent_v1_corrected_tsr.png/.svg` | 第 10 页 | Qwen Plain42 为 85/120（70.8%），高于 GTE/Qwen Frozen 的 72/120（60.0%）；模拟任务，不宣称显著。 |

## 2. 备选/附录

| 图 | 文件 | 用途 | 关键限制 |
|---|---|---|---|
| OOS 附录热力图 | `A01_oos_heatmap_10shot_B0.png/.svg` | P0-1 附录版 | 与 P04 完全相同的 12 格，仅把指标换成 OOS Recall。 |
| Qwen LoRA seed 波动 | `P07_qwen_lora_three_seeds_historical.png/.svg` | 第 7 页备选 | Gate7 **historical exploratory test**，不是新盲测；保留 seed42 负结果。 |
| 业务族成功率 | `A02_agent_family_regraded.png/.svg` | 第 10 页备用子图/附录 | 只展示 8 个 n≥5 的业务族，不是 18 族全量。 |
| 失败归因 | `A03_agent_failure_attribution.png/.svg` | 附录 | 仅 v1 corrected 的 131 个失败；单标签主因，不重复计数。 |

## 3. 视觉规范

- 主色：深海军蓝 `#18304B`；主方法青绿 `#2B8C83`；
  对照橙 `#D38C44`；基线灰 `#9AA5AF`；Qwen 蓝 `#386FB6`。
- 中文：Microsoft YaHei；图内技术标识保留英文。
- 16:9、3999×2250、300 dpi；导出 SVG 保留可编辑文字。
- 不使用 3D、雷达、环形、多层旭日、截断柱状图夸大小差异。

## 4. 使用建议

1. 第 4 页用 `P04`，不要把 12 格说成 72 组总排名；
2. 第 6 页用 `P06`，强调“Fusion 单独训练 gate 不提升”；
3. 第 7 页若时间够，用 `P07` 说明 LoRA seed 波动；
4. 第 8 页用 `P08`，说明每行 n=3、短线是 ±1 sample std，不是 95% CI；
5. 第 10 页用 `P10`，只引用 v1 corrected，不把 v2 的 74.17% 混入；
6. 附录可用 `A01`、`A02`、`A03`；不要用旧 `gate8_1_failure_heatmap.png` 代替 v1 校正结果。

## 5. 不可混淆的口径

- P04：Gate3 历史矩阵，CLINC150 为 historical exploratory；B0/10-shot 固定切片。
- P06：Gate5 BANKING77 GTE Frozen **development validation**，不是最终 test。
- P08：Gate7 dev_select，B4 头 seed 42/43/44；LoRA(42/43/44) 是编码器训练 seed。
- P07：Gate7 阈值修复后的 **historical exploratory test**。
- P10/A02/A03：Gate8.2 v1 **同轨迹离线修正评分**，不是 v2，也不是 DeepSeek API 模式。

## 6. 脚本

```text
src/ppt_final_analysis/common.py
src/ppt_final_analysis/plot_p04.py
src/ppt_final_analysis/plot_p06.py
src/ppt_final_analysis/plot_p08.py
src/ppt_final_analysis/plot_p10.py
src/ppt_final_analysis/plot_p07.py
src/ppt_final_analysis/plot_a02.py
src/ppt_final_analysis/plot_a03.py
src/ppt_final_analysis/audit.py
```

复现：

```powershell
.\.venv\Scripts\python.exe -m src.ppt_final_analysis.plot_p04
.\.venv\Scripts\python.exe -m src.ppt_final_analysis.plot_p06
.\.venv\Scripts\python.exe -m src.ppt_final_analysis.plot_p08
.\.venv\Scripts\python.exe -m src.ppt_final_analysis.plot_p10
.\.venv\Scripts\python.exe -m src.ppt_final_analysis.plot_p07
.\.venv\Scripts\python.exe -m src.ppt_final_analysis.plot_a02
.\.venv\Scripts\python.exe -m src.ppt_final_analysis.plot_a03
.\.venv\Scripts\python.exe -m src.ppt_final_analysis.audit
```
