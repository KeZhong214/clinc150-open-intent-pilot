# Gate 3 PPT 大纲（10–12 页）

**题目**：面向智能客服的少样本开放意图识别：多模型跨场景评测与原型—近邻协同优化

## Slide 1 — 标题与一句话结论

- 3 数据集 × 4 冻结编码器 × 5/10-shot × B0/B3-P/B3-H = 72 格真实矩阵。
- 一句话：**B3-H 在 BANKING77 有效，在 MASSIVE/CLINC 不稳定；Qwen 平均最强，
  GTE 在 BANKING 最有性价比。**
- 视觉：`figures/gate3_heatmap_macro_f1.png` 的缩略图。

## Slide 2 — 研究问题

- RQ1：四个编码器在 3 个场景的少样本开放意图表现差异。
- RQ2：自适应多原型是否改善类内多样性。
- RQ3：原型—近邻融合是否在 B0 之上提高 OOS 拒识。
- RQ4：收益/退化与类别结构、阈值、分数间隔的关系。
- 强调：不是追求 B3 全面 SOTA，而是找适用边界。

## Slide 3 — 三个数据集

| Dataset | Known / dev / final | 场景 |
|---|---:|---|
| CLINC150 | 20 / — / original OOS | 多领域客服历史 Pilot |
| BANKING77 | 40 / 18 / 19 | 单领域细粒度银行客服 |
| MASSIVE | 35 / 12 / 13 | 多领域语音助手 |

- 统一字段、固定 seed=42、support 5/10 嵌套。
- 文本隔离：support/calibration/test 交集 0。
- 视觉：`figures/gate3_dataset_overview.png` 或 `gate3_domain_distribution.png`。

## Slide 4 — 协议与防泄漏

- 训练/support 只来自官方 train；校准只用 validation/dev + dev-unknown。
- 阈值：known acceptance ≥ 0.90，validation F1-first。
- 72 个阈值先 freeze，再 test 一次。
- CLINC test 已查看，标 historical；BANKING/MASSIVE 是新增独立 test。
- 视觉：协议流程图（文字框即可）。

## Slide 5 — 四个编码器

- BGE-small / Qwen3-Embedding-0.6B / GTE-ModernBERT-base / E5-base-v2。
- 固定 revision、prompt/prefix、128 token、L2 归一化。
- 编码 11,011 条/模型；Qwen 160.6 s，GTE 12.7 s，E5 6.0 s，BGE 19.2 s。
- 视觉：`figures/gate3_cost_scatter.png`。

## Slide 6 — 三种方法

- B0：单原型 + 全局 tau。
- B3-P：top-25% 分散度类做 2 个 KMeans 原型。
- B3-H：`0.5 × B3-P + 0.5 × top-2 support neighbor`。
- 三者的唯一变量：类别表示 + 近邻融合；阈值规则一致。
- 视觉：方法示意图（原型、近邻、融合）。

## Slide 7 — 主矩阵：模型 × 数据集

- B0 5-shot F1 热图：Qwen 平均最高，GTE 在 BANKING 最好。
- 10-shot 平均 F1 从 0.7614 到 0.8042。
- 视觉：`figures/gate3_heatmap_macro_f1.png` +
  `figures/gate3_shot_curves.png`。

## Slide 8 — B3 消融：场景条件收益

| Dataset | B0 F1 | B3-H F1 | Δ OOS |
|---|---:|---:|---:|
| BANKING77 | 0.7625 | **0.7725** | **+5.21 pp** |
| CLINC150 | 0.8768 | 0.8676 | −1.11 pp |
| MASSIVE | 0.7091 | 0.6945 | −0.43 pp |

- B3-P 没有稳定收益。
- B3-H 只在 BANKING77 同时提高 routing 和 OOS。
- 视觉：`figures/gate3_ablation.png`。

## Slide 9 — 阈值行为：MASSIVE 的负结果

- F1-first 规则在 MASSIVE 的 BGE/GTE 上选到 known acceptance ≈ 1.0，
  OOS recall 0–2.5%。
- AUROC 仍约 0.77：分数有区分度，但阈值目标不适合。
- 这是方法学发现；没有用 test 重新调阈值。
- 视觉：`figures/gate3_validation_tradeoff.png` 的 MASSIVE 面板。

## Slide 10 — 训练式少样本对照

- SetFit-BGE：200 条 support，BGE fine-tune，283.6 s。
- SetFit open-set F1 0.6696，BGE-B0 0.7505；明显过拟合。
- 投影适配器：0.7394，也未超过 B0。
- 视觉：`figures/gate3_training.png`。

## Slide 11 — 错误案例与 Demo

- BANKING 混淆：`pending_top_up → top_up_failed`，转移/余额类互相混淆。
- 4 类可追踪案例：correct / rejected / misclassified / OOS false accept。
- Streamlit：选数据集、编码器、方法，实时 Top-3 和拒识理由。
- 视觉：`figures/gate3_streamlit_demo.png` 和 confusion 图。

## Slide 12 — 结论与下一步

- B3-H 是 BANKING77 的条件性正收益，不是普适提升。
- MASSIVE 的失败主要是阈值目标，不是 B3 表示。
- 训练式方法在 5-shot 200 条上过拟合。
- 下一步：新的独立 unknown-class 协议 + recall-constrained threshold +
  多 seed/10-shot 稳定性；不在当前 test 上调参。
- 视觉：三条结论 + 一条限制。

---

## 演讲节奏建议

- 1–3 页：2 分钟（问题、数据）。
- 4–6 页：2 分钟（协议、模型、方法）。
- 7–10 页：4 分钟（结果与负结果）。
- 11–12 页：2 分钟（Demo、结论）。
- 总计约 10 分钟，留 2 分钟提问。
