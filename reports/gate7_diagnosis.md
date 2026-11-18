# Gate 7-A 诊断：Qwen LoRA 为什么验证提升、历史测试下降

**分支**：`gate7-qwen-diagnosis`  
**起点**：`gate6-qwen-lora@a0e01f3`  
**本文件不训练、不使用 final-test 标签。**

---

## 1. 双层开发集冻结

从 Gate3 BANKING77 calibration-known 800 + dev-unknown 720 建立两个互斥的
内层开发集：

| Split | Known | Unknown | 用途 |
|---|---:|---:|---|
| `dev_threshold` | 40 类 × 10 = 400 | 9 类 × 40 = 360 | 只选 tau |
| `dev_select` | 40 类 × 10 = 400 | 另外 9 类 × 40 = 360 | 只选训练配方/报告 |

- 随机种子：42；
- known 每类按 sample_id 排序后打乱，10/10 切分；
- unknown 9/9 按整类互斥；
- sample_id overlap 0，text_sha256 overlap 0，unknown class overlap 0；
- `final_unknown` / `known_test` 未参与任何选择。

文件：

- `data/manifest/gate7_dev_threshold.csv` SHA `a6821544...`
- `data/manifest/gate7_dev_select.csv` SHA `07ba9a1f...`
- `results/gate7_dev_split.json`

`dev_threshold` unknown classes：pending_transfer, getting_virtual_card,
declined_card_payment, wrong_amount_of_cash_received,
direct_debit_payment_not_recognised, supported_cards_and_currencies,
exchange_rate, transaction_charged_twice, get_disposable_virtual_card。

`dev_select` unknown classes：failed_transfer, getting_spare_card,
order_physical_card, top_up_limits, request_refund, edit_personal_details,
reverted_card_payment?, pending_cash_withdrawal, card_delivery_estimate。

这是内部稳健性开发集，不是新的独立盲测。

---

## 2. Frozen / Plain / Anchor 在 dev_select 的结果

每个 encoder 的 tau 只在 dev_threshold 上选，再报告 dev_select。

| Encoder | Method | Dev-select F1 | OOS recall | Routing | AUROC |
|---|---|---:|---:|---:|---:|
| Qwen Frozen | B0 | 0.7518 | 0.5500 | 0.8100 | 0.8462 |
| Qwen Frozen | B3-H | 0.7904 | 0.6639 | 0.8250 | 0.8749 |
| Qwen Frozen | B4-RejectOnly | 0.7956 | 0.6667 | 0.8275 | 0.8751 |
| Qwen Plain | B0 | 0.7841 | 0.6444 | 0.8225 | 0.8714 |
| Qwen Plain | B3-H | **0.7984** | **0.6917** | 0.8300 | **0.8896** |
| Qwen Plain | B4-RejectOnly | **0.7995** | **0.7000** | 0.8250 | 0.8810 |
| Qwen Anchor | B0 | 0.7691 | 0.6278 | 0.8050 | 0.8627 |
| Qwen Anchor | B3-H | 0.7942 | 0.6694 | 0.8225 | 0.8784 |

**关键：在 dev_select 上 Plain 确实优于 Frozen**（B3-H .7984 vs .7904），
但 Gate6 的 historical test 是 Plain .7972 < Frozen .8064。  
所以 Gate6 的“验证提升、test 下降”不是随机噪声，也不是简单的训练失败：
**dev_select 与 final-test 的 unknown 类分布/迁移不同**。

---

## 3. H1：microbatch=2 的 SupCon 实际只有 6 个向量

当前 `train_qwen_lora.py`：

- 每个 microbatch 2 个 anchor + 2 个 positive + 2 个 hard negative；
- `supcon_loss(z, y)` 的相似度矩阵是 **6×6**；
- `grad_accum=8` 只是把 8 个 microbatch 的梯度相加，**不会**构造 48×48 的
  InfoNCE 相似度矩阵；
- 400/2=200 microbatches/epoch；200/8=25 optimizer steps/epoch，恰好整除；
- 如果余数不为 0，原循环会丢弃最后一个不足 accum 的 microbatch；本配置
  余数为 0，不是当前 bug 的触发条件，但实现上仍是一般性风险。

toy 文件：`results/gate7_toy_batch.json`；图：`figures/gate7_toy_6x6.png`。

---

## 4. H2：hard negative 过强且重复

`gate6_qwen_hard_negatives.csv`：

| 指标 | 值 |
|---|---:|
| hard negative cosine mean | 0.807 |
| hard negative cosine p90 | 0.883 |
| positive − negative margin mean | **−0.0126** |
| margin < 0 的数量 | **203 / 400** |
| margin p10 | −0.182 |
| hard negative 重复数 | **243** |
| unique hard negative | **157 / 400** |

**超过一半的 anchor 与 hard negative 的余弦高于其 positive**；而 hard negative
只来自 400 support 内的最近异类。这会把“语义非常接近但标签不同”的银行意图
强行推开，造成少样本过拟合。不能把语义接近直接说成标签错误；但训练目标确实
可能过强。

图：`figures/gate7_hard_negative.png`。

---

## 5. H3/H4：分数、校准与 unknown 迁移

- Plain 的 dev_select known 均值 0.882，OOS 均值 0.764，margin 0.097；
- Frozen 的 known 均值 0.892，OOS 均值 0.794，margin 0.077；
- Plain 的 OOS 均值下降，但 tail 分离度提高，dev_select AUROC 0.890 > 0.875；
- 到了 final-test，Plain 的 OOS AUROC 0.868 < Frozen 0.887，OOS recall
  0.562 < 0.586；
- 这说明 **dev-unknown 与 final-unknown 的类语义族不同**，不能只用
  dev_select 的 AUROC 推断 final-test。

图：`figures/gate7_score_distribution.png`、`gate7_acceptance_curve.png`。

---

## 6. H5：Anchor 覆盖范围

Anchor 只对 microbatch 的 anchor 子集做余弦约束，没有约束 positive/negative
和类间结构；support cosine 从 Plain 0.892 提到 Anchor 0.936，但 dev_select
B3-H 从 .7984 降到 .7942。当前证据不支持“更大 Anchor 一定更好”；后续只做
轻量结构约束实验，不盲目扩大矩阵。

---

## 7. A 阶段结论

1. Gate6 的 val/test 分叉是真实存在的分布迁移信号，不是随机波动；
2. microbatch 的 6×6 对比矩阵是实际实现，grad accumulation 不能替代；
3. hard negative 的 203/400 负 margin 和 243 重复是明确风险；
4. Plain 在 dev_select 上优于 Frozen，但 final-test 不优；必须保留这个
   负结果，不能把 dev_select 当最终结论；
5. 下一步 B 阶段只在新 dev_select 上筛选学习率/损失，不碰 final-test。

---

## 8. 复现

```powershell
.\.venv\Scripts\python.exe -m src.gate7.make_dev_split
.\.venv\Scripts\python.exe -m src.gate7.diagnose
.\.venv\Scripts\python.exe -m src.gate7.make_figures
```

产物：

- `results/gate7_dev_diagnosis.csv`
- `results/gate7_dev_acceptance_curves.csv`
- `results/gate7_dev_unknown_class_fpr.csv`
- `results/gate7_hard_negative_stats.json`
- `results/gate7_toy_batch.json`
- `results/gate7_dev_scores.npz`
