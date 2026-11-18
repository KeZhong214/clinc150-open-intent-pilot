# Gate 7 演讲稿（约 10–12 分钟）

## 开场（约 50 秒）

Gate 7 的目标不是再训练一个更大的模型，而是回答：为什么 Qwen LoRA 在 Gate6 的验证集提升、历史 test 却下降？这是 seed 偶然、对比学习实现、表征漂移，还是 unknown 分布迁移？以及核心的 RejectOnly 结论在多种子下是否可靠。

先说结论：**单 seed42 的负结果不是 Qwen LoRA 的普遍失败。3 个 seed 的 Qwen Plain，RejectOnly test 均值 0.8256，比 frozen 0.8130 高 1.26 个百分点，2/3 个 seed 胜出。B4-RejectOnly 在 GTE/Qwen、Frozen/LoRA 上仍然稳定优于 B4-Full。P1/P2 没有超过 P0，P3 完成但 B3-H 仍低于 P0，所以没有优化成功。**

## A：诊断（约 2.5 分钟）

先冻结双层开发集：dev_threshold 400 known + 360 unknown，只选阈值；dev_select 另 400 known + 另 9 个 unknown 类，只选配方。ID、文本和 unknown 类都不重叠，test 从未参与选择。

诊断有三个发现。

第一，`microbatch=2` 时每个 microbatch 是 2 anchor + 2 positive + 2 hard negative，共 6 条；SupCon 相似度矩阵是 **6×6**。`grad_accum=8` 只是累积 8 个 6×6 的梯度，不会产生 48×48 的对比矩阵。这是真实实现，不是“把 batch 调大”的等价实验。

第二，400 条 support 的 hard negative 平均余弦 0.807，positive-negative margin 均值 **−0.0126**；**203/400** 的 anchor 与 hard negative 比 positive 更近；hard negative 重复 243 次，只有 157 个唯一负例。少样本对比目标确实过强。

第三，在 dev_select 上，Qwen Plain B3-H 0.7984 高于 Frozen 0.7904，OOS 0.6917 vs 0.6639。这与 Gate6 historical test seed42 的方向相反，说明不是简单训练失败，而是 seed 和 unknown 迁移问题。

## B：多种子（约 3 分钟）

P0 原始 Plain，P1 lr=3e-5，P2 lr=1e-5，P3 去掉 triplet。P1/P2 在 dev_select 上都低于 P0；P3 首次在 Windows 内存压力下停在 40/50 step，重跑后完成 50 step，dev B3-H 0.7945，仍低于 P0 0.7984，所以 P3 完成但未选中。

然后训练 P0 的 seed 42/43/44。validation B3-H 均值 0.8028，RejectOnly 0.8130；test B3-H 均值 0.8068，RejectOnly **0.8256**。相对 frozen，B3-H 只高 0.18 pp，RejectOnly 高 1.26 pp。每个 seed 的 adapter SHA 都不同，seed43/44 的 test 明显高于 seed42；seed42 的负结果保留，没有用 test 选 seed。

所以“为什么 Gate6 验证提升、test 下降”的答案有两层：一层是这个 seed 在 final-unknown 上确实迁移不好；另一层是不能用一个 seed 推断整个方法失败。

## C：冻结与历史 test（约 1.5 分钟）

所有 adapter、B4 checkpoint、dev 阈值在 `gate7_final_freeze.json` 中冻结，test_evaluated=true 后才评估一次。History test 上：

Frozen B3H 0.8051、RejectOnly 0.8130；Qwen Plain 3-seed B3H 0.8068、RejectOnly 0.8256。RejectOnly 是正结果，B3H 只有很小正结果。

优化配方没有选出来，所以 Optimized = NOT SELECTED；没有把 P1/P2 包装成成功。

## D：Agent 与稳定性（约 3 分钟）

Agent 修复了 5 条应澄清任务：当 accepted intent 是 transfer-not-arrived 语义族，并且文本有 “not arrived / not received” 时，先问澄清，不直接执行。新回归测试真实通过；旧 120 条任务上，GTE Frozen RejectOnly 73/120，Qwen Frozen RejectOnly 81/120，Qwen Plain42 RejectOnly 81/120。安全率都是 100%，Qwen 的 unknown 族从 0.667 提到 1.0。

分类 F1 与业务 TSR 不单调：Qwen Reject 的 3-seed test F1 0.826，低于 GTE Reject 0.843，但 Agent TSR 0.675 高于 GTE 0.608。这说明业务效果还受拒识、工具执行和澄清影响。

稳定性部分：GTE Frozen 的 RejectOnly 3-seed mean 0.8144，Full 0.8105；Qwen Frozen Reject 0.7950，Full 0.7752；GTE-LoRA 42/43/44 的 Reject 都高于 Full。每个 seed 的 checkpoint SHA 不同，说明 seed 参数真的生效。RejectOnly 的结论跨编码器、跨 seed 稳定。

## 结论（约 1 分钟）

四句话：

1. Gate6 的 Qwen LoRA 负结果是 seed42 的真实结果，但不是三种子普遍结论；
2. 3-seed RejectOnly +1.26 pp，B3H 仅 +0.18 pp；
3. B4-RejectOnly 在 GTE/Qwen、Frozen/LoRA 上稳定优于 Full；
4. P1/P2 没有改善，P3 完成但未选中，没有优化成功；LLM API 仍无凭据。

限制：test 已看过；120 任务是旧任务；3 seeds 不是显著性；P3 未进入 test；没有新独立 unknown 协议。

下一步：新的独立 unknown-class 协议、真 batch/负例质量修复、更多 seed 和 bootstrap、dispute 路由与澄清覆盖，而不是继续在当前 test 上调参。

谢谢大家。
