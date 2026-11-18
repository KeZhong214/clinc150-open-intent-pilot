# Gate 6 演讲稿（约 10–12 分钟）

## 开场（约 50 秒）

Gate 6 的目标不是再做一个新分类器，而是把 Qwen3-Embedding-0.6B 从 Gate3 的冻结模型升级成可训练的 LoRA 模型，并回答三个问题：Qwen LoRA 能不能改善已知/未知意图；GTE-LoRA 的退化是不是表征漂移；Qwen 上 RejectOnly 和 Full 的模块归因是否还成立，最后能不能转成业务任务成功率。

先说结论：**Qwen LoRA 在 validation 上有效，但 historical test 没有复现；Anchor 能减小漂移，但没有提高 test；Qwen 上 RejectOnly 仍然优于 Full；Qwen-LoRA-RejectOnly 的 120 任务成功率是 65%，是四个 Qwen 版本里最高的。**

## A：先对齐再训练（约 2 分钟）

Qwen 的推理路径和 GTE 完全不同：Qwen 是 causal 模型，1024 维，last-token pooling，不是 CLS。Gate3 的 SentenceTransformer 用的是 right padding，取最后一个非 pad token。我第一次按 Qwen 官方 README 用 left padding，32 个样本和 Gate3 cache 的 mean cosine 只有 0.412，top1 一致率 0.344。这不是模型问题，是位置/padding 路径不一致。

改成 right padding + last non-pad token 后，mean cosine 0.999998，min 0.999995，top1 一致率 1.0；Gate3 B3-H validation 复跑 0.7414，和 Gate3 的 0.74035 对上了。对齐通过之后才开始训练，没有绕过。

LoRA 的真实模块是 `q_proj/k_proj/v_proj/o_proj`，2,293,760 可训练参数，占 0.38%。2-step smoke 峰值 1.2GB，save/reload 向量差 0。

## B：Plain / Anchor（约 3 分钟）

Plain 只用 support 的 SupCon + hard negative triplet；Anchor 在 Plain 上增加冻结 Qwen 的余弦约束，teacher 只读 detached。两者都是 2 epoch、50 步、seed 42。

Plain 的 support 与 frozen 的余弦是 0.892，validation B3-H 0.7580，比 frozen 0.7404 高 1.76 pp；但 historical test B3-H 0.7972，比 frozen 0.8064 低 0.92 pp。**validation 提升没有复现。**

Anchor 把 support 余弦提高到 0.936，漂移明显更小；但 test 是 0.7968，和 Plain 0.7972 基本一样，OOS 还略低。所以“漂移更小”不等于“泛化更好”。这是本轮最重要的负结果。

## C：Qwen 联合矩阵（约 2 分钟）

Validation 上，Qwen Frozen 的 RejectOnly 0.7501，高于 B3H 0.7404、Full 0.7431；Qwen Plain 的 RejectOnly 0.7690，高于 B3H 0.7580、Full 0.7644。RejectOnly 的收益在 Qwen 上成立。

Historical test 上，Qwen Frozen RejectOnly 0.8127，是 Qwen 内部最好；Qwen Plain RejectOnly 0.8080，Plain B3H 0.7972。**Qwen Plain 的 test F1 低于 Frozen B3H，validation 的增益没有转化为 test。** 同时，GTE Frozen RejectOnly 0.8435 仍然是跨编码器最高。Qwen 0.6B 没有在 BANKING77 分类上超过 GTE。

为什么 GTE-LoRA 和 Qwen-LoRA 都出现 validation 好、test 不好？这更像是 400 条 support 上的少样本适配对 calibration/dev 分布有效，对已看过的 test 分布不稳定，而不是单一“漂移”问题。Anchor 减小漂移但没有解决这个泛化问题。

## D：Qwen Agent（约 3 分钟）

Agent 没有重写。复用了 Gate5 的 30 条虚构政策、SQLite 工具、多轮、确认、所有权和审计。新增的是真实 Qwen 在线编码：三个 Qwen-Plain 路由，B3-H、RejectOnly、B4-Full。没有从历史 CSV 查分。

120 条旧任务的探索性回归：Qwen Frozen-B3H 62.5%，Qwen Plain-B3H 59.2%，Qwen Plain-B4Full 62.5%，Qwen Plain-RejectOnly 65.0%。安全操作率四个都是 100%。Qwen RejectOnly 的工具+参数正确率 76.7%，也是最高。

这里有一个重要区别：Qwen Plain-B3H 的 test F1 比 frozen 低，TSR 也从 62.5 降到 59.2；但 Qwen RejectOnly 的 test F1 只有 0.808，TSR 却升到 65.0。分类 F1 和业务 TSR 不是单调对应。RejectOnly 的显式拒识减少了错误的金融动作，card 族从 0.84 升到 0.88，transfer 从 0.60 升到 0.63；但 dispute 仍然只有 0.15，5 条需要澄清的任务 Qwen 全部直接接受，clarification 是 0。这是真实的 Agent 缺陷，不是模型分数能掩盖的。

真实回放 6 条，5 条走 Qwen-LoRA；截图里 Qwen RejectOnly 取消 TXN-2002 后，Transfers 表是 cancelled；FAQ 显示 KB-013/KB-027 引用。

## 结论（约 1 分钟）

四句话：

1. Qwen LoRA 在 validation 上有效，但 test 没有复现；
2. Anchor 减小漂移，但没有提高 test；
3. Qwen 上 RejectOnly 仍然优于 Full，跨架构一致；
4. Qwen-LoRA-RejectOnly 的业务 TSR 65.0% 是 Gate6 最好，但分类 F1 仍是 GTE-RejectOnly 0.8435 最高，Qwen 不能被称为全面更优。

限制：BANKING77 test 已看过；120 条任务是旧任务；单 seed；没有新盲测；API 仍无凭据，NOT RUN_API_BLOCKED。

下一步，我会在新的独立 unknown-class 协议上复现 RejectOnly，做多 seed 和 Anchor λ 扫描，并修复澄清与 dispute 路由，而不是继续在当前 test 上调参。

谢谢大家。
