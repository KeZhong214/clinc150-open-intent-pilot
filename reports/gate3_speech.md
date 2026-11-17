# Gate 3 演讲稿（约 8–10 分钟）

## 开场（约 45 秒）

各位老师好，我的项目题目是《面向智能客服的少样本开放意图识别：多模型跨场景评测与原型—近邻协同优化》。

前几轮我们已经在 CLINC150 上完成了 B0/B2 的 Pilot，发现 Qwen B0 很强，但 B2 的类别阈值没有稳定超过 B0。这一轮的任务不是继续调 B2，而是把项目扩展成三个场景、四个编码器的完整比较，并检验多原型和近邻融合是否真的有用。

## 数据（约 1 分钟）

我们使用三个数据集。CLINC150 保留原有的 20 类，作为历史 Pilot；新增 BANKING77，选 40 个 known、18 个 dev-unknown、19 个 final-unknown；新增 MASSIVE en-US，选 35/12/13。所有 support 都来自官方 train，5-shot 和 10-shot 严格嵌套；known calibration 和 dev-unknown 只用于阈值选择，final test 在 freeze 后只评一次。support、calibration、test 的规范化文本交集都是 0。

这里要特别说明：CLINC150 的 test 在之前的 Gate 已经看过，所以它只作为历史结果，不作为本轮 B3 的盲测证据。

## 编码器（约 1 分钟）

我们比较四个冻结编码器：BGE-small、Qwen3-Embedding-0.6B、GTE-ModernBERT-base 和 E5-base-v2。每个模型按模型卡固定 revision 和输入前缀；E5 用 `query: `，Qwen 用固定的客服意图 instruction，BGE 和 GTE 不加前缀。四个模型一共编码 11,011 条文本/模型。Qwen 最强但最贵，单次三数据集编码 160.6 秒；GTE 只要 12.7 秒，E5 6 秒。

## 方法（约 1 分钟）

方法有三组。B0 是单原型加全局阈值。B3-P 在 support 上计算类内分散度，对最高的 25% 类做两个 KMeans 原型，其他类保持单原型。B3-H 在 B3-P 的基础上，把原型分数和“到该类 support 的 top-2 近邻余弦均值”按 0.5/0.5 融合。三组方法使用完全相同的 support、编码器和阈值选择规则。阈值在 validation 上选：known acceptance 不低于 0.90，最大化 open-set macro-F1。72 个配置先冻结，再跑 test。

## 主结果（约 2 分钟）

先看 B0 的模型×数据集热图。Qwen 的平均 F1 最高，5-shot 平均 0.793；GTE 在 BANKING77 的 5-shot 最好，达到 0.764；BGE 和 E5 整体弱一些。10-shot 对所有模型都有帮助，B0 的总体平均 F1 从 0.761 提升到 0.804。

再看 B3 消融。这是本轮最重要的结果：**B3-H 的收益是场景条件的**。在 BANKING77 上，B3-H 把 OOS recall 从 0.494 提到 0.546，提升 5.21 个百分点；open-set F1 从 0.7625 提到 0.7725；known routing 也略微提高。这个结果在四个模型上方向一致。

但在 CLINC150 上，B3-H 的 OOS recall 下降 1.11 个百分点，F1 下降 0.92；在 MASSIVE 上，F1 下降 1.46 个百分点。B3-P 在三个数据集上都没有稳定收益。所以不能说“多原型或近邻融合普遍更好”，只能说它在 BANKING77 这种单领域、细粒度、类间高度相似的场景中有用。

## MASSIVE 的负结果（约 1 分 30 秒）

MASSIVE 的结果需要特别解释。BGE 和 GTE 在 MASSIVE 上几乎不拒 OOS，OOS recall 是 0 到 2.5%。但它们的 AUROC 仍有 0.77，说明分数不是完全不可分。问题出在阈值目标：open-set macro-F1 里 OOS 只占 1/36 的权重，F1-first 规则把 known acceptance 选到了 0.99 以上，几乎不拒未知。Qwen 的分离度更好，OOS recall 大约 0.48 到 0.53，但仍然远低于 BANKING。

这是一个真实的方法学发现：**如果阈值目标本身不合适，分类器再改也可能被吞掉。** 我们没有用 test 重新调阈值，因为那会变成 test 调参；如果继续，需要一个新的独立 unknown-class 协议，并改用 recall-constrained 的阈值目标。

## 训练式对照（约 1 分钟）

我们做了两个训练式对照。SetFit-BGE 只用 BANKING77 的 200 条 5-shot support，微调 BGE body。训练 283 秒，support 上很快到 100%，但 test 的 known routing 只有 0.726，OOS recall 0.362，open-set F1 0.670；BGE-B0 是 0.7505。明显过拟合。另一个 residual 投影适配器，F1 0.739，也没有超过 B0。结论是：在这组 5-shot 预算下，训练式适配没有带来开放集收益。

## Demo（约 45 秒）

我们做了一个真实的 Streamlit Demo：可以选择数据集、编码器和 B0/B3-P/B3-H，输入英文客服文本，实时显示 Top-3、融合分数、阈值和接受/拒绝理由。它调用本地模型和缓存，不是预设答案。无 GPU 时 BGE 可以纯 CPU 运行。

## 结论（约 45 秒）

总结三句话：

第一，Qwen 是平均最强模型，GTE 是 BANKING77 上最有性价比的选择。

第二，B3-H 在 BANKING77 是条件性正收益，在 CLINC 和 MASSIVE 不稳定；B3-P 没有稳定收益。

第三，MASSIVE 的主要问题是阈值目标，训练式 5-shot 方法明显过拟合。下一步不应在当前 test 上继续调参，而应使用新的独立 unknown classes 和 recall-constrained 阈值协议，再做多 seed/10-shot 稳定性。

谢谢大家。
