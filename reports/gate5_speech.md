# Gate 5 演讲稿（约 10–12 分钟）

## 开场（约 50 秒）

这一轮 Gate 5 我做了四件事：真实训练 GTE 的 LoRA、把 B4 拆成 RejectOnly 和 Full、把分类器升级成带 RAG 的客服 Agent、以及用 120 条自然语言任务做端到端 benchmark。

先说结论：**真正有效的不是 LoRA，也不是可学习门控，而是固定 0.5 融合、只训练拒识头。** 模型指标的小改善没有转化为业务成功率；Frozen-B3-H 仍然是业务 benchmark 上最好的路由。

## A：LoRA（约 2 分钟）

LoRA 的基座是 GTE-ModernBERT。我先用 `named_modules()` 检查了真实层名，ModernBERT 不是 q_proj/v_proj，而是 `Wqkv`、`Wo`、`Wi`。所以 LoRA 注入了这三类模块：r=8、alpha=16、dropout=0.05，可训练参数 169 万，占 1.12%。

训练只用 400 条 support。损失是 SupCon 加 support 内挖掘的困难负样本 triplet；没有从 calibration、dev 或 test 挖训练样本。2 个 epoch，22.19 秒，峰值 1.37 GB，adapter 6.78 MB。我们做了 forward、backward、optimizer step、保存、重载；重载后同一句话向量最大差是 0。

但效果要诚实：LoRA-B0 的 validation F1 从 0.734 升到 0.753，可是 LoRA-B3-H 从 0.767 降到 0.760，known routing 从 0.851 降到 0.828。第一次 4 epoch 更差，B3-H 只有 0.740；我根据 validation 改成 2 epoch，没有用 test 选 epoch。LoRA 改变了表征，但没有让路由更好。

## B：RejectOnly 与六组矩阵（约 3 分钟）

Gate 4 的 B4-Full 同时训练 gate 和拒识头。这一轮我加了关键对照 B4-RejectOnly：**gate 固定 0.5，只训练 7→16→1 的拒识头**，gate 参数冻结。

Validation 上，Frozen-B3-H 是 0.767；Frozen-RejectOnly 是 0.790；Frozen-B4-Full 是 0.781；B4-Fusion 只有 0.740。也就是：只加 reject head 最好，gate 反而拖后腿。

冻结后做了一次历史探索性 test：Frozen-B3-H F1 0.831，Frozen-RejectOnly 0.843，Frozen-B4-Full 0.837；RejectOnly 的 OOS 0.643，B4-Full 0.658，但 F1 仍低于 RejectOnly；routing RejectOnly 0.876，B3-H 0.863。**RejectOnly 是唯一稳定正贡献的新模块。**

LoRA 的六组里，最好的是 LoRA-RejectOnly 0.814，仍低于 Frozen-RejectOnly 0.843；LoRA-B3-H 0.803，LoRA-B4-Full 0.802。所以 LoRA 和拒识头没有互补，LoRA 反而让路由变差。

这就回答了四个问题：只加 reject head 更好；LoRA 没有改善表示；两者不互补；OOS 提升没有明显伤害正常路由，但 LoRA 明显伤害 routing。

## C：RAG Agent（约 3 分钟）

C 部分保留 B3/B4 作为路由层，不让大模型替代分类。新增 30 条虚构银行政策文档，每条有 doc_id、版本、source ID 和可引用文本；用 TF-IDF 检索，回答显示引用，检索不修改数据库。

Agent 的自然澄清只显示自然语言，例如“checking whether a card has arrived”，不显示 `card_arrival` 这种内部标签。用户澄清时保留前面的 TXN 槽位。每个 Streamlit 会话使用独立 SQLite 文件，切换方法不会清空其他会话。取消转账、激活卡必须先展示对象和金额，用户明确 yes 后才执行；工具层再次校验所有权并写审计。

我们跑了 10 条真实模型演示，其中 7 条 B4-Full：查卡、查转账、取消转账、三个政策 FAQ、非本人卡 handoff、拒绝激活、模糊转账澄清。截图里可以看到 B4 取消 TXN-2002 后，Transfers 表变成 cancelled，工具时间线展开 confirmation_required。

LLM API 接口已经实现，支持 OpenAI 兼容端点，密钥只读环境变量。但当前环境没有 `OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL`，所以 12 条 API 模式是 **NOT RUN_API_BLOCKED**，没有把模板输出冒充 API 输出。

## D：120 条 Benchmark（约 3 分钟）

D 部分新建了 120 条自然语言任务，先冻结文件哈希，再跑三个真实路由：Frozen-B3H、Frozen-B4Full、LoRA-B4Full。任务分六族：card 25、transfer 30、dispute 20、policy 20、unknown/ambiguous 15、permission/safety 10；总 120。每条有独立 ID、初始 DB、对话、期望工具和预期终态。

结果：Frozen-B3H 的 task success rate 是 **62.5%**，Frozen-B4Full 和 LoRA-B4Full 都是 **58.3%**；tool+argument accuracy 是 72.5 / 70.0 / 72.5；安全操作率三个都是 **100%**；routing accuracy 都是 64.2%。P50 延迟 50–97 毫秒。

分任务族看，policy FAQ 三个都是 90%，unknown 上 B3H 100%、B4/LoRA 66.7%；但 transfer 只有 43%，dispute 25%，permission 30%。失败原因主要是 routing：每个 router 都有 43 次 routing failure；工具失败只有 2/7/2 次。也就是说，**业务失败主要不是工具越权，而是自然语言意图路由错误**；RAG 能回答政策，但不能修复金融意图路由。

## 结论（约 1 分钟）

三句话：

第一，RejectOnly 是本轮唯一稳定有效的新模块，Frozen-RejectOnly test F1 0.843，高于 B3-H 0.831。

第二，LoRA 和 B4-Full 的 gate 都没有带来稳定收益；LoRA 甚至降低了 routing；模型指标的小改善没有转化为业务成功率。

第三，Agent 的工具、确认、所有权、审计和 RAG 引用链是真实闭环的，安全操作率 100%；但 120 条 benchmark 显示，下一轮最该解决的是自然语言路由，而不是继续加模块。

限制也要说明：BANKING77 test 已看过，不是新盲测；单 seed、单编码器；120 条是模拟任务；API 12 条因无凭据未运行。

下一步，我会在新的独立 unknown-class 协议上复现 RejectOnly，做多 seed LoRA 和路由错误分析，再决定是否把 LoRA 写进最终方案。

谢谢大家。
