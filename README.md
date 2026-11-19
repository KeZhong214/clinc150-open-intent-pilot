# 少样本开放意图识别：从跨场景矩阵到可训练路由器与客服 Agent

课程项目：**面向智能客服的少样本开放意图识别：多模型跨场景评测、LoRA/B4 归因与真实工具 Agent**

当前状态：**Gate 0–8.1 均已完成真实运行。** Gate 3 完成 3 数据集、4 编码器、
5/10-shot、B0/B3-P/B3-H 的 72 格矩阵；Gate 4 完成 BANKING77 GTE 10-shot
的 B4-AR 与 SQLite Agent；Gate 5 完成真实 LoRA-GTE、RejectOnly 归因、
30 条虚构银行政策 RAG、120 条业务 benchmark；Gate 6 完成 Qwen3-Embedding
0.6B LoRA；Gate 7 完成双层开发集诊断、3-seed 复验和核心 B4 稳定性实验；
Gate 7.1 完成阈值协议修复与 36 组稳定性；Gate 8 完成多轮客服 Agent、
统一 UI 和 102 例新冻结模拟评测；Gate 8.1 完成 GTE 权重—阈值配对、
OOS/LLM 安全修复和 120 例新模拟测试。

Gate 3 的核心结果（best config per dataset；CLINC 为历史测试）：

| Dataset | Best config | Known routing | OOS recall | Open-set macro-F1 |
|---|---|---:|---:|---:|
| CLINC150（历史） | Qwen 10-shot B0 | 0.9817 | 0.9010 | 0.9226 |
| BANKING77 | GTE 10-shot B3-H | 0.8625 | 0.6368 | **0.8306** |
| MASSIVE en-US | Qwen 10-shot B0 | 0.8272 | 0.5298 | 0.7982 |

Gate 4 的核心结果（BANKING77 GTE 10-shot；test 已看过，探索性）：

| Method | Validation F1 | Test F1 | Test OOS recall |
|---|---:|---:|---:|
| B0 | 0.734 | 0.808 | 0.505 |
| B3-H | 0.767 | 0.831 | 0.637 |
| B4-Fusion | 0.740 | 0.820 | 0.555 |
| **B4-Full** | **0.781** | **0.837** | **0.658** |

Gate 5 的核心结果（BANKING77 GTE 10-shot；历史探索性 test）：

| Method | Validation F1 | Test F1 | Test OOS recall |
|---|---:|---:|---:|
| Frozen B3-H | 0.767 | 0.831 | 0.637 |
| **Frozen RejectOnly** | **0.790** | **0.843** | 0.643 |
| Frozen B4-Full | 0.781 | 0.837 | 0.658 |
| LoRA B3-H | 0.760 | 0.803 | 0.584 |
| LoRA RejectOnly | 0.767 | 0.814 | 0.574 |
| LoRA B4-Full | 0.753 | 0.802 | 0.500 |

- **RejectOnly 是唯一稳定正贡献的新模块**；LoRA 没有改善 B3-H/业务路由；
- LoRA 真实训练：1.69M 可训练参数（1.12%），2 epoch，22.19 s，adapter 6.78 MB；
- Agent：30 条虚构政策、RAG 引用、自然语言澄清、独立 session SQLite；
- 120 条真实路由 benchmark：Frozen-B3H **62.5% TSR**，Frozen-B4Full 58.3%，
  LoRA-B4Full 58.3%；三者安全操作率均 100%；
- 12 条 LLM API 模式因无凭据 **NOT RUN_API_BLOCKED**，未伪造调用。

Gate 6 的核心结果（Qwen3-Embedding 0.6B；历史探索性 test）：

| Qwen encoder | B3-H Val | B3-H Test | RejectOnly Test | Agent TSR |
|---|---:|---:|---:|---:|
| Frozen | 0.740 | 0.806 | 0.813 | 0.625 |
| Plain LoRA | **0.758** | 0.797 | 0.808 | 0.592（B3-H）/ **0.650（Reject）** |
| Anchor LoRA | 0.757 | 0.797 | — | NOT RUN |

- 冻结对齐：32 probe mean cosine **0.999998**，top1 1.0；
- LoRA：2.29M 可训练参数（0.38%），2 epoch，adapter 9.20 MB；
- **validation 提升，但 historical test 没有复现**；
- Anchor 把 support 余弦从 0.892 提高到 0.936（漂移更小），但没有 test 收益；
- Qwen 上 **RejectOnly > Full**，支持 Gate5 模块归因；
- Qwen Plain-RejectOnly Agent TSR **65.0%**，四版本最高，安全率 100%；
- 分类 F1 仍是 GTE Frozen-RejectOnly **0.8435** 最高。

Gate 7 的核心结果（双层 dev + 3-seed；历史探索性 test）：

| Qwen Plain seed | Val B3-H | Test B3-H | Test RejectOnly |
|---|---:|---:|---:|
| 42 | 0.7984 | 0.7969 | 0.8091 |
| 43 | 0.8051 | 0.8159 | 0.8356 |
| 44 | 0.8050 | 0.8077 | 0.8321 |
| mean ± std | 0.8028 ± .0032 | 0.8068 ± .0095 | **0.8256 ± .0144** |

- Gate6 的 seed42 test 负结果是真实结果，但不是 3-seed 普遍结论；
- B4-RejectOnly 相对 frozen test +1.26 pp，2/3 seed 胜出；
- dev_threshold 只选 tau，dev_select 只选配方；
- 6×6 microbatch 和 203/400 负 margin hard negative 已诊断；
- P1/P2 没有超过 P0；P3 完成但未选中（dev B3-H 低于 P0）；
- B4-RejectOnly 在 GTE/Qwen、Frozen/LoRA 上稳定优于 B4-Full；
- Agent 修复 5 条澄清后，Qwen RejectOnly 81/120、GTE RejectOnly 73/120；
- 86 个测试通过。

Gate 7.1 修复：

- B0/B3-H/B4-RejectOnly 的 test 阈值统一使用 Gate7 `dev_threshold`；
  原 test/freeze/summary 保留为 `*_pre_fix.*`，v2 单独记录；
- B4 stability 补齐 `qwen_plain42/B4-Full/seed44`，共 **36 组无重复键**；
- P3 记录更新为 completed/not selected；
- `Qwen-Opt-RejectOnly` 更正为 `Qwen-Plain42-RejectOnly`，保留 legacy alias；
- Agent 仍使用 Gate5/6 旧阈值，与分类 C 的 Gate7 dev_threshold 明确分开；
- 详细：[`reports/gate7_1_fix_report.md`](reports/gate7_1_fix_report.md)
  与 [`reports/project_final_research_summary.md`](reports/project_final_research_summary.md)。

Gate 8 的核心结果（新冻结 102 例模拟测试；不是外部盲测）：

| Router | TSR | Tool | Safe | P50 |
|---|---:|---:|---:|---:|
| GTE-Frozen-RejectOnly | **0.951** | 0.980 | 1.000 | 0.060 s |
| Qwen-Frozen-RejectOnly | 0.824 | 0.824 | 1.000 | 0.140 s |
| Qwen-Plain42-RejectOnly | 0.794 | 0.824 | 1.000 | 0.199 s |

- 统一使用 Gate7 `dev_threshold`，不是 Gate5/6 旧阈值；
- 修复 dispute 被 policy fallback 截获、pending slot 吞新话题、确认阶段不能改对象；
- 多轮 slot/纠正/reset/确认/所有权/幂等/OOS 安全测试；
- 9 个可操作 Demo：卡状态、激活、转账、取消、FAQ、dispute、多轮纠正、OOS、越权；
- 离线 **Offline Router-driven Agent**；LLM-assisted 约束模式无凭据
  `NOT RUN_API_BLOCKED`，没有伪造调用；
- Qwen 在 dispute 首句被 RejectOnly 拒绝是真实负结果，没有按测试结果调阈值；
- 详细：[`reports/gate8_agent_final_report.md`](reports/gate8_agent_final_report.md)、
  [`reports/gate8_benchmark_protocol.md`](reports/gate8_benchmark_protocol.md)、
  [`reports/gate8_failure_analysis.md`](reports/gate8_failure_analysis.md)、
  [`reports/gate8_demo_script.md`](reports/gate8_demo_script.md)。

Gate 8.1 修复（新冻结 120 例模拟测试；不是外部盲测）：

- GTE 改为 **Gate5 权重 + Gate7 dev_threshold 重校准**，tau 0.4388361126；
  checkpoint SHA `0e920d0f...`；不再混用 Gate7 另一份 checkpoint 的阈值；
- 三模型注册表与 SHA 校验：`results/gate8_1_model_registry.json`；
- OOS `currency` 子串改为 word-boundary，不再把 cryptocurrency 引导到 KB；
- LLM 槽位必须有原文/可信历史证据；
- 模型切换前关闭旧 sessions，实测 GPU allocated/reserved 回到 0 MB；
- 120 条新案例、120 个 template、18 families；与 Gate5/Gate8 无完全重复；
- 严格 v1 首次运行保留；v2 仅修复 FAQ 期望词后回归，不是第二次盲测。

| 版本 | GTE | Qwen Frozen | Qwen Plain42 |
|---|---:|---:|---:|
| v1 首次冻结 | 0.575 | 0.575 | 0.692 |
| **v2 期望词修复后** | **0.633** | **0.633** | **0.742** |

- 三模型 safe_action_rate 均为 1.000；
- Qwen Plain42 vs GTE +10.83 pp，family-cluster 95% CI −0.64～+23.90，不显著；
- 详细：[`reports/gate8_1_final_report.md`](reports/gate8_1_final_report.md)、
  [`reports/gate8_1_protocol_audit.md`](reports/gate8_1_protocol_audit.md)、
  [`reports/gate8_1_benchmark_protocol.md`](reports/gate8_1_benchmark_protocol.md)、
  [`reports/gate8_1_failure_analysis.md`](reports/gate8_1_failure_analysis.md)、
  [`reports/gate8_1_demo_guide.md`](reports/gate8_1_demo_guide.md)。

B3-H 的场景差异：

- BANKING77：相对 B0 平均 OOS recall +5.21 pp，open-set F1 +0.99 pp；
- CLINC150（历史）：OOS recall −1.11 pp，F1 −0.92 pp；
- MASSIVE：OOS recall −0.43 pp，F1 −1.46 pp。
- **B3-H 在 BANKING77 有用，在 MASSIVE/CLINC 没有稳定收益。**

训练式对照（BANKING77 5-shot，同一 test）：BGE-B0 open-set F1 0.7505；
SetFit-BGE 0.6696；投影适配器 0.7394。两者都未超过冻结 BGE-B0。

Gate 1/2 历史结果保持冻结：

| Scope | Model/Method | Key result |
|---|---|---:|
| Gate 1 CLINC | Qwen B0 | open-set F1 0.9570 |
| Gate 1 CLINC | Qwen B2 | open-set F1 0.9499 |
| Gate 2 hard-OOS | Qwen B0 | hard OOS recall 0.9426 |
| Gate 2 hard-OOS | Qwen B2-v2 | hard OOS recall 0.9482, known routing −1.17 pp |

详细报告：[Gate 0 环境与数据](reports/gate0_environment.md) ·
[Gate 1 Baseline](reports/gate1_baseline.md) ·
[Gate 2 Hard-OOS](reports/gate2_method.md) ·
[Gate 3 多数据集](reports/gate3_final_report.md) ·
[Gate 4 B4+Agent](reports/gate4_final_report.md) ·
[Gate 5 LoRA+Benchmark](reports/gate5_final_report.md) ·
[Gate 6 Qwen+Agent](reports/gate6_final_report.md) ·
[Gate 7 多种子诊断](reports/gate7_final_report.md) ·
[Gate 7.1 协议修复](reports/gate7_1_fix_report.md) ·
[Gate 8 Agent 最终报告](reports/gate8_agent_final_report.md) ·
[Gate 8 评测协议](reports/gate8_benchmark_protocol.md) ·
[Gate 8 失败分析](reports/gate8_failure_analysis.md) ·
[Gate 8 Demo 脚本](reports/gate8_demo_script.md) ·
[Gate 8.1 最终报告](reports/gate8_1_final_report.md) ·
[Gate 8.1 协议审计](reports/gate8_1_protocol_audit.md) ·
[Gate 8.1 评测协议](reports/gate8_1_benchmark_protocol.md) ·
[Gate 8.1 失败分析](reports/gate8_1_failure_analysis.md) ·
[Gate 8.1 Demo Guide](reports/gate8_1_demo_guide.md) ·
[项目最终研究总结](reports/project_final_research_summary.md)  
结果索引：[results/gate6_qwen_method_matrix_historical_test.csv](results/gate6_qwen_method_matrix_historical_test.csv) ·
[results/gate6_qwen_benchmark_summary.csv](results/gate6_qwen_benchmark_summary.csv) ·
[results/gate7_final_historical_test.csv](results/gate7_final_historical_test.csv) ·
[results/gate7_final_summary.csv](results/gate7_final_summary.csv)  
图表：[figures/gate6_qwen_matrix.png](figures/gate6_qwen_matrix.png) ·
[figures/gate6_qwen_benchmark.png](figures/gate6_qwen_benchmark.png) ·
[figures/gate7_final_test.png](figures/gate7_final_test.png) ·
[figures/gate7_stability.png](figures/gate7_stability.png)  
Demo：[app_gate8.py](app_gate8.py) ·
[Gate8 卡状态截图](figures/gate8_ui_card.png) ·
[Gate8 激活截图](figures/gate8_ui_activation.png) ·
[Gate8 dispute 截图](figures/gate8_ui_dispute.png) ·
[Gate8 越权拒绝截图](figures/gate8_ui_ownership.png) ·
[Gate8.1 卡状态截图](figures/gate8_1_ui_card.png) ·
[Gate8.1 激活截图](figures/gate8_1_ui_activation.png) ·
[Gate8.1 dispute 截图](figures/gate8_1_ui_dispute.png) ·
[Gate8.1 越权截图](figures/gate8_1_ui_ownership.png) ·
[app_gate5.py](app_gate5.py) ·
[app_gate6.py](app_gate6.py) ·
[Gate5 B4 查卡截图](figures/gate5_agent_card_b4.png) ·
[Gate5 RAG 截图](figures/gate5_agent_rag.png) ·
[Qwen RejectOnly 取消转账截图](figures/gate6_qwen_agent_cancel.png) ·
[Gate7 澄清截图](figures/gate7_agent_clarification.png) ·
[Qwen RAG 截图](figures/gate6_qwen_agent_rag.png)

---

## 1. 当前已完成的内容

- 从 CLINC150 官方仓库固定 commit 下载 `data_full.json`、`domains.json`、
  `LICENSE`、`README.md`，记录 SHA-256 和逐分区计数。
- 校验官方 JSON 结构、150 个已知意图、10 个业务域、OOS 标签和
  train/val/test 数量。
- 用固定规则选出 20 个已知意图（10 域 × 2 类），写入
  `data/manifest/selected_intents.json`。
- 生成 3 个 support seed（42/43/44）× 2 个 shot（5/10）的 K-shot
  manifest；K=5 是 K=10 的嵌套子集。
- 生成 calibration / final-test 索引，测试集只建立索引，未参与任何选择。
- 完成文本重复与切分重叠审计；报告官方数据中 5 处跨分区重复文本，
  没有删除或改写官方样本。
- 建立 D 盘项目专用 venv，完成 NumPy/Pandas/Scikit-learn/BGE 最小
  CPU 编码冒烟测试。
- Gate 0 只记录 BGE/Qwen 的 Hub revision、许可证和下载体积，未下载 Qwen；
  Gate 1 才下载并加载 Qwen3-Embedding-0.6B，完成 2200 条文本的 GPU 编码：
  batch=8，峰值 allocated 1,155.5 MiB，编码 27.7 s，无 OOM。
- 完成 BGE/Qwen × B0/B2 的 validation 阈值选择和 test 一次评测。
- Gate 2 完成 65/65 hard-OOS 类别划分；1,300 条 hard-OOS val、
  1,950 条 held-out hard-OOS test；hard-OOS 文本隔离重叠为 0。
- Gate 2 实现 B2-v2：corrected `q_c`、hard-OOS predicted-class 上尾 `u_c`、
  向全局阈值收缩，并且类别阈值不低于全局 tau。
- Validation 上冻结 B2-v2 后才编码/评测 held-out test；test 只读一次。
- Gate 3 新建 CLINC150/BANKING77/MASSIVE en-US 统一 manifest；BANKING77
  40/18/19、MASSIVE 35/12/13 类划分，support 5/10 严格嵌套。
- 完成 4 个冻结编码器（BGE、Qwen、GTE-ModernBERT、E5）在 3 数据集上的
  向量提取；总 11,011 条文本/模型。
- 完成 3 数据集 × 4 模型 × 2 shot × B0/B3-P/B3-H 的 72 格矩阵；validation
  选阈值、freeze 后 test 只评一次。
- 完成 BANKING77 SetFit-BGE 训练和投影适配器对照，记录训练曲线；两者均
  未超过 BGE-B0。
- 完成数据集画像、模型×数据集热图、shot 曲线、B3 消融、阈值权衡、
  BANKING77 混淆、成本散点和 4 类可追踪错误案例。
- 完成 Streamlit Demo 并在真实浏览器中运行截图。
- Gate 4 用 GTE BANKING77 10-shot 的 400 条 support 训练 B4-AR：
  4→32→1 共享门控 + 7→16→1 拒识头；24 个 episodic，60+60 epoch，12.55 s。
- B4-Fusion/B4-Full checkpoint、validation、freeze 和一次性探索性 test 落盘；
  B4-Full test F1 0.8369，比 B3-H 0.8306 高 0.62 pp；B4-Fusion 单独退化。
- 建立 SQLite 虚构银行库 + 7 个工具 + 多轮状态机；敏感动作二次确认、所有权
  校验、幂等、审计；16/16 确定性工具测试、6/6 真实路由演示。
- 完成 B4 训练/门控/对照图、Agent 指标图、真实 Streamlit 截图。
- Gate 5 真实训练 GTE-ModernBERT LoRA（Wqkv/Wo/Wi，1.69M 可训练参数，
  2 epoch，22.19 s，adapter 6.78 MB），support-only 重编码 4,280 条。
- 完成 Frozen/LoRA × B3-H/RejectOnly/B4-Full 的 9 行 validation 与一次性
  历史探索性 test；Frozen-RejectOnly 最佳（validation F1 0.790，test 0.843）。
- 30 条虚构银行政策 + TF-IDF RAG 引用 + 自然语言澄清 + 每会话独立 SQLite；
  10 条真实模型演示；LLM API 接口已实现但因无凭据标记
  `NOT RUN_API_BLOCKED`。
- 建立 120 条冻结 Agent benchmark，用 3 个真实路由完成 360 次评测；
  Frozen-B3H TSR 62.5%，B4/LoRA 均为 58.3%，安全操作率 100%。
- Gate 6 完成 Qwen3-Embedding 0.6B 冻结对齐（mean cosine 0.999998）、
  Plain/Anchor LoRA（2.29M 参数、2 epoch、adapter 9.20 MB）。
- 完成 Qwen Frozen/Plain × B0/B3-H/RejectOnly/Full 的 9 行 validation 和
  一次性历史 test；Qwen Frozen-RejectOnly 0.8130，Qwen Plain-B3H 0.7972。
- 完成 Qwen Agent 接入、6 条真实回放和 120 任务 × 4 路由回归；
  Qwen Plain-RejectOnly TSR 65.0%，安全率 100%。
- Gate 7 冻结双层 dev_threshold/dev_select，诊断 6×6 micro-batch 与
  203/400 负 margin 的 hard negative。
- 完成 Qwen Plain seed42/43/44 真实复验；RejectOnly test 均值 0.8256，
  相对 frozen +1.26 pp，2/3 seed 胜出。
- P1/P2 未超过 P0；P3 完成但未选中（dev B3-H 0.7945 < P0 0.7984）。
- 完成 GTE Frozen、Qwen Frozen、GTE-LoRA 42/43/44 的 B4 多种子稳定性；
  RejectOnly 在全部对照中稳定优于 B4-Full。
- 修复 5 条 transfer-not-arrived 澄清 bug；Qwen RejectOnly 81/120，
  GTE RejectOnly 73/120，安全率 100%。
- 64 个协议、配置、可复现性、Gate 1/2/3/4/5/6/7 逻辑测试通过。

详细报告：[Gate 0 环境与数据](reports/gate0_environment.md)  
Gate 1 报告：[B0/B2 Baseline 与动态拒识](reports/gate1_baseline.md)  
Gate 2 报告：[Hard-OOS 与 B2-v2 类别边界校准](reports/gate2_method.md)  
Gate 3 报告：[多数据集、跨模型与 B3 实验](reports/gate3_final_report.md)  
Gate 4 报告：[B4 可训练路由器与客服 Agent](reports/gate4_final_report.md)  
Gate 5 报告：[LoRA、RejectOnly、RAG Agent 与 120 任务 Benchmark](reports/gate5_final_report.md)  
Gate 6 报告：[Qwen3-Embedding LoRA、表征漂移与联合路由](reports/gate6_final_report.md)  
Gate 7 报告：[Qwen LoRA 多种子诊断、优化与稳定性](reports/gate7_final_report.md)  
修改记录：[Gate 0](reports/gate0_vibe_coding.md)、
[Gate 1](reports/gate1_vibe_coding.md)、
[Gate 2](reports/gate2_vibe_coding.md)、
[Gate 3](reports/gate3_vibe_coding.md)、
[Gate 4](reports/gate4_vibe_coding.md)、
[Gate 5](reports/gate5_vibe_coding.md)、
[Gate 6](reports/gate6_vibe_coding.md)、
[Gate 7](reports/gate7_vibe_coding.md)

---

## 2. 项目目录

```text
D:\code\intent-open-set\
├── README.md
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── requirements_lock.txt
├── pip_freeze.txt
├── app.py                           # Gate 3 classifier demo
├── app_gate4.py                     # Gate 4 agent demo
├── app_gate5.py                     # Gate 5 RAG agent demo
├── app_gate6.py                     # Gate 6 Qwen LoRA RAG agent demo
├── configs\
│   ├── pilot.yaml
│   ├── gate2.yaml
│   ├── gate3.yaml
│   ├── gate4.yaml
│   ├── gate5_lora.yaml
│   ├── gate5_methods.yaml
│   ├── gate5_agent.yaml
│   ├── gate5_benchmark.yaml
│   ├── gate6_qwen_lora.yaml
│   ├── gate6_qwen_b4.yaml
│   ├── gate6_qwen_agent.yaml
│   ├── gate6_qwen_benchmark.yaml
│   ├── gate7_dev.yaml
│   ├── gate7_training.yaml
│   ├── gate7_stability.yaml
│   └── gate7_agent.yaml
├── data\
│   ├── raw\
│   │   ├── data_full.json
│   │   ├── domains.json
│   │   ├── LICENSE
│   │   ├── README_clinc.md
│   │   ├── download_manifest.json
│   │   └── model_cards\
│   ├── gate3\
│   │   └── raw\                     # BANKING77 parquet + MASSIVE en-US JSONL
│   ├── gate4\                       # SQLite demo DB (git-ignored)
│   ├── gate5\
│   │   ├── benchmark_cases.jsonl    # frozen 120 tasks
│   │   └── sessions\                # per-session SQLite (git-ignored)
│   ├── gate6\
│   │   ├── sessions\                # Qwen per-session SQLite (git-ignored)
│   │   └── demo_sessions\           # replay DBs (git-ignored)
│   ├── gate7_agent_sessions\        # Gate7 Agent DBs (git-ignored)
│   ├── manifest\
│   │   ├── selected_intents.json
│   │   ├── gate2_*                  # hard-OOS split/manifests
│   │   ├── gate3_*                  # 3 datasets, classes, support/cal/test
│   │   └── ...
│   ├── model_cache\                 # BGE/Qwen/GTE/E5 weights (git-ignored)
│   └── cache_embeddings\            # Gate 1/2/3/5 vectors (git-ignored)
├── knowledge_base\
│   └── fictional_bank\
│       └── policies.jsonl           # 30 citable fictional policy docs
├── src\
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── encode_embeddings.py
│   ├── run_baseline.py
│   ├── gate3\
│   │   └── ...
│   ├── gate4\
│   │   └── ...
│   ├── gate5\
│   │   ├── train_lora_gte.py
│   │   ├── encode_lora.py
│   │   ├── evaluate_lora.py
│   │   ├── train_reject_only.py
│   │   ├── train_b4_lora.py
│   │   ├── run_method_matrix.py
│   │   ├── kb.py
│   │   ├── llm_client.py
│   │   ├── agent_v2.py
│   │   ├── router_lora.py
│   │   ├── benchmark.py
│   │   └── make_figures.py
│   ├── gate6\
│   │   ├── qwen_encoder.py
│   │   ├── inspect_qwen.py
│   │   ├── train_qwen_lora.py
│   │   ├── encode_qwen_lora.py
│   │   ├── train_qwen_b4.py
│   │   ├── run_qwen_matrix.py
│   │   ├── router_qwen.py
│   │   ├── benchmark_qwen.py
│   │   ├── combine_qwen_benchmark.py
│   │   └── make_figures.py
│   ├── gate7\
│   │   ├── make_dev_split.py
│   │   ├── diagnose.py
│   │   ├── dev_eval.py
│   │   ├── final_test.py
│   │   ├── stability_b4.py
│   │   ├── agent_benchmark.py
│   │   ├── combine_agent.py
│   │   └── make_figures.py
│   └── ...                          # Gate 1/2 scripts retained
├── tests\
│   ├── test_gate1_logic.py
│   ├── test_gate2_protocol.py
│   ├── test_gate3_protocol.py
│   ├── test_gate4.py
│   ├── test_gate5.py
│   ├── test_gate6.py
│   ├── test_gate7.py
│   └── ...
├── logs\
├── reports\
│   ├── gate0_*.md
│   ├── gate1_*.md
│   ├── gate2_*.md
│   ├── gate3_*.md
│   ├── gate4_*.md
│   ├── gate5_final_report.md
│   ├── gate5_ppt_outline.md
│   ├── gate5_speech.md
│   ├── gate5_vibe_coding.md
│   ├── gate6_final_report.md
│   ├── gate6_ppt_outline.md
│   ├── gate6_speech.md
│   ├── gate6_vibe_coding.md
│   ├── gate7_final_report.md
│   ├── gate7_ppt_outline.md
│   ├── gate7_speech.md
│   └── gate7_vibe_coding.md
├── results\                         # Gate 1/2/3/4/5/6/7 metrics, predictions, checkpoints
└── figures\                         # Gate 1/2/3/4/5/6/7 PNGs, real screenshots
```

---

## 3. 环境说明（重要）

本机已有可用的 CUDA PyTorch：

```text
Python 3.14.4
torch 2.11.0+cu128
CUDA available=True
NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, compute capability 12.0
```

因此项目 venv 是这样创建的：

```powershell
cd D:\code\intent-open-set
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install `
  "sentence-transformers==6.3.0" `
  "scikit-learn==1.9.1" `
  "PyYAML>=6.0"
```

这是一个项目专用 venv，新增包进入 `.venv`；但 **torch 是从基环境
`D:\Environment` 继承的，不是完全 hermetic 的隔离环境**。这样做是因为
RTX 5070 Laptop 的 sm_120 已经由现有 torch 2.11.0+cu128 验证，重复安装
一套 CUDA torch 会多占数 GB。`pip check` 显示无依赖冲突。

重建到另一台机器时，应先安装与显卡匹配的 PyTorch，再安装
`requirements.txt`。完整版本见 `requirements_lock.txt` / `pip_freeze.txt`。

### 固定缓存目录

为避免写入 C 盘，所有模型和嵌入缓存都在 D 盘：

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TOKENIZERS_PARALLELISM = "false"
```

`src/check_models.py` 也会默认把 `HF_HOME` 指向该目录。

---

## 4. Gate 0 运行顺序

所有命令都在 `D:\code\intent-open-set` 下执行。

### 4.1 数据下载（小文件，可重复运行）

```powershell
.\.venv\Scripts\python.exe -m src.download_data
```

成功输出末尾：

```text
Validated schema and counts: {'oos_val': 100, 'val': 3000, ...}
SOURCE_DOWNLOAD_OK
```

### 4.2 固定类别、K-shot 与 manifest

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data `
  --seed 42 --known-classes 20 --per-domain 2
```

成功输出包含：

```text
Selected 20 intents (2 per domain):
...
PREPARE_DATA_OK
```

### 4.3 协议测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

成功输出末尾：

```text
Ran 86 tests in ...s
OK
```

### 4.4 环境与 BGE 最小冒烟测试

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
.\.venv\Scripts\python.exe -m src.check_environment
.\.venv\Scripts\python.exe -m src.check_models --bge-smoke
```

成功输出末尾分别为 `ENVIRONMENT_CHECK_OK` 和 `MODEL_CHECK_OK`。

`--bge-smoke` **只**下载/加载 BGE-small-en-v1.5（约 128 MiB），不会下载
Qwen；Qwen 是在 Gate 1 单独下载和编码的。BGE 当前结果：3 条文本编码为
`(3, 384)`，向量有限且 L2 norm=1.0；缓存后加载约 2.95 s，3 条文本编码
约 0.13 s（CPU，不能外推为全量耗时）。

### 4.5 第一个验收命令

如果只想快速确认当前 Gate 0 产物，先执行：

```powershell
cd D:\code\intent-open-set
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

预期最后一行是 `OK`，共 86 个测试。请反馈
`logs/gate7_all_tests.log`；如需环境信息，再反馈
`data/manifest/environment.json`。

### 4.6 Gate 1 复现命令（实际已运行）

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# 编码缓存
.\.venv\Scripts\python.exe -m src.encode_embeddings --model bge --device cpu --batch-size 32
.\.venv\Scripts\python.exe -m src.encode_embeddings --model qwen --device cuda --batch-size 8

# seed=42 / 5-shot；B0 先跑，B2 再跑
.\.venv\Scripts\python.exe -m src.run_baseline --model bge  --method B0 --seed 42 --shot 5
.\.venv\Scripts\python.exe -m src.run_baseline --model qwen --method B0 --seed 42 --shot 5
.\.venv\Scripts\python.exe -m src.run_baseline --model bge  --method B2 --seed 42 --shot 5
.\.venv\Scripts\python.exe -m src.run_baseline --model qwen --method B2 --seed 42 --shot 5

# 汇总与图表
.\.venv\Scripts\python.exe -m src.aggregate_results
.\.venv\Scripts\python.exe -m src.make_figures
```

`run_baseline.py` 默认拒绝覆盖已有结果，避免无意中第二次读取 test。
只有明确写 `--force` 才会重新评测；本轮正式结果没有使用 `--force`。

### 4.7 Gate 2 hard-OOS 复现命令（实际已运行）

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# 65/65 hard-OOS 类别划分和 manifest
.\.venv\Scripts\python.exe -m src.prepare_gate2

# 只编码 hard-OOS validation
.\.venv\Scripts\python.exe -m src.encode_gate2_hard_oos --split hard_oos_val --batch-size 8

# validation 校准；检查后再加 --freeze
.\.venv\Scripts\python.exe -m src.run_gate2 --phase validation
.\.venv\Scripts\python.exe -m src.run_gate2 --phase validation --freeze

# 冻结后才编码 held-out hard-OOS test
.\.venv\Scripts\python.exe -m src.encode_gate2_hard_oos --split hard_oos_test --batch-size 8

# 一次性 test
.\.venv\Scripts\python.exe -m src.run_gate2 --phase test

# 图表
.\.venv\Scripts\python.exe -m src.make_gate2_figures
```

`run_gate2.py` 在 freeze 前不会编码或评测 hard-OOS test；test 结果文件
存在后默认拒绝第二次读取。不要用 `--force` 重跑正式 test。

### 4.8 Gate 3 复现命令（实际已运行）

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# 1. 3 数据集统一 manifest
.\.venv\Scripts\python.exe -m src.gate3.prepare

# 2. 四个冻结编码器，每个模型一次性编码 3 个数据集
.\.venv\Scripts\python.exe -m src.gate3.encode --model bge
.\.venv\Scripts\python.exe -m src.gate3.encode --model qwen --device cuda --batch-size 8
.\.venv\Scripts\python.exe -m src.gate3.encode --model gte --device cuda --batch-size 16
.\.venv\Scripts\python.exe -m src.gate3.encode --model e5 --device cuda --batch-size 32

# 3. validation-only 的 72 格阈值选择
.\.venv\Scripts\python.exe -m src.gate3.run_matrix --phase validation

# 4. 冻结后才跑 final test；不要重复
.\.venv\Scripts\python.exe -m src.gate3.freeze
.\.venv\Scripts\python.exe -m src.gate3.run_matrix --phase test

# 5. 汇总与图
.\.venv\Scripts\python.exe -m src.gate3.aggregate
.\.venv\Scripts\python.exe -m src.gate3.make_figures

# 6. 训练式对照（可选，但本轮已运行）
.\.venv\Scripts\python.exe -m src.gate3.train_adapter
.\.venv\Scripts\python.exe -m src.gate3.train_setfit

# 7. Demo
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Gate 3 的 test 已在 `results/gate3_freeze.json` 后评测一次；再次运行
`--phase test` 会被拒绝。SetFit/Adapter 的 BANKING77 test 与矩阵 test 相同，
因此报告中明确标注“非盲测复用”。

### 4.9 Gate 4 复现命令（实际已运行）

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# 1. B4-Fusion + B4-Full 训练与 validation
.\.venv\Scripts\python.exe -m src.gate4.train_b4

# 2. 冻结 checkpoint + 阈值；之后 test 只跑一次
.\.venv\Scripts\python.exe -m src.gate4.freeze_b4
.\.venv\Scripts\python.exe -m src.gate4.run_b4_test

# 3. 16 条工具/状态测试 + 6 条真实路由演示
.\.venv\Scripts\python.exe -m src.gate4.run_agent_cases

# 4. 图表
.\.venv\Scripts\python.exe -m src.gate4.make_figures

# 5. 有状态 Agent Demo
.\.venv\Scripts\python.exe -m streamlit run app_gate4.py
```

Gate 4 的 test 在 `results/gate4_b4_freeze.json` 后只跑一次；BANKING77 test
在 Gate 3 已看过，因此报告中明确标注“探索性、非新盲测”。Agent 的 16 条
确定性测试使用 StubRouter；6 条演示使用真实 GTE/B3-H/B4，两组指标分开报告。

### 4.10 Gate 5 复现命令（实际已运行）

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"

# A. LoRA-GTE
.\.venv\Scripts\python.exe -m src.gate5.train_lora_gte --epochs 2
.\.venv\Scripts\python.exe -m src.gate5.encode_lora
.\.venv\Scripts\python.exe -m src.gate5.evaluate_lora

# B. RejectOnly + LoRA B4 + 9-row matrix
.\.venv\Scripts\python.exe -m src.gate5.train_reject_only --encoder frozen
.\.venv\Scripts\python.exe -m src.gate5.train_reject_only --encoder lora
.\.venv\Scripts\python.exe -m src.gate5.train_b4_lora
.\.venv\Scripts\python.exe -m src.gate5.run_method_matrix --phase validation
.\.venv\Scripts\python.exe -m src.gate5.run_method_matrix --phase test

# C. 30 KB docs / demos / app
.\.venv\Scripts\python.exe -m src.gate5.make_kb
.\.venv\Scripts\python.exe -m src.gate5.run_demos_gate5
.\.venv\Scripts\python.exe -m streamlit run app_gate5.py

# D. 120 benchmark / API status
.\.venv\Scripts\python.exe -m src.gate5.make_benchmark
.\.venv\Scripts\python.exe -c "from src.gate5.benchmark import load_cases, freeze; freeze(load_cases())"
.\.venv\Scripts\python.exe -m src.gate5.benchmark
.\.venv\Scripts\python.exe -m src.gate5.run_llm_api_cases

# Figures / tests
.\.venv\Scripts\python.exe -m src.gate5.make_figures
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Gate 5 的 BANKING77 test 是 `historical_exploratory`，只跑一次；120 条
benchmark 在比较前冻结 SHA。LoRA 4-epoch 第一次尝试保留在
`results/gate5_lora_4ep/`，最终 adapter 是 2 epoch。LLM API 因无凭据标记
`NOT RUN_API_BLOCKED`。

### 4.11 Gate 6 复现命令（实际已运行）

```powershell
$env:HF_HOME = "D:\code\intent-open-set\data\model_cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# A. Qwen 冻结对齐 + smoke
.\.venv\Scripts\python.exe -m src.gate6.inspect_qwen

# B. Plain / Anchor LoRA + 重编码 + validation
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_lora --run plain --epochs 2 --microbatch 2 --grad-accum 8
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_lora --run anchor --epochs 2 --microbatch 2 --grad-accum 8 --lambda-anchor 0.5
.\.venv\Scripts\python.exe -m src.gate6.encode_qwen_lora --run plain
.\.venv\Scripts\python.exe -m src.gate6.encode_qwen_lora --run anchor
.\.venv\Scripts\python.exe -m src.gate6.evaluate_qwen_lora

# C. Qwen B4 与矩阵
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_b4 --encoder frozen --method reject
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_b4 --encoder frozen --method full
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_b4 --encoder plain --method reject
.\.venv\Scripts\python.exe -m src.gate6.train_qwen_b4 --encoder plain --method full
.\.venv\Scripts\python.exe -m src.gate6.run_qwen_matrix

# D. Qwen Agent / benchmark (one process per router on Windows 8GB)
.\.venv\Scripts\python.exe -m src.gate6.run_qwen_demos
.\.venv\Scripts\python.exe -m streamlit run app_gate6.py
foreach ($n in @('Qwen-Frozen-B3H','Qwen-Plain-B3H','Qwen-Plain-RejectOnly','Qwen-Plain-B4Full')) {
  .\.venv\Scripts\python.exe -m src.gate6.benchmark_qwen --router $n
}
.\.venv\Scripts\python.exe -m src.gate6.combine_qwen_benchmark
.\.venv\Scripts\python.exe -m src.gate6.make_figures
```

Qwen 与 Gate3 对齐的关键是 **right padding + 最后一个非 pad token**；左 padding
会破坏 Qwen causal/RoPE 的位置路径，不能用于训练。Qwen test 是
`historical_exploratory`；120 条 Agent 任务是 Gate5 旧任务的探索性回归。

---

## 5. 数据协议（冻结）

| 项目 | 设置 |
|---|---|
| 数据集 | CLINC150 / OOS-Eval，官方 `data_full.json` |
| 官方 revision | `828f8093932c8fe6ca7936c3d2e52903b1c523de` |
| 已知类别 | 20 个，官方 10 个 domain 各 2 个 |
| 选择 seed | 42（SHA-256 排序，非 test 调参） |
| Support | 官方 `train`，每类 K=5/10；seed 42/43/44 |
| Calibration known | 20 类官方 `val` 全部 400 条 |
| Calibration OOS | 官方 `oos_val` 100 条 |
| Final known test | 20 类官方 `test` 全部 600 条 |
| Final OOS test | 官方 `oos_test` 1000 条 |
| `oos_train` | 本 pilot 不使用 |

**测试集不允许用于**：方法选择、prompt 选择、阈值调优、挑 seed、看错误后
反向改方法。Gate 1 只能读取固定参数并在最后跑一次。

### Gate 2 的独立 hard-OOS 协议

Gate 2 不覆盖 Gate 1，也不混用原始 `oos_test`：

- 剩余 130 个未支持类按 `SHA-256("hard-oos|42|<intent>")` 分成 65/65；
- dev 65 类的 1,300 条官方 val = hard-OOS validation；
- held-out 65 类的 1,950 条官方 test = hard-OOS final test；
- hard-OOS val/test 与 known support/val/test、原始 OOS 的文本重叠均为 0；
- B0/B2/B2-v2 在同一 validation 上重新校准；
- B2-v2 的 `kappa`、`u_quantile` 只在 validation 选，test 冻结后只评测一次。

详见 `reports/gate2_method.md`。

### Gate 3 的 3 数据集协议

| Dataset | Known / dev-unknown / final-unknown | Support | Calibration known | Known test | Dev unknown | Final unknown |
|---|---:|---:|---:|---:|---:|---:|
| CLINC150（历史） | 20 / — / — | 100 / 200 | 400 | 600 | 100 | 1,000 |
| BANKING77 | 40 / 18 / 19 | 200 / 400 | 800 | 1,600 | 720 | 760 |
| MASSIVE en-US | 35 / 12 / 13 | 175 / 350 | 1,323 | 1,915 | 407 | 436 |

- 类别选择：固定 SHA-256 seed=42；MASSIVE known 额外要求官方 train ≥10；
- BANKING77 的 dev/final unknown 来自官方 test 中互不重叠的类别；
- MASSIVE 的 dev unknown 用官方 `dev`，final unknown 用官方 `test`；
- 支持集/校准/测试的规范化文本交集为 0；
- CLINC150 原始 test 已查看，标为 historical，仅作追加分析。

### Gate 4 的 B4 与 Agent 协议

- B4 只用 BANKING77 400 条 10-shot support 训练；
- 每类 5 条原型、5 条 query，两折互换；每 episode 屏蔽 6/40 类，
  其 query 作为 pseudo-OOS；不使用 calibration/dev 标签做梯度；
- calibration known 800 + dev unknown 720 只用于 validation、阈值和诊断；
- checkpoint/阈值写入 `results/gate4_b4_freeze.json` 后，探索性 test 只跑一次；
- BANKING77 test 在 Gate 3 已看过，Gate 4 不声称新盲测；
- Agent 使用 `data/gate4/agent.db` 虚构 SQLite，不连接真实银行；
- 16 条 StubRouter 测试只验证工具/状态；6 条真实路由演示单独报告。

### Gate 5 的 LoRA / B4 / Agent / Benchmark 协议

- LoRA 与 B4 只使用 400 条 10-shot support；
- 4-epoch 第一次尝试仅作为 validation 失败记录，最终 2-epoch adapter 由
  validation 选择；test 不参与 epoch 选择；
- B4-RejectOnly 的 gate 参数 `requires_grad=False`，固定 `g=0.5`；
- 9 行矩阵在 `results/gate5_method_freeze.json` 后只做一次历史探索性 test；
- 30 条政策文档明确标记为虚构演示，不访问外部网页；
- 120 条 benchmark 在评测前写入
  `results/gate5_benchmark_freeze.json`，文件 SHA 不被修改；
- 三个真实路由（Frozen-B3H/Frozen-B4Full/LoRA-B4Full）使用相同输入和初始
  DB；StubRouter 不计入；
- LLM API 无凭据时为 `NOT RUN_API_BLOCKED`，离线模板模式不冒充 API。

### Gate 6 的 Qwen LoRA / B4 / Agent 协议

- Qwen revision、prompt、maxlen、right padding、last non-pad token 与 Gate3
  cache 对齐（mean cosine 0.999998）；
- LoRA 与 B4 只使用 400 条 10-shot support；Plain/Anchor 同 seed/epoch；
- Anchor teacher 是冻结 Qwen 的 detached 向量，不参与梯度；
- Qwen B4-RejectOnly 固定 gate 0.5，只训练拒识头；Full 训练 gate+拒识；
- 9 行矩阵在 `results/gate6_qwen_method_freeze.json` 后只做一次历史 test；
- Qwen test 已看过；120 条 Agent 任务是 Gate5 旧任务的探索性回归；
- Qwen benchmark 在 Windows 8GB 上按 router 分进程执行，最后合并；每个
  router 一个 partial CSV，不把 smoke/partial 冒充完整结果；
- 没有新的 LLM API 调用；API 仍为 `NOT RUN_API_BLOCKED`。

### 已知的官方数据重复

官方数据内部没有分区内重复，但规范化文本有 **5 处跨分区碰撞**：

- `train` ↔ `val`：3 处；
- `train` ↔ `test`：2 处。

完整样本 ID、标签和文本见
[data/manifest/leakage_report.json](data/manifest/leakage_report.json)。
没有删除或重标官方样本。构造出的 support 与 calibration/test 之间
ID 和规范化文本交集均为 0；但 `user_name` 的一个 test 文本在 train 中
存在另一个标签的重复，这是上游数据的残余风险，报告中已注明。

---

## 6. 不要做的事

- 不要用 test 结果倒推阈值、prompt、K、seed 或方法。
- 不要把 BGE smoke test 当成 B0 准确率。
- 不要改写或删除官方 CLINC150 样本来“修复”重复文本。
- 不要无意中用 `--force` 重新评估 test；如果必须重跑，要在报告中审计。
- 不要把 `Qwen3-0.6B`（生成模型）与
  `Qwen3-Embedding-0.6B`（嵌入模型）混淆。

---

## 7. 当前边界与后续扩展

已完成：Gate 0–7；3 数据集 × 4 编码器 × 2 shot × B0/B3-P/B3-H 的 72 格
test 矩阵；B4-AR；LoRA-GTE；RejectOnly/Full 归因；Qwen3-Embedding LoRA；
双层开发集与 3-seed 复验；RAG Agent；真实路由 benchmark；真实截图。

Gate 3 的主要判断：

- B3-H 在 BANKING77 有正收益：OOS recall +5.21 pp，open-set F1 +0.99 pp；
- B3-H 在 MASSIVE 和 CLINC150 没有稳定收益；
- Qwen 平均最强，GTE 在 BANKING77 最强且编码成本低；
- MASSIVE 的 F1-first 阈值选择在 BGE/GTE 上把 known acceptance 推到
  0.99+，几乎不拒 OOS，说明阈值目标本身是主要限制。

Gate 4 的主要判断：

- B4-Full 在 validation F1 +1.39 pp、test F1 +0.62 pp、test OOS +2.11 pp；
- B4-Fusion 单独退化：收益主要来自拒识头，不是可学习门控；
- 门控随输入难度变化：Top-1 gate 0.56（最难）→ 0.92（最易）；
- Agent 工具、状态、确认、所有权、审计链真实闭环；16/16 确定性测试、
  6/6 真实路由演示；
- Gate 4 test 是探索性对照，不是新盲测。

Gate 5 的主要判断：

- LoRA 真实训练成功（1.69M 参数、22.19 s、adapter 6.78 MB），但 LoRA-B3-H
  validation 从 0.767 降到 0.760，test 0.803，低于 frozen 0.831；
- **Frozen-B4-RejectOnly 是唯一稳定正贡献**：validation 0.790、test 0.843；
- B4-Full 的门控没有超过 RejectOnly；LoRA 与拒识头不互补；
- 120 条 benchmark：Frozen-B3H TSR 62.5%，B4/LoRA 58.3%；安全率 100%；
  主要失败是 routing，不是工具越权；
- 30 条政策 RAG + 10 条真实模型 demo 可用；API 12 条因无凭据
  `NOT RUN_API_BLOCKED`。

Gate 6 的主要判断：

- Qwen 冻结对齐通过（mean cosine 0.999998，top1 1.0），train 前先固定
  right padding + last non-pad token；
- Qwen Plain/Anchor validation 高于 frozen（B3-H 0.758/0.757 vs 0.740），
  但 historical test 没有复现（0.797/0.797 vs 0.806）；
- Anchor 把 support 余弦从 0.892 提到 0.936，但没有 test 收益；
- Qwen 上 RejectOnly > Full（Frozen 0.813/0.805；Plain 0.808/0.797）；
- Qwen Plain-RejectOnly Agent TSR 65.0%，四版本最高；分类 F1 仍是
  GTE Frozen-RejectOnly 0.8435 最高；
- 5 条应澄清任务 Qwen 全部直接接受，clarification=0，是真实缺陷。

Gate 7 的主要判断：

- 双层 dev 隔离后，Qwen Plain 3-seed 的 B3-H test 均值 0.8068、
  RejectOnly 0.8256；RejectOnly 相对 frozen +1.26 pp，2/3 seed 胜出；
- Gate6 的 seed42 负结果真实，但不是三种子普遍结论；
- 6×6 microbatch / 203/400 负 margin hard negative 是机制风险；
- P1/P2 没有超过 P0；P3 完成但未选中（dev B3-H 低于 P0）；
- B4-RejectOnly 在 GTE Frozen、GTE-LoRA、Qwen Frozen 上稳定优于 Full；
- Agent 修复 5 条澄清后，Qwen RejectOnly 81/120、GTE RejectOnly 73/120；
- 86 个测试通过。

尚未做：

- 新的独立 unknown-class test / 大样本 bootstrap；
- Qwen LoRA 更多 seed、Anchor λ 扫描与显著性检验；
- P3 已完成但未进入最终 test（dev 低于 P0）；
- 更多 BANKING77 intent 的 Agent 槽位；
- 实际 PPTX 排版、演讲录像；
- 12 条真实 LLM API 调用（无凭据）；
- 近域/远域未知的可靠统一定义（NOT RUN）。

不会在当前 BANKING77 test 上继续调 Qwen LoRA/B4 参数；若继续，需要新的
独立 unknown classes 或明确的新协议。
