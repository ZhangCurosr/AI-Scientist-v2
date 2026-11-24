# The AI Scientist-v2 — 技术文档地图

> 本目录文档基于**当前仓库真实源码**逐项阅读后编写，目标：让零上下文的 AI/工程师能直接修改代码、理解算法、排查运行故障。
> 所有类名、函数名、配置项、命令行参数均来自源码核验，关键定义以 `文件:行号` 标注。

---

## 1. 项目定位（一句话）

**The AI Scientist-v2 是一个「无模板、基于 Agentic 树搜索」的全自动科研系统**：给定一个高层研究主题（workshop 描述），LLM 用工具（Semantic Scholar）辅助产出研究想法；随后由 **AIDE 风格的 Best-First Tree Search（BFTS）** 在 4 个递进阶段中，让 Agent 自动生成/调试/改进实验代码、真实执行、用 VLM 分析实验图表，最后用多模型分工完成论文写作（8 页 normal 或 4 页 ICBINB）、文本评审与 VLM 图像审稿。

它生成了**第一篇完全由 AI 撰写、通过同行评审的 workshop 论文**（ICLR2025 ICBINB workshop）。

> 官方说明：v2 不保证比 v1 写出"更好"的论文。有强模板时 v1 成功率高；v2 走开放探索路线、成功率更低，但**不依赖人工模板、可跨 ML 领域**。

安全提示：本系统会执行 LLM 编写的代码，务必在容器/受限网络环境中运行。

---

## 2. 与 v1 的核心演进差别

| 维度 | v1（`AI-Scientist`） | v2（本仓库） |
| --- | --- | --- |
| 想法生成 | 依赖模板 `experiment.py`+`seed_ideas.json`，无工具调用 | **无模板**：`perform_ideation_temp_free.py` 基于 workshop 描述 + `SearchSemanticScholar`/`FinalizeIdea` 工具 |
| 实验组织 | 线性，每想法最多 5 次 run | **BFTS 树搜索**：`Journal`/`Node` 组成的解树，`ParallelAgent` best-first 扩展，含 debug/improve 分支 |
| 代码修改者 | Aider（`Coder`，diff 格式） | 自建 `MinimalAgent`/`ParallelAgent` 生成整段代码 + `Interpreter` 子进程执行；**不再用 Aider** |
| 阶段推进 | 无阶段概念 | `AgentManager` 管理 4 个主阶段（初始实现→基线调优→创新研究→消融实验），每阶段可含 LLM 动态生成的子阶段 |
| 评估方式 | 文本结果 → Aider | LLM 从 `.npy` 文件中提取结构化 `MetricValue`（多数据集/多指标、方向感知），`WorstMetricValue` 表示失败 |
| 图表反馈 | 无 | `ParallelAgent._analyze_plots_with_vlm` 用 VLM 分析每次实验产生的图，反馈进后续决策 |
| 写作 | 单一 ICLR2024 模板，逐节 SEARCH/REPLACE | 双模式：`normal`（8 页 ICML2025）/`icbinb`（4 页 ICLR2025），多模型分工（agg/citation/writeup/writeup_small） |
| 评审 | 仅 LLM 文本评审 | LLM 文本评审 + **`perform_vlm_review.py` 图-标题-正文引用一致性审稿** |
| 编排入口 | `launch_scientist.py` | `launch_scientist_bfts.py` + `bfts_config.yaml` |
| 产物统计 | 无 | `TokenTracker` 全局统计 token 与成本，落盘 `token_tracker*.json` |

---

## 3. 文档地图（阅读顺序建议）

| 文档 | 内容 | 何时读 |
| --- | --- | --- |
| [`architecture.md`](architecture.md) | BFTS 树搜索数据结构、AIDE 集成边界、`AgentManager`/`ParallelAgent`/后端分层与 Mermaid 架构图 | 想理解系统全貌、树搜索原理、模块边界时 |
| [`pipeline_and_dataflow.md`](pipeline_and_dataflow.md) | 无模板想法生成、4 阶段推进、回溯/剪枝、VLM 视觉审稿、ICBINB 写作的完整数据流与产物 | 想改流程、理解阶段切换、追踪中间文件时 |
| [`modules_reference.md`](modules_reference.md) | `treesearch/`、`agent_manager.py`、`perform_vlm_review.py` 等核心类/方法签名与入参出参 | 要动手改某个类/函数、接新模型/新阶段时 |
| [`execution_guide.md`](execution_guide.md) | `bfts_config.yaml` 全部配置、多模型分工、运行命令、可视化与排错 | 部署、调参、复现实验、排错时 |
| [`treesearch_deep_dive.md`](treesearch_deep_dive.md) ★深度 | `Node` 全属性、`Journal` 数据结构、`AgentManager`↔`ParallelAgent`↔`MinimalAgent` 调用链、**4 阶段状态转移判定代码逻辑**、`unified_tree_viz.html` 生成原理与 HTML/JS 结构 | 要**改选点/剪枝策略、阶段判定、多种子/可视化字段，或排查卡死分支**时（零上下文二次开发首选） |
| [`prompts_and_vlm_schemas.md`](prompts_and_vlm_schemas.md) ★深度 | 无模板构思完整 Prompt + IDEA JSON 出参、VLM 审稿 Prompt + Base64 传图 + JSON 契约、ICBINB 4 页写作 Prompt 拆解与三类 reflection | 要**替换 Prompt、修 JSON 解析、改 VLM 审稿/图描述**时 |
| [`troubleshooting.md`](troubleshooting.md) ★深度 | 树搜索卡死分支、`debug_prob` 机制、显存溢出自动降级、VLM 格式错误重试、模型超时处理，逐项「症状→根因→证据→修复」 | 运行卡住/报错、想按证据层级快速定位时 |

---

## 4. 顶层目录速览

```
AI-Scientist-v2/
├── launch_scientist_bfts.py       # 主入口：想法→树搜索实验→聚合图→写作→文本/VLM 评审→进程清理
├── bfts_config.yaml               # BFTS/Agent 核心配置（模型、阶段迭代数、worker 数、搜索策略）
├── ai_scientist/
│   ├── llm.py                     # 统一 LLM 客户端（带 TokenTracker）+ 模型白名单
│   ├── vlm.py                     # VLM 客户端（图像输入）
│   ├── perform_ideation_temp_free.py  # 无模板想法生成（工具调用式）
│   ├── perform_plotting.py        # aggregate_plots()：LLM 生成并执行图表聚合脚本
│   ├── perform_writeup.py         # normal 模式（8 页，ICML2025 模板）
│   ├── perform_icbinb_writeup.py  # icbinb 模式（4 页，ICLR2025 模板）+ gather_citations
│   ├── perform_llm_review.py      # 文本评审（集成 + 反思）
│   ├── perform_vlm_review.py      # 图/标题/正文引用一致性评审、重复图检测
│   ├── tools/                     # BaseTool + SemanticScholarSearchTool
│   ├── utils/token_tracker.py     # 全局 token/成本统计
│   ├── blank_icbinb_latex/        # ICBINB 论文模板（4 页）
│   ├── blank_icml_latex/          # normal 论文模板（8 页）
│   ├── ideas/                     # workshop 描述 .md + 生成的想法 .json + 参考代码 .py
│   ├── fewshot_examples/          # 评审 few-shot 论文样例
│   └── treesearch/                # ★ 核心：AIDE 派生的树搜索子系统
│       ├── perform_experiments_bfts_with_agentmanager.py  # perform_experiments_bfts() 入口
│       ├── agent_manager.py       # AgentManager：4 阶段推进、完成判定、子阶段生成
│       ├── parallel_agent.py      # ParallelAgent/MinimalAgent/GPUManager：树节点生成与并行执行
│       ├── journal.py             # Node/Journal：解树数据结构、最佳节点选择
│       ├── interpreter.py         # Interpreter：子进程执行代码 + ExecutionResult
│       ├── log_summarization.py   # 各阶段日志摘要、overall_summarize
│       ├── journal2report.py      # journal→报告
│       ├── backend/               # LLM 后端抽象（OpenAI/Anthropic）+ FunctionSpec
│       └── utils/                 # config / metric / response / serialize / tree_export / data_preview
├── experiments/                   # 运行时生成：<时间戳>_<Name>_attempt_<id>/
└── requirements.txt
```

> `treesearch/` 源自 AIDE（见根 README Acknowledgement），但本仓库已在其上做了大量 v2 定制（阶段管理、多数据集指标、VLM 反馈等），文档以**本仓库源码为准**。

---

## 5. 一句话理解一次运行

```
① python ai_scientist/perform_ideation_temp_free.py --workshop-file <主题.md> --model <LLM>
      → 生成 <主题>.json（研究想法列表）

② python launch_scientist_bfts.py --load_ideas <主题>.json [--load_code] [--add_dataset_ref] \
      --model_writeup ... --model_citation ... --model_review ... --model_agg_plots ...
      → experiments/<时间戳>_<Name>_attempt_0/
          idea.md / idea.json → 复制 bfts_config.yaml 并重定向路径
          → perform_experiments_bfts()：AgentManager 跑 4 阶段 BFTS（写 logs/0-run/*）
          → aggregate_plots()：LLM 聚合出 figures/
          → perform_[icbinb_]writeup()：生成 <目录名>.pdf（含 reflection 迭代）
          → perform_review() + perform_imgs_cap_ref_review()：review_text.txt / review_img_cap_ref.json
          → token_tracker*.json
```

想深入任一环节，请按上表进入对应文档。

> **零上下文 AI / 二次开发推荐路线**：先读 [`treesearch_deep_dive.md`](treesearch_deep_dive.md) 建立 Node/Journal/AgentManager/ParallelAgent 的源码级调用链与阶段判定认知 → 用 [`prompts_and_vlm_schemas.md`](prompts_and_vlm_schemas.md) 核对真实 Prompt/JSON 契约 → 用 [`troubleshooting.md`](troubleshooting.md) 按证据层级排错 → 需要函数签名与配置项时查 [`modules_reference.md`](modules_reference.md) 与 [`execution_guide.md`](execution_guide.md)。三份深度文档与既有文档交叉引用，行号锚点均可用 `grep -n` 复核。
