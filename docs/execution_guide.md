# The AI Scientist-v2 — 配置、运行与排错

> 基于根 `README.md`、`requirements.txt`、`bfts_config.yaml`、`launch_scientist_bfts.py`、`perform_ideation_temp_free.py` 核验。

---

## 1. 硬件与系统要求

- **Linux + NVIDIA GPU + CUDA + PyTorch**（README 明确；`launch_scientist_bfts.py` 用 `torch.cuda.device_count()` 打印可用 GPU；`GPUManager` 给并行 worker 分卡，无卡可走 CPU 但很慢）。
- 需要 PDF/LaTeX 工具：`poppler`、`chktex`（安装见下）。
- 建议容器化运行（会执行 LLM 写的代码）。

---

## 2. 环境与依赖

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia  # 按机器调整 CUDA 版本
conda install anaconda::poppler
conda install conda-forge::chktex
pip install -r requirements.txt
```

`requirements.txt` 关键项：`anthropic`, `openai`, `backoff`（LLM）；`matplotlib`, `seaborn`, `pypdf`, `pymupdf4llm`（图/PDF）；`numpy`, `transformers`, `datasets`, `tiktoken`, `wandb`, `tqdm`；`rich`, `humanize`, `dataclasses-json`, `funcy`, `black`, `genson`, `shutup`, `python-igraph`, `coolname`, `jsonschema`, `omegaconf`, `botocore`, `boto3`（树搜索/Rich UI/Bedrock）。

按模型补充：
- Bedrock Claude（树搜索默认 `agent.code.model` 是 `anthropic.claude-3-5-sonnet-20241022-v2:0`）：`pip install anthropic[bedrock]` + `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION_NAME`。
- Ollama 本地模型（`AVAILABLE_LLMS` 含 `ollama/...`）：本机启动 Ollama 服务（OpenAI 兼容端点 `http://localhost:11434/v1`）；可选 `OLLAMA_API_KEY`。

### API Key 汇总
| 变量 | 用途 |
| --- | --- |
| `OPENAI_API_KEY` | 写作（`o1`/`o3`/`gpt-4o`）、评审、图聚合、`model_citation`；默认配置几乎处处用到 |
| `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION_NAME` | Bedrock Claude（树搜索代码生成默认） |
| `S2_API_KEY` | Semantic Scholar（想法生成 novelty、写作引用；可选但有速率限制） |
| `GEMINI_API_KEY` / `DEEPSEEK_API_KEY` / `HUGGINGFACE_API_KEY` | 使用对应模型时 |
| `ANTHROPIC_API_KEY` | 直接用 Anthropic API 的 Claude（非 Bedrock） |

```bash
export OPENAI_API_KEY="..."
export AWS_ACCESS_KEY_ID="..." AWS_SECRET_ACCESS_KEY="..." AWS_REGION_NAME="us-east-1"
export S2_API_KEY="..."   # 可选
```

> 重要：`treesearch/backend/backend_anthropic.py` 的 Claude 走 **AnthropicBedrock**（不是普通 Anthropic API），因此树搜索的代码生成模型需要 AWS Bedrock 凭证；`ai_scientist/llm.py` 的 `claude-` 前缀走普通 Anthropic API。两套通道凭据不同。

---

## 3. `bfts_config.yaml` 配置详解

根配置会被 `bfts_utils.edit_bfts_config_file` 复制到每个 idea 目录并改写路径字段。原始内容（`bfts_config.yaml`）：

| 键 | 默认值 | 说明 / 修改建议 |
| --- | --- | --- |
| `data_dir` | `"data"` | 任务数据目录；运行时被改写为 idea 目录内的空 `data/`（Agent 通过 HuggingFace 拉数据） |
| `preprocess_data` | `False` | 是否对 `data_dir` 跑 `preproc_data` |
| `goal` / `eval` | `null` | 任务目标/评估描述；本流程用 `desc_file`（idea.json）代替，二者留空 |
| `log_dir` | `logs` | 日志根目录；`prep_cfg` 下建 `<log_dir>/<索引>-<exp_name>/`（默认 `0-run`） |
| `workspace_dir` | `workspaces` | Agent 工作区根目录；运行时被改写为 idea 目录 |
| `copy_data` | `True` | True=复制数据到工作区（防止 Agent 改原始数据），False=符号链接 |
| `exp_name` | `run` | 实验名；`prep_cfg` 会加索引前缀生成 `0-run`，留空则自动生成 coolname slug |
| `exec.timeout` | `3600` | 单段代码执行超时（秒）；CUDA OOM/长训练可调大 |
| `exec.agent_file_name` | `runfile.py` | worker 执行代码的文件名 |
| `exec.format_tb_ipython` | `False` | traceback 是否用 IPython 详细格式 |
| `generate_report` | `True` | 是否用 `overall_summarize` 产出 4 份 summary JSON（写作依赖） |
| `report.model` / `report.temp` | `gpt-4o-2024-11-20` / `1.0` | 最终报告摘要模型 |
| `experiment.num_syn_datasets` | `1` | 合成数据集数量参数 |
| `debug.stage4` | `False` | 阶段 4 调试开关 |
| `agent.type` | `parallel` | `parallel`/`sequential`（`prep_cfg` 校验） |
| `agent.num_workers` | `4` | 每步并行探索的节点数（进程池大小） |
| `agent.stages.stage1_max_iters` | `20` | 阶段 1（初始实现）最大节点数 |
| `agent.stages.stage2_max_iters` | `12` | 阶段 2（基线调优） |
| `agent.stages.stage3_max_iters` | `12` | 阶段 3（创新研究） |
| `agent.stages.stage4_max_iters` | `18` | 阶段 4（消融） |
| `agent.steps` | `5` | 通用最大步数；若某阶段未设 `stageN_max_iters` 则用此值 |
| `agent.k_fold_validation` | `1` | 提示词中是否让 Agent 用 CV（设为 1 表示禁用；README 称当前版本未使用） |
| `agent.multi_seed_eval.num_seeds` | `3` | 多种子评估次数；README 建议 ≤3 |
| `agent.expose_prediction` / `agent.data_preview` | `False` | 实验性开关（README 称当前未使用） |
| `agent.code.model` / `temp` / `max_tokens` | `anthropic.claude-3-5-sonnet-20241022-v2:0` / `1.0` / `12000` | **代码生成模型**（树搜索，Bedrock） |
| `agent.feedback.model` / `temp` / `max_tokens` | `gpt-4o-2024-11-20` / `0.5` / `8192` | 评估程序输出/traceback、阶段判定、指标解析、摘要 |
| `agent.vlm_feedback.model` / `temp` | `gpt-4o-2024-11-20` / `0.5` | 分析实验图表的 VLM |
| `agent.search.max_debug_depth` | `3` | debug 链最大深度（超过则剪枝） |
| `agent.search.debug_prob` | `0.5` | 每步选择 debug 分支的概率 |
| `agent.search.num_drafts` | `3` | 阶段 1 初始草稿树数量 |
| `agent.summary` / `agent.select_node` | 注释掉（默认 gpt-4o） | 可选：覆盖摘要/选节点使用的模型与温度 |

> 关键关系：`num_workers`（并行宽度）× 各阶段 `max_iters`（深度）决定搜索预算；README 举例 `num_workers=3, steps=21` 大约探索 21 个节点。实际每阶段用 `stageN_max_iters` 优先。

---

## 4. 运行命令

### 4.1 第一步：生成想法（无模板）

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file "ai_scientist/ideas/my_research_topic.md" \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 20 \
  --num-reflections 5
# 输出：ai_scientist/ideas/my_research_topic.json
```

参数：`--workshop-file`（主题 `.md`，默认 `ai_scientist/ideas/i_cant_believe_its_not_better.md`）、`--model`、`--max-num-generations`（默认 1）、`--num-reflections`（默认 5）。

主题 `.md` 参考 `ideas/i_cant_believe_its_not_better.md`（含 Title/Keywords/TL;DR/Abstract）；生成的 JSON 每个想法需含 `Name/Title/Short Hypothesis/Related Work/Abstract/Experiments/Risk Factors and Limitations`。

### 4.2 第二步：运行主流水线

```bash
python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/my_research_topic.json" \
  --load_code \
  --add_dataset_ref \
  --model_writeup o1-preview-2024-09-12 \
  --model_citation gpt-4o-2024-11-20 \
  --model_review gpt-4o-2024-11-20 \
  --model_agg_plots o3-mini-2025-01-31 \
  --num_cite_rounds 20
```

`launch_scientist_bfts.py` 全部 CLI 参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--writeup-type` | `icbinb` | `normal`(8 页)/`icbinb`(4 页) |
| `--load_ideas` | `ideas/i_cant_believe_its_not_better.json` | 想法 JSON 路径 |
| `--load_code` | flag | 加载同名 `.py` 作为可选起始代码 |
| `--idea_idx` | `0` | 用第几个想法 |
| `--add_dataset_ref` | flag | 追加根目录 `hf_dataset_reference.py`（若存在） |
| `--writeup-retries` | `3` | 写作失败重试次数 |
| `--attempt_id` | `0` | 并行重复实验的区分 ID |
| `--model_agg_plots` | `o3-mini-2025-01-31` | 图表聚合脚本模型 |
| `--model_writeup` | `o1-preview-2024-09-12` | 论文主写作模型（big） |
| `--model_citation` | `gpt-4o-2024-11-20` | 引用收集模型 |
| `--num_cite_rounds` | `20` | 引用轮数 |
| `--model_writeup_small` | `gpt-4o-2024-05-13` | 写作草案模型（small） |
| `--model_review` | `gpt-4o-2024-11-20` | 文本/VLM 评审模型 |
| `--skip_writeup` | flag | 跳过写作 |
| `--skip_review` | flag | 跳过评审 |

**多模型分工小结**：树搜索内部模型在 `bfts_config.yaml`（code=feedback=vlm_feedback）；想法生成 `--model`（ideation 脚本）；写作聚合/引用/大/小模型/评审各由上述 CLI 指定。**改"实验用什么模型"要改 yaml，改"写作用什么模型"要改 CLI**。

---

## 5. 输出与可视化

- 结果目录：`experiments/<date>_<Name>_attempt_<id>/`。
- **树可视化**：实验第一阶段完成时看 `experiments/<run>/logs/0-run/unified_tree_viz.html`（跨阶段分页）；每阶段目录下还有 `tree_plot.html` + `tree_data.json`。用浏览器打开。
- 阶段产物：`logs/0-run/stage_<name>/{journal.json, config.yaml, best_solution_<id>.py, best_node_id.txt, tree_plot.html, notes/}`。
- 论文：`<run目录名>.pdf`（及可能的 `*_reflection_*.pdf`）；图在 `figures/`。
- 评审：`review_text.txt`、`review_img_cap_ref.json`；成本：`token_tracker.json`。
- 终端：Rich Live 界面显示任务描述、阶段进度、解树（best 节点绿色高亮）；`Ctrl+C` 可停止。

---

## 6. 排错手册

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| `KeyError: 'Title'... not found in task_desc` | idea JSON 缺 5 个必需键之一 | 补齐 `Title/Abstract/Short Hypothesis/Experiments/Risk Factors and Limitations` |
| `You must provide either goal or desc_file` | `bfts_config.yaml` 的 `desc_file` 未设置（直接跑 `perform_experiments_bfts` 时） | 正常经 `launch_scientist_bfts.py` 跑会由 `edit_bfts_config_file` 注入；单独跑需手动设 `desc_file` |
| Claude 结构化调用报 `NotImplementedError: function calling` | `backend_anthropic.query` 不支持 function call；节点选择/阶段判定用了 Claude | 把 `agent.feedback.model`、`report` 等需要结构化的模型设为 OpenAI 系（gpt-4o 等） |
| Bedrock 认证失败 | 缺 AWS 凭证或 `agent.code.model` 区域不可用 | 配 `AWS_*` 环境变量；或把 `agent.code.model` 改为可用的 Bedrock/其它模型 |
| `CUDA out of memory` | Agent 生成的代码加载了过大的模型/数据 | 调大 `exec.timeout` 无效；按 README 改主题 `.md` 提示 Agent 用更小模型/更小 batch |
| 实验Stage 1 结束即退出："Could not find working implementation" | 20 次迭代内无 good 节点（代码总跑不通或指标解析失败） | 增大 `stage1_max_iters`、降低想法复杂度、换更强 code 模型；检查 `logs/0-run/stage_1_*/journal.json` 里 `is_buggy`/`parse_exc_*` |
| 一直停在阶段 2/3 不推进 | 最佳节点未超过 base 节点；阶段 3 运行时间过短被要求放大实验 | 看终端 `Feedback from _check_stage_completion`；增大 epoch/数据规模；调 `feedback` 模型 |
| 没有 PDF 生成 | 写作 3 次重试都失败；引用收集/页数控制出错 | 看控制台 writeup 日志；检查 `cached_citations.bib`；`--skip_review` 先隔离评审；确认 poppler/chktex 已装 |
| `find_pdf_path_for_review` 报 `UnboundLocalError` | idea 目录里没有任何 `reflection` PDF（该函数只处理 reflection_pdfs 分支） | 确认写作已产出 reflection PDF；否则在自定义流程中兜底返回普通 PDF |
| 图表聚合无 `figures/` | LLM 未返回代码块或聚合脚本运行失败 | 看 `auto_plot_aggregator.py` 与控制台；确认 `.npy` 数据被保存到 `experiment_results` |
| VLM 评审无 `review_img_cap_ref.json` | PDF 图像抽取失败或 VLM 调用失败 | 确认 PyMuPDF 可用、`<run>` PDF 存在；检查 `model_review` 是否支持视觉输入（gpt-4o） |
| 进程未退出/残留 torch 进程 | 结尾 psutil 清理失败 | 手动 `pkill -f bfts`；确认 `psutil` 已安装（脚本末尾 import） |

---

## 7. 成本与耗时（源自根 README）

- 想法生成：几美元（取决于模型与 generation/reflection 数）。
- 主实验流水线：Claude 3.5 Sonnet 约 **$15–20/run**。
- 写作：默认模型约 **+$5**；`model_citation` 用 GPT-4o 可降成本。
- 耗时：整个流水线通常数小时；写作阶段约 20–30 分钟。

---

## 8. 关键源码锚点（改配置/行为时）

- 阶段预算与搜索策略：`bfts_config.yaml`。
- 阶段目标/推进：`ai_scientist/treesearch/agent_manager.py` 的 `main_stage_dict`/`main_stage_goals`/`run`。
- 选点/剪枝：`parallel_agent.py` 的 `_select_parallel_nodes`、`_process_node_wrapper`。
- 指标比较：`utils/metric.py` 的 `MetricValue.__gt__`/`get_mean_value`。
- 可视化：`utils/tree_export.py` 的 `generate`/`create_unified_viz`。
- 写作/评审模型选择：`launch_scientist_bfts.py` 的 `parse_arguments` 与调用处。
