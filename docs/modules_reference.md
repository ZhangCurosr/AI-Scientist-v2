# The AI Scientist-v2 — 核心模块与函数参考

> 基于源码核验。给出类/方法签名、入参、返回、副作用与源码位置。行号基于当前仓库版本，可用 `grep -rn '^def \|^\s*def ' ai_scientist --include=*.py` 复核。

---

## 1. `treesearch/` 子系统

### 1.1 `treesearch/perform_experiments_bfts_with_agentmanager.py`

| 函数 | 位置 | 说明 |
| --- | --- | --- |
| `journal_to_rich_tree(journal, cfg)` | `:34` | 把当前阶段 Journal 转成 Rich Tree，标注 best 节点与 buggy 节点，用于终端 Live 显示 |
| `perform_experiments_bfts(config_path: str)` | `:58` | 树搜索入口：`load_cfg`→`load_task_desc`→`prep_agent_workspace`→建 `AgentManager`→`manager.run(exec_callback, step_callback)`；保存 `manager.pkl`；可选 `overall_summarize` 产出 4 份 summary JSON |

`step_callback(stage, journal)`（内部闭包）：写 `notes/node_<id>_summary.json`、`stage_progress.json`，调 `save_run`；`exec_callback` 包装 `interpreter.run` 并更新状态文本。

### 1.2 `treesearch/agent_manager.py`

**数据类**：
- `Stage(name, description, goals, max_iterations, num_drafts, stage_number)`（`:104`）：一个（子）阶段。`name` 格式 `<主阶段号>_<主阶段名>_<子阶段号>_<子阶段名>`（由 `parse_stage_names` 解析）。
- `StageTransition(from_stage, to_stage, reason, config_adjustments)`（`:114`）：记录阶段切换。

**`AgentManager(task_desc: str, cfg, workspace_dir: Path)`**（`:123`）：
- 入参 `task_desc` 是 idea JSON 的**字符串**，`__init__` 用 `json.loads` 解析，并强制要求 `Title/Abstract/Short Hypothesis/Experiments/Risk Factors and Limitations` 五个键，否则 `ValueError`。
- 关键属性：`stages: list[Stage]`、`journals: dict[str, Journal]`（每阶段一个）、`current_stage`、`stage_history`、`completed_stages`、`main_stage_dict`（1–4 阶段名）、`main_stage_goals`（各阶段目标文本）。
- `__init__` 通过 `_create_initial_stage()` 建立 `1_initial_implementation_1_preliminary` 阶段，`max_iterations` 取 `cfg.agent.stages.stage1_max_iters`（缺省 `cfg.agent.steps`），`num_drafts=cfg.agent.search.num_drafts`。

主要方法：

| 方法 | 位置 | 入参→返回 | 说明 |
| --- | --- | --- | --- |
| `_get_max_iterations(stage_number)` | `:171` | int→int | 读 `stage{n}_max_iters`，缺省 `agent.steps` |
| `_get_task_desc_str()` | `:179` | →str | 组装 Title/Abstract/Short Hypothesis/(Code) 的研究描述 |
| `_create_initial_stage()` | `:200` | →None | 建首阶段与其 Journal |
| `_curate_task_desc(stage)` | `:216` | Stage→str | 阶段 3 追加 Experiments 计划，阶段 4 追加 Risk Factors |
| `_save_checkpoint()` | `:249` | →None | 写 `logs/<run>/stage_<name>/checkpoint.pkl`（journals/stage_history/task_desc/cfg/...） |
| `_create_agent_for_stage(stage)` | `:274` | Stage→ParallelAgent | 复制 cfg、注入子阶段目标与上一阶段 best 节点（best_stage1/2/3_node） |
| `_parse_vlm_feedback(node)` | `:331` | Node→str | 取 `node.plot_analyses[0].analysis` 与 `vlm_feedback_summary` 拼成反馈文本 |
| `_check_substage_completion(current_substage, journal)` | `:343` | Stage,Journal→(bool,str) | 用 VLM 反馈 + 子阶段目标让 LLM（`stage_completion_eval_spec`）判 `is_complete` |
| `_check_stage_completion(stage)` | `:410` | Stage→(bool,str) | 4 阶段完成判定；阶段 1 无可用实现则 `current_stage=None` 结束实验；阶段 3 运行太短会写 `exec_time_feedback` 要求放大实验；到 max_iters 强制完成 |
| `_get_best_implementation(stage_name)` | `:538` | str→Node\|None | deepcopy 最佳节点并断开 parent/children，作为下一阶段起点 |
| `_generate_substage_goal(main_stage_goal, journal)` | `:552` | str,Journal→(goal_str, sub_stage_name) | 依据 `_gather_stage_metrics`/`_identify_issues`/`_analyze_progress` 让 LLM 生成子阶段目标（function call） |
| `_create_next_substage(current_substage, journal, substage_feedback)` | `:638` | →Stage\|None | 子阶段号 +1，`num_drafts=0` |
| `_create_next_main_stage(current_substage, journal)` | `:664` | →Stage\|None | 进入下一主阶段（阶段 4 返回 None）；子阶段号重置 1，名 `first_attempt` |
| `run(exec_callback, step_callback=None)` | `:692` | →None | 主循环：主阶段×子阶段；每步 `agent.step`→完成判定→多种子评估/聚合图→阶段切换；结束 `_save_checkpoint` |
| `_create_stage_analysis_prompt(...)` / `_gather_stage_metrics(journal)` / `_identify_issues(journal)` / `_analyze_progress(journal)` / `_evaluate_stage_progression(...)` | `:831/:1041/:1084/:1127/:1151` | — | 阶段进度分析、指标/问题/进展汇总、是否进入下一阶段的 LLM 评估（内部辅助） |
| `parse_stage_names(stage_name)` | `:927` | str→(int,str,int,str) | 正则拆出主阶段号/名、子阶段号/名 |
| `_save_stage_summary(...)` | `:943` | →None | 写 `notes/stage_<n>_complete/stage_completion_summary.json` |
| `_get_response(prompt)` | `:978` | str→dict | 结构化 LLM 调用（stage config schema） |

模块级 `FunctionSpec`：`stage_config_spec`、`stage_progress_eval_spec`、`stage_completion_eval_spec` —— 定义结构化输出的 JSON Schema（供 OpenAI function calling）。

### 1.3 `treesearch/parallel_agent.py`

**数据类**：`AblationConfig`（`:223`）、`AblationIdea`（`:238`）、`HyperparamTuningIdea`（`:246`）。

**`MinimalAgent(task_desc, cfg, memory_summary=None, evaluation_metrics=None, stage_name=None)`**（`:254`）：运行于 worker 进程，生成代码并解析结果。核心方法：

| 方法 | 位置 | 返回 | 说明 |
| --- | --- | --- | --- |
| `_draft()` | `:453` | Node | 从零起草实验代码（无 parent） |
| `_debug(parent_node)` | `:494` | Node | 依据 traceback 修 bug（parent.is_buggy） |
| `_improve(parent_node)` | `:523` | Node | 改进可运行实现（parent 非 buggy） |
| `_generate_seed_node(parent_node)` | `:549` | Node | 多种子评估：复用 parent 代码 |
| `_generate_hyperparam_tuning_node(parent, idea)` | `:557` | Node | 阶段 2 超参调优（写 `hyperparam_name`） |
| `_generate_ablation_node(parent, ablation_idea)` | `:605` | Node | 阶段 4 消融（写 `ablation_name`） |
| `plan_and_code_query(prompt, retries=3)` | `:658` | (str,str) | 让 LLM 输出 plan+code，提取代码 |
| `parse_exec_result(node, exec_result, workspace)` | `:683` | None | 把执行结果吸收到节点、判断 is_buggy 等 |
| `_generate_plotting_code(node, working_dir, plot_code_from_prev_stage=None)` | `:720` | str | 生成绘图代码（阶段 3/4 参考上一阶段 plot_code） |
| `_determine_datasets_successfully_tested(node)` | `:835` | list[str] | 从指标解析输出判断哪些数据集测试成功 |
| `_analyze_plots_with_vlm(node)` | `:894` | None | VLM 分析节点图（>10 张先让 LLM 选 10 张），写 `plot_analyses`/`vlm_feedback_summary`；`_prompt_environment`/`_prompt_impl_guideline` 等（`:274/:298`）给出全局指标定义与实现约束 |
| `_generate_node_summary(node)` | `:1035` | dict | 生成节点摘要（写入 notes） |

**`GPUManager(num_gpus)`**（`:1091`）：`acquire_gpu(process_id)->int` / `release_gpu(process_id)`；`get_gpu_count()`（`:1120`）读可见 GPU 数。

**`ParallelAgent(task_desc, cfg, journal, stage_name, best_stage1_node=None, best_stage2_node=None, best_stage3_node=None)`**（`:1142`）：阶段内 best-first 调度器。核心方法：

| 方法 | 位置 | 返回 | 说明 |
| --- | --- | --- | --- |
| `_define_global_metrics()` | `:1194` | str | 生成"全局指标定义"注入各 worker 的 prompt |
| `_select_parallel_nodes()` | `:1931` | list[Node\|None] | best-first 选点：草稿→debug（受 debug_prob/max_debug_depth 限制）→improve；阶段 2/4 用上一阶段 best；剪掉整树全 buggy 的分支；按树去重 |
| `step(exec_callback)` | `:2053` | None | 选点→进程池 `executor.submit(_process_node_wrapper,...)`→收集结果 `Node.from_dict`→`journal.append` |
| `_process_node_wrapper(...)`（模块级函数） | `:1410` | dict | worker：建环境→生成 child→执行代码→解析指标（MetricValue/WorstMetricValue）→绘图→VLM 分析→返回 `to_dict()` |
| `_run_multi_seed_evaluation(node)` | `:1261` | list[Node] | 复用最佳节点代码跑 `num_seeds` 次 |
| `_run_plot_aggregation(node, seed_nodes)` | `:1332` | Node | 聚合多种子结果，产出供写作的图 |
| `_generate_hyperparam_tuning_idea()` / `_generate_ablation_idea()` | `:1798/:1860` | Idea\|None | 生成下一个调参/消融方案（含去重状态） |
| `_get_leaves(node)` | `:1921` | list[Node] | 取子树叶节点（用于剪枝判定） |
| `__enter__/__exit__/cleanup` | `:2333/2367/2336` | — | 管理进程池/资源清理 |

### 1.4 `treesearch/journal.py`

| 类/方法 | 位置 | 说明 |
| --- | --- | --- |
| `Node` | `:44` | 解树节点（字段见 architecture 文档 2.1） |
| `Node.stage_name` / `is_leaf` / `debug_depth` | `:159/:192/:203` | draft/debug/improve 判定、叶节点、debug 深度 |
| `Node.absorb_exec_result` / `absorb_plot_exec_result` | `:170/:178` | 吸收 ExecutionResult |
| `Node.to_dict` / `from_dict` | `:214/:294` | 序列化/反序列化（跨进程；`from_dict(data, journal)` 恢复父子关系） |
| `Journal` | `:362` | 节点集合；`append`、`draft_nodes`、`buggy_nodes`、`good_nodes`、`get_node_by_id`、`get_metric_history` |
| `Journal.get_best_node(only_good=True, use_val_metric_only=False, cfg=None)` | `:420` | LLM（`node_selection_spec`）综合选择最佳节点，失败回退 `max(metric)`；`cfg.agent.select_node` 可配模型 |
| `Journal.generate_summary(include_code=False, **model_kwargs)` | `:504` | LLM 总结成功/失败实验，作为 worker 的 memory_summary |
| `Journal.to_dict` / `save_experiment_notes(...)` | `:561/:565` | 序列化/落盘笔记 |

### 1.5 `treesearch/utils/*`

- `utils/config.py`：
  - 数据类 `ThinkingConfig/StageConfig/SearchConfig/DebugConfig/AgentConfig/ExecConfig/ExperimentConfig/Config`（`:30–89`）——仅作 OmegaConf 结构化 schema 提示，**真实配置在 `bfts_config.yaml`**。
  - `load_cfg(path)`（`:134`）= `prep_cfg(_load_cfg(path))`：解析路径、生成带索引的 `exp_name`（coolname）、创建 log/workspace 目录、合并 schema 校验 `agent.type∈{parallel,sequential}`。
  - `prep_cfg(cfg)`（`:139`）：校验 `data_dir`、`desc_file` 或 `goal` 必须存在；`load_task_desc`（`:183`）读 `desc_file` 或由 `goal/eval` 构造；`prep_agent_workspace`（`:209`）建 input/working 并按 `copy_data` 复制/符号链接数据；`save_run(cfg, journal, stage_name)`（`:219`）写 journal.json/config.yaml/tree_plot.html/best_solution_*.py/best_node_id.txt。
- `utils/metric.py`：`MetricValue`（`:114`，`__gt__` `:171`、`get_mean_value` `:302`、`value_npsafe` `:264`）、`WorstMetricValue`（`:328`）、`MetricValue_old`（`:11`）。
- `utils/response.py`：`wrap_code`、`is_valid_python_script`、`extract_jsons`、`trim_long_string`、`extract_code`、`extract_text_up_to_code`、`format_code` —— LLM 输出解析工具。
- `utils/serialize.py`：`dumps_json/dump_json/loads_json/load_json`（dataclasses_json）、`parse_markdown_to_dict`。
- `utils/tree_export.py`：`cfg_to_tree_struct`（`:76`）、`generate_html`（`:362`，用 `viz_templates/template.{html,js}` 注入树数据）、`generate`（`:376`，写 tree_plot.html + tree_data.json + unified viz）、`create_unified_viz`（`:414`，跨阶段分页可视化）、`get_completed_stages`（`:43`）。
- `utils/data_preview.py`：数据目录树/CSV/JSON 预览（`generate` `:121`），供 `data_preview` 配置使用。
- `interpreter.py`：`ExecutionResult`（`:27`）、`Interpreter`（`:81`，`run(code, reset_session=True)` `:213` 返回 ExecutionResult；`_run_session`/`create_process`/`cleanup_session` 管理子进程）、`RedirectQueue`、`exception_summary`。
- `log_summarization.py`：`get_nodes_infos`（`:109`）、`get_summarizer_prompt`（`:138`）、`get_stage_summary`（`:149`）、`update_summary`（`:198`）、`annotate_history`（`:262`）、`overall_summarize(journals, cfg)`（`:299`，产出 4 类 summary）。
- `journal2report.py`：`journal2report(journal, task_desc, rcfg)`（`:6`）。
- `backend/__init__.py`：`get_ai_client(model)`、`query(system_message, user_message, model, temperature=None, max_tokens=None, func_spec=None, **kwargs)`（`:19`，o1/o3 特殊处理）；`backend_openai.py`/`backend_anthropic.py` 的 `query` 返回 `(output, req_time, in_tok, out_tok, info)`；`backend/utils.py` 的 `FunctionSpec`（`:106`，`as_openai_tool_dict`、`openai_tool_choice_dict`）与 `compile_prompt_to_md`（`:44`）。

---

## 2. 入口与辅助模块

### 2.1 `launch_scientist_bfts.py`

| 函数 | 位置 | 说明 |
| --- | --- | --- |
| `print_time()` | `:31` | 时间戳日志 |
| `save_token_tracker(idea_dir)` | `:35` | 写 `token_tracker.json` 和 `token_tracker_interactions.json` |
| `parse_arguments()` | `:42` | CLI 参数（见 execution_guide） |
| `get_available_gpus()` | `:134` | `torch.cuda.device_count()` |
| `find_pdf_path_for_review(idea_dir)` | `:140` | 选 final/编号最高 reflection PDF |
| `redirect_stdout_stderr_to_file(log_file_path)` | `:168` | 上下文管理器重定向日志 |
| `__main__` | `:182` | 全流程编排（含结尾 psutil 子进程清理） |

### 2.2 `treesearch/bfts_utils.py`
- `idea_to_markdown(data: dict, output_path: str, load_code: str) -> None`（`:7`）：idea JSON → Markdown（含可选代码块）。
- `edit_bfts_config_file(config_path, idea_dir, idea_path) -> str`（`:45`）：复制配置并设置 `desc_file`/`workspace_dir`/空 `data_dir`/`log_dir`，返回 run 配置路径。

### 2.3 `ai_scientist/perform_ideation_temp_free.py`
- `generate_temp_free_idea(idea_fname, client, model, workshop_description, max_num_generations=20, num_reflections=5, reload_ideas=True) -> list[dict]`（`:128`）：工具式（SearchSemanticScholar/FinalizeIdea）无模板想法生成，写回 `idea_fname`。

### 2.4 `ai_scientist/perform_plotting.py`
- `aggregate_plots(base_folder, model="o1-2024-12-17", n_reflections=5) -> None`（`:136`）：LLM 生成并执行 `auto_plot_aggregator.py`，产出 `figures/`。辅助：`build_aggregator_prompt`（`:52`）、`extract_code_snippet`（`:89`）、`run_aggregator_script`（`:99`）。

### 2.5 `ai_scientist/perform_icbinb_writeup.py`（4 页）
- `perform_writeup(base_folder, citations_text=None, no_writing=False, num_cite_rounds=20, small_model="gpt-4o-2024-05-13", big_model="o1-2024-12-17", n_writeup_reflections=3, page_limit=4) -> bool`（`:857`）：ICBINB 论文写作主函数（含 VLM 图描述、页数控制、reflection）。
- `gather_citations(base_folder, num_cite_rounds=20, small_model=...) -> str`（`:745`）：收集引用（缓存 `cached_citations.bib`）。
- `load_idea_text`（`:648`）、`load_exp_summaries`（`:665`）、`filter_experiment_summaries(exp_summaries, step_name)`（`:691`）：按 `citation_gathering`/`writeup`/`plot_aggregation` 过滤节点字段。
- `check_page_limit`（`:238`）、`detect_references_position_clean`（`:121`）、`get_reflection_page_info`（`:304`）、`compile_latex`（`:45`）：页数与编译工具。

### 2.6 `ai_scientist/perform_writeup.py`（8 页 normal）
- `perform_writeup(base_folder, no_writing=False, num_cite_rounds=20, small_model=..., big_model=..., n_writeup_reflections=3, page_limit=8) -> bool`（`:455`）：用 `blank_icml_latex` 模板，读 `logs/0-run/{baseline,research,ablation}_summary.json`，写 8 页论文。

### 2.7 `ai_scientist/perform_vlm_review.py`
- `perform_imgs_cap_ref_review(client, client_model, pdf_path) -> dict`（`:372`）：抽图 + VLM 评审，返回 `{img_name: review}`，review 字段 `Img_description/Img_review/Caption_review/Figrefs_review`。
- `extract_figure_screenshots(pdf_path, img_folder_path)`（`:154`）：PyMuPDF 抽图、caption、正文 figref。
- `extract_abstract(text)`（`:311`）、`generate_vlm_img_cap_ref_review(...)`（`:350`）、`generate_vlm_img_review(...)`（`:363`）。
- `detect_duplicate_figures(client, client_model, pdf_path)`（`:389`）、`generate_vlm_img_selection_review(...)`（`:448`）、`perform_imgs_cap_ref_review_selection(...)`（`:464`）：图去重/带页数约束的图选择评审。
- 提示词常量：`img_cap_ref_review_prompt`、`img_cap_selection_prompt`、`img_review_prompt`。

### 2.8 `ai_scientist/perform_llm_review.py` 与 `vlm.py`
- `perform_review(text, model, client, num_reflections=1, num_fs_examples=1, num_reviews_ensemble=1, temperature=0.75, ...) -> dict`（`:125`）：文本评审（与 v1 同构），主入口用默认参数调用。
- `load_paper(pdf_path, num_pages=None, min_size=100)`（`:257`）：三级回退 PDF 解析。
- `vlm.py`：`get_response_from_vlm`（`:132`）、`get_batch_responses_from_vlm`（`:251`）、`make_vlm_call`（`:96`）、`create_client`（`:195`）——图像输入的 LLM 通道，带 `@track_token_usage`。

### 2.9 `ai_scientist/utils/token_tracker.py` 与 `llm.py`
- `TokenTracker`（`:10`）：`add_tokens`/`add_interaction`/`get_summary`（含 cost）/`calculate_cost`；全局 `token_tracker` 单例；`track_token_usage` 装饰器（`:143`）。
- `llm.py`：`AVAILABLE_LLMS`（模型白名单，含 ollama/qwen/deepseek-r1 等）、`create_client(model)`（`:480`）、`get_response_from_llm`（`:267`，带 `@track_token_usage`）、`get_batch_responses_from_llm`（`:87`）、`extract_json_between_markers`（`:452`）。

> 复核提示：`grep -rn '^def \|^    def ' ai_scientist/treesearch ai_scientist/*.py` 可获取最新定义行号；本文行号基于当前仓库版本。
