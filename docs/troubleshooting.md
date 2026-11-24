# The AI Scientist-v2 — 树搜索与 VLM 排错指南

> 基于真实源码核验；每项「症状 → 源码根因 → 定位证据 → 修复」。行号 `文件:行号` 可用 grep 复核。
> 配套：[`treesearch_deep_dive.md`](treesearch_deep_dive.md)（节点/阶段/可视化原理）、 [`prompts_and_vlm_schemas.md`](prompts_and_vlm_schemas.md)（Prompt/JSON）、 [`execution_guide.md`](execution_guide.md)、 [`modules_reference.md`](modules_reference.md)。

---

## 0. 证据层级（v2 排错先看这里）

v2 状态全在 `experiments/<date>_<Name>_attempt_<id>/` 下。按下列顺序取证：

| 证据 | 位置 | 回答 |
| --- | --- | --- |
| 想法 | `idea.md` / `idea.json` | 5 个必需键是否齐全 |
| 每阶段 journal | `logs/0-run/stage_<name>/journal.json` | 有哪些节点、metric、is_buggy、父子 |
| 阶段进度 | `logs/0-run/stage_<name>/notes/stage_progress.json` | total/buggy/good 节点数、best_metric、findings |
| 单节点摘要 | `logs/0-run/stage_<name>/notes/node_<id>_summary.json` | 每节点结论 |
| 可视化 | `logs/0-run/stage_<name>/tree_plot.html`、`tree_data.json`、`logs/0-run/unified_tree_viz.html` | 树形态、buggy 分支、VLM 反馈 |
| 最佳代码 | `logs/0-run/stage_<name>/best_solution_<id>.py`、`best_node_id.txt` | 每阶段最终实现 |
| checkpoint | `logs/0-run/stage_<name>/checkpoint.pkl` | journals/stage_history/cfg（主阶段末写） |
| 实验临时产物 | `logs/0-run/experiment_results/experiment_<id>_proc_<pid>/` | 每节点 experiment_code.py/plotting_code.py/*.npy/*.png |
| 阶段总结 | `logs/0-run/{draft,baseline,research,ablation}_summary.json` | 写作输入的过滤源 |
| 论文/评审 | `<run>.pdf`、`*_reflection_*.pdf`、`review_text.txt`、`review_img_cap_ref.json` | 写作与评审结果 |
| token 成本 | `token_tracker.json`、`token_tracker_interactions.json` | 花费与调用明细 |

控制台 Rich 状态含阶段名、`Stage X not complete. Missing: ...`、`exec_time_minutes`、`Stage ... completed: reached max iterations` 等关键行；并行模式下进程 stdout 可能交叠。

---

## 1. 树搜索卡死 / 分支不再推进

### 1.1 卡在某一阶段迟迟不切换
**根因**（`agent_manager.py`）：
- `_check_stage_completion`（`:410`）：阶段 2 要求 `best_node != journal.nodes[0]`（有改进）且 VLM 判达标；阶段 3 要求 best≠起点且（过半后 exec_time 足够长）；若 best 一直是继承的起点或指标无提升 → 永远 `(False, "No improvement found...")`。
- **子阶段的"跑满 max_iters 强制完成"分支在源码里不可达**：`_check_substage_completion`（`:343-408`）中 `if len(journal.nodes)>=max_iterations` 的 return 位于 `try/except` 之后、已有 `return` 之后（`:399-407`），实际只有 LLM `is_complete=True` 才能结束子阶段。**LLM 一直判 incomplete 就会一直生成新子阶段**（`_create_next_substage` `:638`）。
- 阶段 4 只能靠 `len(nodes)>=max_iters`（`:412`）强制完成；`stage4_max_iters` 在 `bfts_config.yaml` 默认 18，较大。

**定位**：看 `notes/stage_progress.json` 的 `total_nodes` vs `stage*_max_iters`；看控制台 `Missing: ...`；看 `stage_history`（checkpoint.pkl 或控制台）是否在无限追加子阶段。

**修复**：
- 调小 `bfts_config.yaml` 的 `agent.stages.stage{2,3,4}_max_iters`（默认 12/12/18）。
- 临时放宽判定：改 `agent_manager.py:410` 的阶段 2/3 判定（如移除 `best_node==nodes[0]` 检查），或给 `_check_substage_completion` 加一个真正可达的 max_iters 兜底。
- 提高 `cfg.agent.feedback.model` 能力（默认 `gpt-4o-2024-11-20`），让 `is_complete` 判定更果断。

### 1.2 所有分支全 buggy，树"死掉"
**根因**：`_select_parallel_nodes`（`parallel_agent.py:1931`）的 `viable_trees` 只保留"至少一个叶节点非 buggy"的树；整树全 buggy 不再扩展。若所有树都死，`good_nodes` 为空 → 走到 `nodes_to_process.append(None)`（回退新草稿，`:2011`）。若草稿数已达 `num_drafts` 且无 viable 树，循环可能反复 `continue`（依赖 debug 分支救回）。

**定位**：`journal.json` 中统计 `is_buggy=true` 比例；`tree_plot.html` 看红/蓝节点分布；控制台 `Checking debuggable nodes` 是否频繁。

**修复**：
- 增大 `agent.search.num_drafts`（默认 3）播种更多树。
- 增大 `agent.search.debug_prob`（默认 0.5）提高调试概率、`max_debug_depth`（默认 3）放宽调试深度。
- 降低研究想法复杂度（换 idea_idx 或重写 idea 的 Experiments），减小 `exec.timeout`。

### 1.3 阶段 1 跑满仍无可运行实现 → 实验提前结束
**根因**：`_check_stage_completion`（`:412-426`）在阶段 1 到 max_iters 仍无 `good_nodes` 时 `self.current_stage = None`，打印 "Experiment ended: Could not find working implementation ..."。
**修复**：增大 `stage1_max_iters`（默认 20），或用 `--load_code` 提供起始代码（idea 同名 `.py`，注入 `idea["Code"]`），或 `AgentManager._get_task_desc_str` 的 "Code To Use" 直接用。

### 1.4 worker 抛异常导致整步失败
**根因**：`step()`（`:2159-2186`）收 `future.result` 时除 `TimeoutError` 外的异常 `traceback.print_exc(); raise`——**一个 worker 的异常会往上抛，中断 run()**（除非被更外层 `try` 吞掉）。常见：`_generate_hyperparam_tuning_idea()` 返回 None 时 `.name` 触发 `AttributeError`（`:2120`）；`node_data["is_buggy"]` 为 None 时比较异常；pickle 失败（`_safe_pickle_test` 已预检但可能有漏）。
**定位**：控制台 `Error processing node: ...` + traceback。
**修复**：在 `step()` 的 idea 生成处加 None 判断；核实节点 `is_buggy` 已被赋值（buggy 判定在 `parse_exec_result`）；保证自定义节点属性可 pickle。

---

## 2. `debug_prob` 与调试分支机制

### 2.1 机制（`parallel_agent.py:1931-2051`）
- 每轮 `_select_parallel_nodes` 先补草稿（`len(draft_nodes) < cfg.agent.search.num_drafts` → 追加 `None`）。
- 然后 `if random.random() < search_cfg.debug_prob`（`:1962`）：从 `journal.buggy_nodes` 中筛 `is_leaf and debug_depth <= search_cfg.max_debug_depth` 的节点（`:1975-1981`），`random.choice` 选一个调试；若其所属树已被本轮处理且已处理的树数未达 viable 树总数，则跳过换其它。
- `Node.stage_name`（`journal.py:159`）决定 worker 用 `_debug`（parent.is_buggy）还是 `_improve`；`debug_depth = parent.debug_depth + 1`（`:203`）防止无限深挖。

### 2.2 默认值与调优（`bfts_config.yaml`）
```
agent.search:
  max_debug_depth: 3
  debug_prob: 0.5
  num_drafts: 3
```
| 现象 | 调法 |
| --- | --- |
| 小 bug 反复出现、好节点少 | 提高 `debug_prob`（如 0.7~0.8），适当提高 `max_debug_depth` |
| 一直在 debug、不做创新 | 降低 `debug_prob`（如 0.2），让 improve 分支多跑 |
| Debug 到深层仍修不好 | 降 `max_debug_depth`（超限 buggy 节点被剪枝），靠 `num_drafts` 重播种 |
| debug 分支完全不触发 | 检查是否 `buggy_nodes` 为空、或 `debug_depth` 都已超限；看控制台 `Found debuggable nodes` |

> 注意 `_select_parallel_nodes` 中 `debuggable_nodes` 的初始化在 `try` 内，若筛选抛异常会被 `except` 吞掉并打印 `Error getting debuggable nodes`，此时该轮 debug 被跳过。

---

## 3. 显存溢出（CUDA OOM）与自动降级

### 3.1 系统已有的降级路径（源码）
1. **GPU 计数**：`get_gpu_count()`（`parallel_agent.py:1120`）用 `nvidia-smi --query-gpu=gpu_name`；失败则读 `CUDA_VISIBLE_DEVICES`（过滤空串与 `-1`）；都没有返回 0。
2. **CPU 回退**：`ParallelAgent.__init__`（`:1160-1166`）：`num_gpus==0` → 打印 "No GPUs detected, falling back to CPU-only mode"，`gpu_manager=None`。
3. **worker 降级**：`_process_node_wrapper`（`:1437-1442`）：`gpu_id=None` → `os.environ["CUDA_VISIBLE_DEVICES"]=""`（强制 CPU）。
4. **GPU 获取失败兜底**：`step()`/`_run_multi_seed_evaluation` 的 `gpu_manager.acquire_gpu` 抛 `RuntimeError("No GPUs available")` 时 `logger.warning` 后 `gpu_id=None` 走 CPU（`:2077-2083`、`:1266-1274`）。
5. **worker 数限制**：有 GPU 时 `num_workers = min(cfg.agent.num_workers, num_gpus)`（`:1168`），减少并发抢显存。

### 3.2 OOM 的实际表现与处理
- LLM 生成的实验代码若 OOM，`Interpreter.run` 捕获异常 → `exc_type` 非 None → `parse_exec_result` 置 `is_buggy=True` → 该 buggy 节点进入 debug 分支（见 §2），LLM 拿到 stderr 尝试修 bug（通常减小 batch/model）。**这就是"自动降级"的实际路径：靠 debug 分支让 LLM 改小，而非框架主动降 batch。**
- `Interpreter.run`（`interpreter.py:213-311`）在子进程执行：`reset_session=True` 时每段代码新建进程并在结束后 `cleanup_session()`（`:196-208`：terminate→join(2s)→kill→join(2s)），因此显存通常能随进程退出释放；但若子进程未正常结束，可能残留占卡。

### 3.3 人工处理
```bash
nvidia-smi                                  # 找残留占卡进程
kill -9 <pid>                               # 清残留
export CUDA_VISIBLE_DEVICES=0               # 限定单卡，降低并发
```
- 降低 `bfts_config.yaml` `agent.num_workers`（默认 4）到 1~2。
- 在代码生成 guideline（`parallel_agent.py:298` `_prompt_impl_guideline`）里已强制 `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 与 `.to(device)`；若要更激进的省显存，可往该 prompt 追加"使用 gradient checkpointing / 减小 batch / 及时 del + empty_cache"等约束。
- 增大 `exec.timeout`（默认 3600s）时要权衡显存占用时长。
- 无 GPU 环境可纯 CPU 跑（`num_gpus=0` 自动 CPU），但阶段 3 会因 `exec_time_feedback` 判定"运行太短"不断要求放大实验——见 `treesearch_deep_dive.md §5.3`。

---

## 4. VLM 输出格式错误与重试

### 4.1 解析契约
VLM 输出必须含 ```json 围栏，字段由各 Prompt 规定（`prompts_and_vlm_schemas.md §2`）；解析用 `vlm.py:216` 的 `extract_json_between_markers`（正则 → `{}` 兜底 → 去控制字符）。**解析失败返回 `None`**，调用方表现：
- `generate_vlm_img_cap_ref_review`（`perform_vlm_review.py:350`）：返回 `None` → `perform_imgs_cap_ref_review`（`:372`）把 `None` 存进 `img_reviews[img_name]` → 写进 `review_img_cap_ref.json` 的是 `null`。
- ICBINB 写图描述（`:960-970`）：`review_data=None` → `desc_map[pf]="No description found"`（有兜底）。
- 树搜索 `_analyze_plots_with_vlm`（`parallel_agent.py:894`）：用 function calling（`vlm_feedback_spec`），格式由 schema 约束，异常被 `try/except` 包住并记录 `Error analyzing plots`（`:1780-1788`）。

### 4.2 内置重试机制（源码现状）
| 位置 | 重试 |
| --- | --- |
| `MinimalAgent.plan_and_code_query`（`:658`） | `retries=3`，失败往 prompt 加 `Parsing Feedback` 要求用 ```python |
| `_determine_datasets_successfully_tested`（`:835`） | `retry_limit=5`，要求 `REASONING:`/`SUCCESSFULLY_TESTED_DATASETS:` 两行 |
| `_generate_plotting_code` 执行（`_process_node_wrapper:1676`） | 绘图代码异常 `retry_count<3` 重试 |
| `vlm.py` `get_response_from_vlm/get_batch_responses_from_vlm` | `@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))` |
| `Journal.get_best_node` / 各 `_check_*_completion` | LLM 结构化调用异常 → `except` 回退（best → `max(metric)`；判定 → False） |

### 4.3 排查与加固
- **症状**：`review_img_cap_ref.json` 里某图值为 `null`；或 `plot_descriptions_str = "No descriptions available."`；或日志 `Error analyzing plots for node ...`。
- **定位**：`extract_json_between_markers` 无 debug 打印；临时在 `generate_vlm_img_cap_ref_review` 里 `print(content)` 看 VLM 原始输出是否漏了围栏/字段名。
- **修复**：
  1. 换更强 VLM（`--model_review` 默认 `gpt-4o-2024-11-20`；树搜索 `cfg.agent.vlm_feedback.model`）。
  2. 在 Prompt 末尾追加 "Output ONLY the JSON block, no prose outside ```json" 强化格式。
  3. 在 `generate_vlm_img_cap_ref_review` 外包一层重试（当前无重试），失败重发一次或回退 `img_review_prompt`（只要两字段，更易遵守）。
  4. 确认图片有效：`detail:"low"`（`vlm.py:165`）分辨率低，信息量不足时 VLM 可能拒答；`extract_figure_screenshots` 的 `min_text_length=50/min_vertical_gap=30` 可能截错区域，看图文件 `<pdf名>_imgs/*.png`。

---

## 5. 模型超时 / API 错误处理

### 5.1 代码执行超时（`Interpreter`，`interpreter.py:213-311`）
- 每段代码 `timeout=cfg.exec.timeout`（默认 3600s）。超过时发 `SIGINT`（`:284`），`child_in_overtime=True`；若再超 `timeout+60` 秒仍不退出 → `cleanup_session()` kill，返回 `state=(None,"TimeoutError",{},[])`，`exec_time=timeout`。
- 子进程"意外死亡"（未发 overtime 中断就死）→ `RuntimeError("REPL child process died unexpectedly")`（`:262-269`）。
- 启动阶段 10 秒无 `state:ready` → `RuntimeError("REPL child process failed to start execution")`（`:243-249`）。
- **超时影响**：`exc_type="TimeoutError"` 非 None → 节点 is_buggy → 进 debug；控制台 `Execution exceeded the time limit ...`。
- **处理**：增大 `exec.timeout`；或让 LLM 缩小实验（阶段 3 的 `exec_time_feedback` 只在"太短"时触发，太长靠 is_buggy→debug）；OOM 与超时都可能表现为 exc_type 非 None，看 `journal.json` 的 `exc_info` 区分。

### 5.2 `future.result(timeout=self.timeout)` 超时（`parallel_agent.py:2160`）
`step()` 收 worker 结果用 `future.result(timeout=self.timeout)`（=3600s）；超时打印 `Worker process timed out, couldn't get the result`，该节点不入 journal，GPU 在 `finally` 释放。**修复**：确保 `exec.timeout` 与 LLM 调用时长之和 < `future` 超时；减少单个 worker 的 LLM 反思轮数。

### 5.3 LLM/API 限流与超时
- `vlm.py` 的 `get_response_from_vlm` 与 `treesearch/backend` 的 `query`（`backend/__init__.py:19`）都带 `@backoff.on_exception(backoff.expo, (RateLimitError, APITimeoutError))`。
- Semantic Scholar 无 key 时限流：`tools/semantic_scholar.py` 的 `SemanticScholarSearchTool.__init__` 会 `warnings.warn("No Semantic Scholar API key found ...")`；`search_for_papers`（`@backoff.on_exception(..., HTTPError, ConnectionError)`）指数退避。**修复**：`export S2_API_KEY=...`。
- `AgentManager` 各 `_get_response`/`query` 异常 → 回退（如 `_generate_substage_goal` `:620` 回退通用目标）；日志 `Error generating sub-stage goals`。

### 5.4 进程清理（避免僵尸/孤儿占资源）
`launch_scientist_bfts.py` 结尾（`:323-369`）：用 `psutil` 递归 `children(recursive=True)`，先 `SIGTERM`，`wait_procs(timeout=3)` 后仍存活的 `kill()`；再遍历 `process_iter(["name","cmdline"])` 杀含 `python/torch/mp/bfts/experiment` 关键字的进程，最后 `sys.exit(0)`。
**若中途 Ctrl-C 或崩溃**：这些清理代码可能不执行 → 残留 worker/解释器子进程。手动清理：
```bash
ps aux | grep -E 'runfile|experiment|bfts|python' | grep -v grep
pkill -f 'experiment_code.py'   # 谨慎：确认是当前实验的进程
nvidia-smi                      # 查显存残留
```

---

## 6. 其他高频坑（速查表）

| 现象 | 源码根因 | 处理 |
| --- | --- | --- |
| `ValueError: Key X not found in task_desc` | `AgentManager.__init__`（`:126`）要求 5 键；idea JSON 不全 | 补全 `Title/Abstract/Short Hypothesis/Experiments/Risk Factors and Limitations` |
| `find_pdf_path_for_review` 报 `UnboundLocalError: pdf_path` | `launch_scientist_bfts.py:140-166`：仅当存在 `reflection` PDF 时才给 `pdf_path` 赋值，否则未定义 | 保证写作产出 `*_reflection_*.pdf`，或在函数开头 `pdf_path = None` 兜底并让调用方处理 |
| `anthropic` 后端 `NotImplementedError` | `treesearch/backend/backend_anthropic.py` 不支持 function calling | 结构化决策（节点选择/阶段判定/指标解析）必须用 OpenAI 系模型（配 `cfg.agent.feedback/select_node` 等） |
| 想法生成报 `Invalid action` | `perform_ideation_temp_free.py` 的 ACTION 不在 `SearchSemanticScholar/FinalizeIdea` | 确认 system prompt 工具名与输出格式；换更强模型 |
| `No .npy files found in working directory` | LLM 代码没按 guideline 存 `experiment_data.npy`（`parallel_agent.py:1512`） | 检查 `experiment_code.py`；在 `_prompt_impl_guideline` 强化保存要求 |
| `WorstMetricValue` 大量出现、best 为空 | 指标解析失败（无有效 metrics / 解析代码异常，`parallel_agent.py:1630-1667`） | 看 `parse_term_out`/`parse_exc_info`；确认代码 print 了明确指标名与数值 |
| 写作 `--writeup-type icbinb` 超页 | `check_page_limit` 返回超页 + reflection 未收敛 | 看 `get_reflection_page_info` 注入的提示；增大 `n_writeup_reflections` 或让模型更激进移图 |
| `tree_plot.html` 空白/无节点 | `cfg_to_tree_struct` 异常被 `generate` 吞掉只打印（`tree_export.py:379-381`）；或 CDN 不可达 | 看控制台 `Error in cfg_to_tree_struct`；离线环境 vendored p5/highlight.js |
| 统一 viz 缺阶段 tab | `get_completed_stages` 依赖 `tree_data.json/tree_plot.html/journal.json` 存在 | 确认各阶段 `save_run` 成功；`unified_tree_viz.html` 生成失败只打印（`tree_export.py:408-411`） |

---

## 7. 一键自检

```bash
# 环境与依赖
python -c "import torch, psutil, pymupdf, igraph, omegaconf, openai; print(torch.cuda.device_count())"
nvidia-smi | head

# 配置校验（关键池）
grep -nE 'num_workers|max_debug_depth|debug_prob|num_drafts|stage[1-4]_max_iters|timeout|num_seeds' bfts_config.yaml

# 阶段产物完整性
RUN=experiments/<date>_<Name>_attempt_0
ls $RUN/logs/0-run/ | head -30
python - <<PY
import json, glob
for j in glob.glob("$RUN/logs/0-run/stage_*/journal.json"):
    d = json.load(open(j)); ns = d["nodes"]
    print(j, "nodes=", len(ns),
          "buggy=", sum(1 for n in ns if n.get("is_buggy")),
          "good=", sum(1 for n in ns if n.get("is_buggy") is False and n.get("is_buggy_plots") is False))
PY

# 定义行号复核
grep -rn '^def \|^    def \|^class ' ai_scientist/treesearch ai_scientist/*.py | grep -E 'select_parallel_nodes|_check_stage_completion|_analyze_plots_with_vlm|get_best_node|def run|def step|def parse_exec_result'
```

更多树搜索 internals 见 [`treesearch_deep_dive.md`](treesearch_deep_dive.md)；Prompt/JSON 契约见 [`prompts_and_vlm_schemas.md`](prompts_and_vlm_schemas.md)。
