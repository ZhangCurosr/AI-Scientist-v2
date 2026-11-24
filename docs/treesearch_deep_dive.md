# The AI Scientist-v2 — 树搜索机制超深度拆解

> 基于当前仓库真实源码逐行核验；行号以 `文件:行号` 标注（`grep -n '^def \|^    def \|^class ' ai_scientist/treesearch/*.py` 可复核）。
> 目标：让零上下文 AI 能直接改树搜索策略、加阶段、修卡死、读懂可视化。
> 配套：[`architecture.md`](architecture.md)（分层总览）、 [`pipeline_and_dataflow.md`](pipeline_and_dataflow.md)（阶段数据流）、 [`modules_reference.md`](modules_reference.md)（签名表）、 [`troubleshooting.md`](troubleshooting.md)。

---

## 0. 全局心智模型（先看这张调用链）

```
launch_scientist_bfts.__main__
  └─ perform_experiments_bfts(config_path)                 # perform_experiments_bfts_with_agentmanager.py:58
       ├─ load_cfg / load_task_desc / prep_agent_workspace # utils/config.py
       ├─ AgentManager(task_desc, cfg, workspace_dir)      # agent_manager.py:123
       │    └─ _create_initial_stage() → Stage + Journal   # :200
       ├─ exec_callback / step_callback(...)               # step_callback → save_run → tree_export.generate
       └─ manager.run(exec_callback, step_callback)        # agent_manager.py:692
            ├─ _create_agent_for_stage(stage) → ParallelAgent  # :274
            └─ while 每步:
                 ParallelAgent.step(exec_callback)          # parallel_agent.py:2053
                   ├─ _select_parallel_nodes()              # :1931  best-first 选点
                   ├─ executor.submit(_process_node_wrapper) # :1410  多进程 worker
                   │    └─ MinimalAgent._draft/_debug/_improve/...
                   │       → Interpreter.run(code)         # 子进程执行
                   │       → LLM 解析 metric → MetricValue / WorstMetricValue
                   │       → 绘图代码 → VLM 分析 → node.to_dict()
                   ├─ Node.from_dict(result, journal) → journal.append()
                   └─ journal.generate_summary()            # 记忆
                 _check_stage_completion(stage)             # agent_manager.py:410  主阶段判定
                 _check_substage_completion(...)            # :343  子阶段判定
                 达标 → _run_multi_seed_evaluation + _run_plot_aggregation → 阶段切换
```

**三权分立**：`AgentManager` 管"阶段"、`ParallelAgent` 管"一步内并发展开哪些节点"、`MinimalAgent` 管"生成一个节点的代码"。`Journal` 是共享的解树数据库，`Interpreter` 是代码执行沙箱。

---

## 1. `Node` 数据类逐属性剖析（`treesearch/journal.py:44-311`）

`Node` 是 `@dataclass(eq=False)` + `DataClassJsonMixin`，表示解树中的一个候选解（代码 + 执行结果 + 评估）。**所有字段都是 `kw_only=True`**（必须关键字传参）。按源码分组：

### 1.1 代码与计划（`:47-51`）
| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `plan: str` | `""` | 本节点实验设计的自然语言草稿（LLM 在 `plan_and_code_query` 中生成） |
| `overall_plan: str` | `""` | 整体研究计划（供 log summarization / report 使用） |
| `code: str` | `""` | 主实验代码（单文件、可被 `Interpreter.run` 执行） |
| `plot_code: str` | `None` | 绘图代码（执行 `experiment_data.npy` → `*.png`） |
| `plot_plan: str` | `None` | 绘图计划 |

### 1.2 通用/拓扑（`:54-61`）
| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `step: int` | `None` | 在 Journal 中的序号，由 `Journal.append` 设为 `len(nodes)` |
| `id: str` | `uuid.uuid4().hex` | 全局唯一 32 位十六进制 id（父子关系靠 id 序列化） |
| `ctime: float` | `time.time()` | 创建时间戳 |
| `parent: Optional[Node]` | `None` | 父节点；`__post_init__` 自动把 self 加入 `parent.children` |
| `children: set[Node]` | `set()` | 子节点集合；`__deepcopy__` 特意清空防止循环引用 |
| `exp_results_dir: str` | `None` | 该节点代码/数据/图落盘目录（worker 中生成，见 §4.3） |

### 1.3 执行信息（`:64-68`）
`_term_out: list[str]`（终端输出原始行）、`exec_time: float`、`exc_type/exc_info/exc_stack`（异常类型/信息/栈）。`absorb_exec_result()`（`:170`）把 `ExecutionResult` 5 字段吸收进来。`term_out` 属性（`:185`）用 `trim_long_string("".join(_term_out))` 截断后返回。

### 1.4 指标解析信息（`:71-77`）
`parse_metrics_plan`、`parse_metrics_code`（LLM 生成的"从 `experiment_data.npy` 解析指标"代码）、`parse_term_out`、`parse_exc_type/info/stack`（解析代码自身执行的输出/异常）。这套字段使"指标解析"也是一段**可被 LLM 修复的代码**。

### 1.5 绘图执行信息（`:80-84`）
`plot_term_out`、`plot_exec_time`、`plot_exc_type/info/stack`；由 `absorb_plot_exec_result()`（`:178`）吸收。

### 1.6 评估（`:87-91`）
| 字段 | 含义 |
| --- | --- |
| `analysis: str` | LLM 对执行输出的分析摘要（`review_func_spec` 的 `summary`） |
| `metric: MetricValue` | 指标对象；无有效指标时为 `WorstMetricValue()` |
| `is_buggy: bool` | `response["is_bug"] or node.exc_type is not None`（`parallel_agent.py:699`）；**buggy 节点被选点/剪枝淘汰** |
| `is_buggy_plots: bool` | VLM 判定图表是否无效（`vlm_feedback_spec.valid_plots_received` 取反，`:1007-1010`） |

> `Journal.good_nodes`（`:386`）明确要求 `is_buggy is False and is_buggy_plots is False`——**两者都为 False 才算"好节点"**。

### 1.7 绘图产物与 VLM 反馈（`:94-102`）
- `plot_data: dict`（默认 `{}`）、`plots_generated: bool`（默认 False）。
- `plots: List[str]`：相对可视化路径（`../../logs/<run>/experiment_results/...`）。
- `plot_paths: List[str]`：绝对路径，供程序化访问。
- `plot_analyses: List[str]`（实际元素是 dict，每个含 `analysis` 与 `plot_path`）、`vlm_feedback_summary: List[str]`、`datasets_successfully_tested: List[str]`。
- `exec_time_feedback: str`（`:105`）：阶段 3 运行太短时由 `AgentManager` 写入（`agent_manager.py:492-499`），要求放大实验规模。

### 1.8 特殊用途（`:108-114`）
`ablation_name`（消融名）、`hyperparam_name`（调参名）、`is_seed_node`（多种子评估节点）、`is_seed_agg_node`（多种子聚合节点）。

### 1.9 关键属性与方法
| 成员 | 行号 | 逻辑 |
| --- | --- | --- |
| `__post_init__` | `:117-124` | children 若是 list 转 set；parent 非 None 且非 str 时 `parent.children.add(self)` |
| `__deepcopy__` | `:126-141` | 复制所有属性但**跳过 parent/children**；result.parent=原 parent，result.children=空集 → 断开循环引用（供跨阶段继承） |
| `__eq__/__hash__` | `:193-198` | 按 `id` 判等/哈希 → Node 可作 set 元素（children） |
| `stage_name` | `:159-167` | `parent is None`→`"draft"`；`parent.is_buggy`→`"debug"`；否则 `"improve"` |
| `absorb_exec_result` / `absorb_plot_exec_result` | `:170/:178` | 吸收执行结果 |
| `is_leaf` | `:192` | `not self.children` |
| `debug_depth` | `:203-211` | 非 debug→0；否则 `parent.debug_depth + 1`（连续 debug 层数，剪枝用） |
| `to_dict()` | `:214-277` | 序列化为 dict：parent/children 换成 `parent_id`/`children`(id 列表)；路径转相对 `os.getcwd()`；metric 拆成 value/maximize/name/description；VLM 分析的 plot_path 也转相对 |
| `from_dict(data, journal=None)` | `:280-310` | 反向：metric dict → `MetricValue`；若传 `journal`，用 `parent_id` 查 parent 并 `parent.children.add(node)` 恢复拓扑 |

**序列化是跨进程核心**：worker 返回 `child_node.to_dict()`，主进程 `Node.from_dict(result_data, self.journal)` 恢复父子关系（`parallel_agent.py:2164`）。`_safe_pickle_test`（`:31`）在 submit 前验证可 pickle。

---

## 2. `Journal` 数据结构（`journal.py:362-612`）

`Journal` 是"每阶段一个"的有序节点集合（`@dataclass`，唯一字段 `nodes: list[Node]`）。

| 成员 | 行号 | 说明 |
| --- | --- | --- |
| `__getitem__/__len__` | `:365-370` | 支持 `journal[i]`、`len(journal)` |
| `append(node)` | `:372-375` | `node.step = len(nodes)` 后追加 |
| `draft_nodes` | `:378-381` | `parent is None` 的节点（树根） |
| `buggy_nodes` | `:384-386` | `is_buggy is True` 的节点 |
| `good_nodes` | `:389-408` | `is_buggy is False and is_buggy_plots is False`；同时打印所有节点的 `(step,parent.step,id,buggy,buggy_plots)` 供调试 |
| `get_node_by_id` | `:410-414` | 线性查找 |
| `get_metric_history` | `:416-418` | 所有节点的 metric 列表 |
| `get_best_node(only_good=True, use_val_metric_only=False, cfg=None)` | `:420-502` | **best-first 核心**，见下 |
| `generate_summary(include_code=False, **model_kwargs)` | `:504-535` | LLM 总结成功/失败实验 → worker 的 `memory_summary` |
| `generate_summary_old` | `:537-546` | 旧版纯字符串拼接 |
| `to_dict()` | `:561` | `{nodes:[n.to_dict()...]}` |
| `save_experiment_notes(workspace_dir, stage_name, cfg)` | `:565-612` | 写 `experiment_notes/<stage>_node_<id>_summary.json` 与 `<stage>_summary.txt` |

### 2.1 `get_best_node()` 决策逻辑（`:420-502`）逐行
1. `only_good=True`（默认）→ 候选 `self.good_nodes`，空则返回 None；否则用全部 `nodes`。
2. `use_val_metric_only=True` → `max(nodes, key=lambda n: n.metric)`（纯指标比较，靠 `MetricValue.__gt__`）。
3. 候选只有 1 个 → 直接返回。
4. 构造 prompt（dict）：`Introduction`（"experienced AI researcher evaluating different implementations"）、`Task`（"Avoid relying too heavily on the validation loss alone ... consider all of them"）、`Candidates`（逐节点 `ID/Metric/Training Analysis/VLM Feedback`，跳过 seed 节点）。
5. `model = cfg.agent.select_node.model` 若配置，否则默认 `"gpt-4o"`（`temperature=0.3`）；`query(..., func_spec=node_selection_spec)` → `{"selected_id","reasoning"}`。
6. 找到 id 匹配节点返回；任何异常 → 回退 `max(nodes, key=lambda n: n.metric)`（`logger.error`）。

> `node_selection_spec`（`journal.py:21-41`）是 `FunctionSpec`，强制 LLM 用 function call 返回结构化选择。这意味着**节点选择需要支持 function calling 的模型**（OpenAI 系，见 [`troubleshooting.md`](troubleshooting.md)）。

### 2.2 `generate_summary()`（`:504-535`）
- 成功实验：拼接 `Design: node.plan / Results: node.analysis / Metric: str(node.metric)`（含 code 可选）。
- 失败实验：`Design / Error Analysis / Error Type: node.exc_type / Debug Depth: node.debug_depth`。
- `query(system_message=prompt, user_message="...success patterns / failure patterns / recommendations...", model=kwargs.get("model","gpt-4o"), temp=...)`。
- 该摘要经 `step()` 传给每个 worker（`parallel_agent.py:2061-2066`），实现"跨步记忆"。

---

## 3. `AgentManager` 与 `ParallelAgent` 方法调用链

### 3.1 `AgentManager` 数据类与字段（`agent_manager.py`）

- `Stage(name, description, goals, max_iterations, num_drafts, stage_number)`（`:104-111`）。
- `StageTransition(from_stage, to_stage, reason, config_adjustments)`（`:114-121`）。
- `AgentManager.__init__(task_desc, cfg, workspace_dir)`（`:123-169`）：
  - `self.task_desc = json.loads(task_desc)`；强制 5 键 `Title/Abstract/Short Hypothesis/Experiments/Risk Factors and Limitations`，缺则 `ValueError`。
  - 关键属性：`stages: list[Stage]`、`journals: dict[str, Journal]`（**每阶段一个 Journal**）、`current_stage`、`stage_history: list[StageTransition]`、`completed_stages`、`main_stage_dict`（`:143`）、`main_stage_goals`（`:149`）。
  - `_create_initial_stage()`（`:200`）建 `1_initial_implementation_1_preliminary`，`max_iterations = _get_max_iterations(1)`，`num_drafts = cfg.agent.search.num_drafts`。

`main_stage_dict`（`:143-148`）：`1 initial_implementation / 2 baseline_tuning / 3 creative_research / 4 ablation_studies`。
`main_stage_goals`（`:149-168`）：阶段 1 "simple working implementation"；阶段 2 "change hyperparameters, DO NOT change architecture, TWO more HF datasets"；阶段 3 "novel improvements, THREE HF datasets in total"；阶段 4 "systematic component analysis, same datasets"。

### 3.2 `AgentManager.run()` 主循环（`:692-829`）逐步
```
while self.current_stage:                              # 主阶段循环
    current_substage = self.current_stage
    while current_substage:                            # 子阶段循环
        with self._create_agent_for_stage(current_substage) as agent:   # :274
            if self.stage_history:                     # 继承上一阶段最佳
                prev_best = self._get_best_implementation(prev_stage)
                if prev_best: self.journals[cur].append(prev_best)
                else: 结束实验（current_stage=None; break）
            while True:
                agent.step(exec_callback)              # 走一步（并发展展节点）
                step_callback(...)                     # 写 notes/stage_progress + save_run
                main_done, fb = self._check_stage_completion(current_substage)  # :410
                if main_done:
                    best = self._get_best_implementation(cur.name)
                    seed_nodes = agent._run_multi_seed_evaluation(best)      # :1261
                    agent._run_plot_aggregation(best, seed_nodes)            # :1332
                    current_substage = None; break
                sub_done, fb = self._check_substage_completion(...)           # :343
                if sub_done:
                    next_substage = self._create_next_substage(...)           # :638
                    self.stage_history.append(StageTransition(...))
                    self.stages.append(next); self.journals[next.name]=Journal()
                    current_substage = next; break
    self._save_checkpoint()                             # :249 → checkpoint.pkl
    if self.current_stage:
        next_main = self._create_next_main_stage(...)    # :664
        if next_main: 记录 transition、建新 Journal、current_stage=next_main
        else: current_stage = None
```

### 3.3 `_create_agent_for_stage()`（`:274-329`）——跨阶段"继承起点"
- 复制 cfg（`stage_cfg = self.cfg.copy()`），设 `stage_cfg.agent.search.num_drafts = stage.num_drafts`。
- `task_desc = self._curate_task_desc(stage)`（`:216`）：阶段 3 追加 `Experiments` 计划，阶段 4 追加 `Risk Factors and Limitations`；再追加 `Current Main Stage/Sub-stage/Sub-stage goals`。
- 按主阶段号解析出上一阶段 best 节点：阶段 2 用 `best_stage1_node`、阶段 3 用 `best_stage2_node`、阶段 4 用 `best_stage3_node`（找不到对应子阶段 → `ValueError`）。
- 返回 `ParallelAgent(task_desc, stage_cfg, journal=self.journals[stage.name], stage_name, best_stage1/2/3_node=...)`。

`_get_best_implementation(stage_name)`（`:538-551`）：`get_best_node(cfg)` → `copy.deepcopy` → 手动 `parent=None, children=set()`（配合 Node.__deepcopy__ 断开引用）→ 作为下一阶段 journal 的起点节点。

### 3.4 `ParallelAgent.__init__`（`parallel_agent.py:1142-1192`）
- 存 `best_stage1/2/3_node`（注释分别标注用于"阶段 2 调参起点/阶段 3 绘图起点/阶段 4 消融起点"）。
- `num_workers = cfg.agent.num_workers`；`num_gpus = get_gpu_count()`（`:1120`，用 `nvidia-smi` 或 `CUDA_VISIBLE_DEVICES`）；有 GPU 时 `num_workers = min(num_workers, num_gpus)`。
- `self.gpu_manager = GPUManager(num_gpus) if num_gpus>0 else None`；`self.executor = ProcessPoolExecutor(max_workers=num_workers)`。
- `self.evaluation_metrics = self._define_global_metrics()`（`:1194`，LLM 提一个全局指标，注入所有 worker）；`_ablation_state`/`_hyperparam_tuning_state` 记录已试方案名。

`GPUManager`（`:1091-1118`）：`available_gpus=set(range(num_gpus))`，`acquire_gpu(process_id)` 取 `min(available_gpus)` 并移除，`release_gpu` 归还；无可用 GPU 抛 `RuntimeError("No GPUs available")`（step/worker 捕获后走 CPU）。

### 3.5 `ParallelAgent.step(exec_callback)`（`:2053-2190`）逐步
1. `nodes_to_process = self._select_parallel_nodes()`（`:1931`）——选本轮要处理的 N 个节点。
2. 每个节点 `node.to_dict()` + `_safe_pickle_test`（pickle 预检），`None` 表示新草稿。
3. `memory_summary = journal.generate_summary(include_code=False, ...)`（有 `cfg.agent.summary` 时用其 model/temp）。
4. 对每个 node_data：`gpu_id = gpu_manager.acquire_gpu(f"worker_{len(futures)}")`；阶段 2/4 且节点非 buggy 时在主进程生成 `new_hyperparam_idea`/`new_ablation_idea`（避免并行重复）；取出 `best_stage1/2/3_node.plot_code`；`executor.submit(_process_node_wrapper, node_data, task_desc, cfg, gpu_id, memory_summary, evaluation_metrics, stage_name, new_ablation_idea, new_hyperparam_idea, best_*_plot_code, seed_eval=False)`。
5. 收结果：`result_data = future.result(timeout=self.timeout)`；`Node.from_dict(result_data, self.journal)` 恢复父子；`_update_hyperparam_tuning_state`/`_update_ablation_state`；`journal.append(result_node)`。异常：`TimeoutError` 打印；其它异常 `traceback + raise`；`finally` 释放 GPU。

> **阶段 2/4 的 idea 生成在主进程**（见 `_select_parallel_nodes` 注释 `:1934-1940`），阶段 1/3 在 worker 进程——这是"避免并行做重复调参/消融"的设计。

---

## 4. worker 内单节点处理：`_process_node_wrapper()`（`:1410-1797`）

这是真正"跑一次实验"的函数，运行在进程池子进程中。逐步：

### 4.1 环境准备（`:1424-1457`）
- `process_id = multiprocessing.current_process().name`；workspace = `<cfg.workspace_dir>/process_<name>`，其下建 `working/`。
- `CUDA_VISIBLE_DEVICES = str(gpu_id)`（有 GPU）或 `""`（CPU，`:1437-1442`）——**这是"显存溢出自动降级/CPU 回退"的落点**。
- 建 `MinimalAgent(task_desc, cfg, memory_summary, evaluation_metrics, stage_name)` 与 `Interpreter(working_dir=workspace, timeout=cfg.exec.timeout, ...)`。

### 4.2 生成 child 节点（`:1459-1499`）
`Node.from_dict(node_data, journal=None)` 重建 parent，然后按状态分流：
| 条件 | 生成的 child |
| --- | --- |
| `seed_eval=True` | `_generate_seed_node(parent)`（复用代码，`is_seed_node=True`） |
| `parent is None` | `_draft()` |
| `parent.is_buggy` | `_debug(parent)` |
| `new_hyperparam_idea` 且无 ablation（阶段 2） | `_generate_hyperparam_tuning_node(parent, idea)` |
| `new_ablation_idea` 且无 hyperparam（阶段 4） | `_generate_ablation_node(parent, idea)` |
| 否则 | `_improve(parent)` |

### 4.3 执行 + 指标解析（`:1501-1667`）
1. `exec_result = process_interpreter.run(child_node.code, True)` → `cleanup_session()`；`parse_exec_result` 吸收结果并让 LLM（`review_func_spec`）判 `is_bug`，`node.is_buggy = response["is_bug"] or exc_type is not None`。
2. 检查 `working/` 下有无 `.npy`：无 → `logger.warning("No .npy files found ...")`；有 → 非 seed 时让 LLM 写 `parse_metrics_code`（persist 到 `child_node.parse_metrics_code`），seed 时复用 parent 的解析代码。
3. 执行解析代码；成功则 `metrics_prompt` → `query(func_spec=metric_parse_spec)` → 若 `valid_metrics_received=True`：`child_node.metric = MetricValue(value={"metric_names": metrics_response["metric_names"]})`；否则 `WorstMetricValue()` + `is_buggy=True`。解析代码本身异常也走 `WorstMetricValue()`。

### 4.4 绘图 + VLM（`:1670-1790`）
- 非 buggy 才绘图：`_generate_plotting_code(child_node, working_dir, plot_code_from_prev_stage)`（阶段 3/4 参考上一阶段 plot_code），执行；`plot_exc_type` 非空且 `retry_count<3` → 重试（最多 3 次）。
- 成功图：存到 `<logs>/<run>/experiment_results/experiment_<id>_proc_<pid>/`（`plotting_code.py`/`experiment_code.py`/`*.npy`/`*.png`）；`child_node.plots` 存 web 相对路径、`plot_paths` 存绝对路径。
- 有图则 `_analyze_plots_with_vlm(child_node)`（`:894-1033`）：>10 张图先用 `plot_selection_spec` 让 LLM 选 10 张；每张图 `base64.b64encode` 后作为 `data:image/jpeg;base64,...` 的 `image_url` 传给 `query(func_spec=vlm_feedback_spec, model=cfg.agent.vlm_feedback.model)`；结果写 `plot_analyses`/`vlm_feedback_summary`/`datasets_successfully_tested`，`valid_plots_received` 取反写 `is_buggy_plots`。
- 返回 `child_node.to_dict()`。异常 `traceback.print_exc(); raise`。

### 4.5 `MinimalAgent` 的关键 prompt 方法（`:254-1035`）
| 方法 | 行号 | prompt 关键点（真实文本要点） |
| --- | --- | --- |
| `_prompt_environment` | `:274` | 列出 numpy/pandas/sklearn/... 包（`random.shuffle` 后拼串），"all packages are already installed" |
| `_prompt_impl_guideline` | `:298` | CRITICAL GPU 要求（`device = torch.device('cuda' if ...)`、`.to(device)`）；代码必须以 `working_dir = os.path.join(os.getcwd(),'working')` 开头；不得放 `if __name__ == "__main__"`；必须 `np.save(..., 'experiment_data.npy')`；时间限制 `humanize.naturaldelta(cfg.exec.timeout)` |
| `_prompt_resp_fmt` | `:397` | "7-10 sentences + single ```python code block" |
| `_draft` | `:453` | "write a python code to implement a solid baseline ... from data preparation to model training, as well as evaluation and visualization" |
| `_debug` | `:494` | 提供 `Previous (buggy) implementation` + `Execution output` + `vlm_feedback_summary` + `exec_time_feedback` |
| `_improve` | `:523` | "improve it based on the current experimental stage" |
| `_generate_hyperparam_tuning_node` | `:557` | 数据字典键改为 `hyperparam_tuning_type_*`，写 `hyperparam_name` |
| `_generate_ablation_node` | `:605` | 数据字典键改为 `ablation_type_*`，写 `ablation_name` |
| `plan_and_code_query` | `:658` | `query(model=cfg.agent.code.model)` → `extract_code` + `extract_text_up_to_code`，失败重试 3 次并加 `Parsing Feedback` |
| `_determine_datasets_successfully_tested` | `:835` | `REASONING:`/`SUCCESSFULLY_TESTED_DATASETS:` 两行格式，重试 5 次 |

### 4.6 多种子与聚合图（`ParallelAgent`）
- `_run_multi_seed_evaluation(node)`（`:1261`）：对 `range(cfg.agent.multi_seed_eval.num_seeds)` 每个 seed，把 `node_data["code"]` 前面注入 `random.seed/np.random.seed/torch.manual_seed(seed)`，submit 到进程池（`seed_eval=True`），结果 `Node.from_dict(..., self.journal)` 后 append。
- `_run_plot_aggregation(node, seed_nodes)`（`:1332`）：生成 `is_seed_agg_node=True` 的聚合节点，供写作准备跨 seed 统计。

---

## 5. 四个 Stage 的状态转移条件判定（代码逻辑）

每条判定都在 `agent_manager.py`，由 `_check_stage_completion(stage)`（`:410-536`）与 `_check_substage_completion`（`:343-408`）实现；转移由 `run()` 调用 `_create_next_substage`（`:638`）/`_create_next_main_stage`（`:664`）完成。

### 5.1 阶段 1 `initial_implementation`（`stage.stage_number == 1`，`:431-440`）
- 先判 `len(journal.nodes) >= stage.max_iterations`（`:412`）：若达到且**仍无可运行实现** → `logger.error(...)`，`self.current_stage = None`（**整个实验提前结束**），返回 `(True, "Failed to find working implementation")`。
- 否则 `if len(journal.good_nodes) > 0` → `(True, "Found working implementation")`。
- **转移条件一句话**：出现至少一个 `is_buggy=False 且 is_buggy_plots=False` 的节点即进入阶段 2；若跑满 max_iters 都没有 → 终止实验。

### 5.2 阶段 2 `baseline_tuning`（`:442-490`）
- `best_node = journal.get_best_node(cfg)`；无 → `(False,...)`。
- `if best_node == journal.nodes[0]` → `(False, "No improvement found from the base node ...")`（`journal.nodes[0]` 是从阶段 1 继承来的起点，见 §3.3）。
- 否则构造 eval_prompt：`vlm_feedback`（`_parse_vlm_feedback`，取 `plot_analyses[0]['analysis']` + `vlm_feedback_summary`）+ `datasets_successfully_tested`，要求 ①训练曲线稳定收敛 ②至少 2 个数据集 ③图无重大不稳定 → `query(func_spec=stage_completion_eval_spec, model=cfg.agent.feedback.model)`。
- `evaluation["is_complete"]` → `(True, ...)`；否则 `(False, "Missing criteria: "+missing)`。
- **转移条件**：best 节点优于继承起点 且 VLM+数据集判定达标。

### 5.3 阶段 3 `creative_research`（`:492-533`）
- `best_node` 无 / 等于 `journal.nodes[0]` → 未完成（同上）。
- `exec_time_minutes = best_node.exec_time / 60`；若 `len(nodes) > cfg.agent.stages.stage3_max_iters / 2`（过半）且 `exec_time_minutes < cfg.exec.timeout / 60 / 2`（运行太短）：
  - 写 `exec_time_feedback` 到 `journal.nodes[-1]`："Implementation works but runs too quickly ... scale up the experiment by increasing epochs / larger model / bigger datasets"
  - 返回 `(False, exec_time_feedback)` → 继续迭代。
- **注意源码细节**：`exec_time_feedback` 字符串是拼接的 f-string + 普通字符串，其中一段 `"Given that the current execution time is {exec_time_minutes:.2f} minutes..."` **没有 f 前缀**，会字面输出 `{exec_time_minutes:.2f}`——这是源码现状，二次开发可修（见 [`troubleshooting.md`](troubleshooting.md)）。
- 否则落到末尾 `return False, "stage not completed"`，靠 `len(nodes)>=max_iters` 强制完成。
- **转移条件**：best 优于起点 且（过半后运行时间足够长）或跑满 max_iterations。

### 5.4 阶段 4 `ablation_studies`（`:535-536`）
- `if stage.stage_number == 4: pass` → 直接落到末尾 `return False, "stage not completed"`；**只能靠 `len(nodes) >= max_iterations` 强制完成**（`:412`），完成后 `_create_next_main_stage` 因 `main_stage_num==4` 返回 None → 实验结束。

### 5.5 子阶段完成 `_check_substage_completion()`（`:343-408`）
- `best_node = journal.get_best_node(cfg)`；无 → `False`。
- eval_prompt 含 `vlm_feedback` + `Requirements: current_substage.goals` → `query(func_spec=stage_completion_eval_spec, model=cfg.agent.feedback.model)` → `is_complete` 决定。
- ⚠ 源码中 `if len(journal.nodes) >= current_substage.max_iterations` 的"max iterations 强制完成"分支位于 `try/except` **之后且 `return` 之后**（`:399-407`），实际不可达——**子阶段只能由 LLM 判 `is_complete` 结束**（这是源码现状，调试卡死时需注意，见 [`troubleshooting.md`](troubleshooting.md)）。

### 5.6 阶段/子阶段切换与记录
- `_create_next_substage`（`:638-661`）：`name = f"{main}_{name}_{sub+1}_{llm起的名}"`，`num_drafts=0`，`stage_number+1`；goals 由 `_generate_substage_goal`（`:552`，`_gather_stage_metrics/_identify_issues/_analyze_progress` + `substage_goal_spec` function call）生成。
- `_create_next_main_stage`（`:664-690`）：阶段 4 返回 None；否则 `name=f"{main+1}_{next_name}_1_first_attempt"`，`num_drafts=0`，goals=`main_stage_goals[main+1]`。
- 每次切换 `stage_history.append(StageTransition(from_stage, to_stage, reason, config_adjustments={}))`；主阶段结束 `_save_checkpoint()`（`:249`）写 `checkpoint.pkl`。

### 5.7 状态转移速查表
| 阶段 | 完成判定（源码行） | 未完成时的反馈 |
| --- | --- | --- |
| 1 | `len(good_nodes)>0`（`:433`）；或跑满 max_iters 仍无 → 终止（`:412-426`） | 继续 draft/debug/improve |
| 2 | best≠nodes[0] 且 LLM+VLM 判达标（`:442-490`） | "No improvement found..." / "Missing criteria: ..." |
| 3 | best≠nodes[0] 且（过半后 exec_time 足够）（`:492-533`）；或跑满 | `exec_time_feedback` 要求放大实验 |
| 4 | 仅 `len(nodes)>=max_iters`（`:412`） | "stage not completed" |
| 子阶段 | LLM `stage_completion_eval_spec.is_complete`（`:343-397`） | "Missing criteria: ..." |

---

## 6. `unified_tree_viz.html` 生成原理与 HTML 结构

### 6.1 生成链路（`treesearch/utils/tree_export.py`）
```
save_run(cfg, journal, stage_name)          # utils/config.py:219
  └─ tree_export.generate(cfg, journal, save_dir/"tree_plot.html")   # :376
       ├─ cfg_to_tree_struct(cfg, jou, out_path)  # :76  → tree_struct(dict)
       ├─ json.dump(tree_struct, <stage>/tree_data.json)             # :387
       ├─ html = generate_html(json.dumps(tree_struct))             # :362
       │    ├─ 读 viz_templates/template.js，把 '"PLACEHOLDER_TREE_DATA"' 替换为 JSON 字符串
       │    └─ 读 viz_templates/template.html，把 '<!-- placeholder -->' 替换为整段 JS
       │    → 写 <stage>/tree_plot.html
       └─ create_unified_viz(cfg, out_path)       # :414  → logs/<run>/unified_tree_viz.html
```
- `step_callback` 每步调 `save_run`（`perform_experiments_bfts_with_agentmanager.py:152`），所以 `tree_plot.html`/`tree_data.json` 随每步刷新。

### 6.2 `cfg_to_tree_struct()` 输出字段（`:76-360`）
- 边：`get_edges(journal)` = `(node.step, child.step)`（`:14`）；布局：`generate_layout` 用 `igraph.Graph(...).layout("rt")`（Reingold–Tilford，`:20`），再 `normalize_layout` 归一到 [0,1]（`:34`）。
- 每节点列表字段（长度=节点数，按 `journal` 顺序）：`plan`（`textwrap.fill(...,80)`）、`code`、`term_out`、`analysis`、`exc_type/info/stack`、`metrics`（新格式原样 / 旧格式包装成 `metric_names`）、`is_best_node`（`n is best_node`）、`plots`、`plot_paths`、`plot_analyses`、`vlm_feedback_summary`、`exec_time`、`exec_time_feedback`、`datasets_successfully_tested`、`plot_code`、`plot_plan`、`ablation_name`、`hyperparam_name`、`is_seed_node`、`is_seed_agg_node`、`parse_metrics_plan/code`、`parse_term_out`、`parse_exc_*`。
- 全局字段：`edges`、`layout`、`exp_name`、`completed_stages`（`get_completed_stages(log_dir)`，`:43`，扫描每阶段目录是否有 `tree_data.json`/`tree_plot.html`/`journal.json`）。

### 6.3 `create_unified_viz()`（`:414-484`）
- 位置：`log_dir = current_stage_viz_path.parent.parent`（即 `logs/<run>/`）；输出 `logs/<run>/unified_tree_viz.html`。
- 当前阶段名从路径 `stage_<数字>_...` 提取成 `Stage_<数字>`。
- 读 `template.html`/`template.js`；加载当前阶段 `tree_data.json` 作 `base_data`，附加 `current_stage` 与 `completed_stages`（加载失败则构造 `{current_stage, completed_stages, layout:[], edges:[]}` 兜底，防 JS 报错）。
- 同样做两次字符串替换（JS 的 PLACEHOLDER → base_data JSON；HTML 的 placeholder → JS），写文件。`try/except` 包裹：统一 viz 失败**不影响阶段 viz**（`:408-411`）。

### 6.4 `template.html` 结构（`viz_templates/template.html`，约 230 行）
- `<head>`：
  - Live Server 自动刷新脚本：本地 `127.0.0.1`/`localhost` 时每秒 `fetch(HEAD)` 比对 `last-modified`，变化则 `location.reload()`。
  - CDN：p5.js 1.9.0（树画布绘制）、highlight.js 11.9.0（代码高亮，含 python 语言包）。
  - `<!-- placeholder -->`：被替换为整段 JS（见 6.5）。
  - `<style>`：定义 `#canvas-container`（左侧 40vw）、`#text-container`（右侧 50vw，深色 #282c34）、`#plan/#plot_plan/#exc_info/#exec_time/#exec_time_feedback/#metrics/#vlm_feedback/.datasets_successfully_tested/.plots-container/.plot-item img/.metric-table`、以及**标签页样式** `.tabs-container/.tab(.active/.disabled)/.tab-content/.stage-info/.stage-status(.completed/.in-progress/.not-started)`。
- `<body>`：
  - `<div class="tabs-container" id="stage-tabs">`：4 个 `<button class="tab" data-stage="Stage_1..4" onclick="selectStage('Stage_N')">Stage N</button>`。
  - `<div id="canvas-container">`：p5 canvas 挂载点。
  - `<pre id="text-container">`：节点详情面板，含 `#stage-info`、`#plan`、`#exc_info`、`#exec_time`、`#exec_time_feedback`、`#metrics`、`#plot_plan`、`#plots`、`#vlm_feedback`、`#datasets_successfully_tested`、`<code id="code">`、`#plot_code`，各段以 `<hr>` 分隔。

### 6.5 `template.js` 关键机制（695 行）
- `const stageData = {Stage_1:null,...,Stage_4:null}`、`availableStages=[]`、`currentStage`、`currentSketch`（`:15-29`）。
- `class Node`（JS 侧，`:31`）：`x/y/id/visible/appearProgress/popEffect/selected/isRootNode`，`update()` 做出现/弹出动画，`render(p5)` 画节点。
- `selectStage(stageId)`（`:376`）：仅当 `stageData[stageId]` 非空且在 `availableStages` 中才切换，更新 `.tab.active` 并 `startSketch(stageId)`。
- `loadAllStageData(baseTreeData)`（`:394`）：`currentStageId = baseTreeData.current_stage || 'Stage_1'`；把 base 数据放进 `stageData[currentStageId]`；用 `baseTreeData.log_dir_path`（JS 从 `window.location.pathname` 推得，`:674`）fetch 其它阶段；`stageNames2actualNames` 把 `Stage_1..4` 映射到真实目录名 `stage_1_initial_implementation_1_preliminary`、`stage_2_baseline_tuning_1_first_attempt`、`stage_3_creative_research_1_first_attempt`、`stage_4_ablation_studies_1_first_attempt`，仅加载 `completed_stages` 列出的阶段；最后 `updateTabVisibility()` + `selectStage(availableStages[0])`。
- 节点点击（`:329-344`）：把 `treeData.code/plan/plot_code/plot_plan/metrics/exc_type/exc_info/exc_stack/plots/plot_analyses/vlm_feedback_summary/datasets_successfully_tested/exec_time_feedback/exec_time[nodeIndex]` 填进对应 DOM。
- 底部（`:672-694`）：`const treeStructData = "PLACEHOLDER_TREE_DATA"`（被 Python 替换）；补 `log_dir_path`/`current_stage`（从 URL 是否含 `stage_` 推断）；`loadAllStageData(treeStructData)` 启动。

> **二次开发提示**：改可视化字段要同时动 `cfg_to_tree_struct`（Python 侧产字段）与 `template.js` 的 `nodeIndex` 填充段（JS 侧消费字段）；两侧字段名必须逐字一致。网页依赖 CDN（p5/highlight.js），离线环境需把这些脚本 vendored 或替换为本地文件。

---

## 7. 二次开发锚点速查

| 想改什么 | 改哪里 |
| --- | --- |
| 节点选择策略/剪枝 | `ParallelAgent._select_parallel_nodes`（`:1931`）、`Journal.get_best_node`（`:420`）、`MetricValue.__gt__`（`utils/metric.py:171`） |
| debug 概率/最大深度 | `bfts_config.yaml` `agent.search.debug_prob / max_debug_depth`（默认 0.5 / 3） |
| 阶段数/每阶段迭代 | `bfts_config.yaml` `agent.stages.stage{1..4}_max_iters`、`agent.steps`；`AgentManager.main_stage_goals` |
| 完成判定规则 | `AgentManager._check_stage_completion`（`:410`）/_check_substage_completion（`:343`） |
| 代码生成 Prompt | `MinimalAgent._prompt_impl_guideline`（`:298`）等 `:274-451` 系列 |
| 指标 schema | `metric_parse_spec`（`parallel_agent.py:135`）、`MetricValue`（`utils/metric.py:114`） |
| VLM 图表反馈 | `vlm_feedback_spec`（`:107`）、`_analyze_plots_with_vlm`（`:894`） |
| 可视化字段 | `cfg_to_tree_struct`（`tree_export.py:76`）+ `template.js` nodeIndex 段 |
| 并行/GPU | `GPUManager`（`:1091`）、`get_gpu_count`（`:1120`）、`bfts_config.yaml agent.num_workers` |

算法层面的卡死/超时/OOM 排错见 [`troubleshooting.md`](troubleshooting.md)；Prompt 与 JSON schema 见 [`prompts_and_vlm_schemas.md`](prompts_and_vlm_schemas.md)。
