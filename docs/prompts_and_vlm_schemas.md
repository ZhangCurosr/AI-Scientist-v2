# The AI Scientist-v2 — Prompt 模板与 VLM/JSON 数据契约

> 基于当前仓库真实源码逐字核验；行号以 `文件:行号` 标注（`grep -n '^[a-zA-Z_]* = """\|^def ' ai_scientist/*.py` 可复核）。
> 本文给出**逐字 Prompt 原文**（占位符 `{...}` 与转义保留源码形态）与**真实 JSON 出参 schema**，供零上下文 AI 替换 Prompt、修解析、改 VLM 审稿。
> 配套：[`treesearch_deep_dive.md`](treesearch_deep_dive.md)（树搜索内部 Prompt/FunctionSpec）、 [`pipeline_and_dataflow.md`](pipeline_and_dataflow.md)、 [`troubleshooting.md`](troubleshooting.md)。

---

## 0. 两条 LLM 通道与解析方式

| 通道 | 模块 | 用于本文哪些 Prompt | 输出解析 |
| --- | --- | --- | --- |
| 文本 LLM（`llm.py`） | `get_response_from_llm`（`:267`，带 `@track_token_usage`） | 无模板构思、ICBINB 写作/反思 | 正则 ` ```json...``` ` → `extract_json_between_markers`（`:452`）；写作则正则 ` ```latex...``` ` |
| VLM（`vlm.py`） | `get_response_from_vlm`（`:132`） | 图像审稿、图描述 | 同上 `extract_json_between_markers`（`vlm.py:216`） |
| 树搜索 backend（`treesearch/backend`） | `query(..., func_spec=...)` | 见 `treesearch_deep_dive.md` | OpenAI function calling 结构化输出 |

**通用 JSON 解析契约**（`llm.py:452` / `vlm.py:216`）：优先正则 ```` ```json(.*?)``` ````；无匹配退化 `\{.*?\}`；`json.loads` 失败则去控制字符 `[\x00-\x1F\x7F]` 重试；全失败返回 `None`。**所以本文所有要求 JSON 的 Prompt 都显式写了 `REVIEW JSON:`/`IDEA JSON:` + ```json 围栏。**

---

## 1. 无模板构思：`perform_ideation_temp_free.py`

入口：`generate_temp_free_idea(idea_fname, client, model, workshop_description, max_num_generations=20, num_reflections=5, reload_ideas=True)`（`:128`）。CLI `--workshop-file` 默认 `ideas/i_cant_believe_its_not_better.md`，输出同名 `.json`。

### 1.1 工具定义（`:17-58`）
`tools` 列表 = 一个 `BaseTool` 实例 + 一个 dict 工具：
- `SemanticScholarSearchTool`（`tools/semantic_scholar.py`）：`name="SearchSemanticScholar"`，参数 `{"query": str}`，`use_tool(query)` → 格式化论文字符串。
- `FinalizeIdea`（dict 工具，`:20-33`）：description 规定 IDEA JSON 字段。

由二者生成 `tool_descriptions`（拼进 system prompt）与 `tool_names_str`（`"SearchSemanticScholar", "FinalizeIdea"`）。

### 1.2 `system_prompt`（`:61-97`，逐字摘录）
```
You are an experienced AI researcher who aims to propose high-impact research ideas resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions. Clearly clarify how the proposal distinguishes from the existing literature.

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at top ML conferences.

You have access to the following tools:

{tool_descriptions}

Respond in the following format:

ACTION:
<The action to take, exactly one of {tool_names_str}>

ARGUMENTS:
<If ACTION is "SearchSemanticScholar", provide the search query as {"query": "your search query"}. If ACTION is "FinalizeIdea", provide the idea details as {"idea": { ... }} with the IDEA JSON specified below.>

If you choose to finalize your idea, provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{
  "idea": {
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  }
}
```

Ensure the JSON is properly formatted for automatic parsing.

Note: You should perform at least one literature search before finalizing your idea to ensure it is well-informed by existing research.
```
> 注：`{tool_descriptions}`/`{tool_names_str}` 在源码中是 f-string 已展开；上表用占位符表示展开位置。`{{`/`}}` 在源码里是 f-string 转义，输出为单花括号。

### 1.3 `idea_generation_prompt`（`:99-106`，逐字）
```
{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
```

### 1.4 `idea_reflection_prompt`（`:111-126`，逐字）
```
Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_results}
```

### 1.5 调用循环与出参解析（`:128-249`）
```
for gen_idx in range(max_num_generations):
    prev_ideas_string = "\n\n".join(idea_str_archive)
    last_tool_results = ""; idea_finalized = False; msg_history = []
    for reflection_round in range(num_reflections):
        round0 → idea_generation_prompt.format(workshop_description, prev_ideas_string)
        round>0 → idea_reflection_prompt.format(current_round, num_reflections, last_tool_results or "No new results.")
        response_text, msg_history = get_response_from_llm(prompt, client, model, system_prompt, msg_history)
        # 正则解析：
        action_pattern    = r"ACTION:\s*(.*?)\s*ARGUMENTS:"
        arguments_pattern = r"ARGUMENTS:\s*(.*?)(?:$|\nTHOUGHT:|\n$)"
        if arguments_text.startswith("```json"): 去掉围栏
        if action in tools_dict:                 → tool.use_tool(**json.loads(arguments_text)) → last_tool_results
        elif action == "FinalizeIdea":           → idea = json.loads(arguments_text)["idea"]；追加 archive；break
        else: 打印 Invalid action
```
- `reload_ideas=True` 且 `.json` 存在 → 加载已有想法作为去重上下文（`:133-140`）。
- 解析失败（`Exception`）→ 打印响应 + `traceback` + `break`（跳出该想法的反思循环）。
- 最后 `ideas = [json.loads(s) for s in idea_str_archive]` 写回 `.json`（`:243-248`）。

### 1.6 IDEA JSON 出参契约（下游强依赖）
`FinalizeIdea` 产出的 idea 必须含 7 字段：`Name / Title / Short Hypothesis / Related Work / Abstract / Experiments / Risk Factors and Limitations`。**下游 `AgentManager.__init__`（`agent_manager.py:126-133`）强制校验其中 5 个**：`Title / Abstract / Short Hypothesis / Experiments / Risk Factors and Limitations`（缺任一 `ValueError`）；`Name` 用于目录命名，`Related Work` 用于报告。手动构造 ideas JSON 时 7 个字段都要给。`Experiments` 可为 list[str] / list[dict] / str（`AgentManager._curate_task_desc` 三种都处理，`:218-231`）。

---

## 2. VLM 视觉审稿：`perform_vlm_review.py`

入口主用：`perform_imgs_cap_ref_review(client, client_model, pdf_path) -> dict`（`:372`），返回 `{img_name: review}`，写 `review_img_cap_ref.json`（`launch_scientist_bfts.py` 调用处）。

### 2.1 图像抽取（`extract_figure_screenshots`，`:154-308`）
- 用 `pymupdf.open(pdf)`；逐页 `page.get_text("blocks")` 收集文本块（bbox + text）。
- 图注正则：`^(?:Figure)\s+(?P<fig_label>(?:\d+|[A-Za-z]+\.\d+|\(\s*[A-Za-z]+\s*\)\.\d+))(?:\.|:)`（忽略大小写），匹配 "Figure 1:"/"Figure A.1." 等。
- 在图注**上方**找 `len(text)>=min_text_length(50)`、垂直间距 `>=min_vertical_gap(30)`、水平重叠比 `>0.3` 的最大文本块作裁剪上边界，构造 `clip_rect`，`page.get_pixmap(clip=clip_rect, dpi=150)` 截图。
- 文件名：`figure_<label>escaped>_Page_<页码+1>_<md5[:10]>.png`，存 `<pdf名>_imgs/`。
- 正文引用正则：`(?:Fig(?:\.|-\s*ure)?|Figure)\s*<label>(?![0-9A-Za-z])`（负向预查避免 "Figure 11" 误匹配 "Figure 1"），扫描全文除图注块外的块。
- 返回每项：`{"img_name": "figure_<label>", "caption": <图注文本>, "images": [<png路径>], "main_text_figrefs": [<正文引用块>...]}`。

`extract_abstract(text)`（`:311-347`）：Markdown 文本中找 heading 含 "abstract"，收集到下一 heading 前的行。

`encode_image_to_base64(image_data)`（`:15-25`）：支持 str（路径）、list（取 `[0]`）、bytes，输出 `base64.b64encode(...).decode("utf-8")`。

### 2.2 图片如何传给 VLM（`vlm.py`）

`get_response_from_vlm(msg, image_paths, client, model, system_message, ...)`（`:132-193`）：
```python
content = [{"type": "text", "text": msg}]
for image_path in image_paths[:max_images(=25)]:
    base64_image = encode_image_to_base64(image_path)   # vlm.py:35-49：PIL 打开，RGBA→RGB，JPEG 存 bytes，再 b64
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": "low",                            # 降分辨率省 token
        },
    })
new_msg_history = msg_history + [{"role": "user", "content": content}]
response = make_vlm_call(client, model, temperature, system_message, prompt=new_msg_history)
content = response.choices[0].message.content
```
- `make_vlm_call`（`:96-118`）：`model.startswith("ollama/")` 去掉前缀；`"gpt" in model` 直接用；否则 `ValueError`。`AVAILABLE_VLMS`（`:13-32`）含 gpt-4o 三个版本、gpt-4o-mini、o3-mini、及若干 `ollama/*`（llama4/mistral-small3.2/qwen2.5vl 等）。
- `batch` 版 `get_batch_responses_from_vlm`（`:251`）同样用 `data:image/jpeg;base64,...`（`:290-296`）。
- **注意 JPEG 转换**：即使源图是 PNG，`vlm.py.encode_image_to_base64` 也会用 PIL 转 JPEG 再编码；`perform_vlm_review.encode_image_to_base64` 直接对文件 bytes 编码（用于 `detect_duplicate_figures` 的多图消息，`:389`），路径不同但 URL 前缀一致。

### 2.3 `img_cap_ref_review_prompt`（`:33-72`，逐字摘录）
```
The abstract of the paper is:

{abstract}

You will be given an image via the vision API. As a careful scientist reviewer, your task is to:
  1. Examine the provided image closely.
  2. Describe in detail what the image shows in a scientific manner.
  3. Critically analyze whether the image content aligns with the given caption:

{caption}

  4. We also have references in the main text that mention the figure:

{main_text_figrefs}

You should:
  - Examine the figure in detail: conclude elements in figures (e.g. name of axis) and describe what information is shown (e.g. the line of loss decreases monotonically but plateaus after X epochs)
  - Suggest any potential improvements or issues in the figure itself (e.g., missing legend, unclear labeling, no meaningful conclusion, mismatch with what the caption claims).
  - Critique the caption: does it accurately describe the figure? Is it too long/short? Does it include a concise takeaway?
  - Review how well the main text references (figrefs) explain the figure: are they missing? Do they adequately describe the figure's content, context, or purpose?

Finally, respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```
In <JSON>, provide the review in JSON format with the following fields in the order:
- "Img_description": "<Describe the figure's contents here>"
- "Img_review": "<Your analysis of the figure itself, including any suggestions for improvement>"
- "Caption_review": "<Your assessment of how well the caption matches the figure and any suggestions>"
- "Figrefs_review": "<Your thoughts on whether the main text references adequately describe or integrate the figure>"

In <THOUGHT>, first, thoroughly reason through your observations, analysis of alignment, and any suggested improvements. It is okay to be very long.
Then provide your final structured output in <JSON>.
Make sure the JSON is valid and properly formatted, as it will be parsed automatically.
```

### 2.4 `img_cap_selection_prompt`（`:74-124`）与 `img_review_prompt`（`:126-152`）
- `img_cap_selection_prompt`：在 2.3 基础上增加第 5 项 `{reflection_page_info}`（页数信息）；JSON 字段在前 4 个之外再要 `Overall_comments / Containing_sub_figures / Informative_review`，用于判断图是否移附录/合并/是否信息量低。
- `img_review_prompt`：只要 `Img_description / Img_review` 两字段，用于 ICBINB 写作为每张图生成描述（`desc_map`，`perform_icbinb_writeup.py:962-969`）。

### 2.5 VLM 审稿 JSON 契约（出参）
| 字段 | 来源 Prompt | 用途 |
| --- | --- | --- |
| `Img_description` | 全部 3 个 | 图内容描述（写作时用于准确配图文字） |
| `Img_review` | 全部 3 个 | 图本身问题/改进建议 |
| `Caption_review` | cap_ref / selection | 图注是否准确 |
| `Figrefs_review` | cap_ref / selection | 正文 figref 是否充分解释图 |
| `Overall_comments`, `Containing_sub_figures`, `Informative_review` | selection | 是否移附录/合并、是否信息量低 |

`perform_imgs_cap_ref_review`（`:372-386`）：对每张图调 `generate_vlm_img_cap_ref_review`（`:350`，`img` + `abstract` + model → `get_response_from_vlm` → `extract_json_between_markers`），结果按 `img["img_name"]` 存入 dict。`detect_duplicate_figures`（`:389-446`）另走 `client.chat.completions.create` 一次性发所有图，prompt 要求 `Duplicate figures: <list>` 或 `No duplicates found`；`perform_imgs_cap_ref_review_selection`（`:464`）配合 `reflection_page_info` 用于 ICBINB 页数反射。

---

## 3. ICBINB 4 页论文写作 Prompt 拆解（`perform_icbinb_writeup.py`）

入口：`perform_writeup(base_folder, citations_text=None, no_writing=False, num_cite_rounds=20, small_model="gpt-4o-2024-05-13", big_model="o1-2024-12-17", n_writeup_reflections=3, page_limit=4)`（`:857`）。

### 3.1 system message：`writeup_system_message_template`（`:533-600`，要点逐条）
```
You are an ambitious AI researcher who is looking to publish a paper to the "I Can't Believe It's Not Better" (ICBINB) Workshop at ICLR 2025.
This workshop aims to highlight real-world pitfalls, challenges, and negative or inconclusive results in deep learning, encouraging open discussion.
You must accurately represent the results of the experiments.
The main paper is limited to {page_limit} pages in single-column format, not counting references. ...
DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT.
MINIMIZE THE USAGE OF ITEMIZE OR ENUMERATE. ...
- Do not change the overall style which is mandated by the conference. Keep to the current method of including the references.bib file.
- Do not remove the \graphicspath directive or no figures will be found.
- Do not add `Acknowledgements` section to the paper.
Here are some tips for each section of the paper:
- **Title**: catchy and informative, under 2 lines.
- **Abstract**: brief summary of the challenge/pitfall, concise real-world motivation, one continuous paragraph.
- **Introduction**: overview of the issue, importance for practical contexts, contributions (may include negative results/pitfalls/unexpected behaviors).
- **Related Work**: cite relevant papers/approaches tackling similar issues; compare and contrast.
- **Background** (optional): necessary technical/domain background.
- **Method / Problem Discussion**: problem context or method; partial successes/lessons learned if not strictly an improvement.
- **Experiments** (if applicable): present results truthfully (negative/unexpected/inconclusive are valid); include figures/tables; up to 4 figures in main text, rest in appendix.
- **Conclusion**: main lessons learned, next steps.
- **Appendix**: supplementary material; hyperparams/algorithms; more plots/tables; when checking duplicate figures, also review descriptions (e.g. aggregated_training_accuracy.png vs per_seed_training_accuracy.png may be the same data).
Ensure you are always writing good compilable LaTeX code. Common mistakes ... unmatched braces, duplicate figure labels, unescaped & % $ # _ { } ~ ^ \, proper table/figure closure, DO NOT hallucinate new citations or results not in logs.
Ensure proper citation usage:
- Always include references within \begin{{filecontents}}{{references.bib}} ... \end{{filecontents}} ...
- Use citations from the provided references.bib content.
- Each section (especially Related Work) should have multiple citations.
When returning final code, place it in fenced triple backticks with 'latex' syntax highlighting.
```
> 实际调用前 `.format(page_limit=page_limit)`（`perform_writeup.py:973`），把双花括号还原为单花括号。

### 3.2 主写作 prompt：`writeup_prompt`（`:602-646`，逐字）
```
Your goal is to write up the following idea:

```markdown
{idea_text}
```

We have the following experiment summaries (JSON):
```json
{summaries}
```

We also have a script used to produce the final plots (use this to see how the plots are generated and what names are used in the legend):
```python
{aggregator_code}
```
Please also consider which plots can naturally be grouped together as subfigures.

Available plots for the writeup (use these filenames):
```
{plot_list}
```

We also have VLM-based figure descriptions:
```
{plot_descriptions}
```

Your current progress on the LaTeX write-up is:
```latex
{latex_writeup}
```

Produce the final version of the LaTeX manuscript now, ensuring the paper is coherent, concise, and reports results accurately.
Return the entire file in full, with no unfilled placeholders!
This must be an acceptable complete LaTeX writeup, suitable for a 4-page single-column workshop paper.
Make sure to use the citations from the references.bib file.

Please provide the updated LaTeX code for 'template.tex', wrapped in triple backticks
with "latex" syntax highlighting, like so:

```latex
<UPDATED LATEX CODE>
```
```
**填充值**（`perform_writeup.py:988-995`）：
- `idea_text` ← `load_idea_text(base_folder)`（`:648`，读 `idea.md`）
- `summaries` ← `load_exp_summaries` + `filter_experiment_summaries(step_name="writeup")`（`:691-743`，只保留 `overall_plan/analysis/metric/code/plot_analyses/vlm_feedback_summary` 字段，且只取 `BASELINE_SUMMARY`/`RESEARCH_SUMMARY` 的 `best node`）
- `aggregator_code` ← `<base>/auto_plot_aggregator.py`
- `plot_list` ← `figures/*.png` 文件名
- `plot_descriptions` ← 每张图 `generate_vlm_img_review`（`img_review_prompt`）生成的 `Img_description`
- `latex_writeup` ← 当前 `latex/template.tex`

**出参解析**：`re.search(r"```latex(.*?)```", response, re.DOTALL)`（`:1005`）；无匹配 → `return False`；有则写回 `template.tex`（`:1006`）。

### 3.3 引用收集 prompt（`:337-540`）
`citation_first_prompt_template`（`:365`）要求输出 `Description + Query`（```json），含 "No more citations needed" 早停；`citation_second_prompt_template`（`:403`）要求 `Selected`（索引字符串）+ `Description`，从 top10 选 bibtex；`references_format`（`:529`）把 bibtex 交给写作用的小模型，提示 "You do not need to add them yourself. ABSOLUTELY DO NOT ADD IT AGAIN!!!" 并插入 `\end{filecontents}` 前（`:947`）。缓存到 `cached_citations.bib`（`:924-936`）。

### 3.4 三类 reflection prompt（写作后，`n_writeup_reflections` 轮）

**(a) 主反思 `reflection_prompt`（`:1051-1100`）** —— 9 项检查：①chktex 语法 ②清晰/严谨 ③不幻觉 ④短段合并 ⑤补充材料 ⑥未用图列表 `{sorted(unused_figs)}` ⑦无效图引用 `{sorted(invalid_figs)}` ⑧`{reflection_page_info}` ⑨VLM 图/图注/正文评审 `{review_img_cap_ref}`、重复图 `{analysis_duplicate_figs}`。要求"revised complete LaTeX in triple backticks, or repeat the same if no changes"。

**(b) 图像选择反思 `img_reflection_prompt`（`:1126-1160`）**：给出当前用图/未用图、`reflection_page_info`、VLM 图选择评审 `{review_img_selection}`，要求 ①低价值图移附录 ②信息量低的删除 ③稀疏无子图的可合并 ④更新正文讨论 ⑤强化分析；"Be more aggressive with figure selection"；`"I am done"` 早停。

**(c) 最终页数优化 `final_reflection_prompt`（`:1195-1199`）**：
```
{reflection_page_info}
USE MINIMAL EDITS TO OPTIMIZE THE PAGE LIMIT USAGE.
```

### 3.5 编译与页数控制工具（写入 Prompt 的反馈来自这些函数）
- `compile_latex(cwd, pdf_file, timeout=30)`（`:45`）：pdflatex→bibtex→pdflatex→pdflatex。
- `check_page_limit(pdf_file, page_limit=4)`（`:238`）：用 `pymupdf` 数页、去页眉页脚。
- `detect_references_position_clean`（`:121`）、`get_reflection_page_info(reflection_pdf, page_limit)`（`:304`）：判断参考文献/附录位置，生成注入 reflection 的页数提示。
- 每轮 reflection 前都 `compile_latex(..., <run>_reflection<i+1>.pdf)` 并 `perform_imgs_cap_ref_review` + `detect_duplicate_figures` + `get_reflection_page_info` + `chktex`，把结果喂给 Prompt（`:1020-1049`）。

### 3.6 normal（8 页）模式差异
`ai_scientist/perform_writeup.py` 的 `writeup_system_message_template`（`:345`，含 `{page_limit}`）与 `writeup_prompt`（`:410`）结构类似，但模板为 `blank_icml_latex`（ICML2025），`page_limit=8`，且检查 `detect_pages_before_impact`（`:82`，impact statement 前页数）。CLI `--writeup-type normal` 切换（`launch_scientist_bfts.py:44`）。

---

## 4. Prompt / Schema 速查表

| 要改什么 | 文件:行号 |
| --- | --- |
| 无模板构思 system / 反思 | `perform_ideation_temp_free.py:61 / 99 / 111` |
| 构思工具定义 | `perform_ideation_temp_free.py:17-58`、`tools/semantic_scholar.py` |
| VLM 图审稿 prompt | `perform_vlm_review.py:33 / 74 / 126` |
| VLM 消息构造 / base64 | `vlm.py:132-193`（`encode_image_to_base64:35`） |
| 图抽取正则 | `perform_vlm_review.py:154-308` |
| ICBINB system / 主写作 | `perform_icbinb_writeup.py:533 / 602` |
| ICBINB 三类 reflection | `perform_icbinb_writeup.py:1051 / 1126 / 1195` |
| 引用 prompt | `perform_icbinb_writeup.py:365 / 403 / 529` |
| 树搜索内部 prompt / schema | `treesearch/parallel_agent.py:74-220`（FunctionSpec）、`treesearch/journal.py:21-41`、`treesearch/agent_manager.py:21-101` |

> **改 Prompt 的硬约束**：①凡要求 JSON 的输出必须保留 ```json 围栏与字段名（解析靠正则 + `extract_json_between_markers`）；②写作输出必须保留 ```latex 围栏；③VLM 输入必须是 `data:image/jpeg;base64,...` 的 `image_url`；④`FinalizeIdea` 的 7 字段与 `AgentManager.__init__` 的 5 键校验不可删。格式错误的重试策略见 [`troubleshooting.md`](troubleshooting.md)。
