# 13 完整 Agent 最终目标规范

## 13.1 文档定位

这份文档不是阶段性草案，而是 `foodEpic-agent` 的最终目标规范。

后续如果要把“实现完整的 agent”设成正式 goal，这份文档就应作为最直接的目标描述。

它要回答的不是：

- 当前已经实现了什么。
- 当前哪个 baseline 更高。
- 当前哪几个测试通过了。

它要回答的是：

- 最终完整 agent 应该长成什么样。
- 最终完整 agent 包含哪些模块。
- 每个模块的职责是什么。
- 每一类状态、工具、恢复逻辑、记忆策略最终应该如何工作。
- 什么时候才算真的“实现了完整的 agent”。

---

## 13.2 最终目标一句话定义

最终系统应是一个面向第一视角厨房长视频的、工具驱动的、长期记忆增强的、可恢复执行的多模态证据推理 agent。

它不依赖人工整理好的证据包，不依赖一次性大 prompt，不依赖固定抽帧结果直接喂给模型。

它必须具备以下能力：

- 基于问题主动规划。
- 自主调用图谱检索工具。
- 自主回查原始视频、原始帧、局部区域、OCR 和音频证据。
- 把新观察写回长期记忆。
- 在同视频多题中复用之前得到的记忆。
- 在低置信度时继续补证据，而不是过早结束。
- 在工具失败或工具空转时恢复执行，并切换到替代路径。
- 最终输出答案、置信度、证据链和可回查路径。

---

## 13.3 最终系统不应该是什么

为了避免方向跑偏，必须先明确“不是”什么。

最终系统不应是：

1. 单纯的 text-only 大模型猜题器。
2. 把若干关键帧和整理摘要拼进 prompt 的多模态问答器。
3. 先离线整理出题目证据，再让模型做最后一步选择的半人工 pipeline。
4. 只在单个问题内工作，不能跨题复用记忆的单次推理器。
5. 某一步工具失败就整题报错退出的脆弱执行流。
6. 工具一直空转却不会换路径的伪 agent。

如果系统本质上仍然是“看一眼 prompt 直接猜”，那它就不符合本项目最终目标。

---

## 13.4 最终 Agent 的外部行为

最终用户视角下，系统应表现为：

### 13.4.1 对单题的行为

给一个视频问题后，系统会：

1. 分析问题需要哪些证据。
2. 优先检索已有长期记忆。
3. 如果长期记忆不够，再回到原始视频证据。
4. 必要时抽帧、局部放大、OCR、音频峰值定位。
5. 在得到新证据后写回图谱。
6. 检查证据是否真的足够。
7. 若不足，继续规划下一步。
8. 若足够，输出答案、置信度和证据。

### 13.4.2 对同视频多题的行为

针对同一个视频的一组问题，系统应：

1. 复用之前题目已经抽出的帧。
2. 复用之前题目已经识别出的对象、位置、OCR、状态变化和 timeline。
3. 尽量减少重复观察和重复开销。
4. 允许跨进程恢复 session。

### 13.4.3 对异常情况的行为

如果工具失败：

- 不直接整题崩掉。
- 记录失败。
- 识别缺失证据类型。
- 换替代工具继续执行。

如果工具空转：

- 记录空转。
- 避免重复走同一路径。
- 切换到新的证据策略。

---

## 13.5 最终系统总体结构

最终系统应至少包含六层：

1. Memory Layer
2. Tool Layer
3. Planner Layer
4. Executor Layer
5. Verifier / Critic Layer
6. Evaluation Layer

这六层缺一不可。

---

## 13.6 Memory Layer 最终状态

## 13.6.1 目标

Memory Layer 的目标不是保存答案，而是保存可复用、可写回、可回查的结构化观察。

## 13.6.2 节点类型

最终至少包含以下节点：

- `video`
- `segment`
- `frame`
- `region`
- `timeline_event`
- `ingredient_event`
- `state_change`
- `object_state`
- `location_relation`
- `ocr_reading`
- `audio_event`
- `agent_observation`

## 13.6.3 边类型

最终至少包含以下边：

- `contains`
- `before`
- `after`
- `supports`
- `derived_from`
- `same_object`
- `same_ingredient`
- `same_step`
- `refers_to`
- `co_occurs`

## 13.6.4 节点最小字段

每个节点最终至少具备：

- `node_id`
- `node_type`
- `video_id`
- `start_time`
- `end_time`
- `label`
- `attributes`
- `evidence_paths`
- `source_tool`
- `confidence`

## 13.6.5 最终约束

图谱中的每条关键记忆都必须能回到原始证据：

- 原视频
- 原时间点/时间段
- 原始帧
- 原始区域图
- OCR 来源图
- 音频来源

图谱中禁止写入：

- benchmark 标准答案
- 测试真值
- 评测时不可见信息

## 13.6.6 Session Memory

最终系统除了图谱外，还必须有视频级 session memory。

它应保存：

- working memory
- evidence bundle
- retrieved frames
- retrieved nodes
- hypotheses
- open questions
- tool failures
- ineffective tools
- confidence
- question count

并且支持：

- 同视频复用
- 跨进程恢复
- 主动 reset

---

## 13.7 Tool Layer 最终状态

## 13.7.1 设计原则

最终所有证据访问都必须工具化。

模型默认只拿到：

- 问题
- 选项
- 工具 schema
- 当前工作记忆
- 上一步工具返回

模型默认不能直接拿到：

- 全量关键帧描述
- 全量视频摘要
- 人工整理好的证据包

## 13.7.2 图谱检索工具

最终必须稳定支持：

- `query_time`
- `query_object`
- `query_event`
- `query_state`
- `query_location`
- `query_region`
- `query_ocr`
- `get_neighbors`

作用：

- 优先低成本利用已有结构化记忆。
- 为后续原始证据回查定位候选时间段或候选对象。

## 13.7.3 原始视频访问工具

最终必须支持：

- `extract_frame_at_time`
- `extract_frames_for_range`
- `sample_sparse_frames`
- `sample_frames_around_peaks`
- `extract_input_reference_frames`

要求：

- 默认避免连续重复帧。
- 优先稀疏采样。
- 只在必要时对候选区间补帧。

## 13.7.4 空间/区域工具

最终必须支持：

- `render_bbox_overlay`
- `extract_region_with_context`
- `resolve_bbox_reference`

要求：

- 优先画框而不是纯裁剪。
- 保留上下文。
- 支持从 bbox 反查 object track / association / fixture。

## 13.7.5 感知工具

最终必须支持：

- `inspect_visual_evidence`
- `run_ocr_on_image`
- `run_ocr_on_region`
- `detect_audio_peaks`
- `identify_image_ingredients`
- `infer_visual_mcq`
- `infer_action_mechanism`
- `infer_action_intent`
- `infer_gaze_target_with_context`
- `infer_viewpoint_choice`
- `infer_named_fixture_direction`

要求：

- 感知结果尽量结构化返回。
- 不返回无约束长文本。
- 结果可写回图谱，可被后续问题复用。

## 13.7.6 写回工具

最终必须支持：

- `write_observation`
- `write_frame_observation`
- `write_region_observation`
- `write_ocr_reading`
- `write_audio_event`
- `write_timeline_summary`
- `write_state_change`

要求：

- 每次写回都要有来源工具。
- 每次写回都要有时间范围。
- 每次写回都要有证据路径。
- 每次写回都要能再次被检索。

---

## 13.8 Planner Layer 最终状态

## 13.8.1 最终职责

planner 的职责不是直接做答案，而是决策下一步证据动作。

最终 planner 要基于以下状态共同决策：

- task family
- 问题文本
- 当前时间提示
- 当前 bbox 提示
- working memory
- evidence bundle
- hypotheses
- open questions
- tool failures
- ineffective tools
- retrieved frames
- retrieved nodes
- 上一步工具返回

## 13.8.2 最终规划原则

最终规划规则应是：

1. 先判断当前最缺什么证据。
2. 优先使用低成本结构化检索。
3. 如果结构化检索不够，再回查原始证据。
4. 如果当前路径失败或空转，切换路径。
5. 如果证据不足，禁止直接 finish。
6. 如果证据充足，再调用 finish。

## 13.8.3 最终应弱化的内容

最终 planner 不应重度依赖：

- `current_step == 1/2/3`
- 硬编码题型顺序路由
- 过早进入 `rank_choices_from_state`

最终 planner 应更多依赖：

- evidence gap
- failure history
- ineffective history
- evidence coverage
- confidence

## 13.8.4 恢复性规划

当出现以下情况时，planner 必须重规划：

- 最近一次评分置信度低
- 存在未解决 `open_questions`
- 最近工具失败
- 最近工具空转
- 当前证据互相矛盾

## 13.8.5 最终理想能力

最终 planner 应支持：

- 多条候选补证据路径之间的选择
- 基于成本和收益做路径偏好
- 不仅选择“下一工具”，还要隐式比较“为什么选这条路径”

---

## 13.9 Executor Layer 最终状态

## 13.9.1 最终职责

executor 是 agent runtime。

它最终负责：

- 维护 agent 状态
- 执行工具
- 自动合并结果
- 自动写回长期记忆
- 自动更新 hypotheses / open questions
- 自动处理工具失败
- 自动处理工具空转
- 自动触发恢复和继续规划

## 13.9.2 最终状态字段

最终 `AgentState` 应显式维护：

- `video_id`
- `question`
- `choices`
- `task_family`
- `inputs_json`
- `current_step`
- `max_steps`
- `plan_summary`
- `hypotheses`
- `open_questions`
- `retrieved_node_ids`
- `retrieved_nodes`
- `retrieved_frames`
- `evidence_bundle`
- `working_memory`
- `tool_trace`
- `tool_failures`
- `ineffective_tools`
- `final_answer`
- `final_prediction`
- `confidence`

## 13.9.3 Hypotheses

`hypotheses` 用于记录：

- 当前候选解释
- 当前候选答案
- 已确认收集到的关键证据
- 已失败的路径
- 已空转的路径

## 13.9.4 Open Questions

`open_questions` 用于记录当前仍缺失的证据类型，例如：

- `need_time_localization`
- `need_region_grounding`
- `need_ocr_reading`
- `need_state_evidence`
- `need_location_evidence`
- `need_initial_observation`
- `need_disambiguating_evidence`
- `need_alternative_evidence_path`

## 13.9.5 Tool Trace

最终每次工具调用必须保留：

- 工具名
- 参数
- 结果摘要
- 原始返回

如果工具失败，还要记录：

- `error_type`
- `error_message`

如果工具空转，还要记录：

- `reason`

## 13.9.6 工具失败恢复

最终 executor 对工具失败的默认行为：

1. 记录失败。
2. 写入 `tool_failures`。
3. 更新 `working_memory`。
4. 更新 `open_questions`。
5. 不直接整题退出。
6. 在剩余步数内继续规划。

## 13.9.7 工具空转恢复

最终 executor 对无效工具的默认行为：

1. 识别没有带来任何新证据。
2. 写入 `ineffective_tools`。
3. 更新 `working_memory`。
4. 打开 `need_alternative_evidence_path`。
5. 让 planner 避免重复该工具路径。

## 13.9.8 自动写回

最终 executor 不应完全依赖 planner 手工调用写回工具。

对以下结果应自动写回：

- 视觉观察
- OCR 读数
- 音频峰值事件
- 时间线总结

---

## 13.10 Verifier / Critic Layer 最终状态

这是最终完整 agent 必须补齐的一层。

## 13.10.1 最终职责

verifier 负责判断：

- 当前证据是否真的够回答问题
- 当前答案是否只是凭先验猜测
- 当前 `open_questions` 是否已经实质清空
- 当前证据是否覆盖了题目的关键变量
- 当前证据之间是否矛盾

## 13.10.2 最终输出

verifier 最终至少输出：

- `sufficient: true/false`
- `confidence`
- `missing_evidence_types`
- `conflicts`
- `recommend_next_action`

## 13.10.3 最终行为

如果 verifier 认为证据不足：

- 不能直接 finish
- 必须回到 planner 继续补证据

---

## 13.11 Evaluation Layer 最终状态

## 13.11.1 不能只看 accuracy

最终评测至少包括：

- accuracy
- by-task-family accuracy
- tool success rate
- tool failure recovery rate
- ineffective loop avoidance rate
- session memory reuse rate
- raw evidence revisit rate
- average step count
- average latency
- average cost
- answer trace completeness
- evidence grounding quality

## 13.11.2 最终输出文件

最终每轮运行至少输出：

- prediction records
- failure summary
- failure cases
- progress
- summary
- per-video session summary
- session trace

## 13.11.3 最终应支持断点续跑

运行应支持：

- resume
- 跨进程 session 恢复
- 不覆盖旧结果
- 可单视频复跑
- 可单题复跑

---

## 13.12 最终一题的标准执行过程

最终一题应按如下流程运行：

1. 解析问题。
2. 初始化 evidence gaps。
3. 检索 session memory。
4. 检索 graph memory。
5. 判断是否已有足够证据。
6. 若不足，则选择最便宜的补证据路径。
7. 执行工具。
8. 合并结果。
9. 自动写回图谱。
10. 更新 hypotheses。
11. 更新 open questions。
12. 若工具失败，则恢复。
13. 若工具空转，则避环。
14. 让 verifier 判断当前证据是否足够。
15. 若不足，则继续规划。
16. 若足够，则输出最终答案。

---

## 13.13 最终多题 session 的标准行为

同视频多题最终应做到：

1. 前题抽的帧可复用。
2. 前题识别出的 OCR 可复用。
3. 前题写入的 timeline 可复用。
4. 前题失败和空转历史可辅助后题少走弯路。
5. 进程重启后 session 仍能恢复。

---

## 13.14 最终功能完成定义

只有当以下条件同时满足，才能说“完整 agent 已实现”：

1. 不依赖人工整理证据大 prompt。
2. 原始视频、帧、区域、OCR、音频始终可回查。
3. 图谱能保存和复用长期记忆。
4. 同视频多题能复用 session。
5. 低置信度时能继续补证据。
6. 工具失败时能恢复执行。
7. 工具空转时能避环重规划。
8. verifier 能判断证据是否充足。
9. 能输出答案、置信度、证据链、trace。
10. 能在真实小规模视频问题上端到端稳定运行。

如果缺少其中任意一项，都不应宣称“完整 agent 已实现”。

---

## 13.15 当前最关键的未完成项

从最终目标看，当前最关键的未完成项通常会是：

1. planner 进一步去 step 化，减少写死分支。
2. verifier / critic 层正式落地。
3. 多候选路径比较能力。
4. 更真实的小规模端到端运行验证。
5. 更统一的 session memory / graph memory / raw evidence 三层协同策略。

---

## 13.16 建议未来 goal 的写法

如果后续要正式设定 goal，建议直接写成：

“实现一个完整的工具驱动厨房视频 agent。该 agent 具备长期图谱记忆、原始视频/区域/OCR/音频回查、多题 session 复用、低置信度重规划、工具失败恢复、工具空转避环、证据验证器和可追踪输出能力；不依赖人工整理证据 prompt，而通过工具自主检索、写回和复用证据完成视频问答与开放查询。”

这比“继续优化一下 agent”更清晰，也更接近最终目标。
