# 24 Observation-Centric Agent 交接总览

## 24.1 文档目的

这份文档给后续接手模型使用。

它只回答四件事：

- 当前项目真实架构是什么。
- 当前已经做到了什么程度。
- 接下来应该继续做什么。
- 最终想实现一个什么样的 agent。

它不是新的专项提分清单，也不是为了继续堆规则。

---

## 24.2 第一原则：科研规范与证据严谨性

后续所有实现与实验，必须先满足下面这些约束。

### 24.2.1 不允许的行为

- 不允许从 `best_index`、`choice_text`、`runner_up`、`comparison_summary`、`blocking_hypotheses`、`blocking_comparisons` 反推下一步搜证动作。
- 不允许把 benchmark 真值、隐藏标注、离线才知道的信息写入图谱、工作记忆或提示词。
- 不允许为固定题型、固定选项语义、固定错题模式继续堆专项规则，只为了把分数做高。
- 不允许把单元测试通过数当成实验效果，不允许把 regression 数字写成论文结果。
- 不允许把“看起来像答案的中间比较信息”再喂回搜索主链。

### 24.2.2 允许的行为

- 只从原始观测、时间窗覆盖、对象轨迹、空间关系、状态变化和预算状态推断证据缺口。
- 让模型自己判断“当前证据是否足够”，不足时再主动扩窗、补帧、追轨迹、查图谱。
- 保留完整工具轨迹、证据路径、失败记录和可回查原始证据。
- 用小样本真实 trace 审计和分层抽样验证来判断是否真的变好。

### 24.2.3 论文级标准

最终可汇报结果必须满足：

- 推理过程可解释。
- 证据链可回放。
- 改进机制具有泛化性，而不是题库记忆。
- 消融实验能说明每个模块为什么有效。

---

## 24.3 当前真实架构

当前系统已经不是单次 prompt 问答器，而是一个“问题 -> 规划 -> 调工具 -> 检查证据 -> 决定是否继续”的闭环。

### 24.3.1 入口层

- [food_agent/agent/graph_agent.py](/22liushoulong/agent/hd-epic/food_agent/agent/graph_agent.py)

主要职责：

- 对单题或同视频多题建立运行 session。
- 组装 `store + toolbox + executor`。
- 负责结果持久化、evidence report、session trace 和 usage 统计。

### 24.3.2 规划层

- [food_agent/agent/planner.py](/22liushoulong/agent/hd-epic/food_agent/agent/planner.py)

主要职责：

- 基于当前 `AgentState` 和工具 schema 选择下一步动作。
- 在 LLM 规划和启发式 fallback 之间做受控切换。
- 在预算耗尽时收口。

当前问题：

- 仍残留一部分历史性的 `action_intent` 专项恢复逻辑，还没有完全变成纯 observation-centric。

### 24.3.3 执行层

- [food_agent/agent/executor.py](/22liushoulong/agent/hd-epic/food_agent/agent/executor.py)

主要职责：

- 执行多步工具循环。
- 维护 heartbeat、失败恢复、预算更新和结果并入状态。
- 在 finish 前调用 verifier 做证据充分性检查。

### 24.3.4 验证层

- [food_agent/agent/verifier.py](/22liushoulong/agent/hd-epic/food_agent/agent/verifier.py)

主要职责：

- 判断当前证据是否足够。
- 输出缺失证据类型、冲突和建议下一步动作。

当前问题：

- `fine_grained_why_recognition` 上仍保留了一些历史字段和 specialized gap 结构，仍需继续去答案条件化。

### 24.3.5 状态层

- [food_agent/agent/state.py](/22liushoulong/agent/hd-epic/food_agent/agent/state.py)

主要职责：

- 保存 working memory、evidence bundle、tool trace、verification history、search budget、session memory。
- 作为 planner、executor、verifier 共享的唯一运行时状态容器。

注意：

- `ActionIntentHypothesis` 等结构目前仍存在，后续应进一步限制其作用范围，避免重新渗回主搜索链。

### 24.3.6 工具层

- [food_agent/tools/agent_toolbox.py](/22liushoulong/agent/hd-epic/food_agent/tools/agent_toolbox.py)
- [food_agent/tools/graph_tools.py](/22liushoulong/agent/hd-epic/food_agent/tools/graph_tools.py)
- [food_agent/tools/video_tools.py](/22liushoulong/agent/hd-epic/food_agent/tools/video_tools.py)

当前已经具备的能力：

- 图谱检索：`query_time / query_object / query_event / query_state / query_location / query_region / query_ocr / get_neighbors`
- 原始视频回查：`extract_frame_at_time / extract_frames_for_range / sample_sparse_frames`
- 局部证据：`render_bbox_overlay / extract_region_with_context / OCR`
- 音频线索：`detect_audio_peaks / sample_frames_around_peaks`
- 视觉判断：`inspect_visual_evidence`
- 若干结构化任务工具：称量、营养、recipe、object motion 等

### 24.3.7 记忆与图谱层

- [food_agent/memory/store.py](/22liushoulong/agent/hd-epic/food_agent/memory/store.py)
- [food_agent/graph/builder.py](/22liushoulong/agent/hd-epic/food_agent/graph/builder.py)

当前实现：

- 以 SQLite + JSONL 作为轻量图谱存储。
- 从事件索引和已有 frame memory 构建视频级 graph memory。
- 节点已覆盖 `video / segment / frame / ingredient_event / recipe_step / object_track / audio_event / timeline_event` 等。

### 24.3.8 模型访问层

- [food_agent/model_client.py](/22liushoulong/agent/hd-epic/food_agent/model_client.py)

当前实现：

- 已兼容 OpenAI 风格 `chat_completions` 与 `responses` 双通道。
- 已支持 vision 请求能力检查、usage 统计和成本估算。

---

## 24.4 已经完成了什么

这里强调“真实完成”，不是“计划里写过”。

### 24.4.1 基础工程与数据侧

- HD-EPIC 数据已完成本地下载、整理和可访问化。
- 已建立项目路径、输出目录和图谱持久化目录。
- 已有事件索引、graph builder、memory store 和 session 持久化链路。

### 24.4.2 Agent 基本闭环

- 已实现 `planner -> executor -> verifier -> finish` 的多步 agent 闭环。
- 已支持同视频多题 session 复用。
- 已支持结果落盘、trace 持久化、usage 统计、heartbeat 输出。

### 24.4.3 多模态工具化访问

- 已实现图谱检索工具。
- 已实现视频抽帧、稀疏抽帧、bbox 画框、局部放大、OCR、音频峰值检索。
- 已实现让模型通过工具主动看图，而不是默认把整理好的内容直接塞进 prompt。

### 24.4.4 观测驱动方向上的真实进展

- `why / action_intent` 主链上，大量明显的 answer-conditioned consumer 已经被清掉。
- 近期验证过的专项回归基线是：
  - `pytest -q tests/test_graph_agent.py -k 'action_intent'`
  - 最新已知结果：`724 passed, 453 deselected`

这个数字的含义仅是：

- 回归测试当前没有大面积回退。
- 不能把它当成真实任务准确率，更不能当论文主结果。

### 24.4.5 已经明确不能再走的旧路线

- 不能再把 why 题当成“选项语义冲突分类器”去修。
- 不能再用 `best choice` 或 `runner-up` 去定义缺失证据。
- 不能再把固定错题簇写成越来越长的规则表。

---

## 24.5 当前还没有完成的部分

### 24.5.1 observation-centric 改造还未彻底收口

虽然旧链已经切掉很多，但还没有完全结束。

当前最重要的剩余问题有两类：

- `graph_agent.py` 中仍有少量 `action_intent` 残余逻辑没有完全脱离历史 specialized 路径。
- verifier / finalizer / runtime state 里仍存在一些容易回流到答案语义的旧字段和旧测试契约。

### 24.5.2 真实 agent 的“自主补证”还不够通用

用户真正要的能力是：

- 模型先判断当前证据够不够。
- 如果不够，自己决定先看局部前后帧、还是扩更大时间窗、还是跟踪对象后续轨迹。
- 有预算上限，不能无脑看完整段视频。

当前系统已经有这个闭环骨架，但仍需把它从“带历史 why 专项痕迹的闭环”推进成“通用的证据缺口驱动闭环”。

### 24.5.3 研究级实验还没有准备好

当前还缺：

- 完全去答案条件化后的干净主线。
- 小样本随机分层审计。
- 真正可复现实验协议。
- 对照 baseline、成本、token、失败模式和消融的统一表格。

### 24.5.4 多模态扩展还不完整

音频、空间、object track 这些信息已经有接口或基础能力，但还没有完全以“通用搜证工具”方式稳定并入主决策链。

---

## 24.6 接下来应该继续做什么

后续建议只沿下面这个顺序推进。

### 24.6.1 第一优先级：彻底完成去答案条件化

只允许保留：

- observation state
- gap schema
- budgeted search
- evidence sufficiency
- final mapping

继续清理：

- runtime 中残留的 `choice/category` 驱动搜索逻辑
- 由候选比较结果直接驱动 gap 的链路
- 旧测试里仍在保护旧行为的契约

主执行文档只保留：

- [docs/plan/23_observation_centric_agent_execution_master_plan_zh.md](/22liushoulong/agent/hd-epic/docs/plan/23_observation_centric_agent_execution_master_plan_zh.md)

### 24.6.2 第二优先级：把 gap 推理做成通用机制

目标不是继续加规则，而是统一成少量通用 gap：

- `window_coverage_missing`
- `object_track_unclosed`
- `destination_unclosed`
- `relation_unobserved`
- `state_transition_unconfirmed`
- `precondition_missing`
- `immediate_result_missing`

然后只允许 agent 围绕这些 gap 决定：

- 局部补帧
- 扩时间窗
- 跟踪对象
- 查看空间区域
- 停止并回答

### 24.6.3 第三优先级：在小样本真实视频上做 trace 审计

切记：

- 先做小样本、随机分层、保留完整 trace。
- 不要急着跑全量。
- 先确认 agent 的每一步动作是否真的是“因证据不足而补证”，不是“因选项像某类题而补证”。

### 24.6.4 第四优先级：等主链干净后再做实验

顺序必须是：

1. 主链干净
2. 小样本真实 trace 正常
3. 再做 baseline 对比
4. 再做 full evaluation

不能反过来。

---

## 24.7 最终想实现的功能

最终 agent 应该具备下面这组能力。

### 24.7.1 对单题

输入只有：

- 问题
- 选项
- 当前视频 id
- 可用工具

然后 agent 自己：

- 先查已有记忆
- 再判断证据够不够
- 不够就主动补证
- 直到预算耗尽或证据充分
- 最后输出答案、置信度、证据链和工具轨迹

### 24.7.2 对同视频多题

- 复用已经抽过的帧和图谱节点
- 复用对象轨迹、空间关系、OCR 和 timeline
- 降低重复成本

### 24.7.3 对论文与应用

它最终应该是一个：

- 面向第一视角厨房长视频的
- 工具驱动的
- 证据可回查的
- 长期记忆增强的
- 预算受控的
- 多模态问答与过程理解 agent

它的目标不是“刷某一类 why 题”，而是：

- 回答厨房过程问题
- 给出证据
- 记录做饭流程
- 追踪食材、步骤、物体和状态变化
- 支持后续的异常检测、营养追踪和过程复盘

---

## 24.8 当前有效文档建议

后续接手时，优先看这 4 份：

- [01_project_scope_zh.md](/22liushoulong/agent/hd-epic/docs/plan/01_project_scope_zh.md)
- [13_complete_agent_target_spec_zh.md](/22liushoulong/agent/hd-epic/docs/plan/13_complete_agent_target_spec_zh.md)
- [23_observation_centric_agent_execution_master_plan_zh.md](/22liushoulong/agent/hd-epic/docs/plan/23_observation_centric_agent_execution_master_plan_zh.md)
- 本文档 [24_observation_centric_agent_handoff_zh.md](/22liushoulong/agent/hd-epic/docs/plan/24_observation_centric_agent_handoff_zh.md)

辅助背景文档可看：

- [10_graph_tool_agent_execution_plan_zh.md](/22liushoulong/agent/hd-epic/docs/plan/10_graph_tool_agent_execution_plan_zh.md)
- [11_tool_driven_graph_agent_architecture_zh.md](/22liushoulong/agent/hd-epic/docs/plan/11_tool_driven_graph_agent_architecture_zh.md)
- [12_autonomous_tool_graph_agent_plan_zh.md](/22liushoulong/agent/hd-epic/docs/plan/12_autonomous_tool_graph_agent_plan_zh.md)

`16/17` 可保留作历史审计材料，但不再作为主执行清单。
