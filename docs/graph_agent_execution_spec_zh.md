# HD-EPIC 图谱工具型 Agent 实施规格

## 1. 文档目的

这份文档不是泛泛而谈的想法说明，而是后续实现 `foodEpic-agent` 主线系统的实施约束。

目标只有一个：

**把当前“模型直接看输入后回答”的流程，改造成“模型先规划，再调用工具检索图谱和原始视频证据，最后才回答”的真实 agent。**

---

## 2. 强约束

后续实现必须满足以下约束，不能退回旧路线。

### 2.1 不主动把整理好的内容塞给大模型

禁止做法：

- 把整段视频摘要直接拼进 prompt
- 把大量帧描述直接拼进 prompt
- 把人为选好的关键帧列表直接交给模型
- 先人工筛好证据，再让模型只负责“做选择题”

允许做法：

- 只给问题、选项、工具说明
- 允许模型先调用图谱检索工具
- 允许模型再调用原始视频回查工具
- 只把工具返回结果写入工作记忆

即：

`LLM 看到的是环境与工具，不是预整理答案。`

### 2.2 图谱不是终点，原始数据必须可回查

图谱只保存“可检索的结构化记忆”，不能代替原始证据。

系统必须始终保留：

- 按时间抽帧
- 按时间段批量抽帧
- 对 bbox 画框而不是裁切掉上下文
- 对局部区域放大
- OCR / 读数识别
- 重新查看候选时间窗口

### 2.3 图谱中只能存评测允许获得的信息

允许进入图谱：

- 原视频与音频产生的派生证据
- 自动抽帧
- 自动框选区域
- OCR 结果
- 自动提取的对象、动作、状态、位置、步骤、时间事件

禁止进入图谱：

- benchmark 答案
- 标注侧真值
- 只在离线分析阶段知道、但评测时无法观察到的信息

---

## 3. 目标系统定义

最终系统定义为：

`Question -> Planner -> Tool Calls -> Working Memory Update -> Need-More-Evidence Decision -> Final Answer`

而不是：

`Question -> Prompt -> LLM -> Answer`

Agent 至少要具备四种能力：

1. 识别当前问题属于哪类证据需求
2. 主动检索图谱中的长期记忆
3. 必要时主动访问原始视频/图片/OCR
4. 基于证据链而不是直觉输出答案

---

## 4. 系统分层

## 4.1 Memory Layer

职责：

- 保存视频级长期记忆
- 保存帧、事件、对象、步骤、区域、OCR、音频事件
- 为 planner/executor 提供统一查询接口

当前实现基础：

- `food_agent/memory/schema.py`
- `food_agent/memory/store.py`
- `food_agent/graph/builder.py`

当前状态：

- 已有 SQLite + JSONL 的轻量图谱存储
- 已能为单视频构建图谱
- 已修复序列化与节点 ID 冲突问题

### 4.2 Retrieval Layer

职责：

- 对图谱做按时间、按对象、按事件类型的检索
- 后续扩展到按状态、按位置、按 OCR、按区域检索

当前实现基础：

- `food_agent/retrieval/time_retriever.py`
- `food_agent/retrieval/object_retriever.py`
- `food_agent/retrieval/event_retriever.py`

### 4.3 Tool Layer

职责：

- 向 LLM 暴露“可调用能力”
- 所有原始数据访问都必须包装成工具
- 工具返回结构化结果，而不是无约束文本

当前实现基础：

- `food_agent/tools/graph_tools.py`
- `food_agent/tools/video_tools.py`

### 4.4 Agent Layer

职责：

- 问题解析
- 工具规划
- 工具执行
- 工作记忆维护
- 最终回答

当前实现基础：

- `food_agent/agent/state.py`
- `food_agent/agent/planner.py`
- `food_agent/agent/executor.py`
- `food_agent/agent/graph_agent.py`

当前状态：

- planner 已支持 LLM 驱动的下一步工具决策，并保留启发式 fallback
- executor 已能进入多步工具循环，不只检索图谱，也能调抽帧、画框、局部放大、视觉判别和图谱写回
- 当前瓶颈不再是基本闭环缺失，而是任务族覆盖不完整，以及部分任务的专用工具链仍需增强

---

## 5. 图谱设计原则

### 5.1 节点层级

建议固定为以下层级：

- `video`
- `segment`
- `frame`
- `region`
- `object_track`
- `ingredient_event`
- `recipe_step`
- `audio_event`
- `ocr_reading`
- `timeline_event`
- `observation`

### 5.2 边类型

- `contains`
- `supports`
- `refers_to`
- `co_occurs`
- `before`
- `after`
- `derived_from`
- `same_target`

### 5.3 节点最小字段

每个节点至少有：

- `node_id`
- `node_type`
- `video_id`
- `label`
- `start_time`
- `end_time`
- `attributes`
- `evidence_paths`
- `keywords`

### 5.4 图谱中的证据必须可追溯

每个节点都必须能反查到：

- 视频路径
- 时间范围
- 帧路径
- region 路径
- 对应的来源模块

如果一个节点不能回到原始证据，它就不能作为强证据节点。

---

## 6. 工具设计规范

后续任何新能力，都先定义成工具，再考虑是否让 LLM 使用。

### 6.1 图谱检索工具

必须保留：

- `query_time(video_id, start_time, end_time, limit)`
- `query_object(video_id, query, limit)`
- `query_event(video_id, event_types, keyword, start_time, end_time, limit)`
- `get_neighbors(node_ids, edge_types, limit)`

建议继续新增：

- `query_state(video_id, state_keyword, start_time, end_time, limit)`
- `query_location(video_id, location_keyword, start_time, end_time, limit)`
- `query_ocr(video_id, keyword, start_time, end_time, limit)`
- `query_region(video_id, object_hint, start_time, end_time, limit)`

### 6.2 原始视频工具

已支持核心子集，后续建议继续新增或强化：

- `extract_frame_at_time(video_path, time_s)`
- `extract_frames_for_range(video_path, start_time, end_time, stride_s, max_frames)`
- `render_bbox_overlay(image_path, bbox)`
- `render_bbox_sequence(video_path, bbox_track, times)`
- `extract_region_with_context(image_path, bbox, expand_ratio)`
- `run_ocr_on_image(image_path)`
- `run_ocr_on_region(image_path, bbox, expand_ratio)`
- `sample_sparse_frames(video_path, start_time, end_time, sample_count)`

说明：

- 对于定位类题，默认优先画框，不直接裁剪
- 只有在 OCR / 小物体读数时才允许生成局部 crop

### 6.3 图谱写回工具

这是后续 agent 与普通“检索问答器”拉开差距的关键。

建议继续新增：

- `write_frame_observation(...)`
- `write_ocr_reading(...)`
- `write_region_observation(...)`
- `write_object_hypothesis(...)`
- `write_timeline_summary(...)`

也就是说，agent 不是只读外部记忆，而是可以在检索和观察后增量写回。

---

## 7. LLM 在系统中的角色

LLM 不是直接答题器，而是：

1. 任务规划器
2. 工具调用决策器
3. 证据充足性判断器
4. 最终答案汇总器

### 7.1 输入给 LLM 的内容上限

默认只给：

- 问题文本
- 选项
- 已有工作记忆摘要
- 工具 schema
- 上一步工具返回

不允许默认给：

- 全量帧描述
- 全量视频总结
- 全量图谱导出

### 7.2 工作记忆格式

建议固定为：

```json
{
  "task_family": "...",
  "video_id": "...",
  "question": "...",
  "choices": ["..."],
  "hypotheses": [],
  "evidence": [],
  "visited_times": [],
  "visited_nodes": [],
  "open_questions": [],
  "confidence": 0.0
}
```

LLM 每一步只能基于当前工作记忆和工具返回做下一步决策。

---

## 8. 标准执行循环

后续正式 agent 必须按下面的循环工作。

### Step 1. 问题解析

解析出：

- 题型
- 关键对象
- 关键时间
- 是否涉及 bbox
- 是否涉及状态变化
- 是否涉及 OCR / 重量读数

### Step 2. 初始计划

LLM 输出：

- 当前假设
- 首批工具调用计划
- 调这些工具的原因

### Step 3. 图谱检索

优先做低成本检索：

- 时间窗口
- 对象关键词
- 事件类型
- 已知步骤

### Step 4. 证据评估

判断当前证据是否足够区分选项。

如果不能区分，必须明确说明还缺什么：

- 缺时间局部证据
- 缺空间位置确认
- 缺读数
- 缺状态变化
- 缺对象持续跟踪

### Step 5. 原始数据回查

按需调用：

- 抽帧
- 稀疏抽帧
- 画框
- OCR
- 局部放大
- 时间窗口扩展

### Step 6. 写回图谱/工作记忆

把新观察到的结果写入：

- 工作记忆
- 图谱节点
- 时间线摘要

### Step 7. 最终作答

只有当 evidence 足够时，才允许输出：

- 预测选项
- 证据链
- 使用过的工具轨迹

---

## 9. 针对 HD-EPIC VQA 的题型落地

### 9.1 Ingredient Retrieval / Adding Localization

研究意义：

- 检验 agent 是否能把做饭流程与食材操作绑定
- 检验长期记忆而不是单帧猜测

应用价值：

- 做菜流程自动记录
- 食材加入时刻与顺序追踪
- 饮食分析与菜谱反演

工具优先级：

1. `query_time`
2. `query_event(ingredient_event)`
3. `query_object`
4. 必要时 `extract_frames_for_range`

### 9.2 Recipe Multi-step Localization

研究意义：

- 检验 agent 是否能理解长时步骤结构
- 检验图谱是否真能支撑跨时间段检索

应用价值：

- 自动生成做菜步骤时间线
- 厨房教学视频结构化摘要

工具优先级：

1. `query_event(recipe_step)`
2. `query_time`
3. `build_local_timeline`
4. 必要时稀疏补帧

### 9.3 Object Movement Itinerary / Stationary Localization

研究意义：

- 检验 agent 的空间推理与对象轨迹组织能力
- 能体现“图谱 + 原始证据回查”的优势

应用价值：

- 厨房物品追踪
- 人机协作取物
- 作业过程回放与错误定位

工具优先级：

1. `query_object`
2. `query_time`
3. `render_bbox_overlay`
4. `render_bbox_sequence`
5. `build_motion_trace`

### 9.4 Weight / Scale Reading

研究意义：

- 检验 agent 是否能针对难题调用专用工具
- 能显著区分“直接问答模型”和“工具型 agent”

应用价值：

- 营养分析
- 精准烹饪辅助
- 食材称量记录

工具优先级：

1. `query_time`
2. `extract_frame_at_time`
3. `render_bbox_overlay`
4. `extract_region_with_context`
5. `run_ocr_on_region`
6. `write_ocr_reading`

---

## 10. 与当前代码对齐的分阶段实施

## Phase A. 打牢记忆层

目标：

- 让单视频图谱稳定、可追溯、可增量写回

已完成：

- SQLite/JSONL 图谱存储
- 单视频图构建器
- 基础测试

接下来要做：

- 增加 region / OCR 节点
- 增加图谱写回 API
- 增加图谱统计与完整性检查脚本

验收：

- 单视频构图成功
- 节点和边数量稳定
- 任一节点都可追溯到原始证据

## Phase B. 把视频访问能力真正工具化

目标：

- 不再依赖离线手工生成结果
- agent 可自己抽帧、画框、回看

要做：

- 扩充 `food_agent/tools/video_tools.py`
- 支持时间段稀疏抽帧
- 支持 bbox overlay 序列输出
- 支持 OCR 读数

验收：

- 给定视频和时间范围，工具能自动产出一组可回看的证据图

## Phase C. 把 planner 改成真正的工具调用 planner

目标：

- 不再使用现在这种纯启发式硬编码规划

要做：

- 为 planner 定义 tool schema
- 让模型输出结构化 action
- 限制每轮最多调用若干工具
- 引入“证据是否足够”的显式判断

验收：

- planner 能根据不同题型形成不同工具序列
- 工具序列不是写死在 Python `if-else` 里

## Phase D. 把 executor 改成多轮 agent loop

目标：

- 支持检索后继续查、查完写回、再继续查

要做：

- `executor` 支持多轮循环
- 支持工具结果进入工作记忆
- 支持图谱增量写回
- 支持失败恢复与中断续跑

验收：

- 一道题至少可以有 2 到 5 轮工具调用
- 每轮都能看到工作记忆变化

## Phase E. 做真正有研究价值的对比实验

目标：

- 证明 agent 优势来自“主动检索与工具回查”，不是提示词

必须比较：

- `textonly`
- `directevidence`
- `foodstate`
- `ours-foodevidence`
- `graph-agent`

必须记录：

- 准确率
- 每题平均工具次数
- 每题平均成本
- 每题平均耗时
- 原始数据回查率
- OCR 工具触发率
- 失败原因分布

---

## 11. 结果保存与可恢复要求

所有正式运行都必须支持断点续跑。

每道题结果至少保存：

- `prediction`
- `gold`
- `correct`
- `tool_trace`
- `working_memory_snapshot`
- `evidence_bundle`
- `generated_artifact_paths`
- `error_message`

如果某一步失败：

- 不覆盖旧结果
- 不丢失中间证据
- 可以从最近成功步骤继续

---

## 12. 近期开发优先级

按照现实性与研究价值，近期优先级固定如下：

1. 完善 `video_tools`，让 agent 真能自己回看视频
2. 增加 region/OCR/overlay 证据节点写回
3. 把 planner 改成工具调用式
4. 把 executor 改成多轮循环
5. 先在单视频上打通 `ingredient`、`recipe step`、`object motion`
6. 再做 `weight/OCR` 这种能显著体现 agent 优势的难题

不优先做：

- 继续堆 prompt
- 继续把更多人工摘要喂给模型
- 继续单纯加重试次数

---

## 13. 当前代码状态说明

截至当前版本，系统已经具备图谱型 agent 的最小骨架，但还不是最终形态。

已有能力：

- 单视频图谱构建
- 基础图谱检索
- 简单 planner/executor
- 单题 smoke run

已验证：

- `tests/test_graph_agent.py` 通过
- `scripts/build_video_graph.py --video-id P08-20240617-130401` 可运行

当前短板：

- 还没有真正让 LLM 通过 tool schema 决策
- 还没有把视频抽帧/OCR/region 访问纳入主循环
- 还没有形成“查图谱 -> 回看视频 -> 写回图谱 -> 再决策”的完整闭环

因此后续主线不是继续微调提示词，而是继续把 agent 的环境、工具、记忆和执行循环做完整。
