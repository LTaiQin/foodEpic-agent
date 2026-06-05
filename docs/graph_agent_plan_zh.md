# HD-EPIC Food Graph Agent 计划

## 1. 目标重定义

当前系统的主要问题不是“模型不够大”，而是系统形态仍然接近：

- 给模型一组图片
- 给模型一段整理后的文字
- 给模型一个问题
- 让模型直接回答

这本质上还是一个“证据增强问答器”，不是一个真正的 agent。

本项目下一阶段的目标应当改为：

**构建一个面向做饭视频的图谱检索式 agent，使模型不再被动接收整理好的内容，而是主动调用工具，从外部记忆和原始数据中检索证据，再完成回答。**

---

## 2. 核心原则

### 2.1 不主动把内容喂给大模型

后续系统必须严格遵守以下原则：

- 不主动把大段帧描述、视频摘要、题目相关信息直接拼成 prompt 送给模型
- 不主动把人工整理好的“答案候选证据包”喂给模型
- 不把“已经知道哪些帧相关”这件事预先替模型决定好

模型应当只得到：

- 问题本身
- 可用工具列表
- 工具返回的结构化检索结果

也就是说，模型面对的不是“现成信息”，而是“可行动环境”。

### 2.2 必须保留原始数据访问能力

知识图谱不是替代原始数据，而是一个长期记忆层。

系统必须同时支持两类访问：

- **结构化记忆检索**
  - 从图谱中查对象、动作、事件、位置、状态、时间段
- **原始数据回查**
  - 从视频重新抽帧
  - 对局部区域做放大
  - 对显示读数做 OCR
  - 对候选时间段重新观察

也就是说，图谱只负责“组织记忆、缩小搜索空间、提供线索”，不能把系统锁死成只能读图谱文本。

### 2.3 图谱中只能存评测允许的信息

允许进入图谱的信息：

- 原视频
- 音频
- 系统自己抽取的关键帧
- 局部区域图
- OCR 结果
- 自己从可见证据中抽取出的对象、动作、状态、事件

不允许进入图谱的信息：

- benchmark 标准答案
- 人工标注的隐藏真值
- 任何只在评测标注侧存在、但测试时不允许直接观察到的信息

---

## 3. 最终系统形态

未来的系统不应再是：

`问题 -> prompt -> 大模型 -> 答案`

而应当是：

`问题 -> 规划 -> 检索图谱 -> 必要时访问原始视频/图片 -> 追加证据 -> 汇总回答`

建议抽象成以下闭环：

1. 问题解析
2. 生成检索计划
3. 查询知识图谱
4. 判断证据是否足够
5. 若不足，则调用原始数据工具补证据
6. 更新工作记忆
7. 输出答案与证据链

---

## 4. Agent 的能力边界

### 4.1 Agent 输入

- 一个视频级问题
- 多项选择选项
- 当前视频 ID
- 可用工具说明

### 4.2 Agent 输出

- 最终答案
- 所用证据列表
- 工具调用轨迹
- 可选的中间决策摘要

### 4.3 Agent 不应直接拥有的内容

- 全量帧描述拼接文本
- 预先为某个问题精心整理好的总结
- “已经选好的关键帧集合”

这些都应当变成 agent 可检索、可请求、可追加获取的外部资源。

---

## 5. 图谱记忆设计

第一版不建议先上重型图数据库，优先做轻量可控实现：

- `SQLite`
- `JSONL`
- 可选向量索引

### 5.1 节点类型

- `Video`
- `Segment`
- `Frame`
- `Region`
- `Object`
- `Ingredient`
- `Tool`
- `Action`
- `Location`
- `State`
- `RecipeStep`
- `AudioEvent`
- `OCRReading`
- `Observation`

### 5.2 边类型

- `contains`
- `occurs_at`
- `before`
- `after`
- `appears_in`
- `same_object_as`
- `held_by`
- `moves_to`
- `located_at`
- `changes_to`
- `supports`
- `part_of_step`

### 5.3 第一版建议的最小信息单元

最关键的不是“图数据库很酷”，而是让记忆可查、可回溯。

每条事件建议至少保存：

- `video_id`
- `time_start`
- `time_end`
- `source_type`
  - frame
  - region
  - audio
  - ocr
- `objects`
- `actions`
- `locations`
- `state_changes`
- `evidence_paths`
- `confidence`

---

## 6. 原始数据与图谱的关系

### 6.1 图谱不是原始数据替身

图谱里只放“摘要后的结构化记忆”。

例如：

- 在 `00:16:01-00:16:14` 发生了食材加入事件
- 某个物体在 `00:32:16` 出现在某个位置
- 某个时间段出现秤读数候选

但真正的视觉证据仍然应该是：

- 原始帧
- 原始局部放大图
- OCR 区域图

### 6.2 必须支持“从图谱反查原始数据”

每个图谱节点都应该能追溯到：

- 对应视频
- 对应时间段
- 对应帧路径
- 对应 region 路径

否则后面 agent 做不了真正的证据确认。

---

## 7. 工具系统设计

后续系统的重点不是继续写大 prompt，而是把能力拆成工具。

### 7.1 图谱工具

- `query_graph_by_time`
- `query_graph_by_object`
- `query_graph_by_action`
- `query_graph_by_location`
- `query_graph_by_state`
- `query_graph_by_keyword`

### 7.2 原始数据工具

- `extract_frame_at_time`
- `extract_frames_for_segment`
- `extract_region_from_bbox`
- `render_bbox_overlay`
- `render_focus_crop`
- `run_ocr_on_region`
- `expand_time_window`

### 7.3 视频理解工具

- `track_object_candidate`
- `compare_candidate_segments`
- `build_local_timeline`
- `build_motion_trace`
- `summarize_segment`

### 7.4 决策工具

- `select_relevant_evidence`
- `rank_candidate_options`
- `decide_need_more_evidence`

---

## 8. Agent 运行机制

### 8.1 问题解析层

先识别问题属于哪一类：

- 动作识别
- 时间定位
- 空间位置
- 视线估计
- 运动轨迹
- 静止判断
- 食材事件
- 读数/OCR
- 配方步骤

### 8.2 规划层

模型不直接回答，而是先输出一个行动计划：

示例：

```json
{
  "question_type": "object_motion_itinerary",
  "goal": "find start, intermediate and end locations",
  "tool_plan": [
    {"tool": "query_graph_by_object", "args": {"object_hint": "small bottle"}},
    {"tool": "build_motion_trace", "args": {"time_range": ["00:32:10", "00:32:30"]}},
    {"tool": "rank_candidate_options", "args": {"choices": [0,1,2,3,4]}}
  ]
}
```

### 8.3 执行层

执行器逐步调用工具，并维护一个工作记忆：

- 当前已知对象
- 当前已知时间段
- 当前已知地点
- 已调取的帧和局部图
- 当前证据是否足够

### 8.4 回答层

只有当工作记忆中的证据足够时，模型才被允许输出最终答案。

---

## 9. 针对当前任务的专项思路

### 9.1 Motion / Itinerary

这类题最适合做图谱式 agent。

建议流程：

1. 根据问题中的时间和 bbox 建立目标候选
2. 图谱中检索与目标相关的事件和位置变化
3. 若不够，主动回查原始帧并补做目标追踪
4. 构建简化路径
5. 将路径映射到选项中的地点词表

### 9.2 Stationary Localization

建议流程：

1. 读取候选开始时间
2. 针对每个候选开始时间，在后续多个检查点查询图谱
3. 如果图谱缺失，再从原视频抽检查帧
4. 判断该对象位置是否长期不变

### 9.3 Ingredient / Nutrition Change

建议流程：

1. 图谱中检索时间段内新增食材事件
2. 从新增事件关联营养变化
3. 若证据不足，再查对应帧或包装/OCR 信息

### 9.4 Weight / Scale Reading

这类题不适合只靠通用问答。

建议流程：

1. 检索秤出现的时间段
2. 对秤显示区域主动放大
3. 调用 OCR / 数字读数工具
4. 把读数结果写回图谱
5. 再回答问题

---

## 10. 第一阶段实现重点

为了避免再次退回“继续调 prompt”的路线，第一阶段只做以下内容：

### 阶段 1A：图谱 Schema

产物：

- `docs/graph_schema.md`
- `food_agent/memory/schema.py`

### 阶段 1B：单视频图谱构建器

产物：

- `food_agent/graph/build_video_graph.py`
- `food_agent/graph/extract_events.py`
- `food_agent/graph/extract_regions.py`

### 阶段 1C：基础检索器

产物：

- `food_agent/retrieval/time_retriever.py`
- `food_agent/retrieval/object_retriever.py`
- `food_agent/retrieval/event_retriever.py`

### 阶段 1D：最小 Agent Loop

产物：

- `food_agent/agent/planner.py`
- `food_agent/agent/executor.py`
- `food_agent/agent/state.py`

第一阶段先不要追求全部题型。

优先只支持 3 类：

- `ingredient_ingredient_retrieval`
- `object_motion_object_movement_itinerary`
- `recipe_multi_step_localization`

因为这三类最能体现：

- 长时记忆
- 主动检索
- 工具调用
- 证据驱动回答

---

## 11. 第二阶段实现重点

在最小 agent 跑通之后，再扩展：

- `stationary localization`
- `gaze estimation`
- `fixture interaction counting`
- `scale reading`

这一阶段的核心不是更多 prompt，而是更多专用工具。

---

## 12. 评测方式重做

未来评测不能只看“答对没答对”，还应记录：

- 工具调用次数
- 图谱检索命中率
- 补帧次数
- 原始数据回查次数
- 平均每题耗时
- 平均每题成本
- 不同题型的失败原因

建议后续固定比较以下四个版本：

- `Direct VLM`
- `Prompted VLM`
- `Retriever-Only`
- `Graph Agent`

只有这样，才能真正证明“agent 有价值”，而不是“只是换了一种提示词”。

---

## 13. 与当前系统的关系

当前已有内容不需要推翻，但要重新定位：

- 当前关键帧抽取模块
  - 作为图谱构建输入
- 当前局部放大 / bbox overlay
  - 作为原始数据工具
- 当前视频级记忆
  - 作为图谱初始摘要来源
- 当前按题补帧
  - 后续改造成工具，而不是主流程硬编码

也就是说，现有代码里的有用部分要“工具化”，而不是继续堆成单体脚本。

---

## 14. 明确不再做的事情

从现在开始，以下方向不应再作为主线：

- 单纯继续堆 prompt
- 单纯继续给模型更多整理好的文字
- 单纯继续增加“让模型看更多图”
- 单纯继续靠重试和投票追分

这些可以作为对照或临时 baseline，但不应再是主系统路线。

---

## 15. 下一步执行建议

建议下一步直接开始如下落地顺序：

1. 新建图谱 schema 文档
2. 新建图谱构建目录
3. 把单视频关键帧与事件写入图谱
4. 实现最小检索器
5. 实现最小 planner/executor
6. 先在一个视频上跑通 3 类核心题

如果后续实施时发现某类题仍需视觉细模块，例如秤读数或小文本 OCR，那么把它们作为工具子模块补进去，而不是重新退回“大模型直接看图回答”的路线。
