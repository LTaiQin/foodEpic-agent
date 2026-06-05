# foodEpic-agent 多文档实施计划索引

## 项目目标

本项目基于 HD-EPIC 数据集和 LightAgent baseline，构建一个面向第一视角厨房视频的 food-process agent。系统重点不是泛化的视频聊天，而是可证据追溯的厨房过程理解：菜谱步骤、食材加入、称重、营养变化、物体移动、gaze/hand/audio 线索、异常检测和长时序问答。

核心研究问题：

- 结构化 food-process memory 是否比 TextOnly、FrameOnly、普通 RAG 和原始 LightAgent 更适合长视频厨房问答。
- 证据约束是否能降低 agent 幻觉，并提升答案可核查性。
- 状态化 recipe/ingredient memory 是否能支持真实应用中的饮食记录、营养追踪和步骤检查。
- LightAgent 的通用 tool-use 在长视频、多模态、强证据约束任务中有什么瓶颈。

核心应用价值：

- 自动记录做饭过程。
- 追踪食材加入、称重和营养变化。
- 回答厨房过程问题并返回证据时间段。
- 检测漏加、重复加、顺序异常、步骤耗时异常。
- 为烹饪教学、饮食健康管理、家庭厨房复盘和辅助记忆提供基础系统。

## 文档结构

- [01_project_scope_zh.md](01_project_scope_zh.md)：项目边界、研究价值、应用价值和不做什么。
- [02_data_layer_plan_zh.md](02_data_layer_plan_zh.md)：HD-EPIC 数据读取、manifest、格式检查、事件索引。
- [03_lightagent_baseline_plan_zh.md](03_lightagent_baseline_plan_zh.md)：LightAgent baseline、wrapper、工具注册、trace 和公平性控制。
- [04_food_memory_tasks_zh.md](04_food_memory_tasks_zh.md)：recipe、ingredient、nutrition、object、gaze/hand/audio、异常检测任务设计。
- [05_experiments_eval_plan_zh.md](05_experiments_eval_plan_zh.md)：baseline 矩阵、指标、ablation、错误分析和论文实验。
- [06_implementation_phases_zh.md](06_implementation_phases_zh.md)：逐阶段代码实现计划、模块、脚本和完成标准。
- [07_repo_ops_and_commit_zh.md](07_repo_ops_and_commit_zh.md)：GitHub 绑定、SSH、自动验证提交、数据不入库规范。
- [08_risks_and_decisions_zh.md](08_risks_and_decisions_zh.md)：风险、决策记录、暂缓方向和后续扩展路线。
- [10_graph_tool_agent_execution_plan_zh.md](10_graph_tool_agent_execution_plan_zh.md)：工具驱动、图谱记忆、原始视频可回查的真实 agent 实施主线。
- [11_tool_driven_graph_agent_architecture_zh.md](11_tool_driven_graph_agent_architecture_zh.md)：图谱只做外部记忆层、原始视频始终可回查、LLM 通过工具自主检索证据的主架构约束。

## 当前本地状态

- 本地项目路径：`/22liushoulong/agent/hd-epic`
- GitHub 仓库：`git@github.com:LTaiQin/foodEpic-agent.git`
- 数据根目录：`/22liushoulong/agent/hd-epic/data/HD-EPIC`
- 标注根目录：`/22liushoulong/agent/hd-epic/annotations/hd-epic-annotations-main`
- 本地数据体积：约 `571G`
- 重要约束：`data/`、`annotations/`、`.secrets/`、`outputs/` 不进入 git。

## 实施顺序

1. Phase 0：仓库、环境、数据完整性检查。
2. Phase 1：统一数据加载与 manifest。
3. Phase 2：DuckDB/Parquet 事件索引。
4. Phase 3：LightAgent baseline 和 HD-EPIC tools。
5. Phase 4：recipe/ingredient/nutrition 主线 agent。
6. Phase 5：VQA benchmark 评估和 ablation。
7. Phase 6：object/3D/gaze-hand/audio 扩展。
8. Phase 7：demo、报告、论文实验整理。

## 每阶段完成标准

每个阶段完成必须满足：

- 有明确代码或文档交付物。
- 有验证命令和验证结果。
- 有 git commit。
- 如涉及实验，保存 config、结果和 trace。
- 如失败，记录失败原因、影响范围和下一步修复方案。
