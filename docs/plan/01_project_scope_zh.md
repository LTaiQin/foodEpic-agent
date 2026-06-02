# 01 项目范围、研究意义与应用价值

## 1.1 项目定义

foodEpic-agent 是一个基于第一视角厨房视频的长时序感知、记忆和推理 agent。它不是机器人控制系统，也不是通用视频聊天系统，而是围绕厨房过程建立结构化记忆，并在问答或状态追踪时返回可核查证据。

系统输入：

- HD-EPIC 视频、音频、recipe、ingredient、VQA、object movement、gaze、hand、SLAM、Digital Twin 等数据。
- 用户问题，例如“刚才加了什么食材？”、“现在做到哪一步？”、“这个杯子被移动了几次？”。
- 离线评估样本，例如 HD-EPIC VQA benchmark。

系统输出：

- 答案。
- 证据时间段、帧号、事件 id、recipe step、ingredient state 或 object track。
- LightAgent trace、工具调用记录、失败类型。

## 1.2 研究意义

本项目的研究价值不在于单独提高动作分类准确率，而在于把长视频多模态信息组织为可被 agent 使用的 food-process memory。

核心研究点：

- 长时序视频中的结构化过程记忆。
- 多模态事件对齐：视频、音频、recipe、object、gaze、hand、3D。
- 证据约束 agent：回答必须能回溯到具体事件和时间段。
- tool-use agent 在长视频任务中的能力边界。
- 状态化 memory 与 LLM/VLM 推理的职责划分。
- participant-held-out 跨厨房泛化。

可以形成的论文问题：

- Structured food-process memory for egocentric kitchen QA。
- Evidence-grounded lightweight agent for long-horizon cooking videos。
- Tool-use failure analysis for multimodal process agents。
- Recipe and ingredient state tracking with evidence-constrained reasoning。

## 1.3 应用价值

优先应用方向：

- 做饭过程自动记录：识别做了什么菜，每一步何时发生。
- 食材和营养追踪：识别加了什么、称了多少、营养变化如何。
- 厨房过程问答：回答用户关于过去过程、当前状态、下一步动作的问题。
- 步骤检查：检测漏加、重复加、顺序异常和步骤耗时异常。
- 证据回放：把答案关联到可回放视频片段和事件。
- 家庭健康管理：为饮食记录、慢病管理、家庭照护提供数据基础。

## 1.4 不做什么

暂不做机器人控制：

- HD-EPIC 没有机器人 action。
- 没有力控、夹爪状态、动作执行反馈。
- 没有可交互环境。

暂不做纯动作分类刷榜：

- 如果动作分类不服务步骤追踪、食材状态或证据检索，应用价值不足。

暂不做全量点云建图主线：

- semidense 数据体积巨大。
- 初期 3D 只服务 object/fixture grounding，不展开全量点云。

暂不做泛泛视频聊天：

- 项目目标是可证据追溯的厨房过程 agent，而不是对任意视频做开放式闲聊。

## 1.5 任务进入标准

每个任务进入实施前必须回答：

- 研究问题是什么。
- 实际用户价值是什么。
- 用哪些 HD-EPIC 字段支撑。
- 评估指标是什么。
- 和 baseline 相比预期改进在哪里。
- 是否能返回证据。
- 是否能做 participant-held-out 评估。

不满足研究意义和应用价值的任务不进入主线。

