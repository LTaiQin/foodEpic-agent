# 04 Food Memory 核心任务计划

## 4.1 任务原则

每个任务必须同时满足：

- 有研究意义：涉及长时序记忆、多模态对齐、状态估计、证据约束、泛化或 tool-use。
- 有应用价值：能服务饮食记录、营养管理、烹饪教学、厨房问答、步骤检查或辅助记忆。

## 4.2 任务 A：菜谱步骤追踪

目标：

- 判断视频中正在做哪道菜。
- 判断当前处于哪个 recipe step。
- 定位每个步骤的开始和结束时间。
- 回答“现在做到哪一步？”、“下一步是什么？”。

数据来源：

- `complete_recipes.json`
- `activities/PXX_recipe_timestamps.csv`
- narration/action segments
- recipe VQA
- MP4 clip frames

方法：

- 从 recipe 标注构建 step timeline。
- 从 action/narration 构建局部动作事件。
- 用 `get_recipe_state(video_id, time)` 提供状态查询。
- 用 evidence policy 要求返回 step id 和时间段。

指标：

- step recognition accuracy。
- step localization IoU。
- evidence recall。
- participant-held-out accuracy。

应用价值：

- 自动生成做饭过程记录。
- 烹饪教学和复盘。
- 当前步骤提醒。

## 4.3 任务 B：食材加入、称重与营养状态追踪

目标：

- 判断哪些食材已经加入。
- 定位食材加入和称重时间。
- 估计当前营养变化。
- 回答“刚才加了什么？”、“加了多少？”、“当前营养变化是多少？”。

数据来源：

- `complete_recipes.json` 的 `ingredients`、`weigh`、`add`、nutrition 字段。
- ingredient/nutrition VQA。
- narration/action segments。
- audio events。
- MP4 clip frames。

方法：

- 构建 ingredient timeline。
- 对每个 ingredient 建立 `pending -> weighed -> added -> consumed_in_step` 状态。
- 用 recipe step 和 action event 校验加入顺序。
- nutrition delta 从已加入食材累加。

指标：

- ingredient add localization accuracy。
- ingredient timeline F1。
- nutrition QA accuracy。
- add/weigh event recall。
- anomaly detection precision/recall。

应用价值：

- 饮食记录。
- 营养管理。
- 慢病管理。
- 漏加/过量提醒。

## 4.4 任务 C：结构化记忆增强 VQA

目标：

- 在官方 VQA benchmark 上验证 food memory 的价值。
- 不只报告总准确率，而是按任务族分析结构化 memory 的贡献。

任务族：

- recipe recognition。
- step recognition/localization。
- ingredient identification。
- ingredient add/order/weight。
- nutrition estimation/change。
- object movement/location。
- gaze/interaction anticipation。
- audio-related event。

方法：

- 解析 VQA sample。
- 根据 sample task family 调用对应 retriever。
- 运行多个 baseline。
- 输出答案、证据和 trace。

指标：

- multiple-choice accuracy。
- answer-with-evidence rate。
- evidence correctness。
- tool selection accuracy。
- failure type distribution。

应用价值：

- 统一评估 food agent 是否能回答真实厨房过程问题。

## 4.5 任务 D：异常检测与步骤纠错

目标：

- 检测漏加、重复加、顺序异常、步骤耗时异常。
- 给出异常证据和可能修正建议。

数据来源：

- recipe step timeline。
- ingredient add/weigh timeline。
- action events。
- audio events。
- video frames。

异常类型：

- `missing_ingredient`
- `duplicate_ingredient`
- `wrong_order`
- `step_too_long`
- `unexpected_object_movement`
- `evidence_conflict`

方法：

- 从正常 recipe timeline 合成异常样本。
- 对比 observed timeline 和 expected recipe state。
- EvidencePolicy 要求输出缺失或冲突的证据段。

指标：

- precision。
- recall。
- F1。
- false positive rate。
- explanation correctness。

应用价值：

- 厨房步骤检查。
- 烹饪辅助提醒。
- 饮食记录质量控制。

## 4.6 任务 E：物体状态与厨房位置记忆

目标：

- 查询物体在哪里。
- 统计物体被移动几次。
- 生成物体 movement itinerary。
- 判断物体所在 fixture。

数据来源：

- `assoc_info.json`
- `mask_info.json`
- Digital Twin OBJ/BLEND。
- SLAM trajectory。
- gaze priming。

方法：

- 初期只用 object association、mask、fixture。
- 之后再接入 SLAM 和 Digital Twin。
- 不默认读取全量 semidense observations。

指标：

- object location accuracy。
- movement count accuracy。
- itinerary ordering accuracy。
- fixture grounding accuracy。

应用价值：

- 找物体。
- 物品归位提醒。
- 厨房辅助记忆。

## 4.7 任务 F：gaze/hand/audio 辅助意图预测

目标：

- 预测接下来可能拿什么物体。
- 判断当前动作是否发生在手-物交互区域。
- 用音频补充视觉不清楚的事件。

数据来源：

- gaze priming。
- `general_eye_gaze.csv`
- `wrist_and_palm_poses.csv`
- audio events。
- object tracks。

方法：

- gaze -> object priming。
- hand pose -> interaction window。
- audio event -> action cue。
- 与 recipe/ingredient state 结合预测下一步。

指标：

- next-object prediction accuracy。
- interaction anticipation accuracy。
- audio-assisted event accuracy。

应用价值：

- 提前提醒下一步。
- 弱视觉条件下补充判断。
- 家庭厨房安全和辅助。

## 4.8 优先级

第一主线：

- 菜谱步骤追踪。
- 食材加入/称重。
- 营养状态追踪。
- 结构化记忆 VQA。

第二主线：

- 异常检测。
- 物体状态记忆。

第三主线：

- gaze/hand/audio 意图预测。
- 3D fixture grounding。

