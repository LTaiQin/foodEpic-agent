# HD-EPIC Food Agent 中文详细计划

## 当前本地数据状态

- 数据根目录：`/22liushoulong/agent/hd-epic/data/HD-EPIC`
- 标注根目录：`/22liushoulong/agent/hd-epic/annotations/hd-epic-annotations-main`
- 当前数据体积：约 `571G`
- `semidense_points.csv` / `semidense_observations.csv` 已重新压缩回 `312` 个 `.csv.gz`
- 当前无未压缩的 `semidense_*.csv`

主要文件类型：

- `json`：1493 个
- `obj`：826 个
- `csv`：773 个
- `gz`：312 个
- `mp4`：156 个
- `jsonl`：153 个
- `blend`：18 个
- `hdf5`：9 个
- `pdf`：2 个

## 数据格式总览

### 1. 视频数据

路径结构：

```text
Videos/PXX/{video_id}.mp4
Videos/PXX/{video_id}_mp4_to_vrs_time_ns.csv
Videos/PXX/{video_id}_vrs_to_mp4_log.json
Videos/PXX/durations.txt
Videos/PXX/frames.txt
```

MP4 实测格式：

- 视频编码：H.264
- 分辨率：`1408 x 1408`
- 帧率：`30 fps`
- 音频编码：AAC

时间映射 CSV 字段：

- `mp4_time_ns`
- `relative_vrs_device_time_ns`
- `vrs_device_time_ns`

用途：

- Food agent 的主要视觉输入。
- 用于动作识别、菜谱步骤识别、物体状态判断、VQA 问答。
- `mp4_to_vrs_time_ns.csv` 用于把 MP4 时间和 Aria/VRS 传感器时间对齐。

### 2. Narration 与动作片段

路径：

```text
narrations-and-action-segments/HD_EPIC_Narrations.pkl
narrations-and-action-segments/HD_EPIC_verb_classes.csv
narrations-and-action-segments/HD_EPIC_noun_classes.csv
narrations-and-action-segments/HD_EPIC_Narrations_erratum.csv
```

`HD_EPIC_Narrations.pkl` 字段：

- `unique_narration_id`
- `participant_id`
- `video_id`
- `narration`
- `start_timestamp`
- `end_timestamp`
- `nouns`
- `verbs`
- `pairs`
- `main_actions`
- `verb_classes`
- `noun_classes`
- `pair_classes`
- `main_action_classes`
- `hands`
- `narration_timestamp`

当前注意事项：

- 当前 Python 环境读取该 pickle 失败，错误是 `No module named 'numpy._core.numeric'`。
- 后续需要建一个兼容 NumPy/Pandas 的环境，或者把 pickle 一次性转换成 Parquet/JSONL。

用途：

- 这是动作级事件记忆的核心来源，但它的价值不在于单纯训练“动作识别模型”。
- 它真正服务的是可落地任务：判断当前菜谱步骤、识别食材加入/称重/搅拌等关键操作、定位证据时间段、发现步骤遗漏或顺序错误。
- 可以把长视频切成可查询事件：动作文本、开始/结束时间、verb、noun、手部信息。
- 在 food agent 中，它应该作为“事件索引”和“证据检索”的监督信号，而不是孤立优化 action classification accuracy。

### 3. 高层活动、菜谱与营养

路径：

```text
high-level/activities/PXX_recipe_timestamps.csv
high-level/complete_recipes.json
```

活动 CSV 字段：

- `video_id`
- `recipe_id`
- `high_level_activity_label`
- `start_time`
- `end_time`

`complete_recipes.json` 结构：

- 顶层 key 是 recipe id，例如 `P01_R01`
- `participant`
- `name`
- `type`
- `source`
- `steps`
- `captures`
- `ingredients`
- `step_times`
- `prep_times`

Ingredient 字段：

- `name`
- `amount`
- `amount_unit`
- `calories`
- `carbs`
- `fat`
- `protein`
- `weigh`
- `add`

用途：

- 构建菜谱状态机。
- 跟踪当前做到哪一步。
- 跟踪哪些食材已经加入。
- 估计营养变化。
- 支持问题：“现在在做什么菜？”、“下一步是什么？”、“刚才加入了什么？”、“当前营养变化是多少？”

### 4. 音频数据与音频事件

路径：

```text
Audio-HDF5/PXX/PXX_audio.hdf5
audio-annotations/HD_EPIC_Sounds.csv
audio-annotations/HD_EPIC_Sounds.pkl
```

HDF5 结构：

- 每个 `video_id` 是一个 dataset。
- dataset 是一维 `float32` 音频波形。

音频事件 CSV 字段：

- `participant_id`
- `video_id`
- `start_timestamp`
- `stop_timestamp`
- `start_sample`
- `stop_sample`
- `class`
- `class_id`

注意：

- 音频采样率按标注说明为 `48 KHz`。

用途：

- 辅助判断视觉不清楚的动作，例如倒水、开关水龙头、切菜、搅拌、脚步、包装袋声音。
- 可以作为多模态 food agent 的音频证据。

### 5. 场景与物体移动标注

路径：

```text
scene-and-object-movements/assoc_info.json
scene-and-object-movements/mask_info.json
```

`assoc_info.json` 结构：

```json
{
  "video_id": {
    "association_id": {
      "name": "object name",
      "tracks": [
        {
          "track_id": "track id",
          "time_segment": [start_time, end_time],
          "masks": ["mask id"]
        }
      ]
    }
  }
}
```

`mask_info.json` 结构：

```json
{
  "video_id": {
    "mask_id": {
      "frame_number": 10725,
      "3d_location": [x, y, z],
      "bbox": [xmin, ymin, xmax, ymax],
      "fixture": "P01_counter.008"
    }
  }
}
```

用途：

- 追踪物体从哪里拿起、移动到哪里、放到哪个 fixture。
- 支持物体移动计数、物体位置问答、物体 itinerary 推理。
- 是构建 3D object memory 的关键来源。

### 6. Eye Gaze Priming

路径：

```text
eye-gaze-priming/priming_info.json
```

结构：

```json
{
  "video_id": {
    "object_id": {
      "start": {
        "frame": 283,
        "3d_location": [x, y, z],
        "prime_stats": {
          "prime_window_start": 0,
          "frame_primed": 177,
          "gaze_point": [x, y, z],
          "dist_to_cam": 2.31,
          "prime_gap": 3.53
        }
      },
      "end": {}
    }
  }
}
```

`frame_primed` 含义：

- `>= 0`：检测到 gaze priming 的帧。
- `-1`：位置有效，但没有发生 priming。
- `-2`：该样本被排除。

用途：

- 预测人接下来要拿哪个物体。
- 预测接下来要把物体放到哪里。
- 做 interaction anticipation。

### 7. Gaze 与 Hand Tracking 流

路径：

```text
SLAM-and-Gaze/PXX/GAZE_HAND/mps_{video_id}_vrs/eye_gaze/general_eye_gaze.csv
SLAM-and-Gaze/PXX/GAZE_HAND/mps_{video_id}_vrs/hand_tracking/wrist_and_palm_poses.csv
```

Eye gaze CSV 主要字段：

- `tracking_timestamp_us`
- `left_yaw_rads_cpf`
- `right_yaw_rads_cpf`
- `pitch_rads_cpf`
- `depth_m`
- low/high uncertainty range
- left/right eye position
- `session_uid`

Hand tracking CSV 主要字段：

- `tracking_timestamp_us`
- `left_tracking_confidence`
- `tx_left_wrist_device`
- `ty_left_wrist_device`
- `tz_left_wrist_device`
- `tx_left_palm_device`
- `ty_left_palm_device`
- `tz_left_palm_device`
- 右手对应字段

用途：

- 判断当前哪只手在操作。
- 预测短期意图。
- 辅助手-物交互检测。
- 与物体移动、gaze priming 结合，构建 action anticipation 模块。

### 8. SLAM 与几何数据

路径：

```text
SLAM-and-Gaze/PXX/SLAM/multi/{index}/slam/
```

常见文件：

- `closed_loop_trajectory.csv`
- `open_loop_trajectory.csv`
- `online_calibration.jsonl`
- `semidense_points.csv.gz`
- `semidense_observations.csv.gz`
- `summary.json`

Closed-loop trajectory 字段：

- `graph_uid`
- `tracking_timestamp_us`
- `utc_timestamp_ns`
- `tx_world_device`
- `ty_world_device`
- `tz_world_device`
- `qx_world_device`
- `qy_world_device`
- `qz_world_device`
- `qw_world_device`
- 线速度
- 角速度
- 重力
- `quality_score`

Open-loop trajectory 字段：

- `tracking_timestamp_us`
- `utc_timestamp_ns`
- `session_uid`
- odometry translation
- odometry quaternion
- velocity
- angular velocity
- gravity
- `quality_score`

Online calibration：

- JSONL 格式。
- 每行是一个 calibration snapshot。
- 包含 IMU、相机/传感器标定和坐标变换。

Semidense 文件：

- 当前保留为 `.csv.gz`。
- 读取方式：

```python
import pandas as pd
df = pd.read_csv(path, compression="gzip")
```

用途：

- 3D 空间记忆。
- 相机位姿对齐。
- fixture 和物体位置映射。
- gaze/object/hand 的世界坐标推理。

注意：

- `semidense_observations.csv.gz` 极大，建议只在确实需要 3D 点云观测时读取。
- food agent 初期不应该依赖全量 semidense CSV。

### 9. Digital Twin

路径：

```text
Digital-Twin/PXX_final.blend
Digital-Twin/meshes/PXX/*.obj
Digital-Twin/meshes/PXX/*.mtl
```

格式：

- Blender `.blend` 场景文件。
- OBJ/MTL mesh 文件。
- fixture 名称嵌入文件名，例如 `P01_counter.008.obj`。

用途：

- 构建厨房 fixture map。
- 给 object movement 的 fixture 字段提供几何语义。
- 支持 “杯子在哪个台面上？”、“这个物体应该放回哪里？” 这类 3D/场景问题。

### 10. Hand Masks

路径：

```text
Hands-Masks/contours_preds/{video_id}.json
Hands-Masks/contours_cleaned/
Hands-Masks/contours_memory/
```

观察到的格式：

- JSON dict。
- key 是帧号字符串。
- value 是该帧的 hand contour/mask 信息。
- 某些帧可能是空 dict `{}`。

用途：

- 手部分割监督。
- 裁剪 hand-centric visual features。
- 辅助手-物交互检测。

### 11. VQA Benchmark

路径：

```text
vqa-benchmark/*.json
```

每个 JSON 文件对应一种问题类型。每个样本包含：

- `inputs`
- `question`
- `choices`
- `correct_idx`
- `others`

任务类型包括：

- 细粒度动作识别/定位/how/why
- gaze estimation 与 interaction anticipation
- 食材识别、食材加入定位、食材顺序、食材重量
- 营养估计与营养变化
- 物体移动计数、轨迹、静止物体定位
- 菜谱识别、步骤识别、步骤定位
- 3D object/fixture perception

用途：

- 作为 food agent 的第一套离线评估基准。
- 不需要先自建 benchmark，可以直接报告各任务 family 的 accuracy。

## Food Agent 的目标定义

这里建议把目标定义为：

> 一个面向第一视角厨房视频的长时序感知-记忆-推理 agent。

它的应用价值应该优先围绕“厨房过程理解”和“可追溯记录”，而不是泛泛地回答视频问题。优先服务这些场景：

1. 菜谱执行记录：自动记录做了哪道菜、做到了哪一步、每一步何时发生。
2. 食材与营养追踪：识别食材加入、称重、用量变化，并估算营养变化。
3. 步骤检查：发现是否漏加食材、步骤顺序是否异常、是否重复操作。
4. 厨房过程问答：回答“刚才加了什么？”、“这一步什么时候完成？”、“现在应该做什么？”。
5. 证据回放：任何回答都能返回时间段、帧、动作事件或食材记录。
6. 行为预测：预测下一步可能拿什么、加什么、放到哪里。

因此它应该能做：

1. 输入一段厨房视频或一个时间点。
2. 建立动作、物体、食材、菜谱、营养、gaze、hand、audio、3D 位置的结构化记忆。
3. 回答和视频内容相关的问题。
4. 给出证据时间段、帧、物体或菜谱步骤。
5. 预测接下来可能发生的交互或菜谱步骤。

不建议把它定义成机器人控制 agent。

原因：

- HD-EPIC 没有机器人 action。
- 没有力控、夹爪状态、运动轨迹控制信号。
- 没有可交互环境。
- 它强在感知、理解、时序记忆和预测。

应用价值排序：

1. 高价值：菜谱步骤追踪、食材加入追踪、营养估计、证据检索、VQA 问答。
2. 中价值：物体位置记忆、gaze/hand 意图预测、3D fixture grounding。
3. 低价值或暂缓：纯动作分类榜单、全量 semidense 点云建图、机器人控制策略。

## 任务价值审查标准

后续所有任务必须同时满足两个条件：

1. 有研究意义：不是简单查表，而是能提出可验证的问题，例如长时序记忆、多模态对齐、证据可追溯、跨厨房泛化、意图预测、3D grounding。
2. 有应用价值：结果能服务真实厨房场景，例如做饭记录、食材/营养追踪、步骤检查、厨房助理问答、健康管理、烹饪教学或家庭安全。

不满足这两个条件的方向不作为主线：

- 只做普通动作分类，但不服务步骤、食材、错误检测或证据检索，暂缓。
- 只做全量点云建图，但不服务位置问答、物体状态或 3D grounding，暂缓。
- 只做 benchmark 刷分，但无法解释对 food agent 的实际能力提升，暂缓。
- 只做机器人控制，因为 HD-EPIC 没有动作执行数据和环境反馈支撑，排除。

每个任务进入实施前都要写清楚：

- 研究问题是什么。
- 实际用户价值是什么。
- 用哪些 HD-EPIC 数据字段支撑。
- 评估指标是什么。
- 和已有方法相比，创新点或可发表点在哪里。

## 候选任务的研究意义与应用价值

### 任务 A：可证据追溯的菜谱步骤追踪

研究问题：

- 第一视角长视频中，如何把局部动作、食材事件和高层 recipe step 对齐成可解释的长时序状态？
- LLM/VLM 是否能通过结构化事件记忆减少长视频推理中的遗漏和幻觉？

应用价值：

- 自动生成做饭过程记录。
- 告诉用户当前做到哪一步。
- 支持烹饪教学、复盘和家庭健康记录。

数据依据：

- `complete_recipes.json` 的 `steps`、`step_times`、`prep_times`
- `activities/PXX_recipe_timestamps.csv`
- `HD_EPIC_Narrations.pkl`
- MP4 视频片段
- recipe VQA

评估：

- step recognition accuracy
- step localization IoU / recall
- evidence time span recall
- participant-held-out 泛化

优先级：

- 最高。它是 food agent 的核心主线。

### 任务 B：食材加入、称重与营养状态追踪

研究问题：

- 如何从视频、动作事件和 recipe 标注中恢复食材状态变化？
- 能否建立“食材-时间-动作-营养”的可查询记忆，而不仅是回答单个图像问题？

应用价值：

- 自动记录用户吃了什么、加了多少、营养如何变化。
- 服务饮食管理、慢病管理、家庭厨房记录。
- 可用于提醒漏加、重复加、过量添加。

数据依据：

- `complete_recipes.json` 的 `ingredients`、`weigh`、`add`、nutrition 字段
- ingredient/nutrition VQA
- narration/action segments
- audio events
- MP4 clip frames

评估：

- ingredient add localization accuracy
- ingredient retrieval accuracy
- nutrition change multiple-choice accuracy
- ingredient timeline F1
- 错误检测准确率：漏加、重复加、顺序异常

优先级：

- 最高。应用价值明确，研究上也有长时序状态估计和多模态证据融合的问题。

### 任务 C：结构化记忆增强的厨房 VQA

研究问题：

- 长视频 VQA 是否应该先构建结构化 memory，再调用 VLM？
- 对哪些问题，结构化标注和检索比直接抽帧问 VLM 更可靠？

应用价值：

- 用户可以自然询问厨房过程。
- 系统能返回答案和证据，不只是给一个黑盒结果。
- 可以作为后续 agent 能力的统一评测入口。

数据依据：

- `vqa-benchmark/*.json`
- recipe/action/audio/object/gaze 事件库
- MP4 视频片段

评估：

- VQA multiple-choice accuracy
- 按任务族统计 accuracy
- evidence correctness
- answer-with-evidence rate
- structured-only / visual-only / hybrid ablation

优先级：

- 高。适合作为统一 benchmark，但不能只追求总准确率，必须分析 memory 对真实问题的贡献。

### 任务 D：物体状态与厨房 3D 位置记忆

研究问题：

- 第一视角视频中，如何长期维护物体的“拿起-移动-放置-所在 fixture”状态？
- 3D fixture grounding 是否能改善物体位置问答和物体 itinerary 推理？

应用价值：

- 回答“东西放哪了？”、“这个杯子从哪里拿到哪里？”。
- 支持厨房整理、辅助记忆、烹饪复盘。
- 对未来 embodied assistant 有中间层价值。

数据依据：

- `assoc_info.json`
- `mask_info.json`
- Digital Twin OBJ/BLEND
- SLAM trajectory
- object motion VQA
- 3D perception VQA

评估：

- object movement counting accuracy
- itinerary accuracy
- fixture localization accuracy
- 3D object location answer accuracy
- participant-held-out 泛化

优先级：

- 中高。研究意义强，但工程成本比 recipe/ingredient 高，应作为第二主线。

### 任务 E：gaze/hand 条件下的交互意图预测

研究问题：

- gaze priming、hand pose 和历史事件能否提前预测下一次物体交互？
- 结构化状态是否比只看视频帧更能预测“下一步拿什么/放哪”？

应用价值：

- 让厨房助手提前准备提示或辅助。
- 支持烹饪教学中的下一步提醒。
- 可用于安全预警，例如即将接触热锅、刀具、开水。

数据依据：

- `priming_info.json`
- `general_eye_gaze.csv`
- `wrist_and_palm_poses.csv`
- object movement tracks
- recipe state

评估：

- next object top-k accuracy
- pick-up anticipation time gap
- put-down location accuracy
- gaze-conditioned interaction anticipation VQA accuracy

优先级：

- 中。研究意义强，但应用价值依赖前面的 recipe/object memory 是否已经稳定。

### 任务 F：音频辅助的厨房事件识别

研究问题：

- 在视觉遮挡或手部遮挡时，音频事件能否补充识别关键厨房动作？
- 多模态事件融合是否改善食材加入、器具使用和步骤识别？

应用价值：

- 切菜、倒水、搅拌、开关设备等动作常有明显声音。
- 真实厨房中视觉经常被手、锅具、身体遮挡，音频能补充证据。

数据依据：

- `Audio-HDF5/PXX/PXX_audio.hdf5`
- `HD_EPIC_Sounds.csv`
- narration/action segments
- MP4 clips

评估：

- audio event classification accuracy
- action/step recognition 加入 audio 后的提升
- occlusion case performance

优先级：

- 中。适合作为 recipe/ingredient tracker 的增强模块，不建议单独作为主线。

### 任务 G：异常检测与步骤纠错

研究问题：

- 如何基于 recipe state 和事件记忆发现做饭过程中的异常？
- 仅使用演示数据，能否构建弱监督的步骤遗漏、重复、顺序异常检测？

应用价值：

- 提醒漏加、重复加、步骤顺序错误、等待时间过长。
- 面向家庭用户、老人辅助、健康饮食管理都有实际意义。

数据依据：

- recipe steps
- ingredient add/weigh segments
- action segments
- high-level activity timestamps
- VQA step localization

评估：

- 构造合成异常 benchmark：删除、交换、重复事件。
- anomaly detection precision/recall
- correction suggestion accuracy
- 证据时间段准确率

优先级：

- 高，但应在 recipe/ingredient state tracker 之后做。

## 最终推荐研究主线

推荐把项目主线定义为：

> Evidence-Grounded Food Process Agent: 基于 HD-EPIC 的可证据追溯厨房过程记忆与推理。

核心贡献：

1. 一个把 recipe、ingredient、action、audio、object、gaze、3D 信息统一到事件记忆的 food-process memory。
2. 一个面向食材、菜谱、营养、物体状态的 evidence-grounded agent。
3. 一个系统评估：结构化记忆是否优于直接长视频 VLM，在哪些任务上有效，哪里失败。
4. 一个应用导向的异常检测任务：漏加、重复、顺序异常、证据回放。

这条主线同时有研究意义和应用价值：

- 研究意义：长视频多模态记忆、时序状态跟踪、证据可追溯、跨厨房泛化、agent tool-use。
- 应用价值：厨房过程记录、饮食营养管理、步骤提醒、烹饪问答、家庭辅助。

## LightAgent Baseline 与优化路线

本项目以本地 LightAgent 作为 agent runtime baseline：

```text
/22liushoulong/agent/agent-context-isolation/third_party/LightAgent
```

LightAgent 的可用接口：

```python
agent.run(
    query: str,
    tools: list | None = None,
    history: list | None = None,
    user_id: str = "default_user",
    use_skills: bool = True,
    result_format: str = "str",
    trace: bool = False,
)
```

选择 LightAgent 的理由：

- 接口轻量，适合作为可控 baseline。
- 支持显式传入 `history`，便于比较不同记忆策略。
- 支持 Python 工具注册，适合接入 HD-EPIC 数据查询工具。
- 支持 trace，便于分析 agent 是否真的调用了证据工具。

Baseline 设计不能直接把 HD-EPIC 标注塞进 prompt。必须分层比较：

1. `LightAgent-TextOnly`：只给问题和少量上下文，不接数据工具。
2. `LightAgent-FrameOnly`：给抽样视频帧或 clip 描述，不接结构化 memory。
3. `LightAgent-RAG`：接入结构化事件检索工具。
4. `LightAgent-FoodMemory`：接入 recipe、ingredient、object、gaze、audio 多工具 memory。
5. `Ours`：在 LightAgent wrapper 上加入任务规划、证据约束、状态跟踪和错误检测。

这样可以回答研究问题：

- 结构化 food memory 相比直接 VLM/LLM 到底提升了哪些任务？
- 哪些任务依赖视觉，哪些任务依赖 recipe/ingredient memory？
- LightAgent 的通用 tool-use 能力在长视频厨房任务上有什么瓶颈？
- 加入 evidence constraint 后，答案是否更可靠、更可解释？

### LightAgent 改造原则

本项目把 LightAgent 作为明确 baseline，而不是只把它当作普通 LLM 调用器。

基线版本必须冻结：

- `LightAgent-Original`：不接 HD-EPIC 工具，只验证原始 LightAgent 在厨房问答上的能力边界。
- `LightAgent+HDTools`：只接结构化查询工具，不加额外任务路由和证据约束。
- `Ours-LightAgent`：在 LightAgent 外层加入 food-process memory、任务路由、证据约束和状态跟踪。

这样做的意义是把改进来源拆开：到底是 LightAgent 自身 tool-use 有用，还是 HD-EPIC 结构化 memory 有用，还是本文提出的 food-process evidence policy 有用。

第一阶段不直接改 LightAgent 内核。

采用 wrapper：

```text
user query
  -> FoodTaskRouter
  -> FoodMemoryRetriever
  -> selected history / selected tools / selected evidence
  -> EvidencePolicy
  -> LightAgent.run(query, history=..., tools=..., trace=True, result_format="object")
  -> AnswerVerifier
```

原因：

- 保持原始 LightAgent baseline 可复现。
- 修改点清晰，方便 ablation。
- 后续如果 LightAgent 内部限制明显，再做最小侵入式 patch。

必须注意 LightAgent 的一个实验公平性问题：

- 当前 LightAgent 初始化时会自动注册内置工具，包括 `execute_python_code`、`execute_python_file`、`execute_python_code_stream`、`upload_file_to_oss`。
- 如果 `run(tools=None)`，模型仍然可能看到这些默认工具。
- 因此 `LightAgent-TextOnly` 不能简单地“不传工具”，需要 wrapper 或 subclass 显式屏蔽默认工具。
- 对照实验中必须记录每次实际暴露给模型的 tool list，不能只记录配置名。

初期不使用 LightAgent 内置 memory 自动注入。

原因：

- 内置 memory 会自动把检索结果拼进 query，难以控制证据来源。
- 本项目需要比较 structured memory、visual memory、audio memory、object memory 的贡献。
- 应由 `FoodMemoryRetriever` 显式选择证据，再通过 wrapper 注入。

### 需要新增的 HD-EPIC 工具

所有工具都按照 LightAgent `tool_info` 格式注册。

核心工具：

- `get_video_metadata(video_id)`：返回视频路径、时长、帧数、participant。
- `retrieve_events(video_id, start_time, end_time, event_types)`：检索动作、recipe、audio、object、gaze 事件。
- `get_recipe_state(video_id, time)`：返回当前 recipe、step、已完成步骤。
- `get_ingredient_state(video_id, time)`：返回已加入食材、称重、营养变化。
- `get_object_state(video_id, object_name, time)`：返回物体位置、fixture、movement history。
- `get_gaze_hand_context(video_id, time)`：返回 gaze priming 和 hand pose 摘要。
- `get_audio_events(video_id, start_time, end_time)`：返回声音事件。
- `sample_video_frames(video_id, start_time, end_time, fps)`：抽帧给 VLM 或保存证据帧。
- `answer_vqa_with_evidence(vqa_id)`：运行官方 VQA 样本并返回答案、证据和 trace。

这些工具的研究价值：

- 把 agent 从“直接看长视频”改成“先查证据，再推理”。
- 可以量化每个工具对不同任务族的贡献。
- 可以分析 tool-use 失败模式：没检索、检索错、证据对但推理错、视觉证据不足。

应用价值：

- 用户问答时可以返回具体证据。
- 系统能记录 recipe/ingredient 状态，而不是只生成一句描述。
- 后续可以做步骤提醒、错误检测和营养记录。

### LightAgent 优化点

优化 1：任务路由。

- 判断问题属于 recipe、ingredient、nutrition、object、gaze、audio、3D、general VQA。
- 根据任务类型动态选择工具。
- 研究意义：减少无关工具调用，提高长视频任务的 tool-use 精度。
- 应用价值：用户问题响应更快，证据更聚焦。
- 和 LightAgent baseline 的关系：原始 LightAgent 让模型自己在全部工具中选择；本项目先由 `FoodTaskRouter` 缩小候选工具集合，再交给 LightAgent 调用。

优化 2：证据约束回答。

- 要求 LightAgent 每次回答必须引用事件 id、时间段或帧。
- 无证据时返回不确定，而不是猜。
- 研究意义：降低长视频问答幻觉。
- 应用价值：用户可以回放和核查。
- 和 LightAgent baseline 的关系：不修改 LightAgent 生成逻辑，而是在 wrapper 中把证据格式写入 query，并在 `AnswerVerifier` 中检查答案是否包含合法证据。

优化 3：状态化 food memory。

- 把 recipe、ingredient、object 状态维护成时间序列。
- LightAgent 只调用查询工具，不直接维护复杂状态。
- 研究意义：把 agentic reasoning 和状态估计分离，便于评估。
- 应用价值：支持“现在做到哪一步”、“刚才加了什么”、“漏了什么”。
- 和 LightAgent baseline 的关系：LightAgent 负责自然语言推理和工具调用，状态更新由确定性的 `FoodStateStore` 完成，避免让 LLM 在长上下文中隐式记忆状态。

优化 4：工具调用 trace 评估。

- 使用 `trace=True` 记录 LightAgent 的工具选择。
- 分析每个任务族的工具调用准确率。
- 研究意义：评估 agent 是否真正学会用 food memory。
- 应用价值：便于定位失败原因。
- 和 LightAgent baseline 的关系：保留 LightAgent trace 作为统一日志格式，用同一套 trace parser 对比 Original、HDTools、Ours。

优化 5：participant-held-out 泛化。

- 以参与者/厨房为单位划分训练和测试。
- 研究意义：验证跨厨房泛化，而不是记住 fixture 名称。
- 应用价值：面向真实家庭厨房部署。

优化 6：最小侵入式 LightAgent patch，只在必要时做。

- 允许新增 `disable_default_tools=True` 或通过 subclass 清空默认 registry。
- 允许让 `run()` 接收已经构造好的 system prompt 或 evidence block，但不能改变核心 tool dispatch 流程。
- 不改动模型请求、工具 schema、trace 结构，避免 baseline 与 ours 不可比。
- 研究意义：确保对照实验公平。
- 应用价值：保证系统部署时不会暴露无关工具或产生不可控行为。

### LightAgent 实验矩阵

| 模型/系统 | 工具 | Memory | 视觉 | 目标 |
|---|---|---|---|---|
| LightAgent-Original | 默认 LightAgent 工具关闭或记录 | 无 | 无 | 原始 agent 能力边界 |
| LightAgent-TextOnly | 无 | 无 | 无 | 最弱文本 baseline |
| LightAgent-FrameOnly | 抽帧/clip 描述 | 无 | 有 | 测直接视觉能力 |
| LightAgent+HDTools | HD-EPIC 基础查询工具 | 无状态事件检索 | 可选 | 测工具接入收益 |
| LightAgent-RAG | 事件检索 | action/recipe/audio | 可选 | 测结构化检索增益 |
| LightAgent-FoodMemory | 多工具 | recipe/ingredient/object/gaze/audio | 可选 | 主 baseline |
| Ours-LightAgent+Evidence | 多工具 + 证据约束 + 状态跟踪 | food process memory | 可选 | 最终方法 |

主要评估：

- VQA accuracy by task family。
- evidence recall。
- 工具调用成功率。
- recipe/ingredient 状态跟踪 F1。
- 异常检测 precision/recall。
- participant-held-out accuracy。

### 第一版实现目标

第一版不要直接做完整 3D agent。

先实现：

1. `FoodAgentLightWrapper`
2. `HD-EPIC DuckDB/Parquet event index`
3. recipe/ingredient/audio/object 基础查询工具
4. VQA runner
5. LightAgent baseline 对比：
   - Original
   - TextOnly
   - HDTools
   - RAG
   - FoodMemory
   - FoodMemory + evidence constraint

第一版论文/项目问题：

> 结构化 food-process memory 是否能让轻量 agent 在长时序厨房问答、食材追踪和步骤推理上超过直接视觉/文本 baseline？

### 基于 LightAgent 的具体代码落点

建议新增项目目录：

```text
/22liushoulong/agent/hd-epic/food_agent/
```

核心模块：

- `lightagent_wrapper.py`：封装 LightAgent，负责禁用默认工具、注入 history、选择工具、开启 trace。
- `task_router.py`：把问题映射到 recipe、ingredient、nutrition、object、gaze、audio、3D、general VQA。
- `memory_retriever.py`：根据任务类型从 DuckDB/Parquet 取证据。
- `evidence_policy.py`：规定答案必须包含哪些证据字段，证据不足时输出不确定。
- `state_store.py`：维护 recipe、ingredient、nutrition、object 的时间序列状态。
- `hd_epic_tools.py`：实现 LightAgent `tool_info` 格式工具。
- `trace_eval.py`：解析 `result_format="object"` 返回的 trace，统计工具选择与证据使用。
- `run_lightagent_baselines.py`：统一运行 Original、TextOnly、HDTools、RAG、FoodMemory、Ours。

第一版 wrapper 行为：

```text
FoodAgentLightWrapper.run(question, sample)
  -> parse sample inputs: video_id / timestamp / choices / task_family
  -> FoodTaskRouter.select_tools(task_family)
  -> FoodMemoryRetriever.retrieve(video_id, time_window, task_family)
  -> EvidencePolicy.build_prompt(question, choices, evidence)
  -> LightAgent.run(..., tools=selected_tools, use_skills=False, trace=True, result_format="object")
  -> AnswerVerifier.check(answer, evidence)
  -> return answer / evidence / trace / failure_type
```

失败类型要显式分类：

- `no_retrieval`：没有检索到证据。
- `wrong_retrieval`：检索证据不包含正确答案。
- `wrong_tool`：LightAgent 选择了不相关工具。
- `reasoning_error`：证据正确但答案错误。
- `visual_missing`：结构化证据不足，需要视频帧或 VLM。
- `format_error`：没有按要求输出选项、证据或时间段。

这部分是论文可分析的核心，不只是工程日志。

## 推荐系统架构

### 第一层：数据索引层

先建立统一 manifest，核心 key：

- `participant_id`
- `video_id`
- `recipe_id`
- `start_time`
- `end_time`
- `frame_start`
- `frame_end`

建议构建这些表：

- `videos`
- `actions`
- `recipes`
- `ingredients`
- `audio_events`
- `object_tracks`
- `object_masks`
- `gaze_priming`
- `gaze_streams`
- `hand_streams`
- `slam_pose`
- `vqa_examples`

推荐存储：

- 小中型表：Parquet
- 需要 join 查询：DuckDB
- 大型 `semidense_*.csv.gz`：保留原文件，按需读取

### 第二层：事件记忆层

把所有模态转换成统一事件格式：

```json
{
  "event_id": "...",
  "video_id": "...",
  "time_start": 12.3,
  "time_end": 15.7,
  "event_type": "action|ingredient_add|object_move|sound|gaze_prime|recipe_step",
  "text": "...",
  "entities": {
    "verbs": [],
    "objects": [],
    "ingredients": [],
    "fixtures": [],
    "hands": []
  },
  "geometry": {
    "bbox": [],
    "world_xyz": [],
    "fixture": ""
  },
  "evidence": {
    "video_clip": "...",
    "annotation_source": "...",
    "frame_ids": []
  }
}
```

这是 food agent 的核心 memory substrate。

### 第三层：感知模型层

建议从 frozen encoder + retrieval 开始，不要一上来端到端大训练。

可用模型：

- 视频编码：InternVideo、VideoCLIP、Video-LLaVA、Qwen2.5-VL 类模型
- 图像/帧编码：CLIP、SigLIP
- 音频编码：BEATs、PANNs
- 文本编码：sentence-transformers
- 手/物体 crop：利用 hand masks 和 object bbox 做局部视觉特征

### 第四层：推理 agent 层

LLM/VLM 不应该直接盲看整个长视频，而应该调用工具查询结构化记忆。

建议工具接口：

- `retrieve_events(video_id, time_range, event_types)`
- `get_recipe_state(video_id, time)`
- `get_object_state(video_id, object_name, time)`
- `get_ingredient_state(recipe_id, time)`
- `get_gaze_context(video_id, time)`
- `get_audio_context(video_id, time)`
- `get_3d_location(video_id, object_or_fixture, time)`
- `sample_frames(video_id, start, end, fps)`
- `answer_vqa(example_id)`

推理流程：

1. 先查结构化事件。
2. 如果证据不足，再采样视频帧或音频。
3. 输出答案。
4. 同时输出证据来源：时间段、事件、帧、物体、菜谱步骤。

## 分阶段实施计划

### Phase 0：数据加载与索引

目标：

- 让整个 HD-EPIC 数据集可查询。

任务：

- 写 dataset loader。
- 支持 CSV、JSON、JSONL、HDF5、MP4 metadata、gaze/hand CSV、SLAM trajectory、VQA。
- 解决 `HD_EPIC_Narrations.pkl` 的环境兼容问题。
- 构建 `dataset_manifest.parquet`。
- 构建数据完整性检查。

检查项：

- 156 个 MP4。
- 9 个 HDF5。
- 312 个 semidense `.csv.gz`。
- 0 个未压缩 semidense 大 CSV。

交付物：

- `food_agent/loaders.py`
- `food_agent/data_index.py`
- `outputs/dataset_manifest.parquet`
- `outputs/data_format_report.md`

### Phase 1：结构化事件数据库

目标：

- 把多模态标注融合成一个时间轴事件库。

任务：

- 转换 recipe、activity、audio、object movement、gaze priming、VQA。
- 转换 narration/action segments。
- 将所有事件对齐到 MP4 秒。
- 建立 DuckDB 数据库。

建议表：

- `videos`
- `events`
- `objects`
- `recipes`
- `ingredients`
- `vqa`
- `gaze`
- `hands`
- `slam_pose`

交付物：

- `outputs/food_agent.duckdb`
- 统一事件查询 API。

### Phase 2：Retrieval Baseline Agent

目标：

- 先做一个能跑官方 VQA 的 baseline agent，但重点不是刷通用 VQA 分数，而是验证“结构化记忆 + 证据检索”是否能支撑实用厨房问答。

流程：

1. 解析 VQA 输入。
2. 根据 `video_id` 和时间段检索相关事件。
3. 检索菜谱、物体、食材、gaze、audio 上下文。
4. 必要时采样视频帧。
5. 调用 LLM/VLM 做多选回答。
6. 输出答案和证据。

评估：

- 官方 VQA 多选准确率。
- 按 task family 分组统计。
- 记录答案来源：结构化记忆、视觉帧、音频、混合。

交付物：

- `run_vqa_baseline.py`
- `vqa_results_by_task.json`

### Phase 3：Recipe / Ingredient Food Agent

目标：

- 先做 food agent 最核心、最稳的能力：菜谱和食材状态跟踪。
- 这是最高应用价值模块，应优先于纯动作识别和 3D-heavy 模块。

能力：

- 当前正在做哪个菜。
- 当前处于哪个 recipe step。
- 哪些食材已经加入。
- 食材加入时间。
- 食材称重时间。
- 营养变化估计。
- 下一步预测。
- 异常检查：漏加、重复加、顺序异常、步骤时间过长。

使用数据：

- `complete_recipes.json`
- `activities/PXX_recipe_timestamps.csv`
- narration/action segments
- ingredient/nutrition/recipe VQA
- MP4 clip frames

交付物：

- recipe state tracker
- ingredient timeline
- nutrition delta estimator
- recipe/ingredient VQA baseline

### Phase 4：Object / 3D Memory Agent

目标：

- 让 agent 能理解物体在哪里、移动到哪里、和哪个 fixture 相关。

能力：

- 查询物体当前或历史位置。
- 统计物体移动次数。
- 生成物体 movement itinerary。
- 判断物体所在 fixture。
- 预测 put-down location。

使用数据：

- `assoc_info.json`
- `mask_info.json`
- Digital Twin OBJ/BLEND
- SLAM trajectory
- gaze priming
- gaze/hand stream

交付物：

- object state graph
- fixture map
- object movement QA
- 3D object/fixture VQA baseline

### Phase 5：多模态预测

目标：

- 做 next interaction / next step prediction。

任务：

- 预测下一步 recipe step。
- 预测下一个要交互的物体。
- 预测 pick-up。
- 预测 put-down。
- 预测 gaze-conditioned interaction。

特征：

- 最近 N 个动作事件。
- 当前 recipe state。
- hand pose/confidence。
- gaze priming。
- object movement history。
- audio events。

交付物：

- next-step predictor
- next-object predictor
- interaction anticipation model

### Phase 6：Agent 使用接口

目标：

- 让 food agent 可以实际查询和演示。

建议接口：

```bash
ask --video-id P01-20240202-110250 --time 120 --question "What ingredient was just added?"
recipe-state --video-id P01-20240202-110250 --time 120
object-state --video-id P01-20240202-110250 --object "glass" --time 360
```

可选 notebook/dashboard：

- 视频时间轴。
- 事件轨道。
- 菜谱步骤。
- 食材状态。
- 物体位置。
- gaze/hand/audio 证据。

## 评估方案

第一阶段直接使用官方 VQA benchmark。

任务族：

- action recognition/localization/how/why
- recipe recognition/step recognition/step localization
- ingredient recognition/retrieval/order/weight/add localization
- nutrition estimation/change
- object movement counting/itinerary/localization
- gaze estimation/interaction anticipation
- 3D object/fixture perception

指标：

- multiple-choice accuracy
- task family accuracy
- evidence recall
- ingredient state tracking F1
- object movement tracking F1
- next-step top-k accuracy

推荐划分：

- 使用 participant-held-out split。
- 不要只随机切分视频，因为厨房 fixture 和参与者习惯会泄漏。

## 优先级建议

推荐顺序：

1. 数据加载和 manifest。
2. recipe/ingredient state tracker。
3. VQA retrieval baseline。
4. object movement graph。
5. gaze/hand anticipation。
6. SLAM/3D-heavy reasoning。

原因：

- Recipe/ingredient 任务最贴近 food agent。
- 标注直接、结构清晰、工程成本低。
- SLAM/semidense 数据很大，应该后置，先用 trajectory、fixture、object metadata 即可。
- Narration/action segments 应该优先用于事件切分、证据定位和 recipe state 更新，而不是先单独训练一个动作分类器。

## 主要风险

- `HD_EPIC_Narrations.pkl` 需要解决 NumPy/Pandas 兼容问题。
- `semidense_*.csv.gz` 很大，不应该频繁全量读取。
- object bbox 和 mask 可能存在不一致。
- 只有 9 个厨房，泛化评估必须谨慎。
- 数据集适合理解和推理，不适合直接训练机器人控制策略。

## 首轮实验建议

首轮不要做完整 3D agent，建议做：

> Recipe / Ingredient / Nutrition VQA Agent

输入：

- `complete_recipes.json`
- `activities/PXX_recipe_timestamps.csv`
- ingredient/nutrition/recipe VQA
- MP4 clip frames
- audio events

方法：

1. 从 VQA 样本拿到 `video_id` 和时间段。
2. 检索 recipe state 和 ingredient timeline。
3. 检索相关 action/audio/object events。
4. 必要时采样视频帧。
5. LLM/VLM 在五个选项中选择答案。
6. 输出答案和证据。

目标：

- 先跑通官方 VQA。
- 建立结构化 memory。
- 后续自然扩展到 object、gaze、3D reasoning。
- 验证应用价值：能否自动回答食材加入、步骤状态、营养变化和证据回放类问题。
