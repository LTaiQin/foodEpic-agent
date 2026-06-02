# foodEpic-agent 分阶段详细实施计划

## 0. 项目定位

本项目以 HD-EPIC 为数据基础，以 LightAgent 为 agent baseline，目标不是做泛泛的视频聊天机器人，而是做一个面向第一视角厨房视频的 food-process agent。

核心问题：

- 能否把长视频中的动作、菜谱步骤、食材加入、营养变化、物体移动、gaze/hand/audio 线索组织成可查询的结构化过程记忆。
- 能否让轻量 agent 在回答厨房过程问题时先检索证据、再推理、最后返回可核查证据。
- 能否证明这种 food-process memory 相比 TextOnly、FrameOnly、普通 RAG、原始 LightAgent 更可靠。

应用目标：

- 自动生成做饭过程记录。
- 追踪食材加入、称重和营养变化。
- 回答“刚才加了什么”“现在做到哪一步”“下一步是什么”等厨房过程问题。
- 检测漏加、重复加、顺序异常、步骤耗时异常。
- 返回时间段、帧、事件 id、recipe step、ingredient state 等可回放证据。

不作为主线：

- 不做机器人控制，因为 HD-EPIC 没有机器人 action 和环境反馈。
- 不做纯动作分类刷榜，除非它服务步骤追踪、食材追踪或证据检索。
- 不做全量 3D 点云建图，除非它服务物体位置记忆或 3D grounding。

## 1. GitHub 绑定与协作规范

远端仓库：

```text
https://github.com/LTaiQin/foodEpic-agent.git
```

本地项目：

```text
/22liushoulong/agent/hd-epic
```

绑定策略：

- 本地目录初始化为 git 仓库。
- remote 命名为 `origin`，指向 `LTaiQin/foodEpic-agent`。
- 默认分支使用 `main`。
- `data/`、`outputs/`、大视频、压缩包、日志、DuckDB、Parquet 都不进入 git。
- 只提交代码、计划文档、配置、轻量元数据和实验脚本。

后续自动提交规则：

- 每次修改代码后必须先运行对应验证命令。
- 验证通过后自动 `git add -A && git commit -m "..."`
- 如果用户明确要求 push，再执行 `git push origin main`。
- 如果 push 需要 GitHub 认证但本机没有 token，则保留本地 commit，并提示用户配置认证。

推荐提交命令：

```bash
scripts/verify_and_commit.sh "commit message" pytest
```

没有测试时至少运行：

```bash
scripts/verify_and_commit.sh "commit message" python -m compileall food_agent scripts
```

## 2. 总体技术路线

系统结构：

```text
HD-EPIC raw data
  -> dataset loaders
  -> DuckDB/Parquet event index
  -> FoodMemoryRetriever
  -> FoodTaskRouter
  -> EvidencePolicy
  -> LightAgent.run(...)
  -> AnswerVerifier
  -> metrics + trace analysis
```

核心原则：

- LightAgent 是 baseline，先不修改内核。
- 原始 LightAgent、LightAgent+HDTools、Ours-LightAgent 必须分开评估。
- 不把所有标注直接塞进 prompt，必须通过工具和检索暴露证据。
- 每个答案必须尽量返回证据；证据不足时输出不确定。
- 每个任务必须说明研究意义和应用价值。

## 3. Phase 0：仓库、环境与数据完整性

目标：

- 建立可复现的代码仓库和基础工程结构。
- 确认 HD-EPIC 本地数据完整且不会被误提交。
- 解决 narration pickle 读取兼容问题。

任务：

- 初始化 git，绑定 GitHub remote。
- 创建 `.gitignore`，排除 `data/`、大文件、日志和生成物。
- 建立目录结构：

```text
food_agent/
  __init__.py
  config.py
  paths.py
  loaders.py
  data_index.py
  hd_epic_tools.py
  lightagent_wrapper.py
  task_router.py
  memory_retriever.py
  evidence_policy.py
  state_store.py
  answer_verifier.py
  trace_eval.py
scripts/
  build_manifest.py
  build_event_index.py
  run_lightagent_baselines.py
  verify_and_commit.sh
tests/
```

数据检查：

- MP4 数量应为 156。
- HDF5 数量应为 9。
- semidense `.csv.gz` 数量应为 312。
- 未压缩 `semidense_*.csv` 数量应为 0。
- 外层 zip 应不再残留，除非是轻量源码包。

交付物：

- `.gitignore`
- `food_agent_detailed_plan_zh.md`
- `scripts/verify_and_commit.sh`
- `outputs/data_format_report.md`

验证：

- `git status --ignored`
- `python -m compileall scripts`
- 数据完整性脚本输出 JSON report。

## 4. Phase 1：统一数据加载与 Manifest

目标：

- 让所有 HD-EPIC 数据格式可被统一发现、读取和检查。

输入数据：

- `Videos/PXX/*.mp4`
- `Audio-HDF5/PXX/*.hdf5`
- `high-level/complete_recipes.json`
- `high-level/activities/*.csv`
- `audio-annotations/*.csv`
- `scene-and-object-movements/*.json`
- `eye-gaze-priming/priming_info.json`
- `SLAM-and-Gaze/*/GAZE_HAND/*.csv`
- `SLAM-and-Gaze/*/SLAM/**/*.csv.gz`
- `Digital-Twin/**/*.obj`
- `Hands-Masks/**/*.json`
- `vqa-benchmark/*.json`

任务：

- 实现 `paths.py`，集中定义数据根目录和 annotation 根目录。
- 实现 `loaders.py`，按格式读取 JSON、CSV、JSONL、HDF5、pickle、视频 metadata。
- 实现 `build_manifest.py`，扫描所有文件，输出文件类型、大小、participant、video_id、数据域。
- 尝试建立兼容环境读取 `HD_EPIC_Narrations.pkl`，成功后转换为 Parquet/JSONL。
- 如果 pickle 仍不可读，先把 narration 标为 blocked，不阻塞 recipe/ingredient/object/audio 主线。

Manifest 字段：

- `path`
- `relative_path`
- `domain`
- `participant_id`
- `video_id`
- `file_type`
- `size_bytes`
- `row_count`
- `status`
- `notes`

研究意义：

- 为长视频多模态 agent 建立可复用的数据访问层。
- 避免实验时隐式依赖本地路径或手工查表。

应用价值：

- 后续部署到新厨房数据时，能快速检查数据是否可用。

验证：

- Manifest 覆盖所有非 VRS 数据域。
- 每类文件至少抽样读取 3 个样本。
- 输出 `outputs/dataset_manifest.parquet` 和 `outputs/data_format_report.md`。

## 5. Phase 2：事件索引与 Food Process Memory

目标：

- 把 HD-EPIC 的多模态标注转成统一时间轴事件库。

DuckDB 表设计：

- `videos`：video metadata、participant、duration、fps、frame_count。
- `recipe_steps`：recipe_id、step_id、step_text、start_time、end_time。
- `ingredients`：ingredient name、amount、unit、nutrition、add/weigh segment。
- `events`：通用事件表，包含 action、recipe、audio、object、gaze。
- `audio_events`：声音类别、开始结束时间、sample 范围。
- `object_tracks`：object name、track_id、time_segment、mask ids。
- `object_masks`：frame_number、bbox、3d_location、fixture。
- `gaze_priming`：object_id、frame、3d_location、prime stats。
- `vqa_samples`：benchmark 文件、question、choices、correct_idx、inputs。

统一事件字段：

- `event_id`
- `video_id`
- `participant_id`
- `event_type`
- `start_time`
- `end_time`
- `label`
- `text`
- `payload_json`
- `source_file`
- `evidence_ref`

任务：

- 实现 `data_index.py`。
- 实现 recipe/activity/audio/object/gaze/VQA 转换。
- 对齐 MP4 时间单位。
- 建立基础 SQL 查询接口。
- 先不展开 semidense 大 CSV，只保留路径和按需读取接口。

研究意义：

- 把长视频理解转为事件记忆检索和状态估计问题。
- 支持分析不同模态对 agent 推理的贡献。

应用价值：

- 用户问题可以定位到具体时间段、对象和菜谱步骤。

验证：

- 每个 video 至少能查到 metadata。
- recipe video 能查到步骤和食材。
- audio/object/gaze/VQA 表能按 video_id 查询。
- 随机 20 个 VQA sample 能解析出 video_id/time/context。

## 6. Phase 3：LightAgent Baseline 接入

目标：

- 使用 LightAgent 建立公平、可复现的 baseline。

Baseline：

- `LightAgent-Original`：原始 LightAgent，关闭或记录默认工具，不接 HD-EPIC。
- `LightAgent-TextOnly`：只给问题和选项，不给数据工具。
- `LightAgent-FrameOnly`：给抽帧或 clip caption，不给结构化 memory。
- `LightAgent+HDTools`：接 HD-EPIC 查询工具，但无任务路由和证据约束。
- `LightAgent-RAG`：检索事件证据后拼入 prompt。
- `LightAgent-FoodMemory`：接 recipe、ingredient、audio、object、gaze 多工具。
- `Ours-LightAgent+Evidence`：增加任务路由、状态 memory、证据约束、答案验证。

LightAgent 风险处理：

- LightAgent 默认注册 Python 和 OSS 内置工具，TextOnly 时必须显式屏蔽。
- 初期不使用 LightAgent 内置 memory，避免证据来源不可控。
- 所有实验保存 `trace=True` 的结构化 trace。
- 每次记录实际暴露给模型的工具列表。

任务：

- 实现 `lightagent_wrapper.py`。
- 实现 `hd_epic_tools.py` 的 LightAgent `tool_info`。
- 实现 `task_router.py`。
- 实现 `evidence_policy.py`。
- 实现 `answer_verifier.py`。
- 实现 `run_lightagent_baselines.py`。

工具：

- `get_video_metadata(video_id)`
- `retrieve_events(video_id, start_time, end_time, event_types)`
- `get_recipe_state(video_id, time)`
- `get_ingredient_state(video_id, time)`
- `get_object_state(video_id, object_name, time)`
- `get_gaze_hand_context(video_id, time)`
- `get_audio_events(video_id, start_time, end_time)`
- `sample_video_frames(video_id, start_time, end_time, fps)`
- `answer_vqa_with_evidence(vqa_id)`

研究意义：

- 验证轻量 agent 是否能通过结构化工具解决长视频厨房推理。
- 分析 tool-use、retrieval、evidence constraint 分别带来的收益。

应用价值：

- 为厨房助理问答、过程记录、证据回放建立可运行系统。

验证：

- 至少跑通 50 个 VQA sample。
- 输出每个 baseline 的 accuracy、evidence rate、tool success rate。
- 保存错误案例和 trace。

## 7. Phase 4：Recipe / Ingredient / Nutrition 主线

目标：

- 优先实现应用价值最高的 food agent 能力。

能力：

- 判断当前正在做哪道菜。
- 判断当前处于哪个 recipe step。
- 查询某个时间点之前已加入哪些食材。
- 查询食材加入和称重时间。
- 估算当前营养变化。
- 判断是否漏加、重复加或顺序异常。

任务：

- 实现 `state_store.py` 的 recipe state tracker。
- 实现 ingredient timeline。
- 实现 nutrition delta estimator。
- 基于 VQA 和 recipe 标注构建评估脚本。
- 建立异常合成规则：漏加、重复加、顺序交换、步骤过长。

指标：

- step recognition accuracy。
- step localization IoU / recall。
- ingredient add localization accuracy。
- ingredient timeline F1。
- nutrition QA accuracy。
- anomaly detection precision/recall。
- evidence correctness。

研究意义：

- 长时序状态跟踪、多模态证据融合、可解释 agent 推理。

应用价值：

- 饮食记录、营养管理、烹饪指导、健康管理和家庭厨房复盘。

验证：

- participant-held-out 划分。
- recipe/ingredient/nutrition 分任务报告。
- 输出失败类型：检索失败、状态错误、证据不足、推理错误。

## 8. Phase 5：Object / 3D / Gaze-Hand 扩展

目标：

- 在主线稳定后扩展物体位置记忆和交互意图预测。

能力：

- 查询物体当前或历史位置。
- 统计物体移动次数。
- 生成物体 movement itinerary。
- 判断物体所在 fixture。
- 预测短期会拿哪个物体或放到哪里。

任务：

- 对接 `assoc_info.json` 和 `mask_info.json`。
- 对接 Digital Twin fixture 名称。
- 对接 gaze priming。
- 对接 hand tracking。
- 按需读取 SLAM trajectory。
- 不默认读取全量 semidense observations。

指标：

- object location accuracy。
- movement count accuracy。
- itinerary ordering accuracy。
- interaction anticipation accuracy。
- 3D fixture grounding accuracy。

研究意义：

- 第一视角 3D grounding、gaze-hand-object 时序推理、跨厨房泛化。

应用价值：

- 找物体、物品归位提醒、厨房安全、辅助记忆。

验证：

- 只在 recipe/ingredient 主线稳定后进入。
- 先做 object/mask/fixture 轻量版本，再考虑 SLAM semidense。

## 9. Phase 6：系统评估与论文实验

目标：

- 形成可发表、可解释、可复现的实验结果。

核心研究问题：

- 结构化 food-process memory 是否优于 TextOnly、FrameOnly 和普通 RAG。
- evidence constraint 是否降低幻觉并提高可核查性。
- 状态化 memory 是否提升 recipe/ingredient/nutrition 长时序推理。
- LightAgent 的 tool-use 在厨房长视频任务中的瓶颈是什么。
- 哪些任务必须依赖视觉，哪些任务结构化 memory 已足够。

实验划分：

- participant-held-out。
- task-family split。
- short/medium/long video segment split。
- evidence available/unavailable split。

指标：

- VQA accuracy。
- evidence recall。
- answer-with-evidence rate。
- tool selection accuracy。
- state tracking F1。
- anomaly precision/recall。
- latency 和 token cost。
- failure type distribution。

输出：

- `outputs/results/*.json`
- `outputs/reports/*.md`
- `outputs/traces/*.jsonl`
- 表格：baseline 对比、任务族对比、ablation、错误类型。

## 10. Phase 7：工程化与演示

目标：

- 做出可交互 demo 和可复现实验入口。

CLI：

```bash
python scripts/query_food_agent.py --video-id P01-... --time 123.4 --question "刚才加了什么？"
```

Demo 能力：

- 输入 video_id 和时间点。
- 展示当前 recipe step。
- 展示已加入食材和营养变化。
- 回答 VQA 问题。
- 返回证据时间段和帧路径。
- 展示 LightAgent trace 和工具调用。

工程要求：

- 配置集中在 `config.py`。
- 所有输出写入 `outputs/`。
- 不把数据路径硬编码进工具函数。
- 所有随机采样固定 seed。
- 日志中不保存大文件内容。

## 11. 优先级排序

第一优先级：

- GitHub 绑定、`.gitignore`、计划文档。
- Manifest 和数据读取。
- DuckDB 事件索引。
- LightAgent wrapper 和 HD-EPIC tools。
- VQA baseline 跑通。

第二优先级：

- recipe state tracker。
- ingredient timeline。
- nutrition delta estimator。
- evidence verifier。
- baseline ablation。

第三优先级：

- object memory。
- gaze/hand intention。
- Digital Twin fixture grounding。
- SLAM 轻量接入。

暂缓：

- 全量 semidense 点云处理。
- 机器人控制。
- 只为动作分类刷榜的模型训练。

## 12. 每阶段完成标准

阶段完成必须满足：

- 有代码或文档交付物。
- 有验证命令和验证结果。
- 有 git commit。
- 如果涉及实验，必须保存 config、结果和 trace。
- 如果失败，必须记录失败原因和下一步修复路径。

