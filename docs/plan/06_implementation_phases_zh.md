# 06 分阶段代码实施计划

## Phase 0：仓库与工程骨架

目标：

- 建立可提交、可验证、可扩展的工程骨架。

任务：

- 创建 `food_agent/` 包。
- 创建 `scripts/` 实验入口。
- 创建 `tests/`。
- 配置 `.gitignore`。
- 保证数据、密钥、输出不入库。

交付物：

- `food_agent/__init__.py`
- `food_agent/config.py`
- `food_agent/paths.py`
- `tests/test_imports.py`

验证：

```bash
python -m compileall food_agent scripts
pytest
```

## Phase 1：数据 Manifest

目标：

- 扫描并记录所有数据文件，输出可复查 manifest。

任务：

- 实现 `paths.py`。
- 实现 `loaders.py`。
- 实现 `scripts/build_manifest.py`。
- 输出 `outputs/dataset_manifest.parquet`。
- 输出 `outputs/data_format_report.md`。

验证：

```bash
python scripts/build_manifest.py --data-root data/HD-EPIC --out outputs/dataset_manifest.parquet
```

完成标准：

- 能统计 MP4、HDF5、VQA、recipe、object、gaze、SLAM 文件。
- 能抽样读取 JSON/CSV/HDF5/VQA。
- semidense 不全量读取。

## Phase 2：事件索引

目标：

- 构建 DuckDB/Parquet 事件库。

任务：

- 实现 `data_index.py`。
- 实现 recipe/activity/audio/object/gaze/VQA 转换。
- 生成统一 `events` 表。
- 实现基础 SQL 查询。

交付物：

- `outputs/food_agent.duckdb`
- `outputs/events.parquet`
- `scripts/build_event_index.py`

验证：

```bash
python scripts/build_event_index.py --manifest outputs/dataset_manifest.parquet --out outputs/food_agent.duckdb
python scripts/query_event_index.py --video-id VIDEO_ID --time 120
```

完成标准：

- 每个 video 可查 metadata。
- recipe/ingredient/audio/object/gaze/VQA 可按 video_id 查询。
- 至少 20 个 VQA sample 能检索上下文。

## Phase 3：LightAgent Wrapper

目标：

- 让 LightAgent baseline 可控、公平、可追踪。

任务：

- 实现 `lightagent_wrapper.py`。
- 实现关闭默认工具的策略。
- 实现 trace 保存。
- 实现 `hd_epic_tools.py`。
- 实现 `task_router.py`。

交付物：

- `food_agent/lightagent_wrapper.py`
- `food_agent/hd_epic_tools.py`
- `food_agent/task_router.py`
- `scripts/run_lightagent_smoke.py`

验证：

```bash
python scripts/run_lightagent_smoke.py --baseline textonly
python scripts/run_lightagent_smoke.py --baseline hdtools
```

完成标准：

- TextOnly 没有工具泄漏。
- HDTools 能调用 `get_video_metadata`。
- trace 能记录模型请求和工具列表。

## Phase 4：VQA Baselines

目标：

- 跑通官方 VQA benchmark 的第一批 baseline。

任务：

- 实现 `scripts/run_lightagent_baselines.py`。
- 实现 VQA sample parser。
- 实现 evidence prompt。
- 实现 answer parser。
- 实现 metrics。

验证：

```bash
python scripts/run_lightagent_baselines.py --limit 50 --baselines textonly,rag,foodmemory
python scripts/evaluate_vqa.py --pred outputs/results/predictions.jsonl
```

完成标准：

- 50 个样本跑通。
- 输出 accuracy、evidence rate、tool success rate。
- 保存 trace 和 failure cases。

## Phase 5：Recipe / Ingredient Agent

目标：

- 实现应用价值最高的状态追踪能力。

任务：

- 实现 `state_store.py`。
- 实现 recipe step tracker。
- 实现 ingredient timeline。
- 实现 nutrition delta estimator。
- 实现异常检测原型。

验证：

```bash
python scripts/evaluate_recipe_state.py
python scripts/evaluate_ingredient_timeline.py
python scripts/evaluate_anomaly_detection.py
```

完成标准：

- 支持查询当前 step。
- 支持查询已加入食材。
- 支持营养变化估计。
- 支持漏加/重复/顺序异常的合成评估。

## Phase 6：Object / Gaze / Audio 扩展

目标：

- 扩展物体位置、gaze/hand 意图和音频辅助事件。

任务：

- 实现 object memory。
- 实现 fixture grounding。
- 实现 gaze-hand context。
- 实现 audio event retriever。

验证：

```bash
python scripts/evaluate_object_memory.py
python scripts/evaluate_gaze_intention.py
python scripts/evaluate_audio_events.py
```

完成标准：

- object location/movement 可查询。
- gaze priming 可查询。
- audio events 可辅助回答。

## Phase 7：Demo 与报告

目标：

- 提供可展示、可复现的 agent demo。

任务：

- 实现 CLI query。
- 实现结果报告生成。
- 实现 trace 可视化摘要。
- 整理论文实验表格。

验证：

```bash
python scripts/query_food_agent.py --video-id VIDEO_ID --time 120 --question "刚才加了什么？"
python scripts/build_report.py --results outputs/results
```

完成标准：

- 用户能通过 CLI 查询。
- 返回答案、证据、时间段和 trace。
- 输出 Markdown 报告。

