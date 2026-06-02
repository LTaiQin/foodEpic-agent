# 02 数据层与事件索引计划

## 2.1 目标

建立一个可查询、可检查、可复现的数据层，把 HD-EPIC 的多模态数据从“文件集合”转成 agent 可用的结构化 memory。

最终数据流：

```text
raw HD-EPIC files
  -> manifest
  -> loaders
  -> normalized events
  -> DuckDB/Parquet index
  -> FoodMemoryRetriever
  -> LightAgent tools
```

## 2.2 本地数据路径

```text
data root: /22liushoulong/agent/hd-epic/data/HD-EPIC
annotation root: /22liushoulong/agent/hd-epic/annotations/hd-epic-annotations-main
```

当前状态：

- MP4：156 个。
- Audio HDF5：9 个。
- semidense `.csv.gz`：312 个。
- 未压缩 semidense CSV：0 个。
- 数据总体积：约 571G。

## 2.3 需要支持的数据域

视频：

- `Videos/PXX/*.mp4`
- `Videos/PXX/*_mp4_to_vrs_time_ns.csv`
- `Videos/PXX/*_vrs_to_mp4_log.json`
- `durations.txt`
- `frames.txt`

高层 recipe：

- `high-level/complete_recipes.json`
- `high-level/activities/PXX_recipe_timestamps.csv`

动作/narration：

- `narrations-and-action-segments/HD_EPIC_Narrations.pkl`
- `HD_EPIC_verb_classes.csv`
- `HD_EPIC_noun_classes.csv`
- `HD_EPIC_Narrations_erratum.csv`

音频：

- `Audio-HDF5/PXX/PXX_audio.hdf5`
- `audio-annotations/HD_EPIC_Sounds.csv`
- `audio-annotations/HD_EPIC_Sounds.pkl`

物体与场景：

- `scene-and-object-movements/assoc_info.json`
- `scene-and-object-movements/mask_info.json`

gaze/hand：

- `eye-gaze-priming/priming_info.json`
- `SLAM-and-Gaze/PXX/GAZE_HAND/**/general_eye_gaze.csv`
- `SLAM-and-Gaze/PXX/GAZE_HAND/**/wrist_and_palm_poses.csv`

SLAM/3D：

- `closed_loop_trajectory.csv`
- `open_loop_trajectory.csv`
- `online_calibration.jsonl`
- `semidense_points.csv.gz`
- `semidense_observations.csv.gz`
- `summary.json`

Digital Twin：

- `Digital-Twin/*.blend`
- `Digital-Twin/meshes/**/*.obj`
- `Digital-Twin/meshes/**/*.mtl`

Hands Masks：

- `Hands-Masks/contours_preds/*.json`
- `Hands-Masks/contours_cleaned/*.json`
- `Hands-Masks/contours_memory/*.json`

VQA：

- `vqa-benchmark/*.json`

## 2.4 Manifest 设计

输出：

```text
outputs/dataset_manifest.parquet
outputs/data_format_report.md
```

字段：

- `path`
- `relative_path`
- `domain`
- `participant_id`
- `video_id`
- `recipe_id`
- `file_type`
- `size_bytes`
- `row_count`
- `status`
- `notes`

状态：

- `ok`：已发现并可读取。
- `missing`：预期存在但缺失。
- `blocked`：存在但当前环境不可读取。
- `deferred`：存在但初期不展开，例如 semidense observations。

## 2.5 Loader 模块

目标文件：

```text
food_agent/paths.py
food_agent/loaders.py
scripts/build_manifest.py
```

核心接口：

```python
get_data_root() -> Path
get_annotation_root() -> Path
load_json(path) -> dict | list
load_csv(path, **kwargs) -> pd.DataFrame
load_jsonl(path) -> list[dict]
load_hdf5_audio(path, video_id=None) -> metadata | waveform
load_video_metadata(path) -> dict
load_vqa_files(root) -> list[dict]
```

注意事项：

- 不默认读取大音频 waveform，只读取 metadata。
- 不默认解压 `.csv.gz`。
- semidense observations 只登记路径，不做全量读取。
- narration pickle 当前存在 NumPy 兼容问题，需要单独转换。

## 2.6 事件索引设计

输出：

```text
outputs/food_agent.duckdb
outputs/events.parquet
```

核心表：

- `videos`
- `events`
- `recipe_steps`
- `ingredients`
- `audio_events`
- `object_tracks`
- `object_masks`
- `gaze_priming`
- `hand_poses`
- `slam_pose`
- `vqa_samples`

通用事件字段：

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

## 2.7 查询 API

目标文件：

```text
food_agent/data_index.py
food_agent/memory_retriever.py
```

基础查询：

```python
get_video(video_id)
query_events(video_id, start_time=None, end_time=None, event_types=None)
get_recipe_state(video_id, time)
get_ingredient_state(video_id, time)
get_object_state(video_id, object_name=None, time=None)
get_audio_context(video_id, start_time, end_time)
get_gaze_hand_context(video_id, time)
get_vqa_sample(vqa_id)
```

## 2.8 验证标准

Phase 1 完成标准：

- Manifest 覆盖所有主要数据域。
- 每类数据至少抽样读取 3 个文件。
- 不读取全量 semidense observations。
- 输出数据格式报告。

Phase 2 完成标准：

- DuckDB 中 `videos`、`events`、`vqa_samples` 可查询。
- 随机 20 个 VQA 样本能解析 video_id、question、choices。
- recipe 视频能查到当前 step 和 ingredient state。
- object/gaze/audio 至少有基础事件表。

