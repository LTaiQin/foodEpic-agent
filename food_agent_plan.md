# HD-EPIC Food Agent Plan

## Current Local Dataset State

- Data root: `/22liushoulong/agent/hd-epic/data/HD-EPIC`
- Annotation root: `/22liushoulong/agent/hd-epic/annotations/hd-epic-annotations-main`
- Current data size: about `571G`
- `semidense_points.csv` / `semidense_observations.csv`: recompressed back to `312` `.csv.gz` files.
- Remaining file type profile:
  - `1493` JSON
  - `826` OBJ
  - `773` CSV
  - `312` GZ
  - `156` MP4
  - `153` JSONL
  - `9` HDF5
  - `18` BLEND

## Dataset Formats

### Videos

Path pattern:

```text
Videos/PXX/{video_id}.mp4
Videos/PXX/{video_id}_mp4_to_vrs_time_ns.csv
Videos/PXX/{video_id}_vrs_to_mp4_log.json
Videos/PXX/durations.txt
Videos/PXX/frames.txt
```

Observed MP4 format:

- Video codec: H.264
- Resolution: `1408 x 1408`
- Frame rate: `30 fps`
- Audio codec: AAC

Time mapping CSV fields:

- `mp4_time_ns`
- `relative_vrs_device_time_ns`
- `vrs_device_time_ns`

Use in agent:

- Primary visual input for action, recipe, object, and VQA reasoning.
- Time mapping is needed when aligning MP4 frames with Aria-derived SLAM, gaze, hand, and VRS-time streams.

### Narrations And Action Segments

Path:

```text
narrations-and-action-segments/HD_EPIC_Narrations.pkl
narrations-and-action-segments/HD_EPIC_verb_classes.csv
narrations-and-action-segments/HD_EPIC_noun_classes.csv
narrations-and-action-segments/HD_EPIC_Narrations_erratum.csv
```

`HD_EPIC_Narrations.pkl` fields from README:

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

Notes:

- Loading the pickle failed in the current Python environment due to a NumPy module mismatch: `No module named 'numpy._core.numeric'`.
- Use a compatible NumPy/Pandas environment or load through the original environment expected by the dataset.
- The class CSVs are readable and contain normalized `id`, `key`, `instances`, `category`.

Use in agent:

- Core temporal supervision for action memory.
- Converts raw video into structured events: verb, noun, hand, time span, and free-form narration.

### High-Level Recipes And Nutrition

Paths:

```text
high-level/activities/PXX_recipe_timestamps.csv
high-level/complete_recipes.json
```

Activity CSV fields:

- `video_id`
- `recipe_id`
- `high_level_activity_label`
- `start_time`
- `end_time`

Recipe JSON structure:

- top-level key: recipe id, e.g. `P01_R01`
- `participant`
- `name`
- `type`
- `source`
- `steps`: ordered free-form step map
- `captures`
  - `videos`
  - `ingredients`
    - `name`
    - `amount`
    - `amount_unit`
    - `calories`
    - `carbs`
    - `fat`
    - `protein`
    - `weigh`: video time segments
    - `add`: video time segments
  - `step_times`
  - `prep_times`

Use in agent:

- Recipe progress tracker.
- Ingredient state tracker.
- Nutrition estimator.
- Strong source for long-horizon task state and expected next step.

### Audio

Paths:

```text
Audio-HDF5/PXX/PXX_audio.hdf5
audio-annotations/HD_EPIC_Sounds.csv
audio-annotations/HD_EPIC_Sounds.pkl
```

HDF5 structure:

- One dataset per `video_id`.
- Dataset dtype: `float32`.
- Dataset shape: one-dimensional waveform samples.

Sound annotation CSV fields:

- `participant_id`
- `video_id`
- `start_timestamp`
- `stop_timestamp`
- `start_sample`
- `stop_sample`
- `class`
- `class_id`

Important:

- Annotation sample rate is `48 KHz`.

Use in agent:

- Audio event cues for cooking actions: water, chopping, appliance use, stirring, footsteps, rustling.
- Useful when visual signal is ambiguous or hands/object are occluded.

### Scene And Object Movements

Paths:

```text
scene-and-object-movements/assoc_info.json
scene-and-object-movements/mask_info.json
```

`assoc_info.json` structure:

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

`mask_info.json` structure:

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

Use in agent:

- Object permanence.
- Object movement counting.
- Pick-up/place state updates.
- 3D fixture grounding: counter, cupboard, sink, table, etc.

### Eye Gaze Priming

Path:

```text
eye-gaze-priming/priming_info.json
```

Structure:

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

`frame_primed` semantics:

- `>= 0`: priming frame.
- `-1`: valid location but no priming.
- `-2`: excluded sample.

Use in agent:

- Anticipating the next object interaction.
- Predicting pick-up and placement intent before action completion.

### Gaze And Hand Tracking Streams

Path pattern:

```text
SLAM-and-Gaze/PXX/GAZE_HAND/mps_{video_id}_vrs/eye_gaze/general_eye_gaze.csv
SLAM-and-Gaze/PXX/GAZE_HAND/mps_{video_id}_vrs/hand_tracking/wrist_and_palm_poses.csv
```

Eye gaze CSV fields include:

- `tracking_timestamp_us`
- `left_yaw_rads_cpf`
- `right_yaw_rads_cpf`
- `pitch_rads_cpf`
- `depth_m`
- uncertainty ranges
- left/right eye positions in CPF
- `session_uid`

Hand tracking CSV fields include:

- `tracking_timestamp_us`
- `left_tracking_confidence`
- `tx_left_wrist_device`
- `ty_left_wrist_device`
- `tz_left_wrist_device`
- `tx_left_palm_device`
- `ty_left_palm_device`
- `tz_left_palm_device`
- same fields for right hand

Use in agent:

- Short-term intent recognition.
- Hand-object interaction detection.
- Disambiguating active hand, manipulation mode, and target object.

### SLAM And Geometry

Path pattern:

```text
SLAM-and-Gaze/PXX/SLAM/multi/{index}/slam/
```

Common files:

- `closed_loop_trajectory.csv`
- `open_loop_trajectory.csv`
- `online_calibration.jsonl`
- `semidense_points.csv.gz`
- `semidense_observations.csv.gz`
- `summary.json`

Closed-loop trajectory fields include:

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
- linear velocity
- angular velocity
- gravity
- quality and geo fields

Open-loop trajectory fields include:

- `tracking_timestamp_us`
- `utc_timestamp_ns`
- `session_uid`
- odometry translation and quaternion
- velocity
- angular velocity
- gravity
- `quality_score`

Online calibration:

- JSONL, one calibration snapshot per line.
- Contains IMU, camera/sensor calibration and transforms.

Semidense files:

- Stored as `.csv.gz` to control disk usage.
- Read directly with `pandas.read_csv(path, compression="gzip")` or stream with `gzip.open`.

Use in agent:

- 3D spatial memory.
- Mapping fixture/object locations into a shared kitchen coordinate frame.
- Projecting gaze and object states into world coordinates.

### Digital Twin

Paths:

```text
Digital-Twin/PXX_final.blend
Digital-Twin/meshes/PXX/*.obj
Digital-Twin/meshes/PXX/*.mtl
```

Observed OBJ structure:

- Blender-exported OBJ.
- Fixture/object names are embedded in object/file names, e.g. `P01_shelf.005.obj`, `P01_counter.008.obj`.
- Coordinates are in the scene frame used by the digital twin.

Use in agent:

- Fixture map and kitchen topology.
- 3D grounding for object location answers.
- Spatial priors for “where should this object go?” reasoning.

### Hand Masks

Paths:

```text
Hands-Masks/contours_preds/{video_id}.json
Hands-Masks/contours_cleaned/
Hands-Masks/contours_memory/
```

Observed format:

- JSON dict keyed by frame number string.
- Some frames may contain `{}`.
- Values encode hand contour/mask information when present.

Use in agent:

- Hand segmentation supervision.
- Manipulation detection.
- Visual attention cropping around hands.

### VQA Benchmark

Path:

```text
vqa-benchmark/*.json
```

Each file is one question prototype. Each item has:

- `inputs`: video or clip references, often with `id`, `start_time`, `end_time`
- `question`
- `choices`: five answer choices
- `correct_idx`
- `others`: task-specific metadata

Task families:

- Fine-grained action recognition/localization/how/why
- Gaze estimation and interaction anticipation
- Ingredient recognition/retrieval/weight/order/add localization
- Nutrition estimation and nutrition change
- Object movement counting/itinerary/localization
- Recipe recognition, step recognition, step localization, activity recognition
- 3D object/fixture perception

Use in agent:

- Main offline benchmark for the first agent prototype.
- Supports task-specific evaluation without inventing a new metric immediately.

## Food Agent Objective

Build a long-horizon egocentric cooking agent that can:

1. Watch a kitchen video or clip.
2. Build a structured memory of actions, objects, ingredients, recipe progress, gaze, hands, sound, and 3D locations.
3. Answer questions and produce evidence-grounded reasoning.
4. Predict likely next interactions and recipe steps.
5. Track ingredients, nutrition, and object state over time.

This is best framed as a perception-memory-reasoning agent, not a robot control policy. HD-EPIC does not provide low-level robot actions, force control, or interactive environment feedback.

## Agent Architecture

### Layer 1: Data Index

Create a unified manifest table keyed by:

- `participant_id`
- `video_id`
- `recipe_id`
- `capture_id`
- `start_time`
- `end_time`
- `frame_start`
- `frame_end`

Tables to build:

- `videos`: MP4 path, duration, frame count, FPS, participant.
- `actions`: narration/action segments.
- `recipes`: recipe steps, captures, ingredients, nutrition.
- `audio_events`: sound segments.
- `object_tracks`: object movement associations.
- `object_masks`: mask metadata, bbox, 3D location, fixture.
- `gaze_priming`: priming events.
- `gaze_streams`: per-frame/per-timestamp gaze summaries.
- `hand_streams`: hand pose/confidence summaries.
- `slam_pose`: device-to-world trajectory.
- `vqa_examples`: benchmark questions and answer labels.

Recommended storage:

- Parquet for medium tabular indexes.
- SQLite or DuckDB for easy joins.
- Keep massive `semidense_*.csv.gz` on disk and load on demand only.

### Layer 2: Event Memory

Convert all modalities into normalized event records:

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

This becomes the agent’s structured memory.

### Layer 3: Perception Models

Recommended first-pass models:

- Video-language encoder: VideoCLIP/InternVideo/Video-LLaVA/Qwen2.5-VL style clip encoder.
- Image/frame encoder: CLIP/SigLIP for sampled frames.
- Audio encoder: BEATs/PANNs or lightweight audio classifier.
- Object/hand crops: use existing hand masks and object movement boxes as supervision.
- Text encoder: sentence-transformers for narration, recipe steps, and VQA retrieval.

Start with frozen encoders plus retrieval. Do not fine-tune everything at once.

### Layer 4: Reasoning Agent

The LLM/VLM agent should use tools over memory rather than operate only on raw video:

- `retrieve_events(video_id, time_range, event_types)`
- `get_recipe_state(video_id, time)`
- `get_object_state(video_id, object_name, time)`
- `get_ingredient_state(recipe_id, time)`
- `get_gaze_context(video_id, time)`
- `get_audio_context(video_id, time)`
- `get_3d_location(video_id, object_or_fixture, time)`
- `sample_frames(video_id, start, end, fps)`
- `answer_vqa(example_id)`

Reasoning style:

- First retrieve structured evidence.
- Then inspect visual/audio clips only when structured evidence is insufficient.
- Return answer plus evidence spans.

## Development Plan

### Phase 0: Dataset Sanity And Loader Layer

Goal:

- Make the dataset queryable without manual path chasing.

Tasks:

- Build a Python package under `hd-epic/food_agent/`.
- Add loaders for CSV, JSON, HDF5, MP4 metadata, gaze/hand CSV, SLAM trajectory, and VQA.
- Add a compatibility environment for `HD_EPIC_Narrations.pkl`.
- Build a `dataset_manifest.parquet`.
- Build lightweight file integrity checks:
  - expected 156 MP4s
  - 9 HDF5 files
  - 312 semidense `.csv.gz`
  - no `semidense_*.csv` uncompressed files

Deliverable:

- `food_agent/data_index.py`
- `food_agent/loaders.py`
- `outputs/dataset_manifest.parquet`
- `outputs/data_format_report.md`

### Phase 1: Structured Memory Index

Goal:

- Fuse annotations into a single time-indexed event database.

Tasks:

- Convert recipes, activity segments, audio events, object movements, gaze priming, and VQA inputs into normalized events.
- Convert narration/action segments after fixing pickle environment.
- Align all timestamps to MP4 seconds.
- Create DuckDB tables:
  - `videos`
  - `events`
  - `objects`
  - `recipes`
  - `ingredients`
  - `vqa`
  - `gaze`
  - `hands`
  - `slam_pose`

Deliverable:

- `outputs/food_agent.duckdb`
- Query API for event retrieval.

### Phase 2: Baseline Agent With Retrieval

Goal:

- Answer VQA questions using structured memory and minimal visual sampling.

Pipeline:

1. Parse VQA input.
2. Retrieve relevant events by `video_id` and time span.
3. Retrieve recipe/object/ingredient/gaze/audio context.
4. Optionally sample frames from the clip.
5. Prompt a VLM/LLM with choices and evidence.
6. Return selected answer and evidence.

Initial benchmark:

- Use official VQA JSON files.
- Report accuracy by task family.
- Track whether answer came from structured memory, visual frames, or both.

Deliverable:

- `run_vqa_baseline.py`
- `vqa_results_by_task.json`

### Phase 3: Recipe And Ingredient Agent

Goal:

- Build a domain-specific food agent before tackling full 3D reasoning.

Capabilities:

- Identify current recipe.
- Track current step.
- Track ingredients already added.
- Estimate nutrition changes.
- Answer “what was added?”, “when was it added?”, “what remains?”, “what is the next step?”

Data sources:

- `complete_recipes.json`
- high-level activity CSVs
- narrations/action segments
- VQA ingredient/nutrition tasks
- sampled frames around ingredient add/weigh segments

Deliverable:

- Recipe-state tracker.
- Ingredient timeline.
- Nutrition delta estimator.

### Phase 4: Object And 3D Memory Agent

Goal:

- Make object and fixture state queryable over time.

Capabilities:

- Locate object in 2D bbox and 3D coordinates.
- Answer object movement count and itinerary.
- Track object fixture transitions.
- Predict put-down location with gaze priming.

Data sources:

- `assoc_info.json`
- `mask_info.json`
- Digital Twin OBJ fixture names.
- SLAM trajectory.
- Gaze priming.
- Gaze/hand streams.

Deliverable:

- Object state graph:
  - nodes: object associations, fixtures, recipe ingredients.
  - edges: moved_to, picked_from, placed_on, seen_at, gazed_at.

### Phase 5: Multimodal Prediction

Goal:

- Predict next interaction before it happens.

Tasks:

- Next object interaction prediction.
- Next recipe step prediction.
- Pick-up and put-down anticipation.
- Gaze-conditioned interaction anticipation.

Features:

- last N action events
- hand pose/confidence
- gaze priming
- object movement history
- recipe progress
- audio events

Deliverable:

- Next-step model and evaluation on held-out participants.

### Phase 6: Full Food Agent Interface

Goal:

- Provide a practical research interface.

Interfaces:

- CLI:
  - `ask --video-id ... --time ... --question ...`
  - `recipe-state --video-id ... --time ...`
  - `object-state --video-id ... --object ...`
- Notebook:
  - inspect clip, frames, events, object tracks, gaze.
- Optional web dashboard:
  - video timeline
  - event tracks
  - recipe steps
  - object locations
  - evidence panel

## Evaluation Plan

Use the official VQA benchmark first:

- Action: recognition, localization, how, why.
- Recipe: recipe recognition, step recognition/localization, activity recognition.
- Ingredient: recognition, retrieval, order, weight, add localization.
- Nutrition: image/video nutrition estimation and change.
- Object motion: movement count, itinerary, stationary localization.
- Gaze: gaze estimation and interaction anticipation.
- 3D: object/fixture location and interaction counting.

Metrics:

- multiple-choice accuracy
- accuracy by family
- evidence recall against annotated time spans
- state tracking F1 for ingredients/object moves
- next-step top-k accuracy

Validation split:

- Prefer participant-held-out split.
- Avoid evaluating on the same participant kitchens used for tuning, because Digital Twin/fixture priors can leak heavily.

## Practical Priorities

Recommended implementation order:

1. Loader and manifest.
2. Recipe/ingredient state tracker.
3. VQA retrieval baseline.
4. Object movement graph.
5. Gaze/hand anticipation.
6. SLAM/3D-heavy reasoning.

Why:

- Recipe and ingredient tasks have high signal and manageable engineering cost.
- Full SLAM reasoning is valuable but expensive; keep semidense data compressed and use only trajectory/fixture/object metadata first.

## Key Risks

- `HD_EPIC_Narrations.pkl` currently needs a compatible NumPy/Pandas environment.
- Full semidense SLAM CSVs are huge; read `.csv.gz` lazily.
- Some object masks and boxes may be inconsistent because they were produced by different teams.
- Dataset has 9 kitchens; generalization needs participant-held-out evaluation.
- This dataset supports perception, reasoning, memory, and prediction. It does not directly support low-level robotic control.

## First Experiment

Build a recipe/ingredient VQA agent.

Inputs:

- `complete_recipes.json`
- `activities/PXX_recipe_timestamps.csv`
- VQA files under ingredient, nutrition, and recipe families.
- MP4 clips sampled around requested time ranges.

Method:

- Use structured recipe/ingredient timelines for retrieval.
- Use a VLM only for cases requiring visual confirmation.
- Answer multiple-choice VQA and cite evidence.

Target outcome:

- Strong baseline on recipe/ingredient/nutrition VQA.
- A reusable memory substrate for the later object and 3D agent.
