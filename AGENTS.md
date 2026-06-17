# Project Operating Rules

## Repository

- **Remote**: `https://github.com/LTaiQin/foodEpic-agent.git`
- **Language**: Pure Python, no pyproject.toml or requirements.txt. Dependencies include `openai`, `pandas`, `opencv-python` (`cv2`), `h5py`, `Pillow`, `pyarrow`/`fastparquet`.
- **External dependency**: LightAgent at `/22liushoulong/agent/agent-context-isolation/third_party/LightAgent` — auto-added to `sys.path` by `food_agent/lightagent_wrapper.py` if it exists.

## Gitignored data directories (intentionally untracked)

```
data/          data-test/          annotations/          outputs/
.secrets/      *.duckdb  *.parquet  *.hdf5  *.mp4  *.zip  *.gz
```

## Verification

No linter, formatter, type checker, or CI is configured. The only verification is syntax checking:

```bash
python -m compileall food_agent scripts tests
```

Or use the helper script (runs verification, then `git add -A && git commit`):

```bash
scripts/verify_and_commit.sh "commit message" python -m compileall food_agent scripts tests
```

If verification fails, do not commit until fixed.

## Tests

```bash
# Run all tests (pytest required)
pytest tests/ -v

# Run a single test file
pytest tests/test_config.py -v

# Run a single test
pytest tests/test_config.py::test_project_config_defaults -v
```

- `tests/conftest.py` auto-isolates all graph/session/artifact roots to `tmp_path` via `monkeypatch.setenv` — tests never touch real `outputs/`.
- No fixtures, snapshots, or integration test services required.
- Tests import `food_agent` via `sys.path` manipulation in conftest; no install step needed.

## Architecture

### Package structure

```
food_agent/
  config.py          ProjectConfig, ModelConfig, load_env_file — env-driven config
  paths.py           ProjectPaths, VIDEO_ID_RE, infer_participant_id/video_id/domain
  model_client.py    OpenAICompatibleModelClient — wraps OpenAI SDK, supports responses + chat_completions modes
  loaders.py         Low-level loaders: JSON, JSONL, CSV, HDF5, video metadata via cv2
  data_index.py      Builds normalized Parquet event-index tables from annotations
  task_router.py     Rule-based task routing (recipe/ingredient/nutrition/object/gaze/audio)
  vqa.py             VQA sample loading, prediction records, metrics
  comparison.py      Baseline comparison utilities, choice hint ranking
  state_store.py     FoodStateStore — deterministic queries over Parquet event index
  spatial_store.py   SpatialContextStore — object tracks, masks, gaze, audio queries
  hd_epic_tools.py   HDEpicToolset — LightAgent-compatible tool factory
  lightagent_wrapper.py  FoodAgentLightWrapper — wraps LightAgent for fair baselines
  advantage_metrics.py   Composite scoring for agent advantage evaluation
  agent/
    graph_agent.py   GraphAgent + GraphAgentVideoSession — main orchestrator
    planner.py       GraphAgentPlanner — LLM-based planning
    verifier.py      GraphAgentVerifier — evidence sufficiency checks
    executor.py      GraphAgentExecutor — step execution loop
    state.py         AgentState — mutable working state per question
    action_intent.py Action intent resolution for fine-grained why questions
    artifact_policy.py Per-task artifact reuse policies
  graph/
    builder.py       VideoGraphBuilder — builds per-video graph from annotations
  memory/
    schema.py        GraphNodeRecord, GraphEdgeRecord
    store.py         GraphMemoryStore — persistent graph memory per video
  retrieval/
    event_retriever.py   EventRetriever — event-centric retrieval over graph memory
    object_retriever.py  ObjectRetriever — object-centric retrieval
    time_retriever.py    TimeRetriever — time-window retrieval
  tools/
    agent_toolbox.py AgentToolbox — unified tool registry (40+ tools) exposed to LLM
    graph_tools.py   GraphToolbox — graph query primitives
    video_tools.py   VideoToolbox — frame extraction, clip cutting, OCR
```

### Key entry points

- **`GraphAgent`** (`food_agent/agent/__init__.py`): main agent class. `agent.answer_vqa_row(row)` and `agent.answer_open_query(...)` are the primary interfaces.
- **`scripts/run_graph_agent.py`**: run agent on VQA samples. `--video-id` required, `--limit`, `--max-steps`, `--out-file`, `--append-jsonl`, `--resume` flags.
- **`scripts/build_manifest.py`**: scan data/ and annotations/ to produce `outputs/dataset_manifest.parquet`.
- **`scripts/build_event_index.py`**: from manifest + annotations, produce `outputs/event_index/*.parquet` tables.
- **`scripts/run_agent_comparison.py`**: run baseline comparisons.

### Configuration

All config is env-var driven. Load `.env` via `load_env_file(Path(".env"))` before creating config objects.

Key env vars (see `.env.example`):
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `FOOD_AGENT_MODEL` — model endpoint
- `FOOD_AGENT_PROVIDER_MODE` — `responses` (default) or `chat_completions`
- `FOOD_AGENT_VISION_PROVIDER_MODE` — `auto` (default), `responses`, or `chat_completions`
- `HD_EPIC_DATA_ROOT`, `HD_EPIC_ANNOTATION_ROOT`, `FOOD_AGENT_OUTPUT_ROOT` — data paths
- `FOOD_AGENT_GRAPH_MEMORY_ROOT`, `FOOD_AGENT_ARTIFACTS_ROOT`, `FOOD_AGENT_SESSIONS_ROOT`, `FOOD_AGENT_RUNS_ROOT` — output subdirectories
- `FOOD_AGENT_DISABLE_VISION` — set `1` to skip vision requests

### Video ID format

`P\d{2}-\d{8}-\d{6}` (e.g., `P08-20240620-180825`). Regex in `food_agent/paths.py`.

### Data pipeline order

1. Download HD-EPIC data into `data/HD-EPIC/` and annotations into `annotations/hd-epic-annotations-main/`
2. `python scripts/build_manifest.py` → `outputs/dataset_manifest.parquet`
3. `python scripts/build_event_index.py` → `outputs/event_index/*.parquet` (videos, recipes, events, ingredients, audio_events, object_tracks, object_masks, gaze_priming, vqa_samples)
4. Agent runs consume the event index via `FoodStateStore` and `SpatialContextStore`

### Model client quirks

- `OpenAICompatibleModelClient` strips HTTP proxy env vars by default (`use_env_proxy=False`).
- Vision requests auto-detect working mode (responses vs chat_completions) and cache it.
- Provider `right.codes/codex` defaults to `chat_completions` for vision; `cctq.ai` defaults to `responses`.
- Empty vision responses trigger fallback to the other mode.

### Testing conventions

- All tests use `monkeypatch` to isolate output paths; never write to real `outputs/`.
- `test_imports.py` verifies basic config/path inference — good smoke test.
- No network calls in tests; model client tests mock the OpenAI client.

## Workflow for code changes

1. Make the requested code or documentation change.
2. Run `python -m compileall food_agent scripts tests`.
3. If verification passes, commit automatically.
4. Push only when GitHub credentials are available or the user explicitly asks.
