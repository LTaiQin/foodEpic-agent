# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**foodEpic-agent** — a Python autonomous video question-answering (VQA) agent for the HD-EPIC egocentric cooking dataset. The agent builds per-video graph memory from annotations, uses an LLM planner to select tools, retrieves evidence, and verifies sufficiency before answering.

## Commands

```bash
# Syntax check (only verification available — no linter/formatter/CI)
python -m compileall food_agent scripts tests

# Run all tests (auto-isolates outputs to tmp_path, no network calls)
pytest tests/ -v

# Single test file / single test
pytest tests/test_config.py -v
pytest tests/test_config.py::test_project_config_defaults -v

# Verify + commit helper
scripts/verify_and_commit.sh "commit message"

# Run agent on a video
python scripts/run_graph_agent.py --video-id P08-20240620-180825 --limit 5 --max-steps 6
python scripts/run_graph_agent.py --video-id P08-20240620-180825 --limit 10 --out-file results.jsonl --append-jsonl
```

## Architecture

### Plan-Execute-Verify Loop

`GraphAgent` receives a VQA question → `GraphAgentPlanner` (LLM) picks a tool → `GraphAgentExecutor` runs it, updates `AgentState` → `GraphAgentVerifier` checks evidence sufficiency → loop or finalize.

### Core Modules

- **`food_agent/agent/graph_agent.py`** — `GraphAgent` orchestrator; `answer_vqa_row()` and `answer_open_query()` are primary interfaces
- **`food_agent/agent/planner.py`** — `GraphAgentPlanner` with LLM-based planning and heuristic fallbacks
- **`food_agent/agent/executor.py`** — `GraphAgentExecutor` step execution loop
- **`food_agent/agent/verifier.py`** — `GraphAgentVerifier` evidence sufficiency checks
- **`food_agent/agent/state.py`** — `AgentState` mutable working state per question
- **`food_agent/tools/agent_toolbox.py`** — `AgentToolbox` unified registry of 40+ tools exposed to the LLM

### Data Stores

- **`GraphMemoryStore`** (`memory/store.py`) — SQLite-backed per-video graph with JSONL mirrors
- **`FoodStateStore`** (`state_store.py`) — deterministic Parquet queries (recipe, ingredient, nutrition)
- **`SpatialContextStore`** (`spatial_store.py`) — object tracks, masks, gaze, audio queries

### Data Pipeline

1. Download HD-EPIC → `data/HD-EPIC/` and `annotations/hd-epic-annotations-main/`
2. `python scripts/build_manifest.py` → `outputs/dataset_manifest.parquet`
3. `python scripts/build_event_index.py` → `outputs/event_index/*.parquet`
4. Agent runs consume event index via `FoodStateStore` / `SpatialContextStore`

### Configuration

All config is env-var driven via `ProjectConfig.from_env()` / `ModelConfig.from_env()`. Load `.env` first with `load_env_file(Path(".env"))`.

Key env vars: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `FOOD_AGENT_MODEL`, `FOOD_AGENT_PROVIDER_MODE` (`responses` | `chat_completions`), `HD_EPIC_DATA_ROOT`, `HD_EPIC_ANNOTATION_ROOT`, `FOOD_AGENT_OUTPUT_ROOT`, `FOOD_AGENT_DISABLE_VISION` (`1` to skip vision).

Video ID format: `P\d{2}-\d{8}-\d{6}` (e.g. `P08-20240620-180825`). Regex in `paths.py`.

## Conventions

- Pure Python, no `pyproject.toml` or `requirements.txt`. Imports use `sys.path` manipulation.
- External dep: LightAgent at `/22liushoulong/agent/agent-context-isolation/third_party/LightAgent` (auto-added by `lightagent_wrapper.py`).
- `OpenAICompatibleModelClient` strips HTTP proxy env vars by default (`use_env_proxy=False`).
- Vision requests auto-detect mode and cache it; empty responses trigger fallback to the other mode.
- Tests use `monkeypatch` to isolate paths — never write to real `outputs/`. No network calls in tests.
- Workflow: change → `compileall` → commit → push (only when asked).
