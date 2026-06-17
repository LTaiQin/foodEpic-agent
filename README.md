# HD-EPIC Multimodal Agent

Autonomous video question-answering agent for the HD-EPIC egocentric cooking dataset. Seven-dimensional perception (audio, visual, gaze, 3D spatial, hand interaction, nutrition, motion) with LLM-driven tool-calling loop.

## Setup

```bash
conda create -n food-epic python=3.10 -y
conda activate food-epic
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data

Place HD-EPIC data under `data/HD-EPIC/` and annotations under `annotations/hd-epic-annotations-main/`.

## Usage

```bash
# Run agent on a video
python scripts/run_graph_agent.py --video-id P08-20240620-180825 --limit 5

# Run evaluation
python scripts/run_eval.py --benchmark annotations/hd-epic-annotations-main/vqa-benchmark --limit 10
```

## Architecture

```
User Question → Router → Agent Loop (LLM decides tools) → Evidence Fusion → Answer
                           ↓
            AudioAnalyzer | VisualAnalyzer | GazeTracker | SpatialReasoner
            HandInteractor | NutritionEstimator | MotionTracker
                           ↓
            Aggregator → Judge → Generator → Answer + ReasoningTrace
```

## Project Structure

```
food_agent/
  loaders/          # Data loaders (Audio, Video, Gaze, SLAM, DigitalTwin, Hands)
  perception/       # Perception modules (7 analyzers + Evidence + Registry)
  reasoning/        # Router, Aggregator, Judge, Generator, ToolRegistry
  knowledge/        # RecipeKB, NutritionKB, SceneGraphKB, CommonSenseKB
  agent_v2/         # MultimodalAgent (autonomous tool-calling loop)
  evaluation/       # API client, BenchmarkLoader, Metrics
  utils/            # Time alignment, caching
config/             # YAML configurations
scripts/            # Entry point scripts
```
