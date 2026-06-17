"""Full pipeline: wire all modules together for end-to-end agent execution.

This is the glue code that connects:
- Data loaders → Perception modules → Reasoning engine → Agent loop → Answer
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

from food_agent.loaders import (
    AudioLoader, VideoLoader, GazeLoader,
    SLAMLoader, DigitalTwinLoader, HandsLoader,
)
from food_agent.perception import (
    AudioAnalyzer, VisualAnalyzer, GazeTracker,
    SpatialReasoner, HandInteractor, NutritionEstimator, MotionTracker,
    Evidence,
)
from food_agent.perception.registry import ModuleRegistry
from food_agent.reasoning import Router, Aggregator, Judge, Generator
from food_agent.reasoning.tool_registry import ToolRegistry
from food_agent.knowledge import RecipeKB, NutritionKB, SceneGraphKB, CommonSenseKB
from food_agent.evaluation.api_client import MimoClient
from .agent import MultimodalAgent


class Pipeline:
    """End-to-end pipeline that wires all modules together.

    Usage:
        pipeline = Pipeline()
        result = pipeline.answer("What ingredients are in the video?",
                                  video_id="P01-20240202-110250")
    """

    def __init__(self, config_path: Optional[str] = None):
        from food_agent.config import ProjectConfig
        self.config = ProjectConfig.from_env()

        # Initialize data loaders
        data_root = str(self.config.data_root)
        self.audio_loader = AudioLoader(Path(data_root) / "Audio-HDF5")
        self.video_loader = VideoLoader(Path(data_root) / "Videos")
        self.gaze_loader = GazeLoader(Path(data_root) / "SLAM-and-Gaze")
        self.slam_loader = SLAMLoader(Path(data_root) / "SLAM-and-Gaze")
        self.dt_loader = DigitalTwinLoader(Path(data_root) / "Digital-Twin")
        self.hands_loader = HandsLoader(Path(data_root) / "Hands-Masks")

        # Initialize LLM client
        self.mimo_client = MimoClient()

        # Initialize perception modules
        self.audio_analyzer = AudioAnalyzer()
        self.visual_analyzer = VisualAnalyzer(mimo_client=self.mimo_client)
        self.gaze_tracker = GazeTracker(self.gaze_loader)
        self.spatial_reasoner = SpatialReasoner(self.dt_loader, self.slam_loader)
        self.hand_interactor = HandInteractor(self.hands_loader)
        self.nutrition_estimator = NutritionEstimator()
        self.motion_tracker = MotionTracker(slam_loader=self.slam_loader)

        # Initialize knowledge modules
        self.nutrition_kb = NutritionKB()
        self.commonsense_kb = CommonSenseKB()
        self.scene_graph_kb = SceneGraphKB()

        # Build tool registry with real implementations
        self.tool_registry = self._build_tool_registry()

        # Create the agent
        self.agent = MultimodalAgent(
            mimo_client=self.mimo_client,
            tool_registry=self.tool_registry,
        )

        # Current context (set by answer())
        self._current_video_id = ""
        self._current_participant_id = ""

    def _build_tool_registry(self) -> ToolRegistry:
        """Register all tools with actual implementations."""
        registry = ToolRegistry()

        # --- Perception tools ---
        registry.register("query_audio", self._tool_query_audio)
        registry.register("query_video", self._tool_query_video)
        registry.register("query_gaze", self._tool_query_gaze)
        registry.register("query_3d", self._tool_query_3d)
        registry.register("query_hands", self._tool_query_hands)
        registry.register("query_nutrition", self._tool_query_nutrition)
        registry.register("query_motion", self._tool_query_motion)

        # --- Knowledge tools ---
        registry.register("query_recipe", self._tool_query_recipe)
        registry.register("query_nutrition_kb", self._tool_query_nutrition_kb)
        registry.register("query_scene_graph", self._tool_query_scene_graph)
        registry.register("query_commonsense", self._tool_query_commonsense)

        # --- Control tools ---
        registry.register("check_evidence", self._tool_check_evidence)
        registry.register("expand_search", self._tool_expand_search)
        registry.register("synthesize_answer", self._tool_synthesize_answer)

        return registry

    def _get_context(self, kwargs: Dict) -> Dict:
        """Extract video/participant context from tool kwargs."""
        return {
            "video_id": kwargs.get("video_id", self._current_video_id),
            "participant_id": kwargs.get("participant_id", self._current_participant_id),
        }

    # --- Tool implementations ---

    def _tool_query_audio(self, start_time: float = 0, end_time: float = 30, **kwargs) -> List[Evidence]:
        """Query audio events in a time range."""
        ctx = self._get_context(kwargs)
        try:
            return self.audio_analyzer.get_audio_events(
                self.audio_loader, ctx["participant_id"], ctx["video_id"],
                start_time, end_time,
            )
        except Exception as e:
            return [Evidence(source_module="AudioAnalyzer", evidence_type="audio",
                           content={"error": str(e)}, confidence=0)]

    def _tool_query_video(self, timestamp: float = 10, text_prompt: str = "", **kwargs) -> Evidence:
        """Query video frame for visual analysis."""
        ctx = self._get_context(kwargs)
        try:
            frame = self.video_loader.get_frame(ctx["video_id"], timestamp)
            prompt = text_prompt or "knife. cutting board. tomato. pan. plate. pot. spoon. food. hand."
            return self.visual_analyzer.analyze_frame(frame, timestamp, prompt)
        except Exception as e:
            return Evidence(source_module="VisualAnalyzer", evidence_type="visual",
                          content={"error": str(e)}, confidence=0)

    def _tool_query_gaze(self, start_time: float = 0, end_time: float = 30, **kwargs) -> List[Evidence]:
        """Query gaze data in a time range."""
        ctx = self._get_context(kwargs)
        try:
            return self.gaze_tracker.get_fixation_targets(
                ctx["participant_id"], ctx["video_id"], start_time, end_time,
            )
        except Exception as e:
            return [Evidence(source_module="GazeTracker", evidence_type="gaze",
                           content={"error": str(e)}, confidence=0)]

    def _tool_query_3d(self, query_type: str = "layout", timestamp: float = 10, **kwargs) -> Evidence:
        """Query 3D spatial information."""
        ctx = self._get_context(kwargs)
        try:
            return self.spatial_reasoner.query_3d(
                ctx["participant_id"], ctx["video_id"], timestamp, query_type,
            )
        except Exception as e:
            return Evidence(source_module="SpatialReasoner", evidence_type="spatial",
                          content={"error": str(e)}, confidence=0)

    def _tool_query_hands(self, frame_number: int = 300, **kwargs) -> Evidence:
        """Query hand interactions for a frame."""
        ctx = self._get_context(kwargs)
        try:
            return self.hand_interactor.get_hand_interactions(
                ctx["participant_id"], ctx["video_id"], frame_number,
            )
        except Exception as e:
            return Evidence(source_module="HandInteractor", evidence_type="hand",
                          content={"error": str(e)}, confidence=0)

    def _tool_query_nutrition(self, ingredients: List[Dict] = None, **kwargs) -> Evidence:
        """Query nutrition for ingredients."""
        try:
            if not ingredients:
                return Evidence(source_module="NutritionEstimator", evidence_type="nutrition",
                              content={"error": "no ingredients provided"}, confidence=0)
            result = self.nutrition_estimator.calculate_total(ingredients)
            return Evidence(source_module="NutritionEstimator", evidence_type="nutrition",
                          content=result, confidence=0.7)
        except Exception as e:
            return Evidence(source_module="NutritionEstimator", evidence_type="nutrition",
                          content={"error": str(e)}, confidence=0)

    def _tool_query_motion(self, frame_number: int = 300, **kwargs) -> Evidence:
        """Query object motion data."""
        ctx = self._get_context(kwargs)
        try:
            return self.motion_tracker.get_motion_evidence(ctx["video_id"], frame_number)
        except Exception as e:
            return Evidence(source_module="MotionTracker", evidence_type="motion",
                          content={"error": str(e)}, confidence=0)

    def _tool_query_recipe(self, recipe_name: str = "", step_number: int = 0, **kwargs) -> Dict:
        """Query recipe knowledge base."""
        if recipe_name:
            recipe = self.recipe_kb.get_recipe(recipe_name) if hasattr(self, 'recipe_kb') else None
            if recipe:
                return recipe
        return {"error": "recipe not found", "available": []}

    def _tool_query_nutrition_kb(self, ingredient: str = "", **kwargs) -> Dict:
        """Look up nutrition facts."""
        result = self.nutrition_kb.lookup(ingredient)
        return result or {"error": f"ingredient '{ingredient}' not in database"}

    def _tool_query_scene_graph(self, object_type: str = "", start_time: float = 0, end_time: float = 30, **kwargs) -> Dict:
        """Query scene graph."""
        if object_type:
            results = self.scene_graph_kb.query_objects(object_type)
            return {"objects": results, "count": len(results)}
        return self.scene_graph_kb.get_scene_summary(start_time, end_time)

    def _tool_query_commonsense(self, concept: str = "", relation: str = "UsedFor", **kwargs) -> Dict:
        """Query common sense knowledge."""
        results = self.commonsense_kb.get_related_concepts(concept, relation)
        return {"concept": concept, "relation": relation, "related": results}

    def _tool_check_evidence(self, **kwargs) -> Dict:
        """Check evidence sufficiency (placeholder - agent handles this internally)."""
        return {"status": "check handled by agent loop"}

    def _tool_expand_search(self, modules: List[str] = None, start_time: float = 0, end_time: float = 30, **kwargs) -> List[Evidence]:
        """Expand search to more modules."""
        evidence = []
        if modules:
            for mod in modules:
                if mod == "AudioAnalyzer":
                    evidence.extend(self._tool_query_audio(start_time, end_time, **kwargs))
                elif mod == "VisualAnalyzer":
                    evidence.append(self._tool_query_video((start_time + end_time) / 2, **kwargs))
                elif mod == "GazeTracker":
                    evidence.extend(self._tool_query_gaze(start_time, end_time, **kwargs))
        return evidence

    def _tool_synthesize_answer(self, **kwargs) -> Dict:
        """Synthesize answer (placeholder - agent handles this internally)."""
        return {"status": "synthesis handled by agent loop"}

    # --- Main interface ---

    def answer(
        self,
        question: str,
        video_id: str = "",
        participant_id: str = "",
        choices: Optional[List[str]] = None,
    ) -> Dict:
        """Answer a question about a video.

        Args:
            question: The question to answer.
            video_id: HD-EPIC video ID (e.g., "P01-20240202-110250").
            participant_id: Participant ID (e.g., "P01"). Inferred from video_id if empty.
            choices: Multiple choice options (if applicable).

        Returns:
            Dict with answer, confidence, evidence_chain, reasoning_trace.
        """
        if not participant_id and video_id:
            participant_id = video_id.split("-")[0]

        self._current_video_id = video_id
        self._current_participant_id = participant_id

        return self.agent.run(
            question=question,
            video_id=video_id,
            participant_id=participant_id,
            choices=choices,
        )

    def batch_answer(self, questions: List[Dict], limit: int = 10) -> List[Dict]:
        """Answer multiple questions.

        Args:
            questions: List of dicts with 'question', 'video_id', optional 'choices'.
            limit: Max questions to process.

        Returns:
            List of result dicts.
        """
        results = []
        for i, q in enumerate(questions[:limit]):
            print(f"[{i+1}/{min(len(questions), limit)}] {q.get('question', '')[:80]}...")
            result = self.answer(
                question=q["question"],
                video_id=q.get("video_id", ""),
                choices=q.get("choices"),
            )
            results.append(result)
        return results
