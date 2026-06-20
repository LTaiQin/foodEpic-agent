"""Full pipeline: wire all modules together for end-to-end agent execution.

This is the glue code that connects:
- Data loaders → Perception modules → Reasoning engine → Agent loop → Answer
"""

import json
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

# Model weight paths
SAM3_WEIGHT_PATH = "/22liushoulong/sam-weight/"
SAM2_CHECKPOINT = "/22liushoulong/agent/hd-epic/checkpoints/sam2_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GDINO_CONFIG = "/22liushoulong/agent/hd-epic/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "/22liushoulong/agent/hd-epic/weights/groundingdino_swint_ogc.pth"
CLAP_WEIGHTS = "/22liushoulong/agent/hd-epic/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
BEATS_WEIGHTS = "/22liushoulong/agent/hd-epic/weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"


class Pipeline:
    """End-to-end pipeline that wires all modules together.

    Usage:
        pipeline = Pipeline()
        result = pipeline.answer("What ingredients are in the video?",
                                  video_id="P01-20240202-110250")
    """

    def __init__(self, config_path: Optional[str] = None, load_models: bool = True):
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

        # Initialize SAM3 (open-vocabulary segmentation)
        self.sam3 = None
        if load_models:
            try:
                from food_agent.perception.sam3_wrapper import SAM3Segmentor
                self.sam3 = SAM3Segmentor(SAM3_WEIGHT_PATH)
                print("SAM3 loaded")
            except Exception as e:
                print(f"SAM3 not available: {e}")

        # Initialize Grounding DINO
        self.gdino = None
        if load_models:
            try:
                import sys
                sys.path.insert(0, str(Path(GDINO_CONFIG).parent.parent.parent))
                from groundingdino.util.inference import load_model
                self.gdino = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
                print("Grounding DINO loaded")
            except Exception as e:
                print(f"Grounding DINO not available: {e}")

        # Initialize perception modules with models
        # Note: CLAP is loaded lazily (on first use) to speed up init
        self.audio_analyzer = AudioAnalyzer(clap_model_path=None)
        self.visual_analyzer = VisualAnalyzer(
            mimo_client=self.mimo_client,
            sam3_segmentor=self.sam3,
            grounding_dino_model=self.gdino,
        )
        self.gaze_tracker = GazeTracker(self.gaze_loader, self.gdino)
        self.spatial_reasoner = SpatialReasoner(self.dt_loader, self.slam_loader)
        self.hand_interactor = HandInteractor(self.hands_loader, self.gdino)
        self.nutrition_estimator = NutritionEstimator()
        self.motion_tracker = MotionTracker(slam_loader=self.slam_loader)

        # Initialize knowledge modules
        self.nutrition_kb = NutritionKB()
        self.commonsense_kb = CommonSenseKB()
        self.scene_graph_kb = SceneGraphKB()

        # Load recipe data from annotations
        self.recipe_kb = RecipeKB()
        try:
            recipe_path = Path(self.config.annotation_root) / "high-level" / "complete_recipes.json"
            if recipe_path.exists():
                with open(recipe_path) as f:
                    recipe_data = json.load(f)
                # Convert to RecipeKB format
                recipes = {}
                for key, recipe in recipe_data.items():
                    name = recipe.get("name", key)
                    recipes[name.lower()] = {
                        "name": name,
                        "participant": recipe.get("participant", ""),
                        "steps": list(recipe.get("steps", {}).values()),
                        "source": recipe.get("source", ""),
                    }
                self.recipe_kb._recipes = recipes
                print(f"Loaded {len(recipes)} recipes")
        except Exception as e:
            print(f"Failed to load recipes: {e}")

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
        registry.register("segment_objects", self._tool_segment_objects)
        registry.register("describe_frame", self._tool_describe_frame)
        registry.register("identify_ingredients", self._tool_identify_ingredients)
        registry.register("query_gaze", self._tool_query_gaze)
        registry.register("query_3d", self._tool_query_3d)
        registry.register("fixture_clock_position", self._tool_fixture_clock_position)
        registry.register("query_hands", self._tool_query_hands)
        registry.register("query_nutrition", self._tool_query_nutrition)
        registry.register("query_motion", self._tool_query_motion)
        registry.register("count_interactions", self._tool_count_interactions)
        registry.register("track_object", self._tool_track_object)
        registry.register("identify_added_ingredient", self._tool_identify_added_ingredient)

        # --- Knowledge tools ---
        registry.register("query_recipe", self._tool_query_recipe)
        registry.register("list_recipes", self._tool_list_recipes)
        registry.register("check_recipe_ingredients", self._tool_check_recipe_ingredients)
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

    def _tool_query_video(self, timestamp: float = 10, text_prompt: str = "", use_scene_graph: bool = False, **kwargs) -> Evidence:
        """Query video frame for visual analysis.

        Uses SAM3 for object detection when available, falls back to MiMo2.5.
        """
        ctx = self._get_context(kwargs)
        try:
            frame = self.video_loader.get_frame(ctx["video_id"], timestamp)
            prompt = text_prompt or "food ingredient kitchen object"
            return self.visual_analyzer.analyze_frame(
                frame, timestamp, prompt,
                use_sam3=True, use_scene_graph=use_scene_graph,
            )
        except Exception as e:
            return Evidence(source_module="VisualAnalyzer", evidence_type="visual",
                          content={"error": str(e)}, confidence=0)

    def _tool_describe_frame(self, timestamp: float = 10, question: str = "", **kwargs) -> Evidence:
        """Describe a video frame using MiMo2.5 Vision API.

        Useful for open-ended questions about what's happening in the scene.
        """
        ctx = self._get_context(kwargs)
        try:
            frame = self.video_loader.get_frame(ctx["video_id"], timestamp)
            if self.mimo_client is None:
                return Evidence(source_module="VisualAnalyzer", evidence_type="visual",
                              content={"error": "No LLM client"}, confidence=0)

            prompt = (
                f"Look at this kitchen scene image carefully. {question}\n"
                "Describe what you see in detail, focusing on:\n"
                "- What objects are visible\n"
                "- What actions are being performed\n"
                "- Any food ingredients or utensils\n"
                "- The spatial arrangement of objects\n"
                "Be specific and factual."
            )
            response = self.mimo_client.call_vision(frame, prompt)

            return Evidence(
                source_module="VisualAnalyzer",
                evidence_type="visual",
                time_range={"start": timestamp, "end": timestamp},
                content={"description": response, "timestamp": timestamp},
                confidence=0.7,
            )
        except Exception as e:
            return Evidence(source_module="VisualAnalyzer", evidence_type="visual",
                          content={"error": str(e)}, confidence=0)

    def _tool_identify_ingredients(self, timestamp: float = 10, **kwargs) -> Evidence:
        """Identify food ingredients in a video frame using MiMo Vision.

        Specifically designed for ingredient recognition tasks.
        """
        ctx = self._get_context(kwargs)
        try:
            frame = self.video_loader.get_frame(ctx["video_id"], timestamp)
            if self.mimo_client is None:
                return Evidence(source_module="NutritionEstimator", evidence_type="nutrition",
                              content={"error": "No LLM client"}, confidence=0)

            prompt = (
                "Look at this kitchen scene image. Identify ALL food ingredients visible.\n"
                "For each ingredient, provide:\n"
                "- name: the ingredient name\n"
                "- location: where it is in the frame\n"
                "- state: raw/cooked/chopped/etc.\n"
                "Return a JSON array of ingredients. Example:\n"
                '[{"name": "tomato", "location": "on cutting board", "state": "being sliced"}]'
            )
            response = self.mimo_client.call_vision(frame, prompt)

            # Parse ingredients
            ingredients = []
            try:
                import json
                start = response.find("[")
                end = response.rfind("]") + 1
                if start >= 0 and end > start:
                    ingredients = json.loads(response[start:end])
            except Exception:
                pass

            return Evidence(
                source_module="NutritionEstimator",
                evidence_type="nutrition",
                time_range={"start": timestamp, "end": timestamp},
                content={
                    "ingredients": ingredients,
                    "ingredient_count": len(ingredients),
                    "raw_response": response[:500],
                },
                confidence=0.7 if ingredients else 0.3,
            )
        except Exception as e:
            return Evidence(source_module="NutritionEstimator", evidence_type="nutrition",
                          content={"error": str(e)}, confidence=0)

    def _tool_segment_objects(self, timestamp: float = 10, text_prompt: str = "food ingredient", **kwargs) -> Evidence:
        """Segment objects in a video frame using SAM3.

        Returns pixel-level masks for detected objects.
        """
        ctx = self._get_context(kwargs)
        try:
            frame = self.video_loader.get_frame(ctx["video_id"], timestamp)
            detections = self.visual_analyzer.detect_objects(frame, text_prompt, method="sam3")
            return Evidence(
                source_module="SAM3Segmentor",
                evidence_type="visual",
                time_range={"start": timestamp, "end": timestamp},
                content={
                    "objects": [
                        {"label": d["label"], "score": d["score"], "bbox": d["bbox"], "area": d.get("area", 0)}
                        for d in detections
                    ],
                    "count": len(detections),
                    "prompt": text_prompt,
                },
                confidence=max((d.get("score", 0) for d in detections), default=0),
            )
        except Exception as e:
            return Evidence(source_module="SAM3Segmentor", evidence_type="visual",
                          content={"error": str(e)}, confidence=0)

    def _tool_query_gaze(self, start_time: float = 0, end_time: float = 30, **kwargs) -> List[Evidence]:
        """Query gaze data in a time range."""
        ctx = self._get_context(kwargs)
        try:
            evidence_list = self.gaze_tracker.get_fixation_targets(
                ctx["participant_id"], ctx["video_id"], start_time, end_time,
            )
            # Enrich with gaze direction description
            for ev in evidence_list:
                yaw = ev.content.get("mean_yaw", 0)
                pitch = ev.content.get("mean_pitch", 0)
                # Describe gaze direction
                if abs(yaw) < 0.3:
                    h_dir = "straight ahead"
                elif yaw < -0.3:
                    h_dir = "to the left"
                else:
                    h_dir = "to the right"
                if abs(pitch) < 0.2:
                    v_dir = "at eye level"
                elif pitch < -0.2:
                    v_dir = "downward"
                else:
                    v_dir = "upward"
                ev.content["gaze_direction"] = f"Looking {h_dir}, {v_dir}"
                ev.content["yaw_degrees"] = round(float(yaw) * 57.3, 1)
                ev.content["pitch_degrees"] = round(float(pitch) * 57.3, 1)

            # Also get gaze priming data (what the person will interact with next)
            try:
                priming = self.gaze_tracker.get_gaze_priming(
                    ctx["participant_id"], ctx["video_id"], start_time, end_time,
                )
                if priming:
                    evidence_list.append(Evidence(
                        source_module="GazeTracker",
                        evidence_type="gaze_priming",
                        time_range={"start": start_time, "end": end_time},
                        content={
                            "priming_events": [
                                {"object": p.get("object_name", ""), "time": p.get("time", 0)}
                                for p in priming[:5]
                            ],
                            "next_likely_interaction": priming[0].get("object_name", "") if priming else "",
                        },
                        confidence=0.7,
                    ))
            except Exception:
                pass

            return evidence_list
        except Exception as e:
            return [Evidence(source_module="GazeTracker", evidence_type="gaze",
                           content={"error": str(e)}, confidence=0)]

    def _tool_query_3d(self, query_type: str = "layout", timestamp: float = 10, **kwargs) -> Evidence:
        """Query 3D spatial information."""
        ctx = self._get_context(kwargs)
        try:
            evidence = self.spatial_reasoner.query_3d(
                ctx["participant_id"], ctx["video_id"], timestamp, query_type,
            )
            # Enrich with clock direction for spatial queries
            if query_type == "wearer_pose" and "position" in evidence.content:
                pos = evidence.content.get("position", [0, 0, 0])
                facing = evidence.content.get("facing", [0, 0, -1])
                # Convert facing direction to clock position
                import math
                angle = math.degrees(math.atan2(facing[0], -facing[2]))
                if angle < 0:
                    angle += 360
                clock = int(round(angle / 30))
                if clock == 0:
                    clock = 12
                evidence.content["facing_clock"] = f"{clock} o'clock"
                evidence.content["facing_angle_degrees"] = round(angle, 1)
            return evidence
        except Exception as e:
            return Evidence(source_module="SpatialReasoner", evidence_type="spatial",
                          content={"error": str(e)}, confidence=0)

    def _tool_fixture_clock_position(self, fixture_name: str = "", timestamp: float = 10, **kwargs) -> Evidence:
        """Compute the clock position of a fixture relative to the wearer.

        Returns the fixture's position as a clock direction (e.g., "3 o'clock").
        """
        if isinstance(fixture_name, list):
            fixture_name = fixture_name[0] if fixture_name else ""
        fixture_name = str(fixture_name)
        ctx = self._get_context(kwargs)
        try:
            import math
            import numpy as np

            # Get wearer pose - try the requested timestamp first, then find nearest available
            pose = self.slam_loader.get_pose(ctx["participant_id"], ctx["video_id"], timestamp)
            if pose is None:
                # Try to find the nearest available timestamp
                df = self.slam_loader._load_df(ctx["participant_id"], ctx["video_id"])
                if df is not None and len(df) > 0:
                    ts_us = df['tracking_timestamp_us'].values
                    nearest_idx = np.argmin(np.abs(ts_us - timestamp * 1e6))
                    nearest_ts = ts_us[nearest_idx] / 1e6
                    pose = self.slam_loader.get_pose(ctx["participant_id"], ctx["video_id"], nearest_ts)
                    if pose is None:
                        return Evidence(source_module="SpatialReasoner", evidence_type="spatial",
                                      content={"error": "no pose data available"}, confidence=0)

            wearer_pos = pose.position
            facing = pose.facing_direction

            # Find the fixture
            fixtures = self.dt_loader.get_fixtures(ctx["participant_id"])
            target = None
            for f in fixtures:
                if fixture_name.lower() in f.fixture_type.lower() or fixture_name.lower() in f.id.lower():
                    target = f
                    break

            if target is None:
                return Evidence(source_module="SpatialReasoner", evidence_type="spatial",
                              content={"error": f"fixture '{fixture_name}' not found"}, confidence=0)

            # Compute direction from wearer to fixture
            dx = target.position[0] - wearer_pos[0]
            dz = target.position[2] - wearer_pos[2]

            # Compute angle from facing direction
            facing_angle = math.atan2(facing[0], -facing[2])
            target_angle = math.atan2(dx, -dz)

            relative_angle = target_angle - facing_angle
            # Normalize to 0-2pi
            while relative_angle < 0:
                relative_angle += 2 * math.pi
            while relative_angle > 2 * math.pi:
                relative_angle -= 2 * math.pi

            # Convert to clock position
            clock = int(round(relative_angle / (2 * math.pi) * 12))
            if clock == 0:
                clock = 12

            distance = float(np.linalg.norm(target.position - wearer_pos))

            return Evidence(
                source_module="SpatialReasoner",
                evidence_type="spatial",
                time_range={"start": timestamp, "end": timestamp},
                content={
                    "fixture_name": target.id,
                    "fixture_type": target.fixture_type,
                    "clock_position": f"{clock} o'clock",
                    "clock_number": clock,
                    "distance_meters": round(distance, 2),
                    "wearer_facing_clock": f"{int(round(math.degrees(facing_angle) / 30))} o'clock",
                },
                confidence=0.85,
            )
        except Exception as e:
            return Evidence(source_module="SpatialReasoner", evidence_type="spatial",
                          content={"error": str(e)}, confidence=0)

    def _tool_query_hands(self, frame_number: int = 300, **kwargs) -> Evidence:
        """Query hand interactions for a frame."""
        ctx = self._get_context(kwargs)
        try:
            evidence = self.hand_interactor.get_hand_interactions(
                ctx["participant_id"], ctx["video_id"], frame_number,
            )
            # Enrich with action inference
            interactions = evidence.content.get("interactions", [])
            for inter in interactions:
                if inter.get("has_contact"):
                    mask_area = inter.get("mask_area", 0)
                    if mask_area > 100000:
                        inter["likely_action"] = "rubbing or washing hands"
                    elif mask_area > 50000:
                        inter["likely_action"] = "holding or manipulating object"
                    elif mask_area > 10000:
                        inter["likely_action"] = "touching or grasping"
                    else:
                        inter["likely_action"] = "light contact"
            return evidence
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

    def _tool_count_interactions(
        self, bbox: List[float] = None, timestamp: float = 0,
        time_range: float = 15.0, num_samples: int = 10, **kwargs
    ) -> Evidence:
        """Count open/close interactions for an object in a bounding box region.

        Samples frames around the timestamp, analyzes the scene to detect state changes.

        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates.
            timestamp: Center timestamp in seconds.
            time_range: Seconds before/after timestamp to analyze.
            num_samples: Number of frames to sample.
        """
        ctx = self._get_context(kwargs)
        try:
            if not bbox or len(bbox) < 4:
                return Evidence(source_module="VisualAnalyzer", evidence_type="visual",
                              content={"error": "bbox required [x1,y1,x2,y2]"}, confidence=0)

            # First, identify what fixture is at the bounding box
            try:
                center_frame = self.video_loader.get_frame(ctx["video_id"], timestamp)
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                identify_prompt = (
                    f"Look at this egocentric kitchen video frame. "
                    f"What kitchen fixture is at pixel location ({center_x}, {center_y})? "
                    f"Is it a fridge, cabinet, drawer, dishwasher, washing machine, or something else? "
                    f"Reply with just the fixture name (1-2 words)."
                )
                fixture_name = self.mimo_client.call_vision(center_frame, identify_prompt)
                fixture_name = fixture_name.strip().split('\n')[0].strip()[:30]
            except Exception:
                fixture_name = "fixture"

            # Sample frames around timestamp
            t_start = max(0, timestamp - time_range)
            t_end = timestamp + time_range
            timestamps = [t_start + (t_end - t_start) * i / (num_samples - 1) for i in range(num_samples)]

            states = []
            for ts in timestamps:
                try:
                    frame = self.video_loader.get_frame(ctx["video_id"], ts)
                    prompt = (
                        f"Look at this egocentric kitchen video frame. Focus on the {fixture_name} "
                        f"at the center of the image. Is the {fixture_name} OPEN (door/drawer ajar, "
                        f"can see inside) or CLOSED (flush, shut, door closed)? "
                        f"Reply with ONLY one word: open or closed."
                    )
                    response = self.mimo_client.call_vision(frame, prompt)
                    resp_lower = response.lower().strip()
                    if "open" in resp_lower:
                        state = "open"
                    elif "closed" in resp_lower or "shut" in resp_lower:
                        state = "closed"
                    else:
                        state = "unknown"
                    states.append({"timestamp": round(ts, 2), "state": state, "raw": response[:50]})
                except Exception as e:
                    states.append({"timestamp": ts, "state": "error", "error": str(e)[:50]})

            # Count transitions (state changes)
            transitions = 0
            close_count = 0
            open_count = 0
            for i in range(1, len(states)):
                if states[i]["state"] != states[i-1]["state"] and states[i]["state"] != "unknown" and states[i-1]["state"] != "unknown":
                    transitions += 1
                    if states[i]["state"] == "closed":
                        close_count += 1
                    elif states[i]["state"] == "open":
                        open_count += 1

            return Evidence(
                source_module="VisualAnalyzer",
                evidence_type="interaction_count",
                time_range={"start": t_start, "end": t_end},
                content={
                    "fixture_name": fixture_name,
                    "states": states,
                    "transitions": transitions,
                    "close_count": close_count,
                    "open_count": open_count,
                    "likely_interaction_count": close_count,
                },
                confidence=0.7 if all(s["state"] != "unknown" for s in states) else 0.4,
            )
        except Exception as e:
            return Evidence(source_module="VisualAnalyzer", evidence_type="interaction_count",
                          content={"error": str(e)}, confidence=0)

    def _tool_track_object(
        self, bbox: List[float] = None, timestamp: float = 0,
        time_after: float = 15.0, num_samples: int = 8, **kwargs
    ) -> Evidence:
        """Track where an object is placed after being picked up."""
        ctx = self._get_context(kwargs)
        try:
            if not bbox or len(bbox) < 4:
                return Evidence(source_module="VisualAnalyzer", evidence_type="visual",
                              content={"error": "bbox required [x1,y1,x2,y2]"}, confidence=0)

            # Identify object at pickup time
            try:
                pickup_frame = self.video_loader.get_frame(ctx["video_id"], timestamp)
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                obj_prompt = (
                    f"Look at this egocentric kitchen video frame. The person is picking up an object "
                    f"at pixel location approximately ({center_x}, {center_y}). "
                    f"What specific kitchen object is the person picking up? "
                    f"Reply with just the object name (1-3 words)."
                )
                object_name = self.mimo_client.call_vision(pickup_frame, obj_prompt)
                object_name = object_name.strip().split('\n')[0].strip()[:40]
            except Exception:
                object_name = "kitchen object"

            # Sample frames after pickup - more samples, wider range
            timestamps = [timestamp + 2, timestamp + 4, timestamp + 6, timestamp + 8,
                         timestamp + 10, timestamp + 12, timestamp + 15, timestamp + 20]

            locations = []
            for ts in timestamps:
                try:
                    frame = self.video_loader.get_frame(ctx["video_id"], ts)
                    prompt = (
                        f"An egocentric kitchen video frame at {ts:.0f}s. A person picked up a {object_name} "
                        f"at {timestamp:.0f}s. Look carefully at the person's hands and the scene. "
                        f"Is the person still HOLDING the {object_name}? Or has it been PLACED somewhere? "
                        f"Reply in one sentence: 'Holding [object]' or 'Placed on/in [location]'. "
                        f"Be specific about the location (e.g., 'Placed on counter near sink', 'Placed in drawer')."
                    )
                    response = self.mimo_client.call_vision(frame, prompt)
                    is_holding = "holding" in response.lower()[:20]
                    is_placed = "placed" in response.lower()[:20]
                    locations.append({
                        "timestamp": round(ts, 2),
                        "location": response[:200],
                        "holding": is_holding,
                        "placed": is_placed,
                    })
                except Exception:
                    locations.append({"timestamp": ts, "location": "error", "holding": False, "placed": False})

            # Find placement location
            placed_locs = [l["location"] for l in locations if l["placed"]]
            final_location = placed_locs[-1] if placed_locs else "not determined"

            return Evidence(
                source_module="VisualAnalyzer",
                evidence_type="object_tracking",
                time_range={"start": timestamp, "end": timestamp + time_after},
                content={
                    "object_name": object_name[:50],
                    "final_location": final_location[:200],
                    "placed_count": len(placed_locs),
                },
                confidence=0.7 if placed_locs else 0.3,
            )
        except Exception as e:
            return Evidence(source_module="VisualAnalyzer", evidence_type="object_tracking",
                          content={"error": str(e)}, confidence=0)

    def _tool_identify_added_ingredient(
        self, start_time: float = 0, end_time: float = 30,
        candidates: List[str] = None, **kwargs
    ) -> Evidence:
        """Identify which ingredient is being added to a dish during a time range.

        Analyzes video frames and hand interactions to determine which ingredient
        from the candidates is being added.

        Args:
            start_time: Start of the time range in seconds.
            end_time: End of the time range in seconds.
            candidates: List of candidate ingredient names to match against.
        """
        ctx = self._get_context(kwargs)
        try:
            # Sample frames in the middle of the time range
            mid_time = (start_time + end_time) / 2
            timestamps = [mid_time - 2, mid_time, mid_time + 2]

            for ts in timestamps:
                try:
                    frame = self.video_loader.get_frame(ctx["video_id"], ts)
                    if frame is None:
                        continue

                    # Ask what ingredient is being added
                    prompt = (
                        "Look at this egocentric kitchen video frame. "
                        "The person is adding an ingredient to a dish. "
                        "What specific ingredient is being added right now? "
                        "Look at what the person is holding in their hands. "
                        "Reply with just the ingredient name (1-2 words)."
                    )
                    response = self.mimo_client.call_vision(frame, prompt)
                    identified = response.strip().split('\n')[0].strip()[:30]

                    # Match to candidates if provided
                    if candidates:
                        for i, candidate in enumerate(candidates):
                            candidate_lower = candidate.lower()
                            identified_lower = identified.lower()
                            # Check if identified ingredient matches a candidate
                            if (candidate_lower in identified_lower or
                                identified_lower in candidate_lower or
                                any(word in candidate_lower for word in identified_lower.split())):
                                return Evidence(
                                    source_module="VisualAnalyzer",
                                    evidence_type="ingredient_added",
                                    time_range={"start": start_time, "end": end_time},
                                    content={
                                        "identified_ingredient": identified,
                                        "matched_candidate": candidate,
                                        "matched_index": i,
                                        "timestamp": ts,
                                    },
                                    confidence=0.7,
                                )

                    # Return raw identification if no match
                    return Evidence(
                        source_module="VisualAnalyzer",
                        evidence_type="ingredient_added",
                        time_range={"start": start_time, "end": end_time},
                        content={
                            "identified_ingredient": identified,
                            "matched_candidate": None,
                            "matched_index": -1,
                            "timestamp": ts,
                        },
                        confidence=0.5,
                    )
                except Exception:
                    continue

            return Evidence(
                source_module="VisualAnalyzer",
                evidence_type="ingredient_added",
                time_range={"start": start_time, "end": end_time},
                content={"error": "could not identify ingredient", "identified_ingredient": "unknown"},
                confidence=0.0,
            )
        except Exception as e:
            return Evidence(source_module="VisualAnalyzer", evidence_type="ingredient_added",
                          content={"error": str(e)}, confidence=0)

    def _tool_query_recipe(self, recipe_name: str = "", step_number: int = 0, **kwargs) -> Evidence:
        """Query recipe knowledge base."""
        try:
            if isinstance(recipe_name, list):
                recipe_name = recipe_name[0] if recipe_name else ""
            recipe_name = str(recipe_name)
            if recipe_name:
                recipe = self.recipe_kb.get_recipe(recipe_name) if hasattr(self, 'recipe_kb') else None
                if recipe:
                    return Evidence(
                        source_module="RecipeKB",
                        evidence_type="recipe",
                        time_range={"start": 0, "end": 0},
                        content={
                            "recipe_name": recipe.get("name", ""),
                            "steps": recipe.get("steps", []),
                            "step_count": len(recipe.get("steps", [])),
                            "source": recipe.get("source", ""),
                        },
                        confidence=0.9,
                    )
            return Evidence(
                source_module="RecipeKB",
                evidence_type="recipe",
                content={"error": "recipe not found", "available": []},
                confidence=0.0,
            )
        except Exception as e:
            return Evidence(source_module="RecipeKB", evidence_type="recipe",
                          content={"error": str(e)}, confidence=0)

    def _tool_list_recipes(self, **kwargs) -> Evidence:
        """List all available recipes in the knowledge base."""
        try:
            recipes = list(self.recipe_kb._recipes.keys()) if hasattr(self.recipe_kb, '_recipes') else []
            return Evidence(
                source_module="RecipeKB",
                evidence_type="recipe",
                content={
                    "recipe_count": len(recipes),
                    "recipes": recipes[:20],
                },
                confidence=0.9,
            )
        except Exception as e:
            return Evidence(source_module="RecipeKB", evidence_type="recipe",
                          content={"error": str(e)}, confidence=0)

    def _tool_check_recipe_ingredients(self, recipe_name: str = "", ingredients: List[str] = None, **kwargs) -> Evidence:
        """Check which ingredients are used in a recipe.

        Directly compares ingredient list against recipe steps.
        """
        try:
            if isinstance(recipe_name, list):
                recipe_name = recipe_name[0] if recipe_name else ""
            recipe_name = str(recipe_name)
            if not recipe_name or not ingredients:
                return Evidence(source_module="RecipeKB", evidence_type="recipe",
                              content={"error": "need recipe_name and ingredients"}, confidence=0)

            recipe = self.recipe_kb.get_recipe(recipe_name) if hasattr(self, 'recipe_kb') else None
            if not recipe:
                return Evidence(source_module="RecipeKB", evidence_type="recipe",
                              content={"error": f"recipe '{recipe_name}' not found"}, confidence=0)

            all_text = ' '.join(recipe.get('steps', [])).lower()

            # Ingredient aliases (stilton = blue cheese, cheddar = cheese, etc.)
            aliases = {
                'stilton': ['blue cheese', 'stilton'],
                'cheddar': ['cheese', 'cheddar'],
                'parmesan': ['cheese', 'parmesan', 'parmigiano'],
                'mozzarella': ['cheese', 'mozzarella'],
                'feta': ['cheese', 'feta'],
                'yogurt': ['yoghurt', 'yogurt'],
                'cilantro': ['coriander', 'cilantro'],
                'scallion': ['green onion', 'scallion', 'spring onion'],
            }

            results = []
            for ing in ingredients:
                if isinstance(ing, list):
                    ing = ing[0] if ing else ""
                ing_lower = str(ing).lower()
                # Check direct match
                found = ing_lower in all_text
                # Check aliases
                if not found and ing_lower in aliases:
                    for alias in aliases[ing_lower]:
                        if alias in all_text:
                            found = True
                            break
                results.append({
                    "ingredient": ing,
                    "in_recipe": found,
                    "match_type": "direct" if found else "not_found",
                })

            return Evidence(
                source_module="RecipeKB",
                evidence_type="recipe",
                content={
                    "recipe_name": recipe.get("name", ""),
                    "ingredient_check": results,
                    "not_in_recipe": [r["ingredient"] for r in results if not r["in_recipe"]],
                    "in_recipe": [r["ingredient"] for r in results if r["in_recipe"]],
                },
                confidence=0.95,
            )
        except Exception as e:
            return Evidence(source_module="RecipeKB", evidence_type="recipe",
                          content={"error": str(e)}, confidence=0)

    def _tool_query_nutrition_kb(self, ingredient: str = "", **kwargs) -> Dict:
        """Look up nutrition facts."""
        if isinstance(ingredient, list):
            ingredient = ingredient[0] if ingredient else ""
        ingredient = str(ingredient).strip()
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
