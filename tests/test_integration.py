"""Integration tests for multi-module collaboration (Phase 7).

Tests verify that perception modules, reasoning engine, and agent
work together correctly with real HD-EPIC data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- Fixtures ---

@pytest.fixture
def data_root():
    return Path("/22liushoulong/agent/hd-epic/data/HD-EPIC")


@pytest.fixture
def annotation_root():
    return Path("/22liushoulong/agent/hd-epic/annotations/hd-epic-annotations-main")


@pytest.fixture
def audio_loader(data_root):
    from food_agent.loaders import AudioLoader
    return AudioLoader(data_root / "Audio-HDF5")


@pytest.fixture
def video_loader(data_root):
    from food_agent.loaders import VideoLoader
    return VideoLoader(data_root / "Videos")


@pytest.fixture
def gaze_loader(data_root):
    from food_agent.loaders import GazeLoader
    return GazeLoader(data_root / "SLAM-and-Gaze")


@pytest.fixture
def slam_loader(data_root):
    from food_agent.loaders import SLAMLoader
    return SLAMLoader(data_root / "SLAM-and-Gaze")


@pytest.fixture
def dt_loader(data_root):
    from food_agent.loaders import DigitalTwinLoader
    return DigitalTwinLoader(data_root / "Digital-Twin")


@pytest.fixture
def hands_loader(data_root):
    from food_agent.loaders import HandsLoader
    return HandsLoader(data_root / "Hands-Masks")


# Test video: P01-20240202-110250
TEST_VIDEO = "P01-20240202-110250"
TEST_PARTICIPANT = "P01"


# --- D7.1: Two-module collaboration ---

class TestAudioVideoCollaboration:
    """Audio + Video: audio event timestamp -> extract frame -> visual analysis."""

    def test_audio_to_video_frame(self, audio_loader, video_loader):
        """Audio events drive video frame extraction."""
        # Get audio duration
        duration = audio_loader.get_duration(TEST_PARTICIPANT, TEST_VIDEO)
        assert duration > 0

        # Extract a frame at a timestamp within the audio
        timestamp = min(10.0, duration / 2)
        frame = video_loader.get_frame(TEST_VIDEO, timestamp)
        assert frame is not None
        assert frame.shape[2] == 3  # BGR
        assert frame.dtype == np.uint8

    def test_audio_segment_load(self, audio_loader):
        """Audio segment loading works with real data."""
        audio_data, sr = audio_loader.load_segment(TEST_PARTICIPANT, TEST_VIDEO, 0, 5)
        assert len(audio_data) > 0
        assert sr > 0


class TestGazeVideoCollaboration:
    """Gaze + Video: gaze point -> crop region -> object detection."""

    def test_gaze_to_frame(self, gaze_loader, video_loader):
        """Gaze data aligns with video frames."""
        gaze = gaze_loader.get_gaze_at_time(TEST_PARTICIPANT, TEST_VIDEO, 52.0)
        assert gaze is not None
        assert gaze.depth_m >= 0

        # Get the corresponding video frame
        frame = video_loader.get_frame(TEST_VIDEO, gaze.timestamp_s)
        assert frame is not None

    def test_gaze_fixations(self, gaze_loader):
        """Fixation detection works."""
        fixations = gaze_loader.get_fixations(TEST_PARTICIPANT, TEST_VIDEO, min_duration=0.1)
        # Should find some fixations in any real video
        assert isinstance(fixations, list)


class TestSlamDtCollaboration:
    """SLAM + Digital Twin: pose + 3D model -> spatial relations."""

    def test_pose_to_fixture(self, slam_loader, dt_loader):
        """SLAM pose + Digital Twin -> nearest fixture."""
        pose = slam_loader.get_pose(TEST_PARTICIPANT, TEST_VIDEO, 52.0)
        assert pose is not None
        assert len(pose.position) == 3

        # Get fixtures
        fixtures = dt_loader.get_fixtures(TEST_PARTICIPANT)
        assert len(fixtures) > 0

        # Find nearest fixture
        from food_agent.perception.spatial_reasoner import SpatialReasoner
        sr = SpatialReasoner(dt_loader, slam_loader)
        nearest = sr.get_nearest_fixture(TEST_PARTICIPANT, pose.position)
        assert nearest["fixture_type"] != "unknown"

    def test_spatial_relation(self, dt_loader):
        """Spatial relation between fixtures."""
        fixtures = dt_loader.get_fixtures(TEST_PARTICIPANT)
        assert len(fixtures) >= 2

        rel = dt_loader.get_spatial_relation(TEST_PARTICIPANT, fixtures[0].id, fixtures[1].id)
        assert "relation" in rel
        assert "distance" in rel
        assert rel["distance"] >= 0


class TestHandsCollaboration:
    """Hands loader with mask data."""

    def test_hand_mask_loading(self, hands_loader):
        """Hand masks load correctly."""
        frames = hands_loader.get_available_frames(TEST_VIDEO)
        assert len(frames) > 0

        mask = hands_loader.get_mask(TEST_VIDEO, frames[0], "left")
        assert mask is not None
        assert mask.shape[0] > 0 and mask.shape[1] > 0


# --- D7.3: Agent closed-loop ---

class TestAgentClosedLoop:
    """Agent complete closed-loop tests."""

    def test_router_classification(self):
        """Router classifies questions correctly."""
        from food_agent.reasoning.router import Router
        router = Router()

        assert router.classify_question("Where is the sink?") == "3d_perception"
        assert router.classify_question("What ingredient is this?") == "ingredient"
        assert router.classify_question("How many calories?") == "nutrition"

    def test_agent_state(self):
        """AgentState maintains working state."""
        from food_agent.agent_v2 import AgentState
        from food_agent.perception.evidence import Evidence

        state = AgentState(question="test question")
        state.add_evidence(Evidence(source_module="Test", confidence=0.8))
        state.increment_iteration()

        assert state.iteration == 1
        assert len(state.evidence_list) == 1
        assert "test" in state.to_json().lower()

    def test_reasoning_trace(self):
        """ReasoningTrace records decision process."""
        from food_agent.agent_v2 import ReasoningTrace
        from food_agent.agent_v2.reasoning_trace import StepRecord

        trace = ReasoningTrace(question="test")
        trace.add_step(StepRecord(iteration=1, action="tool_call", tool_name="query_audio"))
        trace.finalize("answer", 0.9)

        assert trace.total_iterations == 1
        assert trace.total_tool_calls == 1
        assert "answer" in trace.to_json()

    def test_evidence_roundtrip(self):
        """Evidence serialization roundtrip."""
        from food_agent.perception.evidence import Evidence

        e = Evidence(
            source_module="Test",
            evidence_type="audio",
            confidence=0.85,
            content={"type": "chopping"},
        )
        j = e.to_json()
        e2 = Evidence.from_json(j)

        assert e.source_module == e2.source_module
        assert e.confidence == e2.confidence
        assert e.content == e2.content


# --- D7.4: Fault tolerance ---

class TestFaultTolerance:
    """Agent fault tolerance tests."""

    def test_missing_data_handled(self, audio_loader):
        """Missing data raises appropriate errors."""
        with pytest.raises(KeyError):
            audio_loader.load_segment(TEST_PARTICIPANT, "nonexistent-video", 0, 5)

    def test_aggregator_empty(self):
        """Aggregator handles empty evidence gracefully."""
        from food_agent.reasoning.aggregator import Aggregator
        agg = Aggregator()
        conf = agg.get_confidence()
        assert conf == 0.0
        summary = agg.get_summary()
        assert "No evidence" in summary

    def test_judge_empty(self):
        """Judge handles empty evidence."""
        from food_agent.reasoning.judge import Judge
        judge = Judge()
        result = judge.evaluate_sufficiency([], "test", {"primary": [], "secondary": []})
        assert result["status"] == "full_search"

    def test_generator_no_llm(self):
        """Generator handles missing LLM client."""
        from food_agent.reasoning.generator import Generator
        gen = Generator(mimo_client=None)
        result = gen.generate_answer([], "test question")
        assert "Unable" in result["answer"] or result["confidence"] == 0


# --- D7.5: Prompt validation ---

class TestPrompts:
    """Verify prompt templates work correctly."""

    def test_decision_prompt_format(self):
        """Decision prompt formats without KeyError."""
        from food_agent.agent_v2.prompts import DECISION_PROMPT_TEMPLATE
        prompt = DECISION_PROMPT_TEMPLATE.format(
            question="test",
            category="general",
            iteration=1,
            max_iterations=10,
            primary_modules="VisualAnalyzer",
            secondary_modules="AudioAnalyzer",
            evidence_count=0,
            evidence_summary="None",
            tools_called="None",
            available_tools="query_audio, query_video",
        )
        assert len(prompt) > 0
        assert "test" in prompt

    def test_answer_prompt_format(self):
        """Answer generation prompt formats correctly."""
        from food_agent.reasoning.generator import Generator
        gen = Generator()
        prompt = gen.format_evidence_prompt([], "What is this?", "visual")
        assert len(prompt) > 0
        assert "What is this?" in prompt


# --- Utilities ---

class TestUtilities:
    """Verify utility functions."""

    def test_find_nearest_timestamp(self):
        from food_agent.utils.time_align import find_nearest_timestamp
        assert find_nearest_timestamp(1.5, [1.0, 2.0, 3.0]) == 1.0
        assert find_nearest_timestamp(1.6, [1.0, 2.0, 3.0]) == 2.0

    def test_cache_manager(self):
        from food_agent.utils.cache import CacheManager
        cm = CacheManager()
        cm.put("k", "v")
        assert cm.get("k") == "v"
        cm.invalidate("k")
        assert cm.get("k") is None

    def test_time_align_hub(self):
        from food_agent.perception.time_align_hub import create_default_hub
        hub = create_default_hub()
        # Video seconds -> SLAM microseconds
        assert hub.convert(52.0, "VideoLoader", "SLAMLoader") == 52000000.0

    def test_nutrition_lookup(self):
        from food_agent.knowledge import NutritionKB
        nk = NutritionKB()
        n = nk.lookup("tomato")
        assert n["calories_per_100g"] == 18

    def test_commonsense_inference(self):
        from food_agent.knowledge import CommonSenseKB
        cs = CommonSenseKB()
        result = cs.infer_cooking_purpose(["tomato", "onion", "olive_oil"])
        assert "salad" in result["possible_dishes"]
