"""Module registry: centralized management of all perception module instances."""

from typing import Any, Dict, List, Optional


class ModuleRegistry:
    """Singleton registry for all perception modules.

    Modules register themselves by name; the Agent and tools can
    retrieve any module instance via get().
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._modules: Dict[str, Any] = {}
        return cls._instance

    def register(self, name: str, instance: Any) -> None:
        """Register a module instance by name."""
        self._modules[name] = instance

    def get(self, name: str) -> Optional[Any]:
        """Get a registered module by name."""
        return self._modules.get(name)

    def list_modules(self) -> List[str]:
        """List all registered module names."""
        return list(self._modules.keys())

    def clear(self) -> None:
        """Remove all registrations (useful for testing)."""
        self._modules.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._modules


def register_all_modules(
    audio_loader=None,
    video_loader=None,
    gaze_loader=None,
    slam_loader=None,
    digital_twin_loader=None,
    hands_loader=None,
    beats_path: Optional[str] = None,
    clap_path: Optional[str] = None,
    grounding_dino=None,
    sam2=None,
    sam2_video=None,
    mimo_client=None,
) -> ModuleRegistry:
    """Create and register all perception modules.

    Returns the populated ModuleRegistry.
    """
    from food_agent.perception import (
        AudioAnalyzer, VisualAnalyzer, GazeTracker,
        SpatialReasoner, HandInteractor, NutritionEstimator, MotionTracker,
    )

    registry = ModuleRegistry()

    # Register loaders
    if audio_loader:
        registry.register("AudioLoader", audio_loader)
    if video_loader:
        registry.register("VideoLoader", video_loader)
    if gaze_loader:
        registry.register("GazeLoader", gaze_loader)
    if slam_loader:
        registry.register("SLAMLoader", slam_loader)
    if digital_twin_loader:
        registry.register("DigitalTwinLoader", digital_twin_loader)
    if hands_loader:
        registry.register("HandsLoader", hands_loader)

    # Register perception modules
    registry.register("AudioAnalyzer", AudioAnalyzer(beats_path, clap_path))
    registry.register("VisualAnalyzer", VisualAnalyzer(grounding_dino, sam2, mimo_client))
    if gaze_loader:
        registry.register("GazeTracker", GazeTracker(gaze_loader, grounding_dino))
    if digital_twin_loader and slam_loader:
        registry.register("SpatialReasoner", SpatialReasoner(digital_twin_loader, slam_loader))
    if hands_loader:
        registry.register("HandInteractor", HandInteractor(hands_loader, grounding_dino, sam2))
    registry.register("NutritionEstimator", NutritionEstimator())
    registry.register("MotionTracker", MotionTracker(sam2_video, slam_loader))

    return registry
