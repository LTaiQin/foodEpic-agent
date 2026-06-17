"""Time alignment hub: cross-module timestamp conversion."""

from typing import Dict, Optional


class TimeAlignHub:
    """Centralized time alignment service for all modules.

    Different data sources may use different time bases (e.g., video
    timestamps vs SLAM timestamps vs audio sample indices). This hub
    manages conversions between them.
    """

    def __init__(self):
        self._timebases: Dict[str, Dict] = {}

    def register_timebase(
        self,
        module_name: str,
        offset: float = 0.0,
        scale: float = 1.0,
        unit: str = "seconds",
    ) -> None:
        """Register a module's time base.

        Args:
            module_name: Name of the module.
            offset: Offset in seconds from the reference time base.
            scale: Scale factor (e.g., 1e-6 for microseconds).
            unit: Original unit ('seconds', 'microseconds', 'samples').
        """
        self._timebases[module_name] = {
            "offset": offset,
            "scale": scale,
            "unit": unit,
        }

    def convert(
        self,
        timestamp: float,
        from_module: str,
        to_module: str,
    ) -> float:
        """Convert a timestamp from one module's time base to another.

        Args:
            timestamp: The timestamp to convert.
            from_module: Source module name.
            to_module: Target module name.

        Returns:
            Converted timestamp in the target module's time base.
        """
        if from_module == to_module:
            return timestamp

        from_tb = self._timebases.get(from_module, {"offset": 0, "scale": 1})
        to_tb = self._timebases.get(to_module, {"offset": 0, "scale": 1})

        # Convert to reference time (seconds)
        ref_time = timestamp * from_tb["scale"] + from_tb["offset"]

        # Convert from reference to target
        converted = (ref_time - to_tb["offset"]) / to_tb["scale"]
        return converted

    def to_reference(self, timestamp: float, module_name: str) -> float:
        """Convert a module's timestamp to reference time (seconds)."""
        tb = self._timebases.get(module_name, {"offset": 0, "scale": 1})
        return timestamp * tb["scale"] + tb["offset"]

    def from_reference(self, ref_time: float, module_name: str) -> float:
        """Convert reference time to a module's timestamp."""
        tb = self._timebases.get(module_name, {"offset": 0, "scale": 1})
        return (ref_time - tb["offset"]) / tb["scale"]


# Default hub with standard time bases for HD-EPIC
def create_default_hub() -> TimeAlignHub:
    """Create a TimeAlignHub with default HD-EPIC time base registrations."""
    hub = TimeAlignHub()
    # Video and SLAM use seconds
    hub.register_timebase("VideoLoader", offset=0, scale=1, unit="seconds")
    hub.register_timebase("SLAMLoader", offset=0, scale=1e-6, unit="microseconds")
    hub.register_timebase("GazeLoader", offset=0, scale=1e-6, unit="microseconds")
    hub.register_timebase("AudioLoader", offset=0, scale=1, unit="seconds")
    hub.register_timebase("HandsLoader", offset=0, scale=1, unit="frame_number")
    return hub
