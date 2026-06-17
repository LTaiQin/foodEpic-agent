"""Time alignment utilities for cross-modal timestamp matching."""

import bisect
from typing import List, Optional, Tuple


def find_nearest_timestamp(target: float, timestamps: List[float]) -> float:
    """Find the timestamp in `timestamps` closest to `target`.

    Args:
        target: Query timestamp (seconds).
        timestamps: Sorted list of candidate timestamps (seconds).

    Returns:
        The timestamp value closest to `target`.

    Raises:
        ValueError: If `timestamps` is empty.
    """
    if not timestamps:
        raise ValueError("timestamps list is empty")
    idx = bisect.bisect_left(timestamps, target)
    if idx == 0:
        return timestamps[0]
    if idx >= len(timestamps):
        return timestamps[-1]
    before = timestamps[idx - 1]
    after = timestamps[idx]
    return before if (target - before) <= (after - target) else after


def align_timestamps(
    source_ts: List[float],
    target_ts: List[float],
    max_delta: float = 0.1,
) -> List[Tuple[float, Optional[float]]]:
    """Align source timestamps to target timestamps.

    For each source timestamp, find the closest target timestamp within
    `max_delta` seconds. Returns pairs of (source, matched_target) where
    matched_target is None if no target is within max_delta.

    Args:
        source_ts: Sorted source timestamps (seconds).
        target_ts: Sorted target timestamps (seconds).
        max_delta: Maximum allowed difference in seconds.

    Returns:
        List of (source_ts, target_ts_or_None) pairs.
    """
    result = []
    for s in source_ts:
        nearest = find_nearest_timestamp(s, target_ts)
        if abs(nearest - s) <= max_delta:
            result.append((s, nearest))
        else:
            result.append((s, None))
    return result
