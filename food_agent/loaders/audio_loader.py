"""Audio-HDF5 loader for HD-EPIC dataset."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


class AudioLoader:
    """Load and query audio data from HD-EPIC Audio-HDF5 files.

    Each participant has one HDF5 file containing per-video audio waveforms
    stored as float32 arrays. The default sample rate is 48kHz.
    """

    SAMPLE_RATE = 48000

    def __init__(self, hdf5_dir: str | Path, sample_rate: int = 48000):
        self.hdf5_dir = Path(hdf5_dir)
        self.sample_rate = sample_rate
        self._file_cache: Dict[str, h5py.File] = {}

    def _get_file(self, participant_id: str) -> h5py.File:
        if participant_id not in self._file_cache:
            path = self.hdf5_dir / participant_id / f"{participant_id}_audio.hdf5"
            if not path.exists():
                raise FileNotFoundError(f"Audio HDF5 not found: {path}")
            self._file_cache[participant_id] = h5py.File(path, "r")
        return self._file_cache[participant_id]

    def get_video_ids(self, participant_id: str) -> List[str]:
        """Return all video IDs available for a participant."""
        f = self._get_file(participant_id)
        return list(f.keys())

    def load_segment(
        self,
        participant_id: str,
        video_id: str,
        start_time: float,
        end_time: float,
    ) -> Tuple[np.ndarray, int]:
        """Load a raw audio segment by time range.

        Args:
            participant_id: e.g. "P01".
            video_id: e.g. "P01-20240202-110250".
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            Tuple of (audio_samples, sample_rate).
        """
        f = self._get_file(participant_id)
        if video_id not in f:
            raise KeyError(f"Video {video_id} not in {participant_id} HDF5")
        audio = f[video_id]
        sr = self.sample_rate
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        return np.array(audio[start_sample:end_sample], dtype=np.float32), sr

    def get_duration(self, participant_id: str, video_id: str) -> float:
        """Return total duration of a video's audio in seconds."""
        f = self._get_file(participant_id)
        return len(f[video_id]) / self.sample_rate

    def get_all_events(self, participant_id: str, video_id: str) -> List[Dict]:
        """Return all audio events from the annotation data.

        This reads from the annotation CSV rather than the raw waveform.
        Each event has type, start_time, end_time fields.
        """
        from food_agent.loaders._annotation_helper import load_audio_annotations
        return load_audio_annotations(participant_id, video_id)

    def get_events_in_range(
        self,
        participant_id: str,
        video_id: str,
        start_time: float,
        end_time: float,
    ) -> List[Dict]:
        """Return audio events overlapping the given time range."""
        all_events = self.get_all_events(participant_id, video_id)
        return [
            e for e in all_events
            if e["end_time"] > start_time and e["start_time"] < end_time
        ]

    def close(self) -> None:
        for f in self._file_cache.values():
            f.close()
        self._file_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
