"""Audio analyzer using BEATs and CLAP for kitchen sound classification."""

from typing import Dict, List, Optional

import numpy as np

from .evidence import Evidence


# Kitchen-relevant sound categories for BEATs (527 AudioSet classes)
KITCHEN_SOUND_CLASSES = [
    "chopping", "cutting", "slicing", "dicing",
    "water running", "pouring water", "filling", "draining",
    "frying", "sizzling", "boiling", "simmering", "stirring",
    "microwave", "blender", "food processor", "kettle",
    "placing down", "picking up", "opening", "closing",
    "crunching", "cracking egg", "peeling",
    "exhaust fan", "refrigerator", "dishwasher",
]

# CLAP zero-shot labels for kitchen sounds
CLAP_KITCHEN_LABELS = [
    "sound of chopping vegetables",
    "sound of water running from faucet",
    "sound of food frying in oil",
    "sound of stirring in a pot",
    "sound of knife cutting on cutting board",
    "sound of boiling water",
    "sound of pouring liquid",
    "sound of placing dish on counter",
    "sound of opening refrigerator",
    "sound of microwave operating",
    "sound of blender running",
    "sound of sizzling food",
]


class AudioAnalyzer:
    """Analyze audio events using BEATs (classification) and CLAP (zero-shot).

    Both models run locally on GPU. BEATs provides 527-class AudioSet
    classification; CLAP provides text-aligned zero-shot classification.
    """

    def __init__(self, beats_model_path: Optional[str] = None, clap_model_path: Optional[str] = None):
        self._beats = None
        self._clap = None
        self._beats_path = beats_model_path
        self._clap_path = clap_model_path
        self._device = "cuda"

    def _load_beats(self):
        if self._beats is not None:
            return
        if self._beats_path is None:
            raise RuntimeError("BEATs model path not configured")
        import torch
        checkpoint = torch.load(self._beats_path, map_location=self._device)
        # Try to import BEATs from the standard location
        try:
            from BEATs import BEATs, BEATsConfig
        except ImportError:
            # Try microsoft/unilm path
            import sys
            beats_dir = "/22liushoulong/agent/hd-epic/third_party/BEATs"
            if beats_dir not in sys.path:
                sys.path.insert(0, beats_dir)
            from BEATs import BEATs, BEATsConfig
        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint["model"])
        model.to(self._device)
        model.eval()
        self._beats = model

    def _load_clap(self):
        if self._clap is not None:
            return
        if self._clap_path is None:
            raise RuntimeError("CLAP model path not configured")
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        model.load_ckpt(self._clap_path)
        model.to(self._device)
        self._clap = model

    def classify_audio(
        self, audio_data: np.ndarray, sample_rate: int = 48000
    ) -> Dict:
        """Classify audio using BEATs.

        Args:
            audio_data: Audio samples as float32 array.
            sample_rate: Sample rate (will resample to 16kHz for BEATs).

        Returns:
            Dict with 'type' (predicted class), 'confidence', 'top_k' list.
        """
        import torch
        import librosa

        self._load_beats()

        # Resample to 16kHz for BEATs
        if sample_rate != 16000:
            audio_16k = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio_data

        audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).to(self._device)

        with torch.no_grad():
            probs = self._beats.extract_features(audio_tensor)[0]

        # Get top predictions
        if isinstance(probs, tuple):
            probs = probs[0]
        probs = probs.cpu().numpy().flatten()

        # Map to AudioSet classes
        top_indices = np.argsort(probs)[::-1][:5]
        top_k = [{"index": int(i), "score": float(probs[i])} for i in top_indices]

        return {
            "type": f"audio_class_{top_indices[0]}",
            "confidence": float(probs[top_indices[0]]),
            "top_k": top_k,
        }

    def zero_shot_classify(
        self, audio_data: np.ndarray, text_labels: List[str], sample_rate: int = 48000
    ) -> Dict:
        """Zero-shot audio classification using CLAP.

        Args:
            audio_data: Audio samples as float32 array.
            text_labels: List of text descriptions to match against.
            sample_rate: Sample rate.

        Returns:
            Dict with 'type' (best matching label), 'confidence', 'scores' dict.
        """
        import torch

        self._load_clap()

        audio_input = audio_data.reshape(1, -1).astype(np.float32)

        audio_embed = self._clap.get_audio_embedding_from_data(
            x=audio_input, use_tensor=False
        )
        text_embed = self._clap.get_text_embedding(text_labels)

        # Normalize
        audio_embed = audio_embed / np.linalg.norm(audio_embed, axis=-1, keepdims=True)
        text_embed = text_embed / np.linalg.norm(text_embed, axis=-1, keepdims=True)

        similarity = (audio_embed @ text_embed.T).flatten()
        best_idx = int(np.argmax(similarity))

        return {
            "type": text_labels[best_idx],
            "confidence": float(similarity[best_idx]),
            "scores": {label: float(similarity[i]) for i, label in enumerate(text_labels)},
        }

    def get_audio_events(
        self,
        audio_loader,
        participant_id: str,
        video_id: str,
        start_time: float,
        end_time: float,
        window_size: float = 5.0,
        stride: float = 2.5,
    ) -> List[Evidence]:
        """Analyze audio in a time range using sliding window.

        Args:
            audio_loader: AudioLoader instance.
            participant_id: e.g. "P01".
            video_id: e.g. "P01-20240202-110250".
            start_time: Start time in seconds.
            end_time: End time in seconds.
            window_size: Analysis window size in seconds.
            stride: Step between windows in seconds.

        Returns:
            List of Evidence objects, one per window.
        """
        evidence_list = []
        t = start_time
        while t < end_time:
            window_end = min(t + window_size, end_time)
            try:
                audio_data, sr = audio_loader.load_segment(
                    participant_id, video_id, t, window_end
                )
                if len(audio_data) < sr * 0.5:  # Skip very short segments
                    t += stride
                    continue

                # Use CLAP zero-shot with kitchen labels
                result = self.zero_shot_classify(audio_data, CLAP_KITCHEN_LABELS, sr)

                evidence_list.append(Evidence(
                    source_module="AudioAnalyzer",
                    evidence_type="audio",
                    time_range={"start": t, "end": window_end},
                    content={
                        "sound_type": result["type"],
                        "all_scores": result["scores"],
                    },
                    confidence=result["confidence"],
                ))
            except Exception:
                pass
            t += stride

        return evidence_list

    def cluster_events(self, events: List[Evidence], max_gap: float = 3.0) -> List[Dict]:
        """Cluster nearby audio events into activity segments.

        Args:
            events: List of audio Evidence objects.
            max_gap: Maximum gap (seconds) between events to merge.

        Returns:
            List of activity segment dicts with label, time_range, dominant_sounds.
        """
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e.time_range["start"])
        segments = []
        seg_start = sorted_events[0].time_range["start"]
        seg_end = sorted_events[0].time_range["end"]
        seg_sounds = [sorted_events[0].content.get("sound_type", "")]

        for ev in sorted_events[1:]:
            if ev.time_range["start"] - seg_end <= max_gap:
                seg_end = ev.time_range["end"]
                seg_sounds.append(ev.content.get("sound_type", ""))
            else:
                from collections import Counter
                dominant = Counter(seg_sounds).most_common(3)
                segments.append({
                    "label": dominant[0][0] if dominant else "unknown",
                    "time_range": [seg_start, seg_end],
                    "dominant_sounds": [s for s, _ in dominant],
                })
                seg_start = ev.time_range["start"]
                seg_end = ev.time_range["end"]
                seg_sounds = [ev.content.get("sound_type", "")]

        from collections import Counter
        dominant = Counter(seg_sounds).most_common(3)
        segments.append({
            "label": dominant[0][0] if dominant else "unknown",
            "time_range": [seg_start, seg_end],
            "dominant_sounds": [s for s, _ in dominant],
        })

        return segments
