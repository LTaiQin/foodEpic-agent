"""Raw video access tools for the graph agent."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw


class VideoToolbox:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def extract_frame_at_time(self, *, video_path: Path, time_s: float, output_name: str) -> Path:
        output_path = self.workspace / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{max(0.0, time_s):.3f}",
                "-i",
                video_path.as_posix(),
                "-frames:v",
                "1",
                output_path.as_posix(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return output_path

    def extract_frames_for_range(
        self,
        *,
        video_path: Path,
        start_time: float,
        end_time: float,
        stride_s: float = 2.0,
        max_frames: int = 8,
        prefix: str,
    ) -> list[Path]:
        paths: list[Path] = []
        duration = max(0.0, end_time - start_time)
        if duration <= 0:
            return [self.extract_frame_at_time(video_path=video_path, time_s=start_time, output_name=f"{prefix}_000.jpg")]
        frame_count = min(max_frames, max(1, int(duration / max(stride_s, 0.1)) + 1))
        if frame_count == 1:
            times = [start_time]
        else:
            step = duration / (frame_count - 1)
            times = [start_time + index * step for index in range(frame_count)]
        for index, time_s in enumerate(times):
            paths.append(
                self.extract_frame_at_time(
                    video_path=video_path,
                    time_s=time_s,
                    output_name=f"{prefix}_{index:03d}_{time_s:09.3f}s.jpg",
                )
            )
        return paths

    def sample_sparse_frames(
        self,
        *,
        video_path: Path,
        start_time: float,
        end_time: float,
        sample_count: int,
        prefix: str,
    ) -> list[Path]:
        duration = max(0.0, end_time - start_time)
        if duration <= 0 or sample_count <= 1:
            return [self.extract_frame_at_time(video_path=video_path, time_s=start_time, output_name=f"{prefix}_000.jpg")]
        margin = min(0.4, duration / max(sample_count * 2, 2))
        usable_start = start_time + margin
        usable_end = end_time - margin
        if usable_end <= usable_start:
            usable_start = start_time
            usable_end = end_time
        times = np.linspace(usable_start, usable_end, num=max(1, sample_count)).tolist()
        paths: list[Path] = []
        for index, time_s in enumerate(times):
            paths.append(
                self.extract_frame_at_time(
                    video_path=video_path,
                    time_s=float(time_s),
                    output_name=f"{prefix}_{index:03d}_{float(time_s):09.3f}s.jpg",
                )
            )
        return paths

    def render_bbox_overlay(self, *, image_path: Path, bbox: list[float], output_name: str) -> Path:
        output_path = self.workspace / output_name
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(self._clamp_bbox(bbox, image.size), outline=(255, 64, 64), width=8)
        image.save(output_path, quality=92)
        return output_path

    def extract_region_with_context(self, *, image_path: Path, bbox: list[float], expand_ratio: float, output_name: str) -> Path:
        output_path = self.workspace / output_name
        image = Image.open(image_path).convert("RGB")
        x1, y1, x2, y2 = self._clamp_bbox(bbox, image.size)
        width = x2 - x1
        height = y2 - y1
        expand_x = width * expand_ratio
        expand_y = height * expand_ratio
        crop_box = self._clamp_bbox([x1 - expand_x, y1 - expand_y, x2 + expand_x, y2 + expand_y], image.size)
        cropped = image.crop(crop_box)
        cropped.save(output_path, quality=92)
        return output_path

    def detect_audio_peaks(
        self,
        *,
        video_path: Path,
        start_time: float,
        end_time: float,
        window_s: float,
        top_k: int,
    ) -> list[dict[str, float]]:
        start_time = max(0.0, float(start_time))
        end_time = max(start_time, float(end_time))
        window_s = max(0.05, float(window_s))
        wav_path = self.workspace / f"audio_{start_time:09.3f}_{end_time:09.3f}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start_time:.3f}",
                "-to",
                f"{end_time:.3f}",
                "-i",
                video_path.as_posix(),
                "-ac",
                "1",
                "-ar",
                "16000",
                wav_path.as_posix(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        samples, sample_rate = self._read_wav_pcm16(wav_path)
        if samples.size == 0:
            return []
        window_size = max(1, int(sample_rate * window_s))
        peaks: list[dict[str, float]] = []
        for offset in range(0, len(samples), window_size):
            chunk = samples[offset : offset + window_size]
            if chunk.size == 0:
                continue
            score = float(np.mean(np.abs(chunk)))
            peak_offset = float(np.argmax(np.abs(chunk))) / float(sample_rate)
            chunk_start = start_time + (offset / float(sample_rate))
            peaks.append(
                {
                    "time_s": chunk_start + peak_offset,
                    "window_start": chunk_start,
                    "window_end": min(end_time, chunk_start + window_s),
                    "score": score,
                }
            )
        peaks.sort(key=lambda item: (-item["score"], item["time_s"]))
        return peaks[: max(1, top_k)]

    def _clamp_bbox(self, bbox: list[float], size: tuple[int, int]) -> list[float]:
        width, height = size
        x1, y1, x2, y2 = bbox
        x1 = max(0.0, min(float(width - 1), float(x1)))
        y1 = max(0.0, min(float(height - 1), float(y1)))
        x2 = max(x1 + 1.0, min(float(width), float(x2)))
        y2 = max(y1 + 1.0, min(float(height), float(y2)))
        return [x1, y1, x2, y2]

    def _read_wav_pcm16(self, path: Path) -> tuple[np.ndarray, int]:
        import wave

        with wave.open(path.as_posix(), "rb") as wav_file:
            sample_rate = int(wav_file.getframerate())
            frames = wav_file.readframes(wav_file.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data, sample_rate
