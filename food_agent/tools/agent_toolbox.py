"""Complete tool environment for the graph-based food agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from food_agent.memory import GraphEdgeRecord, GraphMemoryStore, GraphNodeRecord
from food_agent.model_client import OpenAICompatibleModelClient
from food_agent.paths import ProjectPaths
from food_agent.spatial_store import SpatialContextStore
from food_agent.state_store import FoodStateStore
from food_agent.tools.graph_tools import GraphToolbox
from food_agent.tools.video_tools import VideoToolbox


TIME_PATTERN = re.compile(r"<TIME\s+(\d+:\d+:\d+(?:\.\d+)?)")
BBOX_PATTERN = re.compile(r"<BBOX\s+([0-9.\s]+)>")


class AgentToolbox:
    """Unified tool registry exposed to the LLM planner/executor."""

    def __init__(self, *, store: GraphMemoryStore, paths: ProjectPaths, model_client: OpenAICompatibleModelClient, video_id: str):
        self.store = store
        self.paths = paths
        self.model_client = model_client
        self.video_id = video_id
        self.graph = GraphToolbox(store)
        self.state_store = FoodStateStore(self.paths.output_root / "event_index")
        self.spatial_store = SpatialContextStore(self.paths.output_root / "event_index")
        self.workspace = self.paths.output_root / "graph_agent_artifacts" / video_id
        self.video = VideoToolbox(self.workspace)

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "query_time",
                "description": "按时间窗口从图谱检索相关节点。",
                "arguments": {"start_time": "float|None", "end_time": "float|None", "limit": "int"},
            },
            {
                "name": "query_object",
                "description": "按对象、地点、动作关键词从图谱检索节点。",
                "arguments": {"query": "str", "limit": "int"},
            },
            {
                "name": "query_event",
                "description": "按事件类型和关键词从图谱检索节点。",
                "arguments": {
                    "event_types": "list[str]|None",
                    "keyword": "str|None",
                    "start_time": "float|None",
                    "end_time": "float|None",
                    "limit": "int",
                },
            },
            {
                "name": "query_ingredient_measurement",
                "description": "检索某个食材的称量记录，并解析 amount 与 unit。",
                "arguments": {"ingredient_name": "str", "start_time": "float|None", "end_time": "float|None", "limit": "int"},
            },
            {
                "name": "compute_nutrition_change",
                "description": "根据 ingredient add 事件直接计算给定时间窗口内营养变化。",
                "arguments": {"start_time": "float", "end_time": "float"},
            },
            {
                "name": "compare_choice_nutrition",
                "description": "根据数据集中同名食材的结构化营养记录，比较选项的营养高低。",
                "arguments": {"choices": "list[str]", "nutrient": "str"},
            },
            {
                "name": "query_spatial_context",
                "description": "查询当前时间附近的 object tracks、mask fixtures、gaze priming 和 audio context。",
                "arguments": {"time_s": "float", "object_name": "str|None", "limit": "int"},
            },
            {
                "name": "get_neighbors",
                "description": "读取已知节点的邻接边，理解节点关系。",
                "arguments": {"node_ids": "list[str]", "edge_types": "list[str]|None", "limit": "int"},
            },
            {
                "name": "resolve_bbox_reference",
                "description": "把题目中的 bbox + 参考时间解析成 object mask、association 和 object track 证据。",
                "arguments": {"bbox": "list[float]", "reference_time": "float", "limit": "int"},
            },
            {
                "name": "estimate_object_movement_count",
                "description": "根据已解析到的 object association / tracks 估计物体在视频中的位置变化次数。",
                "arguments": {"bbox": "list[float]", "reference_time": "float", "choices": "list[str]"},
            },
            {
                "name": "estimate_stationary_start",
                "description": "根据 object tracks 判断从哪个候选起始时间开始，物体保持静止超过阈值秒数。",
                "arguments": {"bbox": "list[float]", "reference_time": "float", "choices": "list[str]", "threshold_s": "float"},
            },
            {
                "name": "extract_frame_at_time",
                "description": "从原视频在指定时刻抽取单帧。",
                "arguments": {"time_s": "float", "tag": "str"},
            },
            {
                "name": "extract_frames_for_range",
                "description": "从原视频在指定时间段稀疏抽帧。",
                "arguments": {"start_time": "float", "end_time": "float", "stride_s": "float", "max_frames": "int", "tag": "str"},
            },
            {
                "name": "extract_input_reference_frames",
                "description": "根据 inputs_json 中给出的 image/video 引用，跨视频提取对应参考帧。",
                "arguments": {"tag": "str"},
            },
            {
                "name": "render_bbox_overlay",
                "description": "在图片上画出 bbox，保留上下文。",
                "arguments": {"image_path": "str", "bbox": "list[float]", "tag": "str"},
            },
            {
                "name": "extract_region_with_context",
                "description": "对 bbox 区域做保留上下文的局部放大。",
                "arguments": {"image_path": "str", "bbox": "list[float]", "expand_ratio": "float", "tag": "str"},
            },
            {
                "name": "inspect_visual_evidence",
                "description": "让模型查看指定图片并输出保守的结构化观察结果。",
                "arguments": {"prompt": "str", "image_paths": "list[str]"},
            },
            {
                "name": "rank_choices_from_state",
                "description": "基于当前工作记忆、证据和题目选项，对 0-4 选项进行评分排序。",
                "arguments": {"question": "str", "choices": "list[str]", "evidence": "list[str]", "working_memory": "list[str]"},
            },
            {
                "name": "sample_choice_frames",
                "description": "针对题目选项中出现的时间点或时间段，为每个选项抽取少量关键帧。",
                "arguments": {"choice_index": "int", "choices": "list[str]", "frames_per_choice": "int", "tag": "str"},
            },
            {
                "name": "count_visual_candidates",
                "description": "根据参考目标图和候选事件时刻，判断哪些时刻发生了目标交互并给出计数。",
                "arguments": {
                    "reference_image_paths": "list[str]",
                    "candidate_times": "list[float]",
                    "choices": "list[str]",
                    "action_hint": "str",
                    "max_candidates": "int",
                    "tag": "str",
                },
            },
            {
                "name": "infer_viewpoint_choice",
                "description": "基于当前视角图像直接在候选方位/位置选项中做视觉定位判断。",
                "arguments": {"question": "str", "choices": "list[str]", "image_paths": "list[str]"},
            },
            {
                "name": "infer_named_fixture_direction",
                "description": "先识别题目中的具名 fixture 在当前厨房语境里最可能对应什么，再映射到钟表方向选项。",
                "arguments": {"question": "str", "choices": "list[str]", "image_paths": "list[str]"},
            },
            {
                "name": "infer_gaze_target_with_context",
                "description": "结合当前视角帧和空间上下文，在注视目标候选中做判断。",
                "arguments": {"question": "str", "choices": "list[str]", "image_paths": "list[str]", "spatial_context": "dict"},
            },
            {
                "name": "identify_image_ingredients",
                "description": "识别一组参考图里分别是什么食材，用于多图营养对比题。",
                "arguments": {"image_paths": "list[str]"},
            },
            {
                "name": "infer_visual_mcq",
                "description": "基于一组按时间顺序排列的关键帧，直接回答与该片段相关的多选题。",
                "arguments": {"question": "str", "choices": "list[str]", "image_paths": "list[str]"},
            },
            {
                "name": "infer_action_mechanism",
                "description": "针对 fine-grained how 题，判断动作是通过按按钮、拉门、推压还是移动把手完成的。",
                "arguments": {"question": "str", "choices": "list[str]", "image_paths": "list[str]"},
            },
            {
                "name": "infer_action_intent",
                "description": "针对 fine-grained why 题，结合短时上下文与活动语境判断动作目的。",
                "arguments": {"question": "str", "choices": "list[str]", "image_paths": "list[str]", "context_notes": "list[str]"},
            },
            {
                "name": "write_observation",
                "description": "把新的观察写回图谱，供后续继续检索。",
                "arguments": {
                    "label": "str",
                    "start_time": "float|None",
                    "end_time": "float|None",
                    "attributes": "dict",
                    "evidence_paths": "list[str]",
                    "keywords": "list[str]|None",
                },
            },
            {
                "name": "finish",
                "description": "当证据足够时结束，并给出最终答案编号与证据摘要。",
                "arguments": {"prediction": "int", "answer": "str", "confidence": "float"},
            },
        ]

    def run(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, tool_name, None)
        if fn is None:
            raise RuntimeError(f"unknown tool: {tool_name}")
        return fn(**self._normalize_args(tool_name, args))

    def query_time(self, start_time: float | None = None, end_time: float | None = None, limit: int = 20) -> dict[str, Any]:
        nodes = self.graph.query_time(video_id=self.video_id, start_time=start_time, end_time=end_time, limit=limit)
        return {"nodes": nodes, "count": len(nodes)}

    def query_object(self, query: str, limit: int = 20) -> dict[str, Any]:
        nodes = self.graph.query_object(video_id=self.video_id, query=query, limit=limit)
        return {"nodes": nodes, "count": len(nodes)}

    def query_event(
        self,
        event_types: list[str] | None = None,
        keyword: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        nodes = self.graph.query_event(
            video_id=self.video_id,
            event_types=event_types,
            keyword=keyword,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        return {"nodes": nodes, "count": len(nodes)}

    def query_ingredient_measurement(
        self,
        ingredient_name: str,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        nodes = self.graph.query_event(
            video_id=self.video_id,
            event_types=["ingredient_event"],
            keyword=ingredient_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        matches: list[dict[str, Any]] = []
        ingredient_tokens = self._name_tokens(ingredient_name)
        for node in nodes:
            attrs = node.get("attributes", {})
            payload = self._parse_payload_json(attrs.get("payload_json"))
            action_type = str(payload.get("action_type") or attrs.get("event_type") or "").lower()
            label = str(attrs.get("label") or node.get("label") or "").lower()
            if "weigh" not in action_type:
                continue
            if ingredient_tokens and not all(token in label for token in ingredient_tokens):
                continue
            amount = payload.get("amount")
            unit = payload.get("amount_unit")
            matches.append(
                {
                    "node_id": node.get("node_id"),
                    "label": node.get("label"),
                    "start_time": node.get("start_time"),
                    "end_time": node.get("end_time"),
                    "amount": amount,
                    "amount_unit": unit,
                    "normalized_answer": self._normalize_measurement_answer(amount, unit),
                    "payload": payload,
                }
            )
        return {"matches": matches, "count": len(matches)}

    def get_neighbors(self, node_ids: list[str], edge_types: list[str] | None = None, limit: int = 50) -> dict[str, Any]:
        edges = self.graph.get_neighbors(node_ids=node_ids, edge_types=edge_types, limit=limit)
        return {"edges": edges, "count": len(edges)}

    def query_spatial_context(self, time_s: float, object_name: str | None = None, limit: int = 20) -> dict[str, Any]:
        context = self.spatial_store.combined_context(
            self.video_id,
            time=float(time_s),
            object_name=object_name,
            audio_window=5.0,
            limit=limit,
        )
        return {
            "object_tracks": context.object_tracks,
            "object_masks": context.object_masks,
            "gaze_priming": context.gaze_priming,
            "audio_events": context.audio_events,
            "count": len(context.object_tracks) + len(context.object_masks) + len(context.gaze_priming) + len(context.audio_events),
        }

    def compute_nutrition_change(self, start_time: float, end_time: float) -> dict[str, Any]:
        rows = self.state_store.ingredient_interval(self.video_id, start_time, end_time)
        totals = {key: 0.0 for key in ("calories", "carbs", "fat", "protein")}
        contributing: list[dict[str, Any]] = []
        for row in rows:
            payload = self._parse_payload_json(row.get("payload_json"))
            if str(payload.get("action_type") or "").lower() != "add":
                continue
            item = {
                "event_id": row.get("event_id"),
                "label": row.get("label"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
            }
            for key in totals:
                value = self._float_or_none(payload.get(key))
                if value is not None:
                    totals[key] += value
                    item[key] = value
            contributing.append(item)
        return {
            "totals": totals,
            "events": contributing,
            "count": len(contributing),
            "start_time": start_time,
            "end_time": end_time,
        }

    def compare_choice_nutrition(self, choices: list[str], nutrient: str = "carbs") -> dict[str, Any]:
        nutrient_key = str(nutrient).strip().lower()
        if nutrient_key not in {"calories", "carbs", "fat", "protein"}:
            raise ValueError(f"unsupported nutrient: {nutrient}")
        ingredients = pd.read_parquet(self.paths.output_root / "event_index" / "ingredients.parquet")
        scored: list[dict[str, Any]] = []
        for index, choice in enumerate(choices):
            label = str(choice).strip().lower()
            subset = ingredients[ingredients["label"].astype(str).str.lower() == label].copy()
            values: list[float] = []
            evidence_ids: list[str] = []
            for _, row in subset.iterrows():
                payload = self._parse_payload_json(row.get("payload_json"))
                value = self._float_or_none(payload.get(nutrient_key))
                if value is None:
                    continue
                values.append(value)
                if row.get("event_id"):
                    evidence_ids.append(str(row.get("event_id")))
            representative = max(values) if values else None
            scored.append(
                {
                    "index": index,
                    "choice": str(choice),
                    "nutrient": nutrient_key,
                    "value": representative,
                    "evidence_ids": evidence_ids[:10],
                    "support_count": len(values),
                }
            )
        valid = [item for item in scored if item.get("value") is not None]
        if valid:
            best = max(valid, key=lambda item: float(item["value"]))
            best_index = int(best["index"])
        else:
            best_index = 0
        return {
            "nutrient": nutrient_key,
            "scores": scored,
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": 0.7 if valid else 0.1,
        }

    def resolve_bbox_reference(self, bbox: list[float], reference_time: float, limit: int = 5) -> dict[str, Any]:
        mask_refs = self.spatial_store.resolve_object_reference(self.video_id, bbox, time=reference_time, limit=limit)
        tracks = pd.read_parquet(self.paths.output_root / "event_index" / "object_tracks.parquet")
        matched_tracks: list[dict[str, Any]] = []
        association_id = None
        object_name = None
        fixture = None
        for ref in mask_refs:
            mask_id = ref.get("mask_id")
            if fixture is None:
                fixture = ref.get("fixture")
            if not mask_id:
                continue
            video_tracks = tracks[tracks["video_id"] == self.video_id].copy()
            for _, row in video_tracks.iterrows():
                mask_ids = json.loads(row.get("masks_json") or "[]")
                if mask_id not in mask_ids:
                    continue
                record = {
                    "association_id": row.get("association_id"),
                    "object_name": row.get("object_name"),
                    "track_id": row.get("track_id"),
                    "track_index": row.get("track_index"),
                    "start_time": row.get("start_time"),
                    "end_time": row.get("end_time"),
                    "masks": mask_ids,
                }
                matched_tracks.append(record)
                association_id = str(row.get("association_id"))
                object_name = str(row.get("object_name"))
                break
            if association_id:
                break
        all_tracks: list[dict[str, Any]] = []
        if association_id:
            subset = tracks[
                (tracks["video_id"] == self.video_id)
                & (tracks["association_id"].astype(str) == association_id)
            ].copy().sort_values(["start_time", "end_time", "track_index"])
            all_tracks = [
                {
                    "association_id": row.get("association_id"),
                    "object_name": row.get("object_name"),
                    "track_id": row.get("track_id"),
                    "track_index": row.get("track_index"),
                    "start_time": row.get("start_time"),
                    "end_time": row.get("end_time"),
                    "masks": json.loads(row.get("masks_json") or "[]"),
                }
                for _, row in subset.iterrows()
            ]
        return {
            "reference_mask_matches": mask_refs,
            "matched_tracks": matched_tracks,
            "association_id": association_id,
            "object_name": object_name,
            "fixture": fixture,
            "tracks": all_tracks,
            "count": len(all_tracks),
        }

    def estimate_object_movement_count(self, bbox: list[float], reference_time: float, choices: list[str]) -> dict[str, Any]:
        resolved = self.resolve_bbox_reference(bbox=bbox, reference_time=reference_time, limit=5)
        tracks = resolved.get("tracks") or []
        movement_count = len(tracks)
        best_index = self._best_choice_for_count(movement_count, choices)
        return {
            "association_id": resolved.get("association_id"),
            "object_name": resolved.get("object_name"),
            "movement_count": movement_count,
            "tracks": tracks,
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": 0.75 if tracks else 0.1,
        }

    def estimate_stationary_start(
        self,
        bbox: list[float],
        reference_time: float,
        choices: list[str],
        threshold_s: float = 150.0,
    ) -> dict[str, Any]:
        resolved = self.resolve_bbox_reference(bbox=bbox, reference_time=reference_time, limit=5)
        tracks = sorted(
            [
                track for track in resolved.get("tracks") or []
                if track.get("start_time") is not None and track.get("end_time") is not None
            ],
            key=lambda item: (float(item.get("start_time") or 0.0), float(item.get("end_time") or 0.0)),
        )
        first_future_movement = next(
            (track for track in tracks if float(track["start_time"]) > float(reference_time)),
            None,
        )
        candidate_floor = float(first_future_movement["end_time"]) if first_future_movement else float(reference_time)
        choice_times = [(index, self._extract_time_points_from_text(str(choice))) for index, choice in enumerate(choices)]
        choice_points = [(index, values[0]) for index, values in choice_times if values]
        valid_candidates: list[dict[str, Any]] = []
        for index, time_s in choice_points:
            if float(time_s) < candidate_floor:
                continue
            next_track = next((track for track in tracks if float(track["start_time"]) > float(time_s)), None)
            if next_track is None:
                continue
            gap = float(next_track["start_time"]) - float(time_s)
            if gap > float(threshold_s):
                valid_candidates.append(
                    {
                        "choice_index": index,
                        "choice_time": time_s,
                        "next_movement_start": float(next_track["start_time"]),
                        "gap_seconds": gap,
                    }
                )
        if valid_candidates:
            best = min(valid_candidates, key=lambda item: float(item["choice_time"]))
            best_index = int(best["choice_index"])
            confidence = 0.8
        else:
            best_index = 0
            confidence = 0.1
        return {
            "association_id": resolved.get("association_id"),
            "object_name": resolved.get("object_name"),
            "tracks": tracks,
            "candidate_floor": candidate_floor,
            "valid_candidates": valid_candidates,
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
        }

    def extract_frame_at_time(self, time_s: float, tag: str = "frame") -> dict[str, Any]:
        video_path = self._video_path()
        output_name = f"{self._safe_tag(tag)}_{time_s:09.3f}s.jpg"
        path = self.video.extract_frame_at_time(video_path=video_path, time_s=time_s, output_name=output_name)
        return {"artifact_path": path.as_posix(), "time_s": time_s}

    def extract_frames_for_range(
        self,
        start_time: float,
        end_time: float,
        stride_s: float = 2.0,
        max_frames: int = 8,
        tag: str = "range",
    ) -> dict[str, Any]:
        video_path = self._video_path()
        paths = self.video.extract_frames_for_range(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            stride_s=stride_s,
            max_frames=max_frames,
            prefix=self._safe_tag(tag),
        )
        return {"artifact_paths": [path.as_posix() for path in paths], "start_time": start_time, "end_time": end_time}

    def extract_input_reference_frames(self, tag: str = "inputs") -> dict[str, Any]:
        payload = self.default_hints("", "{}").get("inputs")
        if not payload:
            payload = {}
        references = self._extract_image_like_inputs(payload)
        artifact_paths: list[str] = []
        items: list[dict[str, Any]] = []
        for index, item in enumerate(references):
            video_id = str(item["video_id"])
            time_s = float(item["time_s"])
            store = self._ensure_video_store(video_id)
            node = store.get_node(f"video:{video_id}")
            if not node:
                continue
            video_path = Path(node["attributes"].get("path") or (node.get("evidence_paths") or [None])[0])
            output_name = f"{self._safe_tag(tag)}_{video_id}_{time_s:09.3f}s_{index:02d}.jpg"
            path = self.video.extract_frame_at_time(video_path=video_path, time_s=time_s, output_name=output_name)
            artifact_paths.append(path.as_posix())
            items.append({"video_id": video_id, "time_s": time_s, "artifact_path": path.as_posix()})
        return {"artifact_paths": artifact_paths, "items": items, "count": len(items)}

    def render_bbox_overlay(self, image_path: str, bbox: list[float], tag: str = "bbox") -> dict[str, Any]:
        source_path = Path(image_path)
        output_name = f"{self._safe_tag(tag)}_{source_path.stem}_bbox.jpg"
        path = self.video.render_bbox_overlay(image_path=source_path, bbox=bbox, output_name=output_name)
        return {"artifact_path": path.as_posix()}

    def extract_region_with_context(self, image_path: str, bbox: list[float], expand_ratio: float = 0.25, tag: str = "crop") -> dict[str, Any]:
        source_path = Path(image_path)
        output_name = f"{self._safe_tag(tag)}_{source_path.stem}_crop.jpg"
        path = self.video.extract_region_with_context(
            image_path=source_path,
            bbox=bbox,
            expand_ratio=expand_ratio,
            output_name=output_name,
        )
        return {"artifact_path": path.as_posix()}

    def inspect_visual_evidence(self, prompt: str, image_paths: list[str]) -> dict[str, Any]:
        response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in image_paths], temperature=0.0)
        text = response.content.strip()
        payload: dict[str, Any]
        try:
            payload = self.model_client._extract_json_object(text)
        except Exception:  # noqa: BLE001
            payload = {"raw_output": text}
        payload["raw_output"] = text
        return payload

    def rank_choices_from_state(
        self,
        question: str,
        choices: list[str],
        evidence: list[str],
        working_memory: list[str],
    ) -> dict[str, Any]:
        prompt = (
            "你是视频问答 agent 的选项评分器。"
            "你不能使用题外知识，只能根据给定证据和工作记忆给 0-4 每个选项打分。"
            "输出 JSON，格式固定为 "
            '{"scores":[{"index":0,"score":0.0,"reason":""}],"best_index":0,"answer":"","confidence":0.0}。'
        )
        payload = {
            "question": question,
            "choices": choices,
            "evidence": evidence[:30],
            "working_memory": working_memory[:30],
        }
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        try:
            response = self.model_client.complete_json(messages, temperature=0.0)
            best_index = int(response.get("best_index", 0))
            return {
                "scores": response.get("scores", []),
                "best_index": best_index,
                "answer": str(response.get("answer") or choices[best_index]),
                "confidence": float(response.get("confidence") or 0.0),
            }
        except Exception:  # noqa: BLE001
            return self._fallback_rank_choices(question=question, choices=choices, evidence=evidence, working_memory=working_memory)

    def sample_choice_frames(self, choice_index: int, choices: list[str], frames_per_choice: int = 3, tag: str = "choice") -> dict[str, Any]:
        if choice_index < 0 or choice_index >= len(choices):
            raise ValueError(f"invalid choice index: {choice_index}")
        choice = str(choices[choice_index])
        ranges = self._extract_time_ranges_from_text(choice)
        if not ranges:
            points = self._extract_time_points_from_text(choice)
            ranges = [(point, point) for point in points]
        all_paths: list[str] = []
        for range_index, (start_time, end_time) in enumerate(ranges[:3]):
            if start_time == end_time:
                sampled = [
                    self.extract_frame_at_time(time_s=start_time, tag=f"{tag}_choice{choice_index}_{range_index}")["artifact_path"]
                ]
            else:
                sampled = self.extract_frames_for_range(
                    start_time=start_time,
                    end_time=end_time,
                    stride_s=max(0.5, (end_time - start_time) / max(frames_per_choice, 1)),
                    max_frames=frames_per_choice,
                    tag=f"{tag}_choice{choice_index}_{range_index}",
                )["artifact_paths"]
            all_paths.extend(sampled)
        return {"artifact_paths": all_paths, "choice_index": choice_index}

    def count_visual_candidates(
        self,
        reference_image_paths: list[str],
        candidate_times: list[float],
        choices: list[str],
        action_hint: str = "close the target item",
        max_candidates: int = 8,
        tag: str = "count",
    ) -> dict[str, Any]:
        sampled_times = [float(value) for value in candidate_times[:max_candidates]]
        sampled_groups: list[dict[str, Any]] = []
        sampled_paths: list[str] = []
        for index, time_s in enumerate(sampled_times):
            before_time = max(0.0, time_s - 0.7)
            after_time = time_s + 0.7
            before_path = str(self.extract_frame_at_time(time_s=before_time, tag=f"{tag}_{index:02d}_before")["artifact_path"])
            center_path = str(self.extract_frame_at_time(time_s=time_s, tag=f"{tag}_{index:02d}_center")["artifact_path"])
            after_path = str(self.extract_frame_at_time(time_s=after_time, tag=f"{tag}_{index:02d}_after")["artifact_path"])
            sampled_groups.append(
                {
                    "index": index,
                    "event_time": time_s,
                    "before_time": before_time,
                    "after_time": after_time,
                    "before_path": before_path,
                    "center_path": center_path,
                    "after_path": after_path,
                }
            )
            sampled_paths.extend([before_path, center_path, after_path])
        prompt_lines = [
            "你在做厨房视频中的目标交互计数。",
            "前几张图是参考目标图。",
            "后面的候选事件每个都给出前一帧、事件帧、后一帧三张图，用来判断同一个目标是否发生了开合变化。",
            f"请判断哪些候选事件真正显示了同一个目标发生了“{action_hint}”相关的动作。",
            "只统计与参考目标一致的交互，不要统计无关物体。",
            '输出 JSON，字段固定为 {"matching_event_indices":[],"count":0,"best_index":0,"confidence":0.0,"reason":""}。',
            "候选图编号从 0 开始，按输入顺序对应。",
        ]
        prompt_lines.append(f"候选事件摘要: {json.dumps([{'index': item['index'], 'event_time': item['event_time']} for item in sampled_groups], ensure_ascii=False)}")
        content_paths = [Path(path) for path in reference_image_paths] + [Path(path) for path in sampled_paths]
        response = self.model_client.inspect_images(prompt="\n".join(prompt_lines), image_paths=content_paths, temperature=0.0)
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
        except Exception:  # noqa: BLE001
            payload = {}
        matching = payload.get("matching_event_indices")
        if not isinstance(matching, list):
            matching = []
        matching = [int(index) for index in matching if isinstance(index, (int, float, str)) and str(index).isdigit()]
        count = int(payload.get("count", len(matching) if matching else 0))
        best_index = self._best_choice_for_count(count, choices)
        return {
            "matching_event_indices": matching,
            "count": count,
            "candidate_times": sampled_times,
            "candidate_groups": sampled_groups,
            "artifact_paths": sampled_paths,
            "best_index": best_index,
            "answer": choices[best_index],
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
        }

    def infer_viewpoint_choice(self, question: str, choices: list[str], image_paths: list[str]) -> dict[str, Any]:
        clockface_hint = ""
        if any("o'clock" in str(choice).lower() for choice in choices):
            clockface_hint = (
                "\n如果选项是钟表方向，严格使用以下映射："
                "\n- 正前方 = 12 o'clock"
                "\n- 正右方 = 3 o'clock"
                "\n- 正后方 = 6 o'clock"
                "\n- 正左方 = 9 o'clock"
                "\n- 只有当目标明显位于前右/前左且不是更接近正右/正左时，才选 1/2 点或 10/11 点方向。"
                "\n- 如果多帧显示视角正在转动，以中间帧对应的主视线为准，前后帧只用于辅助判断转向趋势。"
            )
        prompt = (
            "你在看厨房第一视角视频的当前视角图像。"
            "这些图片按时间顺序排列。"
            "请根据视线方向、空间布局和题目要求，在给定选项中选择最符合的位置/方位答案。"
            "不要使用题外知识。"
            + clockface_hint
            + '\n输出 JSON，字段固定为 {"best_index":0,"answer":"","confidence":0.0,"reason":""}。'
            + f"\n问题: {question}\n选项:\n"
            + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(choices))
        )
        response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in image_paths], temperature=0.0)
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
            best_index = self._resolve_choice_index(
                choices=choices,
                best_index=payload.get("best_index"),
                answer=payload.get("answer"),
            )
        except Exception:  # noqa: BLE001
            return self._fallback_rank_choices(question=question, choices=choices, evidence=[], working_memory=[text])
        return {
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
        }

    def infer_named_fixture_direction(self, question: str, choices: list[str], image_paths: list[str]) -> dict[str, Any]:
        prompt = (
            "你在看厨房第一视角视频的当前视角图像，这些图片按时间顺序排列。"
            "请先判断题目中的具名 fixture/object 在当前厨房语境里最可能对应画面中的哪个具体设备或容器，"
            "然后再把它映射到给定的钟表方向选项。"
            "\n要求："
            "\n1. 先输出你认为题目实体最可能对应的 visible target。"
            "\n2. 再根据中间帧主视线做严格钟表方向判断：正前=12，正右=3，正后=6，正左=9。"
            "\n3. 如果题目名词在英式/口语厨房语境里可能有别称，优先结合当前可见的厨房 fixture 做匹配。"
            '\n输出 JSON，字段固定为 {"target_match":"","best_index":0,"answer":"","confidence":0.0,"reason":""}。'
            f"\n问题: {question}\n选项:\n"
            + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(choices))
        )
        response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in image_paths], temperature=0.0)
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
            best_index = self._resolve_choice_index(
                choices=choices,
                best_index=payload.get("best_index"),
                answer=payload.get("answer"),
            )
        except Exception:  # noqa: BLE001
            return self._fallback_rank_choices(question=question, choices=choices, evidence=[], working_memory=[text])
        return {
            "target_match": str(payload.get("target_match") or ""),
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
        }

    def infer_gaze_target_with_context(
        self,
        question: str,
        choices: list[str],
        image_paths: list[str],
        spatial_context: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "你在看厨房第一视角视频的瞬时注视片段关键帧。"
            "请结合图像和给定的空间上下文，判断说话者/佩戴者此时最可能在看哪个目标。"
            "\n空间上下文中的 fixture、object track 和 gaze priming 只是候选线索，不要脱离图像瞎猜。"
            f"\n空间上下文: {json.dumps(spatial_context, ensure_ascii=False)}"
            '\n输出 JSON，字段固定为 {"best_index":0,"answer":"","confidence":0.0,"reason":""}。'
            f"\n问题: {question}\n选项:\n"
            + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(choices))
        )
        response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in image_paths], temperature=0.0)
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
            best_index = self._resolve_choice_index(
                choices=choices,
                best_index=payload.get("best_index"),
                answer=payload.get("answer"),
            )
        except Exception:  # noqa: BLE001
            return self._fallback_rank_choices(question=question, choices=choices, evidence=[json.dumps(spatial_context, ensure_ascii=False)], working_memory=[text])
        return {
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
        }

    def identify_image_ingredients(self, image_paths: list[str]) -> dict[str, Any]:
        prompt = (
            "你会看到若干张单独的厨房食材参考图。"
            "请按输入顺序识别每张图最可能展示的主要食材名称。"
            "只能输出保守结果。"
            '\n输出 JSON，字段固定为 {"items":[{"index":0,"ingredient":"","confidence":0.0,"reason":""}]}。'
        )
        response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in image_paths], temperature=0.0)
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
        except Exception:  # noqa: BLE001
            payload = {"items": []}
        items = payload.get("items")
        if not isinstance(items, list):
            items = []
        normalized: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except Exception:  # noqa: BLE001
                continue
            normalized.append(
                {
                    "index": index,
                    "ingredient": str(item.get("ingredient") or "").strip().lower(),
                    "confidence": float(item.get("confidence") or 0.0),
                    "reason": str(item.get("reason") or ""),
                }
            )
        return {"items": normalized, "raw_output": text}

    def infer_visual_mcq(self, question: str, choices: list[str], image_paths: list[str]) -> dict[str, Any]:
        prompt = (
            "你在看厨房第一视角视频片段抽取出的关键帧，这些图片按时间顺序排列。"
            "请只根据这些图像回答给定多项选择题，不要使用题外知识。"
            "如果题目是在问下一步交互对象，就忽略当前正在进行的交互，预测接下来最可能交互的对象。"
            "如果题目是在问 how/why/step，就综合前后帧变化、手部动作、目标对象和上下文。"
            '\n输出 JSON，字段固定为 {"best_index":0,"answer":"","confidence":0.0,"reason":""}。'
            f"\n问题: {question}\n选项:\n"
            + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(choices))
        )
        response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in image_paths], temperature=0.0)
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
            best_index = self._resolve_choice_index(
                choices=choices,
                best_index=payload.get("best_index"),
                answer=payload.get("answer"),
            )
        except Exception:  # noqa: BLE001
            return self._fallback_rank_choices(question=question, choices=choices, evidence=[], working_memory=[text])
        return {
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
        }

    def infer_action_mechanism(self, question: str, choices: list[str], image_paths: list[str]) -> dict[str, Any]:
        prompt = (
            "你在看厨房第一视角视频中某个短动作片段的关键帧，这些图片按时间顺序排列。"
            "请专门判断这个动作是通过什么机械方式完成的。"
            "\n重点关注："
            "\n1. 手指是否按下按钮/卡扣"
            "\n2. 手是否抓住把手并拉开"
            "\n3. 是否是向下推压"
            "\n4. 把手本身是否被移动/旋转"
            "\n如果画面里已经显示门处于打开状态，也要根据打开前后帧推断触发方式。"
            '\n输出 JSON，字段固定为 {"best_index":0,"answer":"","confidence":0.0,"reason":""}。'
            f"\n问题: {question}\n选项:\n"
            + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(choices))
        )
        response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in image_paths], temperature=0.0)
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
            best_index = self._resolve_choice_index(
                choices=choices,
                best_index=payload.get("best_index"),
                answer=payload.get("answer"),
            )
        except Exception:  # noqa: BLE001
            return self._fallback_rank_choices(question=question, choices=choices, evidence=[], working_memory=[text])
        return {
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
        }

    def infer_action_intent(self, question: str, choices: list[str], image_paths: list[str], context_notes: list[str]) -> dict[str, Any]:
        prompt = (
            "你在看厨房第一视角视频中某个动作前后的关键帧，这些图片按时间顺序排列。"
            "请判断这个动作的最直接目的。"
            "\n重点关注："
            "\n1. 拿起物体后是否立刻用于擦拭台面/器具"
            "\n2. 是否拿来擦手/干手"
            "\n3. 是否只是收起、挪开、放回"
            "\n4. 当前活动语境是否在清洗、收纳、做饭准备"
            f"\n上下文线索: {context_notes}"
            '\n输出 JSON，字段固定为 {"best_index":0,"answer":"","confidence":0.0,"reason":""}。'
            f"\n问题: {question}\n选项:\n"
            + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(choices))
        )
        response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in image_paths], temperature=0.0)
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
            best_index = self._resolve_choice_index(
                choices=choices,
                best_index=payload.get("best_index"),
                answer=payload.get("answer"),
            )
        except Exception:  # noqa: BLE001
            return self._fallback_rank_choices(question=question, choices=choices, evidence=context_notes, working_memory=[text])
        return {
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
        }

    def write_observation(
        self,
        label: str,
        start_time: float | None = None,
        end_time: float | None = None,
        attributes: dict[str, Any] | None = None,
        evidence_paths: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        safe_label = self._safe_tag(label)[:64]
        node_id = f"observation:{self.video_id}:{safe_label}:{self._node_time_token(start_time, end_time)}"
        node = GraphNodeRecord(
            node_id=node_id,
            node_type="observation",
            label=label,
            video_id=self.video_id,
            start_time=start_time,
            end_time=end_time,
            attributes=attributes or {},
            evidence_paths=evidence_paths or [],
            keywords=keywords or self._keywords_from_payload(label, attributes or {}),
        )
        self.store.upsert_node(node)
        self.store.upsert_edge(
            GraphEdgeRecord(
                edge_id=f"supports:{node_id}",
                source_id=f"video:{self.video_id}",
                target_id=node_id,
                edge_type="supports",
                video_id=self.video_id,
                attributes={"source": "agent_writeback"},
            )
        )
        return {"node_id": node_id}

    def finish(self, prediction: int, answer: str, confidence: float = 0.0) -> dict[str, Any]:
        return {"prediction": int(prediction), "answer": str(answer), "confidence": float(confidence), "done": True}

    def default_hints(self, question: str, inputs_json: str) -> dict[str, Any]:
        try:
            inputs = json.loads(inputs_json or "{}")
        except json.JSONDecodeError:
            inputs = {}
        times = [self._parse_hms(match.group(1)) for match in TIME_PATTERN.finditer(question)]
        bbox_match = BBOX_PATTERN.search(question)
        bbox = None
        if bbox_match:
            parts = [float(token) for token in bbox_match.group(1).strip().split()]
            if len(parts) == 4:
                bbox = parts
        return {
            "times": times,
            "bbox": bbox,
            "input_times": self._extract_times_from_inputs(inputs if isinstance(inputs, dict) else {}),
            "ingredient_name": self._extract_ingredient_name(question),
            "inputs": inputs if isinstance(inputs, dict) else {},
        }

    def _video_path(self) -> Path:
        node = self.store.get_node(f"video:{self.video_id}")
        if not node:
            raise RuntimeError(f"video node missing for video_id={self.video_id}")
        path = node["attributes"].get("path") or (node.get("evidence_paths") or [None])[0]
        if not path:
            raise RuntimeError(f"video path missing for video_id={self.video_id}")
        return Path(path)

    def _ensure_video_store(self, video_id: str) -> GraphMemoryStore:
        root = self.paths.output_root / "graph_memory" / video_id
        store = GraphMemoryStore(root)
        if not store.query_nodes(video_id=video_id, limit=1):
            from food_agent.graph import VideoGraphBuilder

            return VideoGraphBuilder(self.paths).build(video_id)
        return store

    def _safe_tag(self, value: str) -> str:
        compact = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
        return compact or "artifact"

    def _node_time_token(self, start_time: float | None, end_time: float | None) -> str:
        if start_time is None and end_time is None:
            return "na"
        start = "na" if start_time is None else f"{start_time:.3f}"
        end = "na" if end_time is None else f"{end_time:.3f}"
        return f"{start}_{end}"

    def _keywords_from_payload(self, label: str, attributes: dict[str, Any]) -> list[str]:
        tokens = {label.strip().lower()}
        for value in attributes.values():
            if value is None:
                continue
            if isinstance(value, list):
                tokens.update(str(item).strip().lower() for item in value if str(item).strip())
            else:
                text = str(value).strip().lower()
                if text:
                    tokens.add(text)
        return sorted(token for token in tokens if token)

    def _parse_hms(self, text: str) -> float:
        hours, minutes, seconds = text.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    def _best_choice_for_count(self, count: int, choices: list[str]) -> int:
        normalized_count = str(int(count))
        for index, choice in enumerate(choices):
            if str(choice).strip() == normalized_count:
                return index
        return 0

    def _resolve_choice_index(self, *, choices: list[str], best_index: Any, answer: Any) -> int:
        try:
            idx = int(best_index)
            if 0 <= idx < len(choices):
                answer_text = str(answer).strip().lower() if answer is not None else ""
                if not answer_text or str(choices[idx]).strip().lower() == answer_text:
                    return idx
        except Exception:  # noqa: BLE001
            pass
        answer_text = str(answer).strip().lower() if answer is not None else ""
        if answer_text:
            for index, choice in enumerate(choices):
                if str(choice).strip().lower() == answer_text:
                    return index
        return 0

    def _fallback_rank_choices(self, *, question: str, choices: list[str], evidence: list[str], working_memory: list[str]) -> dict[str, Any]:
        corpus = " ".join([question, *evidence, *working_memory]).lower()
        scores: list[dict[str, Any]] = []
        best_index = 0
        best_score = float("-inf")
        for index, choice in enumerate(choices):
            score = 0.0
            normalized_choice = str(choice).strip().lower()
            if normalized_choice and normalized_choice in corpus:
                score += 3.0
            for token in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", normalized_choice):
                if len(token) >= 2 and token in corpus:
                    score += 1.0
            scores.append({"index": index, "score": score, "reason": "fallback lexical overlap"})
            if score > best_score:
                best_score = score
                best_index = index
        confidence = 0.2 if best_score <= 0 else min(0.75, 0.25 + 0.1 * best_score)
        return {
            "scores": scores,
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
        }

    def _extract_time_points_from_text(self, text: str) -> list[float]:
        return [self._parse_hms(match.group(1)) for match in TIME_PATTERN.finditer(text)]

    def _extract_time_ranges_from_text(self, text: str) -> list[tuple[float, float]]:
        points = self._extract_time_points_from_text(text)
        if len(points) < 2:
            return []
        if " to " in text.lower():
            paired: list[tuple[float, float]] = []
            for index in range(0, len(points) - 1, 2):
                paired.append((min(points[index], points[index + 1]), max(points[index], points[index + 1])))
            return paired
        return []

    def _extract_times_from_inputs(self, payload: dict[str, Any]) -> list[float]:
        values: list[float] = []
        for value in payload.values():
            if isinstance(value, dict):
                for key in ("time", "start_time", "end_time"):
                    raw = value.get(key)
                    if isinstance(raw, str) and ":" in raw:
                        values.append(self._parse_hms(raw))
        return values

    def _extract_image_like_inputs(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            video_id = value.get("id")
            raw_time = value.get("time")
            if not video_id or not isinstance(raw_time, str) or ":" not in raw_time:
                continue
            items.append({"video_id": str(video_id), "time_s": self._parse_hms(raw_time)})
        return items

    def _extract_ingredient_name(self, question: str) -> str | None:
        lowered = question.strip()
        match = re.search(r"weigh of (.+?) in this video\??$", lowered, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _parse_payload_json(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _float_or_none(self, value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _normalize_measurement_answer(self, amount: Any, unit: Any) -> str:
        amount_text = str(amount).strip()
        unit_text = str(unit).strip()
        if not amount_text or amount_text.lower() == "n/a":
            return ""
        if amount_text.endswith(".0"):
            amount_text = amount_text[:-2]
        if unit_text and unit_text.lower() != "n/a":
            return f"{amount_text} {unit_text}".strip()
        return amount_text

    def _name_tokens(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z]+", text.lower()) if len(token) >= 2]

    def _normalize_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(args)
        float_keys = {
            "query_time": ["start_time", "end_time"],
            "query_event": ["start_time", "end_time"],
            "query_ingredient_measurement": ["start_time", "end_time"],
            "compute_nutrition_change": ["start_time", "end_time"],
            "query_spatial_context": ["time_s"],
            "resolve_bbox_reference": ["reference_time"],
            "estimate_object_movement_count": ["reference_time"],
            "estimate_stationary_start": ["reference_time", "threshold_s"],
            "extract_frame_at_time": ["time_s"],
            "extract_frames_for_range": ["start_time", "end_time", "stride_s"],
            "extract_region_with_context": ["expand_ratio"],
            "write_observation": ["start_time", "end_time"],
            "finish": ["confidence"],
        }
        int_keys = {
            "query_time": ["limit"],
            "query_object": ["limit"],
            "query_event": ["limit"],
            "query_ingredient_measurement": ["limit"],
            "query_spatial_context": ["limit"],
            "resolve_bbox_reference": ["limit"],
            "get_neighbors": ["limit"],
            "extract_frames_for_range": ["max_frames"],
            "sample_choice_frames": ["choice_index", "frames_per_choice"],
            "count_visual_candidates": ["max_candidates"],
            "finish": ["prediction"],
        }
        for key in float_keys.get(tool_name, []):
            if key in normalized and normalized[key] is not None:
                normalized[key] = float(normalized[key])
        for key in int_keys.get(tool_name, []):
            if key in normalized and normalized[key] is not None:
                normalized[key] = int(normalized[key])
        if tool_name in {"render_bbox_overlay", "extract_region_with_context", "resolve_bbox_reference", "estimate_object_movement_count", "estimate_stationary_start"} and "bbox" in normalized:
            normalized["bbox"] = [float(value) for value in normalized["bbox"]]
        if tool_name == "compare_choice_nutrition":
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
        if tool_name == "query_spatial_context" and "object_name" in normalized and normalized["object_name"] is not None:
            normalized["object_name"] = str(normalized["object_name"])
        if tool_name == "inspect_visual_evidence" and "image_paths" in normalized:
            normalized["image_paths"] = [str(path) for path in normalized["image_paths"]]
        if tool_name == "identify_image_ingredients":
            normalized["image_paths"] = [str(path) for path in normalized.get("image_paths", [])]
        if tool_name == "rank_choices_from_state":
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
            normalized["evidence"] = [str(item) for item in normalized.get("evidence", [])]
            normalized["working_memory"] = [str(item) for item in normalized.get("working_memory", [])]
        if tool_name == "sample_choice_frames" and "choices" in normalized:
            normalized["choices"] = [str(choice) for choice in normalized["choices"]]
        if tool_name == "count_visual_candidates":
            normalized["reference_image_paths"] = [str(path) for path in normalized.get("reference_image_paths", [])]
            normalized["candidate_times"] = [float(value) for value in normalized.get("candidate_times", [])]
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
        if tool_name in {"estimate_object_movement_count", "estimate_stationary_start"}:
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
        if tool_name == "infer_viewpoint_choice":
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
            normalized["image_paths"] = [str(path) for path in normalized.get("image_paths", [])]
        if tool_name == "infer_named_fixture_direction":
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
            normalized["image_paths"] = [str(path) for path in normalized.get("image_paths", [])]
        if tool_name == "infer_gaze_target_with_context":
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
            normalized["image_paths"] = [str(path) for path in normalized.get("image_paths", [])]
        if tool_name == "infer_visual_mcq":
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
            normalized["image_paths"] = [str(path) for path in normalized.get("image_paths", [])]
        if tool_name == "infer_action_mechanism":
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
            normalized["image_paths"] = [str(path) for path in normalized.get("image_paths", [])]
        if tool_name == "infer_action_intent":
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
            normalized["image_paths"] = [str(path) for path in normalized.get("image_paths", [])]
            normalized["context_notes"] = [str(item) for item in normalized.get("context_notes", [])]
        return normalized
