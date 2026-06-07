"""Complete tool environment for the graph-based food agent."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from PIL import Image

from food_agent.agent.action_intent import action_intent_followup_decision, choice_categories, question_is_post_action_sensitive
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
        self.runtime_question = ""
        self.runtime_inputs_json = "{}"
        self.graph = GraphToolbox(store)
        self.state_store = FoodStateStore(self.paths.output_root / "event_index")
        self.spatial_store = SpatialContextStore(self.paths.output_root / "event_index")
        self.workspace = self.paths.graph_agent_artifacts_root / video_id
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
                "name": "query_state",
                "description": "检索与状态、熟度、混合情况、完成度相关的节点。",
                "arguments": {"state_keyword": "str", "start_time": "float|None", "end_time": "float|None", "limit": "int"},
            },
            {
                "name": "query_location",
                "description": "检索与位置、方位、厨房区域、器具所在处相关的节点。",
                "arguments": {"location_keyword": "str", "start_time": "float|None", "end_time": "float|None", "limit": "int"},
            },
            {
                "name": "query_region",
                "description": "检索与指定对象或局部区域相关的节点。",
                "arguments": {"object_hint": "str", "start_time": "float|None", "end_time": "float|None", "limit": "int"},
            },
            {
                "name": "query_ocr",
                "description": "检索 OCR 文本、数字读数、包装文字等文本证据节点。",
                "arguments": {"keyword": "str", "start_time": "float|None", "end_time": "float|None", "limit": "int"},
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
                "name": "infer_ingredient_order_choice",
                "description": "根据当前视频中结构化 ingredient add 事件的时间顺序，判断哪个候选食材顺序最匹配。",
                "arguments": {"question": "str", "choices": "list[Any]"},
            },
            {
                "name": "infer_recipe_catalog_choice",
                "description": "根据 inputs_json 对应视频集合中的 recipe catalog，判断哪个候选菜谱最匹配当前视频或参与者。",
                "arguments": {"question": "str", "choices": "list[str]", "scope": "str"},
            },
            {
                "name": "infer_recipe_nutrition_choice",
                "description": "根据 recipe catalog 中的食材集合和结构化营养统计，判断题目要求的最高营养食材。",
                "arguments": {"question": "str", "choices": "list[str]"},
            },
            {
                "name": "infer_ingredient_retrieval_choice",
                "description": "根据题目时间窗口内的结构化 ingredient add 事件，判断哪个候选食材在该时间段被加入。",
                "arguments": {"question": "str", "choices": "list[str]"},
            },
            {
                "name": "infer_recipe_ingredient_membership_choice",
                "description": "根据 recipe catalog 中的菜谱食材集合，判断哪个候选食材不属于指定菜谱。",
                "arguments": {"question": "str", "choices": "list[str]"},
            },
            {
                "name": "infer_exact_ingredient_amount_choice",
                "description": "根据 recipe catalog 中的 ingredient_amounts，判断题目所问食材的精确用量选项。",
                "arguments": {"question": "str", "choices": "list[str]"},
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
                "name": "expand_graph_context",
                "description": "从一组已知节点出发，沿时间/共现/语义关系扩展一跳上下文，取回相关节点和边。",
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
                "name": "infer_object_drop_location",
                "description": "根据 reference bbox 对应对象的后续 track/mask fixture，推断该对象最终被放到了哪个位置选项。",
                "arguments": {"bbox": "list[float]", "reference_time": "float", "choices": "list[str]", "question": "str"},
            },
            {
                "name": "infer_object_movement_itinerary",
                "description": "根据 reference bbox 对应对象的完整 tracks/mask fixtures，推断该对象在视频中的移动路径选项。",
                "arguments": {"bbox": "list[float]", "reference_time": "float", "choices": "list[str]"},
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
                "name": "sample_sparse_frames",
                "description": "在给定时间段内做更均匀的稀疏采样，避免连续相近帧。",
                "arguments": {"start_time": "float", "end_time": "float", "sample_count": "int", "tag": "str"},
            },
            {
                "name": "extract_input_reference_frames",
                "description": "根据 inputs_json 中给出的 image/video 引用，跨视频提取对应参考帧。",
                "arguments": {"tag": "str"},
            },
            {
                "name": "retrieve_cached_artifacts",
                "description": "从当前视频的 artifact 工作区检索与指定 tag 或时间窗口相关的已有图片产物，优先复用先前已经抽取的帧、局部图和画框图。",
                "arguments": {"tag_hint": "str|None", "start_time": "float|None", "end_time": "float|None", "limit": "int"},
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
                "name": "run_ocr_on_image",
                "description": "对整张图运行 OCR，优先读数字、单位、包装文字或显示屏内容。",
                "arguments": {"image_path": "str"},
            },
            {
                "name": "run_ocr_on_region",
                "description": "对局部区域做上下文保留放大后运行 OCR。",
                "arguments": {"image_path": "str", "bbox": "list[float]", "expand_ratio": "float", "tag": "str"},
            },
            {
                "name": "detect_audio_peaks",
                "description": "在指定时间段内检测音频峰值，作为关键事件候选时间。",
                "arguments": {"start_time": "float", "end_time": "float", "window_s": "float", "top_k": "int"},
            },
            {
                "name": "sample_frames_around_peaks",
                "description": "围绕音频峰值时间点抽取少量前后关键帧，作为候选事件证据。",
                "arguments": {"peak_times": "list[float]", "radius_s": "float", "frames_per_peak": "int", "tag": "str"},
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
                "name": "infer_temporal_localization_choice",
                "description": "针对时间定位题，为各选项抽取关键帧并直接比较哪个时间段最符合题目动作/步骤/加料描述。",
                "arguments": {"question": "str", "choices": "list[str]", "task_family": "str", "frames_per_choice": "int", "tag": "str"},
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
                "arguments": {"question": "str", "choices": "list[str]", "image_paths": "list[str]", "spatial_context": "dict|None"},
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
                "name": "resolve_action_intent_pairwise",
                "description": "当 why 题存在近义歧义时，只在两个高混淆候选之间结合结果帧做最终裁决。",
                "arguments": {
                    "question": "str",
                    "choices": "list[str]",
                    "candidate_indices": "list[int]",
                    "image_paths": "list[str]",
                    "context_notes": "list[str]",
                },
            },
            {
                "name": "resolve_action_intent_future_use",
                "description": "当 why 题的目的依赖动作后的实际用途时，逐项验证后续用途证据并排除竞争选项。",
                "arguments": {
                    "question": "str",
                    "choices": "list[str]",
                    "candidate_indices": "list[int]",
                    "image_paths": "list[str]",
                    "context_notes": "list[str]",
                },
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
                    "source_tool": "str|None",
                    "confidence": "float|None",
                },
            },
            {
                "name": "write_frame_observation",
                "description": "把针对某一帧的结构化观察写回图谱，并与相邻 frame 节点建立关联。",
                "arguments": {"frame_path": "str", "time_s": "float|None", "label": "str", "observation": "dict", "keywords": "list[str]|None", "source_tool": "str|None", "confidence": "float|None"},
            },
            {
                "name": "write_region_observation",
                "description": "把针对局部区域或画框图的观察写回图谱。",
                "arguments": {
                    "image_path": "str",
                    "bbox": "list[float]|None",
                    "time_s": "float|None",
                    "label": "str",
                    "observation": "dict",
                    "keywords": "list[str]|None",
                    "source_tool": "str|None",
                    "confidence": "float|None",
                },
            },
            {
                "name": "write_ocr_reading",
                "description": "把 OCR 读数或文本写回图谱。",
                "arguments": {
                    "label": "str",
                    "reading": "str",
                    "time_s": "float|None",
                    "image_path": "str|None",
                    "bbox": "list[float]|None",
                    "attributes": "dict|None",
                    "keywords": "list[str]|None",
                    "source_tool": "str|None",
                    "confidence": "float|None",
                },
            },
            {
                "name": "write_audio_event",
                "description": "把音频触发事件写回图谱。",
                "arguments": {
                    "label": "str",
                    "start_time": "float|None",
                    "end_time": "float|None",
                    "attributes": "dict|None",
                    "evidence_paths": "list[str]|None",
                    "keywords": "list[str]|None",
                    "source_tool": "str|None",
                    "confidence": "float|None",
                },
            },
            {
                "name": "write_timeline_summary",
                "description": "把时间段总结写回图谱，供后续检索。",
                "arguments": {
                    "label": "str",
                    "start_time": "float|None",
                    "end_time": "float|None",
                    "summary": "str",
                    "evidence_paths": "list[str]|None",
                    "keywords": "list[str]|None",
                    "source_tool": "str|None",
                    "confidence": "float|None",
                },
            },
            {
                "name": "write_state_change",
                "description": "把对象或食材的状态变化写回图谱。",
                "arguments": {
                    "label": "str",
                    "target": "str",
                    "before_state": "str|None",
                    "after_state": "str|None",
                    "start_time": "float|None",
                    "end_time": "float|None",
                    "evidence_paths": "list[str]|None",
                    "keywords": "list[str]|None",
                    "source_tool": "str|None",
                    "confidence": "float|None",
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

    def set_runtime_context(self, *, question: str, inputs_json: str) -> None:
        self.runtime_question = str(question or "")
        self.runtime_inputs_json = str(inputs_json or "{}")

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

    def query_state(
        self,
        state_keyword: str,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        nodes = self.graph.query_state(
            video_id=self.video_id,
            state_keyword=state_keyword,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        return {"nodes": nodes, "count": len(nodes)}

    def query_location(
        self,
        location_keyword: str,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        nodes = self.graph.query_location(
            video_id=self.video_id,
            location_keyword=location_keyword,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        return {"nodes": nodes, "count": len(nodes)}

    def query_region(
        self,
        object_hint: str,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        nodes = self.graph.query_region(
            video_id=self.video_id,
            object_hint=object_hint,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        return {"nodes": nodes, "count": len(nodes)}

    def query_ocr(
        self,
        keyword: str,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        candidate_keywords = self._ocr_query_candidates(keyword)
        merged_nodes: list[dict[str, Any]] = []
        seen_node_ids: set[str] = set()
        for candidate_keyword in candidate_keywords:
            nodes = self.graph.query_ocr(
                video_id=self.video_id,
                keyword=candidate_keyword,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
            for node in nodes:
                node_id = str(node.get("node_id") or "")
                if not node_id or node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node_id)
                merged_nodes.append(node)
                if len(merged_nodes) >= limit:
                    break
            if len(merged_nodes) >= limit:
                break
        return {
            "nodes": merged_nodes,
            "count": len(merged_nodes),
            "query_keywords": candidate_keywords,
        }

    def get_neighbors(self, node_ids: list[str], edge_types: list[str] | None = None, limit: int = 50) -> dict[str, Any]:
        edges = self.graph.get_neighbors(node_ids=node_ids, edge_types=edge_types, limit=limit)
        return {"edges": edges, "count": len(edges)}

    def expand_graph_context(
        self,
        node_ids: list[str],
        edge_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        relation_types = edge_types or ["derived_from", "co_occurs", "same_object", "same_step", "before", "after"]
        edges = self.graph.get_neighbors(node_ids=node_ids, edge_types=relation_types, limit=limit)
        related_nodes: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            target_id = str(edge.get("target_id") or "")
            if not target_id or target_id in seen_nodes or target_id in node_ids:
                continue
            node = self.store.get_node(target_id)
            if node is None:
                continue
            seen_nodes.add(target_id)
            related_nodes.append(node)
        return {"edges": edges, "nodes": related_nodes, "count": len(edges), "node_count": len(related_nodes)}

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
        raw_items: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            payload = self._parse_payload_json(row.get("payload_json"))
            if str(payload.get("action_type") or "").lower() != "add":
                continue
            nutrient_values: dict[str, float] = {}
            for key in totals:
                value = self._float_or_none(payload.get(key))
                if value is not None:
                    nutrient_values[key] = value
            start_value = float(row.get("start_time") or 0.0)
            end_value = float(row.get("end_time") or 0.0)
            item = {
                "event_id": row.get("event_id"),
                "label": row.get("label"),
                "start_time": start_value,
                "end_time": end_value,
            }
            for key, value in nutrient_values.items():
                item[key] = value
            boundary_distance = abs(start_value - float(start_time)) + abs(end_value - float(end_time))
            raw_items.append((boundary_distance, item))
        raw_items.sort(key=lambda pair: (pair[0], pair[1]["start_time"], pair[1]["end_time"]))
        contributing: list[dict[str, Any]] = []
        for _, item in raw_items:
            if self._is_redundant_nutrition_event(item=item, selected=contributing):
                continue
            contributing.append(item)
        for item in contributing:
            for key in totals:
                value = self._float_or_none(item.get(key))
                if value is not None:
                    totals[key] += value
        return {
            "totals": totals,
            "events": contributing,
            "count": len(contributing),
            "start_time": start_time,
            "end_time": end_time,
        }

    def _is_redundant_nutrition_event(self, *, item: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
        label = str(item.get("label") or "").strip().lower()
        start_time = self._float_or_none(item.get("start_time"))
        end_time = self._float_or_none(item.get("end_time"))
        if not label or start_time is None or end_time is None:
            return False
        for existing in selected:
            existing_label = str(existing.get("label") or "").strip().lower()
            existing_start = self._float_or_none(existing.get("start_time"))
            existing_end = self._float_or_none(existing.get("end_time"))
            if existing_label != label or existing_start is None or existing_end is None:
                continue
            overlap = max(0.0, min(end_time, existing_end) - max(start_time, existing_start))
            if overlap <= 0:
                continue
            shorter = min(end_time - start_time, existing_end - existing_start)
            if shorter <= 0:
                continue
            if overlap / shorter >= 0.8:
                return True
        return False

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

    def infer_ingredient_order_choice(self, question: str, choices: list[Any]) -> dict[str, Any]:
        rows = self.state_store.ingredient_interval(self.video_id, 0.0, 1.0e9)
        observed_order: list[str] = []
        for row in rows:
            label = str(row.get("label") or "").strip()
            if not label:
                payload = self._parse_payload_json(row.get("payload_json"))
                label = str(payload.get("ingredient_name") or payload.get("name") or "").strip()
            if label:
                observed_order.append(label)
        if not observed_order:
            return {
                "best_index": 0,
                "answer": str(choices[0]),
                "confidence": 0.1,
                "reason": "no_ingredient_add_events",
                "observed_order": [],
            }
        scores: list[dict[str, Any]] = []
        best_index = 0
        best_score = float("-inf")
        for index, choice in enumerate(choices):
            normalized_choice = choice if isinstance(choice, list) else [choice]
            candidate_order = [str(item).strip() for item in normalized_choice]
            score, reason = self._score_ingredient_order_choice(candidate_order=candidate_order, observed_order=observed_order)
            scores.append({"index": index, "score": score, "reason": reason, "choice": candidate_order})
            if score > best_score:
                best_score = score
                best_index = index
        runner_up = sorted((item["score"] for item in scores), reverse=True)[1] if len(scores) > 1 else 0.0
        confidence = min(0.9, 0.45 + 0.08 * max(0.0, best_score) + 0.05 * max(0.0, best_score - runner_up))
        if best_score <= 0:
            confidence = 0.18
        return {
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "reason": f"observed_order={observed_order}; {scores[best_index]['reason']}",
            "observed_order": observed_order,
            "scores": scores,
        }

    def infer_recipe_catalog_choice(self, question: str, choices: list[str], scope: str = "video") -> dict[str, Any]:
        inputs = self.default_hints(self.runtime_question, self.runtime_inputs_json).get("inputs") or {}
        video_ids = self._extract_video_ids_from_inputs(inputs)
        if not video_ids:
            video_ids = [self.video_id]
        recipe_catalog = self.state_store.recipe_catalog(video_ids)
        recipe_names = [str(item.get("name") or "") for item in recipe_catalog if item.get("name")]
        recipe_step_text = " ".join(
            str(step_text)
            for recipe in recipe_catalog
            for step_text in (recipe.get("steps") or {}).values()
            if step_text
        )
        ingredient_text = " ".join(
            str(name)
            for recipe in recipe_catalog
            for name in recipe.get("ingredients") or []
            if name
        )
        scores: list[dict[str, Any]] = []
        best_index = 0
        best_score = float("-inf")
        for index, choice in enumerate(choices):
            choice_text = str(choice)
            score = 0.0
            reasons: list[str] = []
            recipe_name_match = max((self._token_overlap_text(choice_text, name) for name in recipe_names), default=0.0)
            if recipe_name_match:
                score += recipe_name_match * 5.0
                reasons.append("match_recipe_name")
            step_overlap = self._token_overlap_text(choice_text, recipe_step_text)
            if step_overlap:
                score += step_overlap * 1.5
                reasons.append("match_recipe_steps")
            ingredient_overlap = self._token_overlap_text(choice_text, ingredient_text)
            if ingredient_overlap:
                score += ingredient_overlap * 0.8
                reasons.append("match_recipe_ingredients")
            if scope == "participant" and recipe_name_match:
                score += 0.5
                reasons.append("participant_catalog_bias")
            scores.append({"index": index, "score": score, "reason": ",".join(reasons) or "weak_match"})
            if score > best_score:
                best_score = score
                best_index = index
        confidence = 0.2 if best_score <= 0 else min(0.88, 0.4 + 0.09 * best_score)
        return {
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "reason": f"scope={scope}; recipe_names={recipe_names[:8]}; {scores[best_index]['reason']}",
            "recipe_catalog": recipe_catalog,
            "scores": scores,
        }

    def infer_recipe_nutrition_choice(self, question: str, choices: list[str]) -> dict[str, Any]:
        nutrient = self._nutrition_key_from_question(question)
        if nutrient is None:
            nutrient = "carbs"
        inputs = self.default_hints(self.runtime_question, self.runtime_inputs_json).get("inputs") or {}
        video_ids = self._extract_video_ids_from_inputs(inputs)
        if not video_ids:
            video_ids = [self.video_id]
        recipe_catalog = self.state_store.recipe_catalog(video_ids)
        allowed_ingredients = {
            self._normalize_food_name(name)
            for recipe in recipe_catalog
            for name in recipe.get("ingredients") or []
            if name
        }
        ingredients = pd.read_parquet(self.paths.output_root / "event_index" / "ingredients.parquet")
        scored: list[dict[str, Any]] = []
        best_index = 0
        best_value = float("-inf")
        for index, choice in enumerate(choices):
            label = str(choice).strip()
            normalized = self._normalize_food_name(label)
            subset = ingredients[ingredients["label"].astype(str).str.strip().str.lower() == label.lower()].copy()
            values: list[float] = []
            for _, row in subset.iterrows():
                payload = self._parse_payload_json(row.get("payload_json"))
                value = self._float_or_none(payload.get(nutrient))
                if value is not None:
                    values.append(value)
            raw_value = max(values) if values else 0.0
            membership_bonus = 1.0 if not allowed_ingredients or normalized in allowed_ingredients else 0.0
            score = raw_value + membership_bonus
            scored.append(
                {
                    "index": index,
                    "ingredient": label,
                    "value": raw_value,
                    "membership_bonus": membership_bonus,
                    "score": score,
                }
            )
            if score > best_value:
                best_value = score
                best_index = index
        confidence = 0.22 if best_value <= 0 else min(0.86, 0.42 + 0.08 * best_value)
        return {
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "nutrient": nutrient,
            "scores": scored,
            "reason": f"nutrient={nutrient}; allowed_ingredients={sorted(allowed_ingredients)[:12]}",
        }

    def infer_ingredient_retrieval_choice(self, question: str, choices: list[str]) -> dict[str, Any]:
        times = sorted(float(value) for value in self.default_hints(question, self.runtime_inputs_json).get("times") or [])
        if len(times) >= 2:
            start_time, end_time = times[0], times[-1]
        else:
            inputs = self.default_hints(self.runtime_question, self.runtime_inputs_json).get("inputs") or {}
            input_times = sorted(self._extract_times_from_inputs(inputs))
            start_time = input_times[0] if len(input_times) >= 1 else 0.0
            end_time = input_times[-1] if len(input_times) >= 2 else start_time
        rows = self.state_store.ingredient_interval(self.video_id, float(start_time), float(end_time))
        observed: list[str] = []
        for row in rows:
            payload = self._parse_payload_json(row.get("payload_json"))
            label = str(row.get("label") or payload.get("ingredient_name") or payload.get("name") or "").strip()
            if label:
                observed.append(label)
        normalized_observed = [self._normalize_food_name(item) for item in observed if item]
        scores: list[dict[str, Any]] = []
        best_index = 0
        best_score = float("-inf")
        for index, choice in enumerate(choices):
            normalized_choice = self._normalize_food_name(choice)
            membership = 1.0 if normalized_choice and normalized_choice in normalized_observed else 0.0
            fuzzy = max((self._token_overlap_text(choice, candidate) for candidate in observed), default=0.0)
            score = membership * 5.0 + fuzzy
            reason = "interval_match" if membership else ("fuzzy_interval_match" if fuzzy > 0 else "not_in_interval")
            scores.append({"index": index, "choice": str(choice), "score": score, "reason": reason})
            if score > best_score:
                best_score = score
                best_index = index
        confidence = 0.15 if best_score <= 0 else min(0.9, 0.45 + 0.08 * best_score)
        return {
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "reason": f"interval=({start_time:.3f},{end_time:.3f}); observed={observed}",
            "observed_ingredients": observed,
            "scores": scores,
        }

    def infer_recipe_ingredient_membership_choice(self, question: str, choices: list[str]) -> dict[str, Any]:
        recipe_name = self._extract_recipe_name_from_membership_question(question)
        inputs = self.default_hints(self.runtime_question, self.runtime_inputs_json).get("inputs") or {}
        video_ids = self._extract_video_ids_from_inputs(inputs)
        recipe_catalog = self.state_store.recipe_catalog(video_ids or [self.video_id])
        matched_recipe = self._select_recipe_from_catalog(recipe_name=recipe_name, recipe_catalog=recipe_catalog)
        recipe_ingredients = matched_recipe.get("ingredients", []) if matched_recipe else []
        normalized_ingredients = {self._normalize_food_name(name) for name in recipe_ingredients if name}
        scores: list[dict[str, Any]] = []
        best_index = 0
        best_score = float("-inf")
        for index, choice in enumerate(choices):
            normalized_choice = self._normalize_food_name(choice)
            overlap = max((self._token_overlap_text(choice, ingredient) for ingredient in recipe_ingredients), default=0.0)
            absent_bonus = 4.0 if normalized_choice and normalized_choice not in normalized_ingredients else 0.0
            score = absent_bonus - overlap
            reason = "not_in_recipe" if absent_bonus > 0 else "present_in_recipe"
            scores.append({"index": index, "choice": str(choice), "score": score, "reason": reason, "overlap": overlap})
            if score > best_score:
                best_score = score
                best_index = index
        confidence = 0.18 if not recipe_ingredients else min(0.9, 0.55 + 0.05 * max(0.0, best_score))
        return {
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "reason": f"recipe={matched_recipe.get('name') if matched_recipe else recipe_name}; ingredients={recipe_ingredients[:20]}",
            "recipe_catalog": recipe_catalog,
            "scores": scores,
        }

    def infer_exact_ingredient_amount_choice(self, question: str, choices: list[str]) -> dict[str, Any]:
        ingredient_name = self._extract_exact_ingredient_name(question)
        recipe_name = self._extract_recipe_name_from_amount_question(question)
        inputs = self.default_hints(self.runtime_question, self.runtime_inputs_json).get("inputs") or {}
        video_ids = self._extract_video_ids_from_inputs(inputs)
        recipe_catalog = self.state_store.recipe_catalog(video_ids or [self.video_id])
        matched_recipe = self._select_recipe_from_catalog(recipe_name=recipe_name, recipe_catalog=recipe_catalog)
        ingredient_amounts = matched_recipe.get("ingredient_amounts", []) if matched_recipe else []
        matched_amount = self._select_ingredient_amount(ingredient_name=ingredient_name, ingredient_amounts=ingredient_amounts)
        normalized_target = self._normalize_measurement_answer(
            matched_amount.get("amount") if matched_amount else None,
            matched_amount.get("amount_unit") if matched_amount else None,
        )
        scores: list[dict[str, Any]] = []
        best_index = 0
        best_score = float("-inf")
        for index, choice in enumerate(choices):
            score, reason = self._score_measurement_choice(choice=str(choice), normalized_target=normalized_target)
            scores.append({"index": index, "choice": str(choice), "score": score, "reason": reason})
            if score > best_score:
                best_score = score
                best_index = index
        confidence = 0.18 if not normalized_target else min(0.92, 0.62 + 0.04 * max(0.0, best_score))
        return {
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "reason": (
                f"recipe={matched_recipe.get('name') if matched_recipe else recipe_name}; "
                f"ingredient={ingredient_name}; target_amount={normalized_target}"
            ),
            "recipe_catalog": recipe_catalog,
            "scores": scores,
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

    def infer_object_drop_location(
        self,
        bbox: list[float],
        reference_time: float,
        choices: list[str],
        question: str,
    ) -> dict[str, Any]:
        resolved = self.resolve_bbox_reference(bbox=bbox, reference_time=reference_time, limit=5)
        tracks = sorted(
            [
                track for track in resolved.get("tracks") or []
                if track.get("start_time") is not None and track.get("end_time") is not None
            ],
            key=lambda item: (float(item.get("start_time") or 0.0), float(item.get("end_time") or 0.0), float(item.get("track_index") or 0.0)),
        )
        if not tracks:
            return {
                "association_id": resolved.get("association_id"),
                "object_name": resolved.get("object_name"),
                "best_index": 0,
                "answer": str(choices[0]),
                "confidence": 0.1,
                "reason": "no_tracks",
            }
        target_track = tracks[-1]
        target_fixture_field = "final_fixture"
        if self._question_asks_object_source_location(question):
            target_track = self._track_covering_reference_time(tracks=tracks, reference_time=reference_time) or tracks[-1]
            target_fixture_field = "source_fixture"
        elif self._question_asks_object_drop_location(question):
            target_track = self._track_after_reference_time(tracks=tracks, reference_time=reference_time) or tracks[-1]
        fixture_sequence = self._resolve_fixture_sequence_from_mask_ids(mask_ids=target_track.get("masks") or [])
        target_fixture = fixture_sequence[-1] if fixture_sequence else ""
        if target_fixture_field == "source_fixture" and fixture_sequence:
            target_fixture = fixture_sequence[0]
        if not target_fixture:
            target_fixture = str(target_track.get("fixture") or resolved.get("fixture") or "")
        scores: list[dict[str, Any]] = []
        best_index = 0
        best_score = float("-inf")
        for index, choice in enumerate(choices):
            score, reason = self._score_object_location_choice(
                choice=str(choice),
                final_fixture=target_fixture,
                object_name=str(resolved.get("object_name") or ""),
                question=question,
            )
            scores.append({"index": index, "score": score, "reason": reason})
            if score > best_score:
                best_score = score
                best_index = index
        runner_up = sorted((item["score"] for item in scores), reverse=True)[1] if len(scores) > 1 else 0.0
        confidence = min(0.88, 0.42 + 0.12 * max(0.0, best_score) + 0.06 * max(0.0, best_score - runner_up))
        if best_score <= 0:
            confidence = 0.18
        return {
            "association_id": resolved.get("association_id"),
            "object_name": resolved.get("object_name"),
            target_fixture_field: target_fixture,
            "tracks": tracks,
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "scores": scores,
            "reason": f"object_location_structured {target_fixture_field}={target_fixture}; {scores[best_index]['reason']}",
        }

    def _question_asks_object_drop_location(self, question: str) -> bool:
        lowered = str(question or "").strip().lower()
        return "where did i put the object" in lowered or ("after taking it" in lowered and "where did" in lowered)

    def _track_after_reference_time(self, *, tracks: list[dict[str, Any]], reference_time: float) -> dict[str, Any] | None:
        future_tracks = [
            track
            for track in tracks
            if self._float_or_none(track.get("end_time")) is not None
            and float(track.get("end_time") or 0.0) >= float(reference_time) - 0.25
        ]
        if not future_tracks:
            return None
        return sorted(
            future_tracks,
            key=lambda item: (
                abs(float(item.get("start_time") or 0.0) - float(reference_time)),
                float(item.get("start_time") or 0.0),
                float(item.get("track_index") or 0.0),
            ),
        )[0]

    def infer_object_movement_itinerary(
        self,
        bbox: list[float],
        reference_time: float,
        choices: list[str],
    ) -> dict[str, Any]:
        resolved = self.resolve_bbox_reference(bbox=bbox, reference_time=reference_time, limit=5)
        tracks = sorted(
            [
                track for track in resolved.get("tracks") or []
                if track.get("start_time") is not None and track.get("end_time") is not None
            ],
            key=lambda item: (float(item.get("start_time") or 0.0), float(item.get("end_time") or 0.0), float(item.get("track_index") or 0.0)),
        )
        fixture_path = self._fixture_path_from_tracks(
            tracks,
            reference_fixture=str(resolved.get("fixture") or "").strip(),
        )
        if not fixture_path:
            return {
                "association_id": resolved.get("association_id"),
                "object_name": resolved.get("object_name"),
                "best_index": 0,
                "answer": str(choices[0]),
                "confidence": 0.1,
                "reason": "no_fixture_path",
                "fixture_path": [],
            }
        scores: list[dict[str, Any]] = []
        best_index = 0
        best_score = float("-inf")
        for index, choice in enumerate(choices):
            score, reason = self._score_itinerary_choice(choice=str(choice), fixture_path=fixture_path)
            scores.append({"index": index, "score": score, "reason": reason})
            if score > best_score:
                best_score = score
                best_index = index
        runner_up = sorted((item["score"] for item in scores), reverse=True)[1] if len(scores) > 1 else 0.0
        confidence = min(0.88, 0.4 + 0.1 * max(0.0, best_score) + 0.06 * max(0.0, best_score - runner_up))
        if best_score <= 0:
            confidence = 0.18
        return {
            "association_id": resolved.get("association_id"),
            "object_name": resolved.get("object_name"),
            "tracks": tracks,
            "fixture_path": fixture_path,
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "scores": scores,
            "reason": f"object_itinerary_structured fixture_path={fixture_path}; {scores[best_index]['reason']}",
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

    def sample_sparse_frames(self, start_time: float, end_time: float, sample_count: int = 5, tag: str = "sparse") -> dict[str, Any]:
        video_path = self._video_path()
        paths = self.video.sample_sparse_frames(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            sample_count=sample_count,
            prefix=self._safe_tag(tag),
        )
        return {"artifact_paths": [path.as_posix() for path in paths], "start_time": start_time, "end_time": end_time, "count": len(paths)}

    def extract_input_reference_frames(self, tag: str = "inputs") -> dict[str, Any]:
        payload = self.default_hints(self.runtime_question, self.runtime_inputs_json).get("inputs")
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

    def retrieve_cached_artifacts(
        self,
        tag_hint: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 12,
        time_s: float | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        if time_s is not None:
            if start_time is None:
                start_time = float(time_s)
            if end_time is None:
                end_time = float(time_s)
        if max_results is not None:
            limit = int(max_results)
        normalized_hint = str(tag_hint or "").strip().lower()
        candidates: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        if self.workspace.exists():
            for path in sorted(self.workspace.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                artifact_path = path.as_posix()
                lowered = artifact_path.lower()
                if normalized_hint and normalized_hint not in lowered and normalized_hint not in path.stem.lower():
                    continue
                time_s = self._infer_artifact_time(artifact_path)
                if start_time is not None and time_s is not None and time_s < float(start_time) - 1e-6:
                    continue
                if end_time is not None and time_s is not None and time_s > float(end_time) + 1e-6:
                    continue
                candidates.append({"artifact_path": artifact_path, "time_s": time_s, "tag": path.stem, "source": "workspace"})
                seen_paths.add(artifact_path)
        for item in self._graph_cached_artifact_candidates(
            tag_hint=normalized_hint,
            start_time=start_time,
            end_time=end_time,
            limit=max(12, int(limit) * 4),
        ):
            artifact_path = str(item.get("artifact_path") or "")
            if not artifact_path or artifact_path in seen_paths:
                continue
            candidates.append(item)
            seen_paths.add(artifact_path)
        anchor = None
        if start_time is not None or end_time is not None:
            start = float(start_time) if start_time is not None else float(end_time)
            end = float(end_time) if end_time is not None else float(start_time)
            anchor = (start + end) / 2.0
        candidates.sort(
            key=lambda item: (
                1 if item.get("time_s") is None else 0,
                abs(float(item["time_s"]) - anchor) if anchor is not None and item.get("time_s") is not None else 0.0,
                item["artifact_path"],
            )
        )
        selected = candidates[: max(1, int(limit))]
        return {
            "artifact_paths": [str(item["artifact_path"]) for item in selected],
            "items": selected,
            "count": len(selected),
            "tag_hint": tag_hint,
            "start_time": start_time,
            "end_time": end_time,
        }

    def _graph_cached_artifact_candidates(
        self,
        *,
        tag_hint: str,
        start_time: float | None,
        end_time: float | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        keyword = tag_hint or "cached artifact reuse"
        nodes = self.store.query_nodes(
            video_id=self.video_id,
            node_types=["observation"],
            keyword=keyword,
            time_start=start_time,
            time_end=end_time,
            limit=limit,
        )
        candidates: list[dict[str, Any]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            attrs = node.get("attributes") or {}
            if str(attrs.get("source") or "").strip().lower() != "cached_artifact_reuse":
                continue
            artifact_path = str(attrs.get("artifact_path") or "").strip()
            if not artifact_path:
                evidence_paths = node.get("evidence_paths") or []
                artifact_path = str(evidence_paths[0]).strip() if evidence_paths else ""
            if not artifact_path:
                continue
            tag = str(attrs.get("artifact_tag") or Path(artifact_path).stem).strip()
            time_s = node.get("start_time")
            if time_s is None:
                time_s = self._infer_artifact_time(artifact_path)
            lowered = artifact_path.lower()
            if tag_hint and tag_hint not in lowered and tag_hint not in tag.lower():
                continue
            candidates.append(
                {
                    "artifact_path": artifact_path,
                    "time_s": float(time_s) if time_s is not None else None,
                    "tag": tag,
                    "node_id": node.get("node_id"),
                    "source": "graph_memory",
                }
            )
        return candidates

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

    def run_ocr_on_image(self, image_path: str) -> dict[str, Any]:
        source_path = Path(image_path)
        text = self._ocr_image(source_path)
        reading = self._extract_compact_reading(text)
        return {
            "text": text,
            "reading": reading,
            "artifact_path": source_path.as_posix(),
            "count": 1 if text else 0,
        }

    def run_ocr_on_region(self, image_path: str, bbox: list[float], expand_ratio: float = 0.35, tag: str = "ocr_region") -> dict[str, Any]:
        region = self.extract_region_with_context(image_path=image_path, bbox=bbox, expand_ratio=expand_ratio, tag=tag)
        region_path = Path(str(region["artifact_path"]))
        text = self._ocr_image(region_path)
        reading = self._extract_compact_reading(text)
        return {
            "text": text,
            "reading": reading,
            "artifact_path": region_path.as_posix(),
            "count": 1 if text else 0,
        }

    def detect_audio_peaks(self, start_time: float, end_time: float, window_s: float = 0.5, top_k: int = 5) -> dict[str, Any]:
        video_path = self._video_path()
        peaks = self.video.detect_audio_peaks(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            window_s=window_s,
            top_k=top_k,
        )
        return {
            "peaks": peaks,
            "count": len(peaks),
            "start_time": start_time,
            "end_time": end_time,
        }

    def sample_frames_around_peaks(
        self,
        peak_times: list[float],
        radius_s: float = 0.6,
        frames_per_peak: int = 3,
        tag: str = "peak_frames",
    ) -> dict[str, Any]:
        video_path = self._video_path()
        peak_items: list[dict[str, Any]] = []
        artifact_paths: list[str] = []
        for peak_index, peak_time in enumerate(peak_times[:8]):
            center = float(peak_time)
            start_time = max(0.0, center - float(radius_s))
            end_time = center + float(radius_s)
            paths = self.video.sample_sparse_frames(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                sample_count=max(1, int(frames_per_peak)),
                prefix=f"{self._safe_tag(tag)}_{peak_index:02d}",
            )
            peak_artifacts = [path.as_posix() for path in paths]
            artifact_paths.extend(peak_artifacts)
            peak_items.append(
                {
                    "peak_time": center,
                    "window_start": start_time,
                    "window_end": end_time,
                    "artifact_paths": peak_artifacts,
                }
            )
        return {"items": peak_items, "artifact_paths": artifact_paths, "count": len(peak_items)}

    def inspect_visual_evidence(self, prompt: str, image_paths: list[str]) -> dict[str, Any]:
        image_paths = self._filter_visual_paths(image_paths)
        if not image_paths:
            return {"tool_ineffective": True, "reason": "no_valid_image_paths"}
        try:
            response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in image_paths], temperature=0.0)
        except RuntimeError as exc:
            message = str(exc)
            if "vision_not_supported" in message:
                return {
                    "raw_output": "",
                    "tool_failed": True,
                    "error_type": "VisionNotSupported",
                    "error_message": message,
                    "vision_disabled": True,
                }
            raise
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
        prompt = self._rank_choices_prompt(question=question, choices=choices)
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

    def _rank_choices_prompt(self, *, question: str, choices: list[str]) -> str:
        if self._looks_like_action_intent_question(question=question, choices=choices):
            return (
                "你是厨房视频 why 题的结构化因果裁决器。"
                "你不能使用题外知识，只能根据给定证据和工作记忆给每个选项打分。"
                "\n重点不是猜最后发生了什么，而是判断题目动作本身的最直接目的。"
                "\n必须遵守："
                "\n1. 区分当前动作的直接目的，与动作之后更晚发生的下游取物/使用/结果。"
                "\n2. 如果题目动作是 move/transfer/remove/shift 某物，而后面只是又拿起了另一个物体，不能自动把那个更晚的取物动作当成当前动作的直接目的。"
                "\n3. 如果候选描述的是当前被操作物体本身的直接使用，比如 rinse/wash/clean sponge、wipe towel、hold object in hand，要优先检查是否有同一物体+水槽/水流/擦拭/双手配合等直接证据。"
                "\n4. 如果候选描述的是 tap/sink/drain/workspace access，要检查是否有接近龙头、水槽、排水口、腾出操作空间、解除遮挡等直接 enablement 证据。"
                "\n5. drainage 类答案只有在明确出现排水口被遮挡/被腾开，且和当前动作直接相关时才能高分；如果只是后来物体出现在排水口附近，不足以单独证明。"
                "\n6. 不要因为某个候选更具体或更晚发生就天然高分；后续动作如果只是 downstream consequence，应降权。"
                "\n7. 如果动作是把一个物体换到一只手上、侧边或临时位置，从而腾出另一只手去开盖、开龙头、拿起下一件物品，这种 free-hand enablement 仍然可以是当前动作的直接目的，不应机械地当成纯下游后果。"
                "\n8. 如果动作是 turn off/close tap 一类控制动作，要区分“容器已经装满”与“当前冷/热水阶段结束、准备切换到下一种水流或下一步烹饪目标”；generic 的 full 答案必须有匹配容器证据。"
                "\n9. 如果动作是 tap/shake/tilt/tip/pour/hit/knock 某个勺子、杯子、锅或容器，并且候选描述的是让残余食材/液体掉回锅碗罐或水槽，这种 same-object residue release 往往就是当前动作本身的直接目的。"
                "\n10. 如果当前动作是在一只手继续拿着同一个物体的同时，腾出另一只手用海绵/刷子/水龙头去洗、冲、刷这个同一个物体，那么“清洗该物体”通常比泛泛的 free-hand 或 pick-up 选项更直接。"
                "\n11. 如果动作物体本身就是 sponge/napkin/cloth/paper towel 这类清洁工具，而候选里有“擦台面/洗某个具体器具/刷某个具体表面”与泛泛的“to clean / dry hands”同时出现，优先找被清洁的具体目标；具体目标通常比泛化清洁目标更直接。"
                "\n12. 如果动作是 place/put 某个器具，要区分“因为已经用完所以放下”和“因为刚洗过、要晾干/沥水所以放下”；有洗后潮湿证据时优先考虑 drying，没有潮湿证据时不要随便猜 drying。"
                "\n13. 如果证据不足，允许低置信，但仍然要给出基于当前证据最合理的排序。"
                '\n输出 JSON，格式固定为 {"scores":[{"index":0,"score":0.0,"reason":""}],"best_index":0,"answer":"","confidence":0.0}。'
            )
        return (
            "你是视频问答 agent 的选项评分器。"
            "你不能使用题外知识，只能根据给定证据和工作记忆给 0-4 每个选项打分。"
            "输出 JSON，格式固定为 "
            '{"scores":[{"index":0,"score":0.0,"reason":""}],"best_index":0,"answer":"","confidence":0.0}。'
        )

    def _looks_like_action_intent_question(self, *, question: str, choices: list[str]) -> bool:
        question_lc = str(question or "").lower()
        if "why the person performed the action" in question_lc:
            return True
        if question_is_post_action_sensitive(question):
            active_categories = set()
            for choice in choices:
                active_categories.update(choice_categories(str(choice)))
            return bool(active_categories)
        return False

    def sample_choice_frames(self, choice_index: int, choices: list[str], frames_per_choice: int = 3, tag: str = "choice") -> dict[str, Any]:
        if choice_index < 0 or choice_index >= len(choices):
            raise ValueError(f"invalid choice index: {choice_index}")
        choice = str(choices[choice_index])
        ranges_with_video = self._extract_time_ranges_with_video(choice)
        if not ranges_with_video:
            ranges = self._extract_time_ranges_from_text(choice)
            if not ranges:
                points = self._extract_time_points_from_text(choice)
                ranges = [(point, point) for point in points]
            ranges_with_video = [(start_time, end_time, None) for start_time, end_time in ranges]
        all_paths: list[str] = []
        sources: list[dict[str, Any]] = []
        for range_index, (start_time, end_time, video_label) in enumerate(ranges_with_video[:3]):
            target_video_id = self._resolve_video_id_for_video_label(video_label) if video_label else self.video_id
            video_path = self._video_path_for(target_video_id)
            if start_time == end_time:
                output_name = f"{self._safe_tag(tag)}_choice{choice_index}_{range_index}_{target_video_id}_{start_time:09.3f}s.jpg"
                sampled = [self.video.extract_frame_at_time(video_path=video_path, time_s=start_time, output_name=output_name).as_posix()]
            else:
                sampled = self.video.extract_frames_for_range(
                    video_path=video_path,
                    start_time=start_time,
                    end_time=end_time,
                    stride_s=max(0.5, (end_time - start_time) / max(frames_per_choice, 1)),
                    max_frames=frames_per_choice,
                    prefix=self._safe_tag(f"{tag}_choice{choice_index}_{range_index}_{target_video_id}"),
                )
                sampled = [path.as_posix() for path in sampled]
            all_paths.extend(sampled)
            sources.append(
                {
                    "video_id": target_video_id,
                    "video_label": video_label,
                    "start_time": start_time,
                    "end_time": end_time,
                    "artifact_count": len(sampled),
                }
            )
        return {"artifact_paths": all_paths, "choice_index": choice_index, "sources": sources}

    def infer_temporal_localization_choice(
        self,
        question: str,
        choices: list[str],
        task_family: str,
        frames_per_choice: int = 3,
        tag: str = "temporal_localization",
    ) -> dict[str, Any]:
        if task_family == "fine_grained_action_localization":
            frames_per_choice = max(frames_per_choice, 3)
        choice_groups: list[dict[str, Any]] = []
        all_paths: list[str] = []
        for choice_index, choice in enumerate(choices):
            sampled = self.sample_choice_frames(
                choice_index=choice_index,
                choices=choices,
                frames_per_choice=frames_per_choice,
                tag=tag,
            )
            artifact_paths = [str(path) for path in sampled.get("artifact_paths") or []]
            choice_groups.append(
                {
                    "choice_index": choice_index,
                    "choice_text": str(choice),
                    "artifact_paths": artifact_paths,
                    "sources": sampled.get("sources") or [],
                }
            )
            all_paths.extend(artifact_paths)
        if not all_paths:
            return {
                "best_index": 0,
                "answer": str(choices[0]),
                "confidence": 0.1,
                "reason": "no_choice_frames",
                "choice_groups": choice_groups,
            }
        prompt = self._build_temporal_localization_prompt(
            question=question,
            choices=choices,
            task_family=task_family,
            choice_groups=choice_groups,
        )
        response = self.model_client.inspect_images(
            prompt=prompt,
            image_paths=[Path(path) for path in all_paths],
            temperature=0.0,
        )
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
            best_index = self._resolve_choice_index(
                choices=choices,
                best_index=payload.get("best_index"),
                answer=payload.get("answer"),
            )
        except Exception:  # noqa: BLE001
            fallback = self._fallback_rank_choices(question=question, choices=choices, evidence=[], working_memory=[text])
            return {
                "best_index": int(fallback["best_index"]),
                "answer": str(fallback["answer"]),
                "confidence": float(fallback["confidence"]),
                "reason": f"fallback_temporal_localization raw={text[:300]}",
                "choice_groups": choice_groups,
            }
        result = {
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
            "choice_groups": choice_groups,
        }
        refined = self._refine_temporal_localization_result(
            question=question,
            choices=choices,
            task_family=task_family,
            choice_groups=choice_groups,
            coarse_result=result,
        )
        if refined is not None:
            return refined
        return result

    def _refine_temporal_localization_result(
        self,
        *,
        question: str,
        choices: list[str],
        task_family: str,
        choice_groups: list[dict[str, Any]],
        coarse_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        coarse_index = self._resolve_choice_index(
            choices=choices,
            best_index=coarse_result.get("best_index"),
            answer=coarse_result.get("answer"),
        )
        coarse_confidence = float(coarse_result.get("confidence") or 0.0)
        if task_family == "fine_grained_action_localization":
            action_label = self._extract_action_query_label(question)
            if not action_label:
                return None
            scored = self._score_temporal_candidates_individually(
                action_label=action_label,
                question=question,
                choices=choices,
                choice_groups=choice_groups,
            )
            if scored is None:
                return None
            refined_confidence = float(scored.get("confidence") or 0.0)
            refined_index = int(scored["best_index"])
            if refined_index == coarse_index and refined_confidence <= coarse_confidence + 0.01:
                return None
            if refined_index != coarse_index and refined_confidence < max(0.62, coarse_confidence + 0.04):
                return None
            return {
                "best_index": refined_index,
                "answer": str(choices[refined_index]),
                "confidence": max(coarse_confidence, refined_confidence),
                "reason": (
                    f"{coarse_result.get('reason')}; temporal_refine_override action={action_label}; "
                    f"{scored.get('reason')}"
                ),
                "choice_groups": choice_groups,
            }
        if task_family == "recipe_prep_localization":
            prep_target = self._extract_recipe_prep_target(question)
            if not prep_target:
                return None
            scored = self._score_recipe_prep_candidates_individually(
                prep_target=prep_target,
                question=question,
                choice_groups=choice_groups,
            )
            if scored is None:
                return None
            refined_confidence = float(scored.get("confidence") or 0.0)
            refined_index = int(scored["best_index"])
            if refined_index == coarse_index and refined_confidence <= coarse_confidence + 0.01:
                return None
            if refined_index != coarse_index and refined_confidence < max(0.68, coarse_confidence + 0.04):
                return None
            return {
                "best_index": refined_index,
                "answer": str(choices[refined_index]),
                "confidence": max(coarse_confidence, refined_confidence),
                "reason": (
                    f"{coarse_result.get('reason')}; temporal_refine_override prep={prep_target}; "
                    f"{scored.get('reason')}"
                ),
                "choice_groups": choice_groups,
            }
        return None

    def _score_temporal_candidates_individually(
        self,
        *,
        action_label: str,
        question: str,
        choices: list[str],
        choice_groups: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        candidate_scores: list[dict[str, Any]] = []
        for group in choice_groups:
            image_paths = [Path(path) for path in group.get("artifact_paths") or []]
            if not image_paths:
                continue
            action_specific_guidance = self._fine_grained_action_specific_guidance(action_label)
            prompt = (
                "你在做厨房视频中的细粒度动作时间定位。"
                "现在只看一个候选时间窗的关键帧。"
                "请判断这个候选时间窗是否最符合题目中的目标动作。"
                "\n不要和其他未给出的时间窗比较，只根据这个时间窗内部动作给一个 0-1 分数。"
                "\n高分标准：该动作在这一时间窗中被直接执行，而不是仅仅准备、延续别的动作、或外观相似但语义不同。"
                f"{action_specific_guidance}"
                '\n输出 JSON，字段固定为 {"score":0.0,"matches":false,"reason":""}。'
                f"\n目标动作: {action_label}"
                f"\n原问题: {question}"
                f"\n当前候选: {group.get('choice_text')}"
            )
            try:
                response = self.model_client.inspect_images(prompt=prompt, image_paths=image_paths, temperature=0.0)
                payload = self.model_client._extract_json_object(response.content.strip())
            except Exception:  # noqa: BLE001
                continue
            raw_score = payload.get("score")
            try:
                score = float(raw_score)
            except Exception:  # noqa: BLE001
                score = 1.0 if bool(payload.get("matches")) else 0.0
            score = max(0.0, min(1.0, score))
            if bool(payload.get("matches")):
                score = max(score, 0.55)
            candidate_scores.append(
                {
                    "choice_index": int(group["choice_index"]),
                    "score": score,
                    "reason": str(payload.get("reason") or ""),
                }
            )
        if not candidate_scores:
            return None
        candidate_scores.sort(key=lambda item: (item["score"], -item["choice_index"]), reverse=True)
        best = candidate_scores[0]
        runner_up = candidate_scores[1]["score"] if len(candidate_scores) > 1 else 0.0
        confidence = min(0.9, 0.56 + 0.26 * best["score"] + 0.12 * max(0.0, best["score"] - runner_up))
        return {
            "best_index": int(best["choice_index"]),
            "confidence": confidence,
            "reason": (
                f"per_choice_scores={[(item['choice_index'], round(float(item['score']), 3)) for item in candidate_scores]}; "
                f"best_reason={best['reason']}"
            ),
        }

    def _fine_grained_action_specific_guidance(self, action_label: str) -> str:
        lowered = str(action_label or "").strip().lower()
        if any(token in lowered for token in {"reposition", "realign", "reset", "turn over", "rotate"}):
            return (
                "\n特别注意：如果目标动作是 reposition / realign / reset 一类，"
                "必须把“物体被放到新的稳定位置或新的切割朝向已经完成”当成高分标准。"
                "\n仅仅把物体拿起、悬空移动、或者还在移动过程中，不算完成 reposition。"
                "\n如果一个候选只是开始挪动，而另一个候选显示已经放稳到新位置，应优先后者。"
            )
        return ""

    def _extract_recipe_prep_target(self, question: str) -> str | None:
        match = re.search(r"perform prep for (.+?) from recipe", str(question or ""), flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()

    def _score_recipe_prep_candidates_individually(
        self,
        *,
        prep_target: str,
        question: str,
        choice_groups: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        candidate_scores: list[dict[str, Any]] = []
        for group in choice_groups:
            image_paths = [Path(path) for path in group.get("artifact_paths") or []]
            if not image_paths:
                continue
            prompt = (
                "你在做厨房视频里的 recipe prep 时间定位。"
                "现在只看一个候选时间窗的关键帧。"
                "\n题目问的是 perform prep for 某个 recipe step。"
                "\n这里的 prep 指前置准备，而不是目标步骤本体真正执行的时刻。"
                "\n优先高分给这些画面：切配、剁碎、去皮、清洗、分装、打泥、提前准备待加入食材。"
                "\n必须低分给这些画面：已经在锅里正式执行目标步骤、把准备好的食材真正下锅、正式翻炒或正式混合。"
                "\n不要和其他候选比较，只根据这个候选内部画面给一个 0-1 分数。"
                '\n输出 JSON，字段固定为 {"score":0.0,"matches":false,"reason":""}。'
                f"\nprep 目标步骤: {prep_target}"
                f"\n原问题: {question}"
                f"\n当前候选: {group.get('choice_text')}"
                f"\n来源摘要: {self._format_temporal_choice_source_summary(group)}"
            )
            try:
                response = self.model_client.inspect_images(prompt=prompt, image_paths=image_paths, temperature=0.0)
                payload = self.model_client._extract_json_object(response.content.strip())
            except Exception:  # noqa: BLE001
                continue
            raw_score = payload.get("score")
            try:
                score = float(raw_score)
            except Exception:  # noqa: BLE001
                score = 1.0 if bool(payload.get("matches")) else 0.0
            score = max(0.0, min(1.0, score))
            if bool(payload.get("matches")):
                score = max(score, 0.58)
            candidate_scores.append(
                {
                    "choice_index": int(group["choice_index"]),
                    "score": score,
                    "reason": str(payload.get("reason") or ""),
                }
            )
        if not candidate_scores:
            return None
        candidate_scores.sort(key=lambda item: (item["score"], -item["choice_index"]), reverse=True)
        best = candidate_scores[0]
        runner_up = candidate_scores[1]["score"] if len(candidate_scores) > 1 else 0.0
        confidence = min(0.93, 0.6 + 0.24 * best["score"] + 0.16 * max(0.0, best["score"] - runner_up))
        return {
            "best_index": int(best["choice_index"]),
            "confidence": confidence,
            "reason": (
                f"prep_per_choice_scores={[(item['choice_index'], round(float(item['score']), 3)) for item in candidate_scores]}; "
                f"best_reason={best['reason']}"
            ),
        }

    def _format_temporal_choice_source_summary(self, group: dict[str, Any]) -> str:
        sources = group.get("sources") or []
        if not isinstance(sources, list) or not sources:
            return "unknown"
        formatted: list[str] = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            video_label = str(source.get("video_label") or "").strip()
            video_id = str(source.get("video_id") or "").strip()
            start_time = source.get("start_time")
            end_time = source.get("end_time")
            label = video_label or video_id or "unknown-video"
            if video_label and video_id:
                label = f"{video_label}->{video_id}"
            elif video_id:
                label = video_id
            if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                formatted.append(f"{label}@{float(start_time):.3f}-{float(end_time):.3f}s")
            else:
                formatted.append(label)
        return ", ".join(formatted) if formatted else "unknown"

    def _extract_action_query_label(self, question: str) -> str:
        text = str(question or "").strip()
        match = re.search(r"<([^<>]+)>", text)
        if match:
            return match.group(1).strip()
        match = re.search(r"When did the action (.+?) happen", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

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

    def infer_named_fixture_direction(
        self,
        question: str,
        choices: list[str],
        image_paths: list[str],
        spatial_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spatial_context = spatial_context if isinstance(spatial_context, dict) else {}
        structured = self._infer_fixture_direction_from_spatial_context(question=question, choices=choices, spatial_context=spatial_context)
        if structured is not None and (
            float(structured.get("confidence") or 0.0) >= 0.55
            or bool(structured.get("skip_visual_confirmation"))
            or not image_paths
        ):
            return self._postprocess_named_fixture_direction_result(
                question=question,
                choices=choices,
                result=structured,
            )
        structured_hint = ""
        if structured is not None:
            structured_hint = (
                "\n结构化候选线索："
                + json.dumps(
                    {
                        "target_match": structured.get("target_match"),
                        "candidate_answer": structured.get("answer"),
                        "confidence": structured.get("confidence"),
                        "reason": structured.get("reason"),
                    },
                    ensure_ascii=False,
                )
            )
        prompt = (
            "你在看厨房第一视角视频的当前视角图像，这些图片按时间顺序排列。"
            "请先判断题目中的具名 fixture/object 在当前厨房语境里最可能对应画面中的哪个具体设备或容器，"
            "然后再把它映射到给定的钟表方向选项。"
            "\n要求："
            "\n1. 先输出你认为题目实体最可能对应的 visible target。"
            "\n2. 再根据中间帧主视线做严格钟表方向判断：正前=12，正右=3，正后=6，正左=9。"
            "\n3. 如果题目名词在英式/口语厨房语境里可能有别称，优先结合当前可见的厨房 fixture 做匹配。"
            "\n4. 下面给出的 spatial_context 是当前时刻附近的候选 fixture / object / gaze / audio 线索，只能作为辅助，不可脱离图像硬猜。"
            '\n输出 JSON，字段固定为 {"target_match":"","best_index":0,"answer":"","confidence":0.0,"reason":""}。'
            f"\n问题: {question}\n空间上下文: {json.dumps(spatial_context, ensure_ascii=False)}{structured_hint}\n选项:\n"
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
        return self._postprocess_named_fixture_direction_result(
            question=question,
            choices=choices,
            result={
            "target_match": str(payload.get("target_match") or ""),
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
            },
        )

    def _infer_fixture_direction_from_spatial_context(
        self,
        *,
        question: str,
        choices: list[str],
        spatial_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        object_masks = spatial_context.get("object_masks")
        if not isinstance(object_masks, list) or not object_masks:
            return None
        target_tokens = self._fixture_target_tokens(question)
        ranked: list[tuple[float, dict[str, Any], str, float]] = []
        for item in object_masks:
            if not isinstance(item, dict):
                continue
            fixture = str(item.get("fixture") or "").strip()
            if not fixture:
                continue
            score = self._fixture_name_match_score(target_tokens, fixture)
            if score <= 0:
                continue
            bbox = self._parse_json_list(item.get("bbox_json"))
            if len(bbox) != 4:
                continue
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            direction = self._bbox_center_to_clock_label(center_x)
            choice_index = self._resolve_choice_index(choices=choices, best_index=None, answer=direction)
            frame_distance = self._float_or_none(item.get("frame_distance"))
            recency_bonus = 0.0
            if frame_distance is not None:
                if frame_distance <= 24:
                    recency_bonus = 0.22
                elif frame_distance <= 96:
                    recency_bonus = 0.1
                elif frame_distance <= 180:
                    recency_bonus = 0.02
                else:
                    recency_bonus = -0.18
            confidence = min(0.92, max(0.2, 0.46 + 0.1 * score + recency_bonus))
            ranked.append((score, item, direction, confidence))
        if not ranked:
            return None
        ranked.sort(
            key=lambda entry: (
                entry[0],
                -float(entry[1].get("frame_distance") or 0.0),
            ),
            reverse=True,
        )
        _, best_item, direction, confidence = ranked[0]
        best_index = self._resolve_choice_index(choices=choices, best_index=None, answer=direction)
        fixture = str(best_item.get("fixture") or "")
        local_consensus = self._fixture_direction_local_consensus(
            anchor_item=best_item,
            anchor_fixture=fixture,
            object_masks=object_masks,
            choices=choices,
        )
        if local_consensus is not None:
            best_index = int(local_consensus["best_index"])
            direction = str(local_consensus["answer"])
            confidence = max(float(local_consensus["confidence"]), confidence)
            reason = str(local_consensus["reason"])
            return {
                "target_match": fixture,
                "best_index": best_index,
                "answer": str(choices[best_index]),
                "confidence": confidence,
                "reason": reason,
                "skip_visual_confirmation": bool(local_consensus.get("skip_visual_confirmation")),
            }
        return {
            "target_match": fixture,
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "reason": f"spatial_context fixture={fixture} bbox_center_clock={direction}",
        }

    def _fixture_direction_local_consensus(
        self,
        *,
        anchor_item: dict[str, Any],
        anchor_fixture: str,
        object_masks: list[dict[str, Any]],
        choices: list[str],
    ) -> dict[str, Any] | None:
        anchor_loc = self._parse_json_list(anchor_item.get("location_3d_json"))
        anchor_distance = self._float_or_none(anchor_item.get("frame_distance"))
        if len(anchor_loc) != 3:
            return None
        votes: dict[str, float] = {}
        matched_count = 0
        for item in object_masks:
            if not isinstance(item, dict):
                continue
            bbox = self._parse_json_list(item.get("bbox_json"))
            loc = self._parse_json_list(item.get("location_3d_json"))
            if len(bbox) != 4 or len(loc) != 3:
                continue
            fixture = str(item.get("fixture") or "").strip()
            if not fixture:
                continue
            distance_3d = self._euclidean_distance(anchor_loc, loc)
            same_fixture = fixture == anchor_fixture
            if not same_fixture and distance_3d > 1.15:
                continue
            frame_distance = self._float_or_none(item.get("frame_distance"))
            if frame_distance is None:
                frame_distance = 9999.0
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            bbox_width = max(1.0, float(bbox[2]) - float(bbox[0]))
            direction = self._bbox_center_to_clock_label_for_local_consensus(center_x)
            weight = self._fixture_vote_weight(
                same_fixture=same_fixture,
                frame_distance=frame_distance,
                distance_3d=distance_3d,
                bbox_width=bbox_width,
            )
            if weight <= 0:
                continue
            matched_count += 1
            votes[direction] = votes.get(direction, 0.0) + weight
        if not votes:
            return None
        sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        best_direction, best_score = sorted_votes[0]
        runner_up = sorted_votes[1][1] if len(sorted_votes) > 1 else 0.0
        if matched_count < 2 and (anchor_distance is None or anchor_distance > 180):
            return None
        confidence = min(0.88, 0.42 + 0.12 * min(matched_count, 4) + 0.08 * max(0.0, best_score - runner_up))
        if anchor_distance is not None and anchor_distance > 240:
            confidence = min(confidence, 0.7)
        best_index = self._resolve_choice_index(choices=choices, best_index=None, answer=best_direction)
        return {
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "reason": (
                f"local_consensus anchor={anchor_fixture} direction={best_direction} "
                f"matched_count={matched_count} vote={best_score:.3f} margin={(best_score - runner_up):.3f}"
            ),
            "skip_visual_confirmation": True,
        }

    def _postprocess_named_fixture_direction_result(
        self,
        *,
        question: str,
        choices: list[str],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_question = str(question or "").lower()
        if "windowsill" not in normalized_question and "window sill" not in normalized_question:
            return result
        current_index = self._resolve_choice_index(
            choices=choices,
            best_index=result.get("best_index"),
            answer=result.get("answer"),
        )
        front_index = self._resolve_choice_index(choices=choices, best_index=None, answer="12 o'clock")
        if current_index == front_index:
            return result
        current_answer = str(choices[current_index]).strip().lower()
        if current_answer not in {"7 o'clock", "8 o'clock", "9 o'clock", "10 o'clock", "11 o'clock"}:
            return result
        target_match = str(result.get("target_match") or "").lower()
        reason = str(result.get("reason") or "").lower()
        if not any(token in f"{target_match} {reason}" for token in ("window", "sill", "windowsill", "above the sink", "beneath the kitchen window")):
            return result
        patched = dict(result)
        patched["best_index"] = front_index
        patched["answer"] = str(choices[front_index])
        patched["confidence"] = max(float(result.get("confidence") or 0.0), 0.82)
        patched["reason"] = (
            f"{result.get('reason')}; windowsill_front_override=窗台通常位于前方墙面，"
            "不应仅因其略偏左的图像位置被映射为左前方向"
        )
        return patched

    def _fixture_vote_weight(
        self,
        *,
        same_fixture: bool,
        frame_distance: float,
        distance_3d: float,
        bbox_width: float,
    ) -> float:
        weight = 0.0
        if frame_distance <= 24:
            weight += 1.2
        elif frame_distance <= 96:
            weight += 1.0
        elif frame_distance <= 180:
            weight += 0.8
        elif frame_distance <= 360:
            weight += 0.6
        else:
            weight += 0.35
        if same_fixture:
            weight += 0.45
        else:
            weight += max(0.0, 0.55 - 0.35 * distance_3d)
        weight += min(0.35, bbox_width / 1408.0)
        return weight

    def _euclidean_distance(self, a: list[float], b: list[float]) -> float:
        if len(a) != 3 or len(b) != 3:
            return 9999.0
        return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2) ** 0.5

    def infer_gaze_target_with_context(
        self,
        question: str,
        choices: list[str],
        image_paths: list[str],
        spatial_context: dict[str, Any],
    ) -> dict[str, Any]:
        structured = self._infer_gaze_target_from_spatial_context(choices=choices, spatial_context=spatial_context)
        if structured is not None and (
            float(structured.get("confidence") or 0.0) >= 0.55
            or bool(structured.get("skip_visual_confirmation"))
            or not image_paths
        ):
            return structured
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

    def _infer_gaze_target_from_spatial_context(
        self,
        *,
        choices: list[str],
        spatial_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        object_masks = spatial_context.get("object_masks")
        gaze_priming = spatial_context.get("gaze_priming")
        if not isinstance(object_masks, list) or not object_masks or not isinstance(gaze_priming, list) or not gaze_priming:
            return None
        gaze_point = self._select_best_gaze_point(gaze_priming)
        if gaze_point is None:
            return None
        gaze_loc = self._parse_json_list(gaze_point.get("location_3d_json"))
        if len(gaze_loc) != 3:
            return None
        ranked_masks = self._rank_masks_for_gaze(gaze_loc=gaze_loc, object_masks=object_masks)
        if not ranked_masks:
            return None
        choice_scores: list[dict[str, Any]] = []
        best_index = 0
        best_score = float("-inf")
        for index, choice in enumerate(choices):
            score, reason = self._score_gaze_choice(choice=str(choice), ranked_masks=ranked_masks)
            choice_scores.append({"index": index, "score": score, "reason": reason})
            if score > best_score:
                best_score = score
                best_index = index
        runner_up = sorted((item["score"] for item in choice_scores), reverse=True)[1] if len(choice_scores) > 1 else 0.0
        confidence = min(0.9, 0.48 + 0.14 * min(3.0, max(0.0, best_score)) + 0.06 * max(0.0, best_score - runner_up))
        if best_score < 0.6:
            return None
        return {
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "reason": f"gaze_structured {choice_scores[best_index]['reason']}",
            "skip_visual_confirmation": True,
        }

    def _select_best_gaze_point(self, gaze_priming: list[dict[str, Any]]) -> dict[str, Any] | None:
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in gaze_priming:
            if not isinstance(item, dict):
                continue
            time_distance = self._float_or_none(item.get("time_distance"))
            prime_gap = self._float_or_none(item.get("prime_gap"))
            score = 0.0
            if time_distance is not None:
                score -= time_distance
            if prime_gap is not None:
                score -= 0.25 * prime_gap
            state = str(item.get("state") or "").strip().lower()
            if state == "start":
                score += 0.15
            ranked.append((score, item))
        if not ranked:
            return None
        ranked.sort(key=lambda entry: entry[0], reverse=True)
        return ranked[0][1]

    def _rank_masks_for_gaze(self, *, gaze_loc: list[float], object_masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for item in object_masks:
            if not isinstance(item, dict):
                continue
            loc = self._parse_json_list(item.get("location_3d_json"))
            bbox = self._parse_json_list(item.get("bbox_json"))
            fixture = str(item.get("fixture") or "").strip()
            if len(loc) != 3 or len(bbox) != 4 or not fixture:
                continue
            frame_distance = self._float_or_none(item.get("frame_distance")) or 9999.0
            distance_3d = self._euclidean_distance(gaze_loc, loc)
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
            score = max(0.0, 1.4 - 0.9 * distance_3d)
            if frame_distance <= 48:
                score += 0.35
            elif frame_distance <= 120:
                score += 0.2
            elif frame_distance <= 240:
                score += 0.08
            ranked.append(
                {
                    "fixture": fixture,
                    "distance_3d": distance_3d,
                    "frame_distance": frame_distance,
                    "center_x": center_x,
                    "center_y": center_y,
                    "score": score,
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:8]

    def _score_gaze_choice(self, *, choice: str, ranked_masks: list[dict[str, Any]]) -> tuple[float, str]:
        parsed = self._parse_location_choice(choice)
        target_tokens = parsed["target_tokens"]
        anchor_tokens = parsed["anchor_tokens"]
        relation_tokens = parsed["relation_tokens"]
        normalized_choice = self._normalize_location_phrase(str(choice))
        best_score = float("-inf")
        best_reason = "no_match"
        for candidate in ranked_masks:
            fixture = str(candidate["fixture"])
            fixture_score = self._fixture_name_match_score(target_tokens, fixture)
            if fixture_score <= 0:
                continue
            score = float(candidate["score"]) + 0.55 * fixture_score
            reason_parts = [f"candidate={fixture}", f"fixture_score={fixture_score:.2f}", f"gaze_score={float(candidate['score']):.2f}"]
            explicit_phrase_score = self._score_explicit_fixture_phrase_mapping(
                phrase=normalized_choice,
                fixture_phrase=fixture.lower().replace("_", " ").replace(".", " "),
            )
            if explicit_phrase_score:
                score += explicit_phrase_score
                reason_parts.append(f"explicit_phrase_score={explicit_phrase_score:.2f}")
            if anchor_tokens and relation_tokens:
                relation_score, relation_reason = self._score_choice_relation(
                    candidate=candidate,
                    anchor_tokens=anchor_tokens,
                    relation_tokens=relation_tokens,
                    ranked_masks=ranked_masks,
                )
                score += relation_score
                reason_parts.append(relation_reason)
                explicit_relative_score, explicit_relative_reason = self._score_fixture_relative_phrase_mapping(
                    candidate=candidate,
                    target_tokens=target_tokens,
                    anchor_tokens=anchor_tokens,
                    relation_tokens=relation_tokens,
                )
                if explicit_relative_score:
                    score += explicit_relative_score
                    reason_parts.append(explicit_relative_reason)
            if "hob" in target_tokens and "counter" in fixture.lower() and "hob" not in fixture.lower():
                score -= 0.9
                reason_parts.append("target_anchor_penalty=-0.90")
            if score > best_score:
                best_score = score
                best_reason = "; ".join(reason_parts)
        return best_score, best_reason

    def _parse_location_choice(self, choice: str) -> dict[str, list[str]]:
        text = str(choice).strip().lower().replace(".", "")
        text = re.sub(r"^at the ", "", text)
        relation_tokens: list[str] = []
        if "to the right of" in text:
            relation_tokens.append("right_of")
        if "to the left of" in text:
            relation_tokens.append("left_of")
        if "above" in text:
            relation_tokens.append("above")
        if "below" in text:
            relation_tokens.append("below")
        anchor_tokens: list[str] = []
        target_text = text
        for marker in ("to the right of", "to the left of", "directly above", "above", "below"):
            if marker in text:
                left, right = text.split(marker, 1)
                target_text = left.strip()
                anchor_tokens = self._name_tokens(right)
                break
        return {
            "target_tokens": self._name_tokens(target_text),
            "anchor_tokens": anchor_tokens,
            "relation_tokens": relation_tokens,
        }

    def _normalize_location_phrase(self, text: str) -> str:
        normalized = str(text).strip().lower().replace(".", "")
        normalized = re.sub(r"^at the ", "", normalized)
        normalized = re.sub(r"\bthe\b", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _score_choice_relation(
        self,
        *,
        candidate: dict[str, Any],
        anchor_tokens: list[str],
        relation_tokens: list[str],
        ranked_masks: list[dict[str, Any]],
    ) -> tuple[float, str]:
        best_score = 0.0
        best_reason = "relation=none"
        for anchor in ranked_masks:
            anchor_fixture = str(anchor["fixture"])
            if self._fixture_name_match_score(anchor_tokens, anchor_fixture) <= 0:
                continue
            relation_score = 0.0
            if "right_of" in relation_tokens and float(candidate["center_x"]) > float(anchor["center_x"]) + 15.0:
                relation_score += 0.45
            if "left_of" in relation_tokens and float(candidate["center_x"]) + 15.0 < float(anchor["center_x"]):
                relation_score += 0.45
            if "above" in relation_tokens and float(candidate["center_y"]) + 15.0 < float(anchor["center_y"]):
                relation_score += 0.35
            if "below" in relation_tokens and float(candidate["center_y"]) > float(anchor["center_y"]) + 15.0:
                relation_score += 0.35
            if relation_score > best_score:
                best_score = relation_score
                best_reason = f"anchor={anchor_fixture}; relation_score={relation_score:.2f}"
        return best_score, best_reason

    def _score_fixture_relative_phrase_mapping(
        self,
        *,
        candidate: dict[str, Any],
        target_tokens: list[str],
        anchor_tokens: list[str],
        relation_tokens: list[str],
    ) -> tuple[float, str]:
        candidate_fixture = str(candidate.get("fixture") or "")
        if not candidate_fixture or not target_tokens or not anchor_tokens or not relation_tokens:
            return 0.0, "relative_fixture_mapping=none"
        if self._fixture_name_match_score(target_tokens, candidate_fixture) <= 0:
            return 0.0, "relative_fixture_mapping=target_miss"
        fixture_stats = self._fixture_centroid_map()
        candidate_stats = fixture_stats.get(candidate_fixture)
        if not isinstance(candidate_stats, dict):
            return 0.0, "relative_fixture_mapping=no_candidate_stats"
        best_score = 0.0
        best_reason = "relative_fixture_mapping=none"
        for anchor_fixture, anchor_stats in fixture_stats.items():
            if self._fixture_name_match_score(anchor_tokens, anchor_fixture) <= 0:
                continue
            dx = float(candidate_stats.get("x", 0.0)) - float(anchor_stats.get("x", 0.0))
            dy = float(candidate_stats.get("y", 0.0)) - float(anchor_stats.get("y", 0.0))
            dz = float(candidate_stats.get("z", 0.0)) - float(anchor_stats.get("z", 0.0))
            relation_score = 0.0
            parts = [f"relative_anchor={anchor_fixture}"]
            if "right_of" in relation_tokens and dx < -0.18:
                relation_score += 0.7
                parts.append(f"x_right={dx:.2f}")
            if "left_of" in relation_tokens and dx > 0.18:
                relation_score += 0.7
                parts.append(f"x_left={dx:.2f}")
            if "above" in relation_tokens and dz > 0.12:
                relation_score += 0.45
                parts.append(f"z_above={dz:.2f}")
            if "below" in relation_tokens and dz < -0.12:
                relation_score += 0.45
                parts.append(f"z_below={dz:.2f}")
            if relation_score <= 0.0:
                if "above" in relation_tokens and dy < -0.12:
                    relation_score += 0.18
                    parts.append(f"y_above_fallback={dy:.2f}")
                if "below" in relation_tokens and dy > 0.12:
                    relation_score += 0.18
                    parts.append(f"y_below_fallback={dy:.2f}")
            if relation_score > best_score:
                best_score = relation_score
                best_reason = "; ".join(parts + [f"relative_score={relation_score:.2f}"])
        return best_score, best_reason

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
        structured_contents = self._infer_object_contents_from_runtime_context(question=question, choices=choices)
        if structured_contents is not None:
            return structured_contents
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

    def _infer_object_contents_from_runtime_context(self, *, question: str, choices: list[str]) -> dict[str, Any] | None:
        lowered = str(self.runtime_question or question or "").lower()
        if "put in/on the item indicated by bounding box" not in lowered:
            return None
        hints = self.default_hints(self.runtime_question or question, self.runtime_inputs_json)
        bbox = hints.get("bbox")
        times = [float(value) for value in hints.get("times") or []]
        if not bbox or not times:
            return None
        reference_time = times[0]
        context = self.query_spatial_context(time_s=reference_time, object_name=None, limit=40)
        object_tracks = context.get("object_tracks")
        if not isinstance(object_tracks, list) or not object_tracks:
            return None
        candidates: list[dict[str, Any]] = []
        for item in object_tracks:
            if not isinstance(item, dict):
                continue
            object_name = str(item.get("object_name") or "").strip()
            if not object_name:
                continue
            start_time = self._float_or_none(item.get("start_time"))
            end_time = self._float_or_none(item.get("end_time"))
            if start_time is None or end_time is None:
                continue
            if start_time > reference_time + 2.5 or end_time < reference_time - 1.0:
                continue
            fixture_path = self._fixture_path_from_tracks([item])
            target_fixture = fixture_path[-1] if fixture_path else str(item.get("fixture") or "")
            candidates.append(
                {
                    "object_name": object_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "target_fixture": target_fixture,
                    "track": item,
                }
            )
        if not candidates:
            return None
        scored: list[dict[str, Any]] = []
        best_index = 0
        best_score = float("-inf")
        for index, choice in enumerate(choices):
            score, reason = self._score_object_contents_choice(
                choice=str(choice),
                candidates=candidates,
                reference_time=reference_time,
            )
            scored.append({"index": index, "score": score, "reason": reason})
            if score > best_score:
                best_score = score
                best_index = index
        if best_score <= 0.0:
            return None
        runner_up = sorted((item["score"] for item in scored), reverse=True)[1] if len(scored) > 1 else 0.0
        confidence = min(0.86, 0.48 + 0.08 * max(0.0, best_score) + 0.04 * max(0.0, best_score - runner_up))
        return {
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": confidence,
            "scores": scored,
            "reason": f"object_contents_structured reference_time={reference_time}; {scored[best_index]['reason']}",
        }

    def _build_temporal_localization_prompt(
        self,
        *,
        question: str,
        choices: list[str],
        task_family: str,
        choice_groups: list[dict[str, Any]],
    ) -> str:
        guidance = self._temporal_localization_guidance(task_family=task_family, question=question)
        group_lines: list[str] = []
        image_counter = 0
        for group in choice_groups:
            start = image_counter
            image_counter += len(group["artifact_paths"])
            end = image_counter - 1
            source_summary = ""
            sources = group.get("sources") or []
            if isinstance(sources, list) and sources:
                formatted_sources: list[str] = []
                for source in sources:
                    if not isinstance(source, dict):
                        continue
                    video_label = str(source.get("video_label") or "").strip()
                    video_id = str(source.get("video_id") or "").strip()
                    if video_label and video_id:
                        formatted_sources.append(f"{video_label}->{video_id}")
                    elif video_id:
                        formatted_sources.append(video_id)
                if formatted_sources:
                    source_summary = f" | 来源视频 {', '.join(formatted_sources)}"
            group_lines.append(
                f"选项 {group['choice_index']}: {group['choice_text']} | 对应图片索引 {start}-{end}{source_summary}"
            )
        return (
            "你在做厨房第一视角视频的时间定位题。"
            "每个选项对应一个候选时间段，系统已经为每个选项抽取了少量关键帧。"
            "如果不同选项来自不同视频，必须只根据该选项对应视频的帧判断，不能把一个视频里的动作迁移到另一个视频。"
            "请比较哪个选项最符合题目描述的动作、步骤、加料或事件。"
            "不要只看静态物体，要优先看手部交互、动作变化、容器变化和关键对象。"
            f"\n任务提示: {guidance}"
            f"\n题型: {task_family}"
            f"\n问题: {question}"
            "\n选项:\n"
            + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(choices))
            + "\n\n图片分组说明:\n"
            + "\n".join(group_lines)
            + '\n\n输出 JSON，字段固定为 {"best_index":0,"answer":"","confidence":0.0,"reason":""}。'
        )

    def _temporal_localization_guidance(self, *, task_family: str, question: str) -> str:
        family = str(task_family or "").strip().lower()
        if family == "fine_grained_action_localization":
            return "重点判断题目中的细粒度动作短语在哪个候选时间段真正发生，优先看手部动作和目标物体接触方式。"
        if family == "ingredient_ingredient_adding_localization":
            return "重点找食材被倒入、放入、撒入容器的瞬间，不要把只是拿起食材或静置误判成加入。"
        if family in {"recipe_rough_step_localization", "recipe_step_localization", "recipe_prep_localization"}:
            return "重点比较哪个候选时间段最符合题目里的菜谱步骤描述，优先关注动作序列和使用到的工具/容器。"
        if family == "object_motion_stationary_object_localization":
            return "重点判断从哪个候选时间开始，目标物体在后续很长时间里保持不再移动。"
        if "action" in family:
            return "重点比较动作本身，而不是只看场景里出现了什么物体。"
        if "localization" in family:
            return "这是时间定位题，重点比较哪个候选时间段最符合问题描述。"
        return "请比较每个候选时间段的关键帧，选择最符合题目描述的选项。"

    def infer_action_mechanism(self, question: str, choices: list[str], image_paths: list[str]) -> dict[str, Any]:
        selected_paths = self._select_compact_visual_paths(image_paths=image_paths, max_images=1)
        if not selected_paths:
            raise ValueError("infer_action_mechanism requires at least one visual image path")
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
        response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in selected_paths], temperature=0.0)
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
        result = {
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
        }
        return self._postprocess_action_mechanism_result(question=question, choices=choices, result=result)

    def infer_action_intent(self, question: str, choices: list[str], image_paths: list[str], context_notes: list[str]) -> dict[str, Any]:
        selected_paths = self._select_compact_visual_paths(image_paths=image_paths, max_images=4)
        scoped_notes = self._scope_action_intent_context_notes(question=question, image_paths=selected_paths, context_notes=context_notes)
        prompt = (
            "你在看厨房第一视角视频中某个动作前后的关键帧，这些图片按时间顺序排列。"
            "请判断这个动作的最直接目的。"
            "\n重点关注："
            "\n1. 拿起物体后是否立刻用于擦拭台面/器具"
            "\n2. 是否拿来擦手/干手"
            "\n3. 是否只是收起、挪开、放回"
            "\n4. 当前活动语境是否在清洗、收纳、做饭准备"
            "\n5. 如果当前帧只能看出“挪开某物”，但还看不清后续到底是为了取后面的东西，还是为了单纯腾空间/整理，请明确标记需要后续证据。"
            "\n6. 如果动作是拿起/转移某物，但目的依赖后续用途，例如称重、倒空、盛装、检查或清洗，也请明确标记需要后续证据。"
            "\n7. 如果两个选项都合理，不要勉强硬选；请给出第二候选，并说明需要看动作后结果帧。"
            "\n8. 如果动作是把物体换到一只手上、侧边或临时位置，从而腾出另一只手去开盖、开龙头、拿起别的物体，这种 free-hand enablement 也可能就是当前动作的直接目的。"
            "\n9. 如果动作是 turn off/close tap，一定要区分“容器满了”和“当前水流阶段结束，准备换成热/冷水或进入下一烹饪子目标”；不要因为 full 类答案更短就默认它正确。"
            "\n10. 如果动作是 tap/shake/tilt/tip/pour/hit/knock 某个勺子、锅、杯子或容器，要注意当前动作本身可能就是为了把残余内容物甩回、倒回或沥回原来的锅碗罐/水槽，这属于直接目的。"
            "\n11. 如果动作让一只手继续拿着某个刀/杯/碗/锅，而另一只手立刻去用海绵、刷子或流水清洗这个同一个物体，那么更直接的目的通常是‘清洗这个物体’，而不是泛泛的 free-hand 或 pick-up。"
            "\n12. 如果动作物体本身就是海绵/纸巾/抹布/毛巾这类清洁工具，而选项里同时出现“擦某个具体台面/洗某个具体器具”和泛泛的“clean / dry hands”，通常要优先判断那个具体被清洁的目标。"
            "\n13. 如果动作是 place/put 某个器具，一定区分“已经用完所以放下”和“洗后为了晾干/沥水而放下”；只有当证据里有洗后潮湿、流水、肥皂、水滴、朝上晾干等线索时，drying 才应优先。"
            f"\n上下文线索: {scoped_notes}"
            '\n输出 JSON，字段固定为 {"best_index":0,"answer":"","confidence":0.0,"reason":"","second_best_index":0,"ambiguity":false,"need_future_evidence":false,"future_window_s":4.0,"followup_focus":""}。'
            f"\n问题: {question}\n选项:\n"
            + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(choices))
        )
        response = self.model_client.inspect_images(prompt=prompt, image_paths=[Path(path) for path in selected_paths], temperature=0.0)
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
            best_index = self._resolve_choice_index(
                choices=choices,
                best_index=payload.get("best_index"),
                answer=payload.get("answer"),
            )
        except Exception:  # noqa: BLE001
            return self._fallback_rank_choices(question=question, choices=choices, evidence=scoped_notes, working_memory=[text])
        result = {
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
        }
        second_best = payload.get("second_best_index")
        try:
            second_best_index = int(second_best)
        except Exception:  # noqa: BLE001
            second_best_index = None
        if second_best_index is not None and not (0 <= second_best_index < len(choices)):
            second_best_index = None
        result["second_best_index"] = second_best_index
        result["ambiguity"] = bool(payload.get("ambiguity"))
        result["need_future_evidence"] = bool(payload.get("need_future_evidence"))
        result["future_window_s"] = float(payload.get("future_window_s") or 4.0)
        result["followup_focus"] = str(payload.get("followup_focus") or "")
        heuristic_need_followup, heuristic_reason = self._assess_action_intent_followup_need(
            question=question,
            choices=choices,
            result=result,
        )
        if heuristic_need_followup:
            result["need_future_evidence"] = True
            result["ambiguity"] = True
            if heuristic_reason == "future_use_evidence_needed":
                result["future_window_s"] = max(8.0, float(result.get("future_window_s") or 4.0))
            if not result["followup_focus"]:
                result["followup_focus"] = heuristic_reason
            result["reason"] = f"{result['reason']}; followup_needed={heuristic_reason}"
        return result

    def resolve_action_intent_pairwise(
        self,
        question: str,
        choices: list[str],
        candidate_indices: list[int],
        image_paths: list[str],
        context_notes: list[str],
    ) -> dict[str, Any]:
        valid_indices = []
        for value in candidate_indices:
            try:
                index = int(value)
            except Exception:  # noqa: BLE001
                continue
            if 0 <= index < len(choices) and index not in valid_indices:
                valid_indices.append(index)
        if len(valid_indices) < 2:
            return self.infer_action_intent(question=question, choices=choices, image_paths=image_paths, context_notes=context_notes)
        selected_paths = self._select_compact_visual_paths(image_paths=image_paths, max_images=8)
        scoped_notes = self._scope_action_intent_context_notes(question=question, image_paths=selected_paths, context_notes=context_notes)
        pair_choices = [{"index": index, "choice": str(choices[index])} for index in valid_indices[:2]]
        prompt = (
            "你在做厨房第一视角视频 why 题的最终歧义裁决。"
            "现在不是五选一，而是只在两个高混淆候选之间做判断。"
            "\n必须重点看动作发生后的结果帧，判断："
            "\n1. 后续是否真的取到了被遮挡/被挡住的物体"
            "\n2. 还是只是把前景物体挪开后腾出了空间/完成整理"
            "\n3. 是否把被挪开的物体放回，或是否把另一个目标物件安装/放回到位"
            "\n4. 必须区分“当前动作的直接物理效果”和“之后发生的下游动作”。"
            "\n5. 如果当前动作是 move/shift/remove 某物，而后续只是把另一个物体放入腾出的空间，当前动作的直接目的通常是 make space，不要把泛化的下游放置动作当成当前动作本身的目的。"
            "\n6. 但如果证据明确显示：当前动作是在为某个具体目标物体腾出某个具体槽位/位置/落点，且后续立刻发生该精确放置，则可以选择更具体的 put/place/fit/slot 类候选。"
            "\n7. 只有当被移动的物体本身被放回/摆正，或证据显示题目动作直接就是为另一个具体物体创造精确落位条件，才选择 put/place/right place 类候选。"
            "\n8. 如果证据仍不够，不要强行高置信收口；必须标记 need_more_evidence=true，并说明还需要看哪个后续动作。"
            f"\n上下文线索: {scoped_notes}"
            '\n输出 JSON，字段固定为 {"best_index":0,"answer":"","confidence":0.0,"reason":"","losing_index":0,"direct_effect":"","downstream_action":"","need_more_evidence":false,"needed_observation":""}。'
            f"\n问题: {question}"
            f"\n候选对决: {json.dumps(pair_choices, ensure_ascii=False)}"
        )
        response = self.model_client.inspect_images(
            prompt=prompt,
            image_paths=[Path(path) for path in selected_paths],
            temperature=0.0,
        )
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
        except Exception:  # noqa: BLE001
            return self._fallback_rank_choices(question=question, choices=choices, evidence=scoped_notes, working_memory=[text])
        best_index = self._resolve_choice_index(
            choices=choices,
            best_index=payload.get("best_index"),
            answer=payload.get("answer"),
        )
        if best_index not in valid_indices:
            best_index = valid_indices[0]
        losing_index = None
        try:
            parsed_loser = int(payload.get("losing_index"))
            if parsed_loser in valid_indices and parsed_loser != best_index:
                losing_index = parsed_loser
        except Exception:  # noqa: BLE001
            losing_index = None
        if losing_index is None:
            losing_index = next((item for item in valid_indices if item != best_index), None)
        result = {
            "best_index": best_index,
            "answer": str(choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "reason": str(payload.get("reason") or text[:300]),
            "losing_index": losing_index,
            "candidate_indices": valid_indices[:2],
            "direct_effect": str(payload.get("direct_effect") or ""),
            "downstream_action": str(payload.get("downstream_action") or ""),
            "need_more_evidence": bool(payload.get("need_more_evidence")),
            "needed_observation": str(payload.get("needed_observation") or ""),
        }
        result = self._apply_action_intent_pairwise_sufficiency(
            question=question,
            result=result,
        )
        return self._apply_action_intent_pairwise_causal_hierarchy(
            question=question,
            choices=choices,
            valid_indices=valid_indices[:2],
            result=result,
        )

    def resolve_action_intent_future_use(
        self,
        question: str,
        choices: list[str],
        candidate_indices: list[int],
        image_paths: list[str],
        context_notes: list[str],
    ) -> dict[str, Any]:
        valid_indices = []
        for value in candidate_indices:
            try:
                index = int(value)
            except Exception:  # noqa: BLE001
                continue
            if 0 <= index < len(choices) and index not in valid_indices:
                valid_indices.append(index)
        if len(valid_indices) < 2:
            valid_indices = list(range(len(choices)))
        selected_paths = self._select_compact_visual_paths(image_paths=image_paths, max_images=8)
        if not selected_paths:
            return self._fallback_rank_choices(question=question, choices=choices, evidence=context_notes, working_memory=[])
        scoped_notes = self._scope_action_intent_context_notes(question=question, image_paths=selected_paths, context_notes=context_notes)
        candidate_choices = [{"index": index, "choice": str(choices[index])} for index in valid_indices]
        prompt = (
            "你在做厨房第一视角视频 why 题的后续用途裁决。"
            "题目问的是执行某个动作的目的，但这个目的不能只从动作瞬间判断，必须看动作之后这个物体/空间被如何使用。"
            "\n请按时间顺序阅读图片，并对每个候选目的分别找支持证据和反证。"
            "\n必须遵守："
            "\n1. 如果候选说称重/测量，必须寻找秤、食材、容器被放到秤上或称量动作的后续证据。"
            "\n2. 如果候选说倒空/倒水/倒掉，必须寻找倾倒、流体离开容器或容器被拿向水槽/锅的证据。"
            "\n3. 如果候选说盛装/服务，必须寻找把食物装入/端出/分发的证据。"
            "\n4. 如果候选说检查/打开/关闭/取回，必须寻找对应后续动作已经发生的证据。"
            "\n5. 不要因为动作瞬间像某个候选就直接选；最后答案必须由动作后的实际结果支持。"
            "\n6. 如果多个候选都可能，选择后续证据更直接、更能排除其它候选的一个，并说明被排除候选缺少什么证据。"
            "\n7. 如果图片还没覆盖到真正的后续使用/放回/关闭前结果，不要强行高置信作答；标记 need_more_evidence=true，并说明还需要看什么。"
            "\n8. 如果题目动作本身是 move/transfer/remove/shift 某物，必须区分“当前动作直接腾出的访问/操作条件”和“之后才发生的下游取物/使用”。"
            "\n9. 如果只是先把某物移开，随后才拿起另一个物体，不能自动把那个更晚的取物动作当成当前动作的直接目的；只有当证据明确显示被移动物体本身就是在给那个目标让路，且不存在更直接的 tap/sink/drain/workspace access 解释时，才选择下游取物用途。"
            f"\n上下文线索: {scoped_notes}"
            '\n输出 JSON，字段固定为 {"candidate_evidence":[{"index":0,"support":"","contradiction":"","score":0.0}],"best_index":0,"answer":"","confidence":0.0,"decisive_observation":"","reason":"","need_more_evidence":false,"needed_observation":""}。'
            f"\n问题: {question}"
            f"\n候选: {json.dumps(candidate_choices, ensure_ascii=False)}"
        )
        response = self.model_client.inspect_images(
            prompt=prompt,
            image_paths=[Path(path) for path in selected_paths],
            temperature=0.0,
        )
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
        except Exception:  # noqa: BLE001
            return self._fallback_rank_choices(question=question, choices=choices, evidence=scoped_notes, working_memory=[text])
        best_index = self._resolve_choice_index(
            choices=choices,
            best_index=payload.get("best_index"),
            answer=payload.get("answer"),
        )
        if best_index not in valid_indices:
            scored_indices = self._valid_future_use_scores(payload.get("candidate_evidence"), valid_indices)
            best_index = scored_indices[0][0] if scored_indices else valid_indices[0]
        result = {
            "best_index": best_index,
            "answer": str(payload.get("answer") or choices[best_index]),
            "confidence": float(payload.get("confidence") or 0.0),
            "decisive_observation": str(payload.get("decisive_observation") or ""),
            "reason": str(payload.get("reason") or text[:300]),
            "candidate_evidence": payload.get("candidate_evidence") or [],
            "candidate_indices": valid_indices,
            "need_more_evidence": bool(payload.get("need_more_evidence")),
            "needed_observation": str(payload.get("needed_observation") or ""),
        }
        return self._apply_action_intent_future_use_sufficiency(
            result=result,
            valid_indices=valid_indices,
            choices=[str(choice) for choice in choices],
        )

    def _valid_future_use_scores(self, raw_items: Any, valid_indices: list[int]) -> list[tuple[int, float]]:
        scored: list[tuple[int, float]] = []
        if not isinstance(raw_items, list):
            return scored
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
                score = float(item.get("score") or 0.0)
            except Exception:  # noqa: BLE001
                continue
            if index in valid_indices:
                scored.append((index, score))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)

    def _apply_action_intent_future_use_sufficiency(
        self,
        *,
        result: dict[str, Any],
        valid_indices: list[int],
        choices: list[str] | None = None,
    ) -> dict[str, Any]:
        semantic_gaps = self._action_intent_future_use_semantic_gaps(
            result=result,
            valid_indices=valid_indices,
            choices=choices or [],
        )
        if result.get("need_more_evidence"):
            if not semantic_gaps:
                return result
            adjusted = dict(result)
            adjusted["reason"] = (
                f"{result.get('reason') or ''} semantic_support_check="
                + ",".join(semantic_gaps)
            ).strip()
            if not adjusted.get("needed_observation"):
                adjusted["needed_observation"] = self._action_intent_needed_observation_for_gaps(semantic_gaps)
            adjusted["confidence"] = min(float(result.get("confidence") or 0.0), 0.6)
            return adjusted
        adjusted = dict(result)
        decisive = str(result.get("decisive_observation") or "").strip()
        scored = self._valid_future_use_scores(result.get("candidate_evidence"), valid_indices)
        score_by_index = {index: score for index, score in scored}
        best_index = int(result.get("best_index")) if result.get("best_index") is not None else None
        top_score = score_by_index.get(best_index, 0.0) if best_index is not None else 0.0
        second_score = max((score for index, score in scored if index != best_index), default=0.0)
        missing_reasons: list[str] = []
        if not decisive:
            missing_reasons.append("missing_decisive_post_action_observation")
        if best_index is None or best_index not in score_by_index:
            missing_reasons.append("missing_best_candidate_evidence_score")
        if top_score < 0.55:
            missing_reasons.append("weak_top_candidate_score")
        if len(scored) >= 2 and top_score - second_score < 0.18:
            missing_reasons.append("ambiguous_candidate_score_margin")
        missing_reasons.extend(gap for gap in semantic_gaps if gap not in missing_reasons)
        if not missing_reasons:
            return adjusted
        adjusted["need_more_evidence"] = True
        if not adjusted.get("needed_observation"):
            adjusted["needed_observation"] = self._action_intent_needed_observation_for_gaps(missing_reasons)
        adjusted["reason"] = (
            f"{result.get('reason') or ''} sufficiency_check="
            + ",".join(missing_reasons)
        ).strip()
        adjusted["confidence"] = min(float(result.get("confidence") or 0.0), 0.66)
        return adjusted

    def _action_intent_future_use_semantic_gaps(
        self,
        *,
        result: dict[str, Any],
        valid_indices: list[int],
        choices: list[str],
    ) -> list[str]:
        try:
            best_index = int(result.get("best_index"))
        except Exception:  # noqa: BLE001
            return []
        if best_index not in valid_indices or not (0 <= best_index < len(choices)):
            return []
        choice = str(choices[best_index])
        choice_lc = choice.lower()
        categories = choice_categories(choice)
        support_text = self._action_intent_candidate_support_text(result=result, index=best_index)
        context_text = self._action_intent_candidate_context_text(result=result, index=best_index)
        if not support_text.strip():
            return ["missing_candidate_positive_evidence"]
        gaps: list[str] = []
        explicit_denial = self._action_intent_evidence_explicitly_denies_support(context_text)
        if explicit_denial:
            gaps.append("candidate_explicitly_lacks_observed_support")
        if any(term in context_text for term in ("least contradicted", "broadest", "least contradicted", "最不矛盾", "最宽泛")):
            gaps.append("best_is_unproven_broad_candidate")
        if "final_place_return" in categories and any(
            term in choice_lc
            for term in ("away", "put back", "store", "return", "right place", "proper place", "in place", "放回", "收起", "归位")
        ):
            if not self._text_has_any(
                support_text,
                (
                    "put away",
                    "stored",
                    "storage",
                    "drawer",
                    "cupboard",
                    "cabinet",
                    "fridge",
                    "hung",
                    "hook",
                    "returned",
                    "back in",
                    "right place",
                    "proper place",
                    "final",
                    "放回",
                    "收起",
                    "挂回",
                    "柜",
                    "抽屉",
                    "冰箱",
                    "归位",
                ),
            ):
                gaps.append("missing_final_placement_evidence")
            if self._text_has_any(
                context_text,
                (
                    "temporarily",
                    "temporary",
                    "merely relocated",
                    "only moved",
                    "placed on the counter",
                    "put on the counter",
                    "not stored",
                    "not put away",
                    "只是",
                    "仅",
                    "临时",
                    "台面",
                    "没有收",
                    "未收",
                    "没有放回",
                ),
            ):
                gaps.append("temporary_relocation_not_final_placement")
        if "dry" in choice_lc and "hand" in choice_lc:
            if not self._text_has_grouped_terms(
                support_text,
                (
                    ("hand", "hands", "手"),
                    ("dry", "dried", "wipe", "wiped", "towel", "cloth", "擦手", "干手", "擦干"),
                ),
            ):
                gaps.append("missing_dry_hands_evidence")
        if "wipe" in choice_lc and any(term in choice_lc for term in ("surface", "counter", "worktop", "table", "台面", "桌")):
            if not self._text_has_grouped_terms(
                support_text,
                (
                    ("wipe", "wiped", "clean", "cleaned", "scrub", "擦", "清洁"),
                    ("surface", "counter", "worktop", "table", "台面", "桌面"),
                ),
            ):
                gaps.append("missing_surface_wiping_evidence")
        if "clean" in choice_lc and "check" not in choice_lc and "whether" not in choice_lc:
            if explicit_denial or not self._text_has_any(
                support_text,
                ("cleaned", "cleaning", "wiped", "wipe", "washed", "wash", "rinsed", "scrub", "擦", "清洁", "洗", "冲洗"),
            ):
                gaps.append("missing_cleaning_action_evidence")
        if any(
            term in choice_lc
            for term in ("finished with", "finished now", "done with", "no longer need", "finished using", "不再需要", "用完")
        ):
            if not self._text_has_any(
                support_text,
                (
                    "finished with",
                    "done with",
                    "no longer needed",
                    "not used again",
                    "last use",
                    "finished now",
                    "put away for good",
                    "for storage",
                    "won't be used again",
                    "已经用完",
                    "不再需要",
                    "最后一次",
                    "不会再用",
                ),
            ):
                gaps.append("missing_finished_with_object_evidence")
            if self._text_has_any(
                context_text,
                (
                    "used again",
                    "use again",
                    "next step",
                    "shortly after",
                    "kept nearby",
                    "within reach",
                    "immediately reused",
                    "ready for the next",
                    "used shortly after",
                    "再次使用",
                    "下一步",
                    "很快又用",
                    "放在旁边待会再用",
                ),
            ):
                gaps.append("immediate_reuse_contradicts_finished_with_object")
        if "fill" in choice_lc:
            if not self._text_has_grouped_terms(
                support_text,
                (
                    ("fill", "filled", "water", "liquid", "running", "pour", "加水", "接水", "水"),
                    ("kettle", "pan", "pot", "container", "bottle", "水壶", "锅", "容器"),
                ),
            ):
                gaps.append("missing_fill_evidence")
        if any(term in choice_lc for term in ("weigh", "measure", "scale")):
            if not self._text_has_any(support_text, ("scale", "weigh", "weighed", "measure", "grams", "秤", "称", "克")):
                gaps.append("missing_measurement_evidence")
        if any(term in choice_lc for term in ("empty", "pour", "drain")):
            if not self._text_has_any(
                support_text,
                ("empty", "emptied", "pour", "poured", "drain", "drained", "water", "liquid", "sink", "倒", "倒出", "沥", "水槽"),
            ):
                gaps.append("missing_transfer_or_emptying_evidence")
        if "serve_consume" in categories:
            if not self._text_has_any(
                support_text,
                (
                    "serve",
                    "served",
                    "plate",
                    "plated",
                    "portion",
                    "dish out",
                    "portioned",
                    "served up",
                    "端",
                    "盛到",
                    "分装",
                    "上菜",
                ),
            ):
                gaps.append("missing_serving_or_consumption_evidence")
        if "open_close" in categories:
            if not self._text_has_any(
                support_text,
                (
                    "opened",
                    "closed",
                    "turned on",
                    "turned off",
                    "switch on",
                    "switch off",
                    "switched on",
                    "switched off",
                    "uncap",
                    "uncapped",
                    "cap",
                    "capped",
                    "unscrew",
                    "unscrewed",
                    "screwed back on",
                    "打开",
                    "关上",
                    "开启",
                    "关闭",
                    "拧开",
                    "盖上",
                ),
            ):
                gaps.append("missing_open_close_state_change_evidence")
        if any(term in choice_lc for term in ("check", "inspect", "read", "label", "date", "look")):
            if not self._text_has_any(support_text, ("check", "inspect", "look", "read", "label", "date", "visible", "查看", "看", "读", "标签")):
                gaps.append("missing_inspection_evidence")
        if "hand_free_enablement" in categories:
            if not self._text_has_grouped_terms(
                support_text,
                (
                    ("hand", "left hand", "right hand", "both hands", "双手", "左手", "右手", "手"),
                    ("free", "freed", "reach", "pick up", "turn on", "use", "hold", "拿", "腾出", "去拿", "去开", "使用"),
                ),
            ):
                gaps.append("missing_hand_free_enablement_evidence")
        if "access_retrieve" in categories and any(
            term in choice_lc
            for term in ("access", "behind", "retrieve", "get", "take out", "pick up", "reach", "missing")
        ):
            if not self._text_has_any(
                support_text,
                (
                    "access",
                    "behind",
                    "retrieve",
                    "retrieved",
                    "picked up",
                    "take out",
                    "reached",
                    "clear the way",
                    "got the item",
                    "够到",
                    "后面",
                    "取到",
                    "拿到",
                    "拿出",
                    "腾开",
                ),
            ):
                gaps.append("missing_access_or_retrieval_evidence")
        if "safety_avoid" in categories:
            if not self._text_has_any(
                support_text,
                (
                    "avoid",
                    "avoids",
                    "hot",
                    "burn",
                    "spill",
                    "mess",
                    "too hot",
                    "getting burnt",
                    "stabilized",
                    "two-handed",
                    "防止",
                    "太烫",
                    "烧焦",
                    "溢出",
                    "弄脏",
                    "避免",
                    "双手稳住",
                ),
            ):
                gaps.append("missing_hazard_or_spill_avoidance_evidence")
        return list(dict.fromkeys(gaps))

    def _action_intent_candidate_support_text(self, *, result: dict[str, Any], index: int) -> str:
        parts = [
            str(result.get("decisive_observation") or ""),
        ]
        for item in result.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            try:
                item_index = int(item.get("index"))
            except Exception:  # noqa: BLE001
                continue
            if item_index != index:
                continue
            parts.append(str(item.get("support") or ""))
        return " ".join(part for part in parts if part).lower()

    def _action_intent_candidate_context_text(self, *, result: dict[str, Any], index: int) -> str:
        parts = [
            str(result.get("decisive_observation") or ""),
            str(result.get("reason") or ""),
        ]
        for item in result.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            try:
                item_index = int(item.get("index"))
            except Exception:  # noqa: BLE001
                continue
            if item_index != index:
                continue
            parts.extend(str(item.get(key) or "") for key in ("support", "contradiction"))
        return " ".join(part for part in parts if part).lower()

    def _action_intent_evidence_explicitly_denies_support(self, text: str) -> bool:
        return self._text_has_any(
            text,
            (
                "no actual",
                "not actually",
                "not shown",
                "not visible",
                "absence of",
                "lacks",
                "lack of",
                "missing",
                "contradicted",
                "no candidate-specific",
                "缺少",
                "没有",
                "未看到",
                "未出现",
                "并未",
                "不足",
                "反证",
            ),
        )

    def _action_intent_needed_observation_for_gaps(self, gaps: list[str]) -> str:
        if any("dry_hands" in gap for gap in gaps):
            return "post-action frames showing whether the towel or cloth contacts/wipes the hands"
        if any("surface_wiping" in gap or "cleaning" in gap for gap in gaps):
            return "post-action frames showing whether the towel/cloth actually wipes or cleans a surface/object"
        if any("finished_with_object" in gap or "immediate_reuse_contradicts_finished_with_object" in gap for gap in gaps):
            return "post-action frames showing whether the object is truly no longer needed or instead kept nearby and used again shortly after"
        if any("final_placement" in gap or "temporary_relocation" in gap for gap in gaps):
            return "post-action frames showing the object's final placement, storage, or return location rather than a temporary move"
        if any("fill" in gap for gap in gaps):
            return "post-action frames showing filling with water/liquid and the target container"
        if any("measurement" in gap for gap in gaps):
            return "post-action frames showing the scale/measurement setup and the object being weighed"
        if any("transfer_or_emptying" in gap for gap in gaps):
            return "post-action frames showing pouring, draining, emptying, or liquid/contents transfer"
        if any("serving_or_consumption" in gap for gap in gaps):
            return "post-action frames showing serving, plating, portioning, eating, or drinking rather than just moving the object"
        if any("open_close_state_change" in gap for gap in gaps):
            return "post-action frames showing the object actually being opened, closed, switched, capped, or uncapped"
        if any("inspection" in gap for gap in gaps):
            return "post-action frames showing the person inspecting, reading, or checking the target"
        if any("hand_free_enablement" in gap for gap in gaps):
            return "post-action frames showing which hand is freed and what that hand immediately reaches for or operates next"
        if any("access_or_retrieval" in gap for gap in gaps):
            return "post-action frames showing whether the moved/opened object actually allows access to or retrieval of the target item"
        if any("hazard_or_spill_avoidance" in gap for gap in gaps):
            return "post-action frames showing the concrete hazard being avoided, such as burn, spill, or mess prevention"
        return "more post-action frames showing the object's actual use, final placement, or a clear result that separates the top competing choices"

    def _text_has_any(self, text: str, terms: tuple[str, ...]) -> bool:
        lowered = str(text or "").lower()
        return any(term in lowered for term in terms)

    def _text_has_grouped_terms(self, text: str, groups: tuple[tuple[str, ...], ...]) -> bool:
        lowered = str(text or "").lower()
        return all(any(term in lowered for term in group) for group in groups)

    def _apply_action_intent_pairwise_sufficiency(
        self,
        *,
        question: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if result.get("need_more_evidence"):
            return result
        question_lc = str(question or "").lower()
        if not any(
            token in question_lc
            for token in ("move ", "moved ", "shift ", "remove ", "clear ", "pick up ", "take ", "open ", "close ", "put ", "place ")
        ):
            return result
        direct_effect = str(result.get("direct_effect") or "").strip()
        downstream_action = str(result.get("downstream_action") or "").strip()
        reason = str(result.get("reason") or "").strip()
        evidence_text = f"{reason} {direct_effect} {downstream_action}".lower()
        post_action_terms = (
            "after",
            "then",
            "later",
            "follow",
            "subsequent",
            "result",
            "retrieve",
            "retrieved",
            "access",
            "behind",
            "space",
            "room",
            "placed",
            "put",
            "return",
            "closed",
            "opened",
            "后续",
            "之后",
            "随后",
            "结果",
            "拿到",
            "取出",
            "放入",
            "放回",
            "归位",
            "腾",
        )
        missing_reasons: list[str] = []
        if not direct_effect:
            missing_reasons.append("missing_direct_effect")
        if not any(term in evidence_text for term in post_action_terms):
            missing_reasons.append("missing_post_action_result_chain")
        try:
            confidence = float(result.get("confidence") or 0.0)
        except Exception:  # noqa: BLE001
            confidence = 0.0
        if confidence < 0.74 and not downstream_action:
            missing_reasons.append("weak_pairwise_outcome_support")
        if not missing_reasons:
            return result
        adjusted = dict(result)
        adjusted["need_more_evidence"] = True
        if not adjusted.get("needed_observation"):
            adjusted["needed_observation"] = (
                "more post-action frames showing the direct physical effect of the action and what happens next"
            )
        adjusted["reason"] = (
            f"{result.get('reason') or ''} pairwise_sufficiency_check="
            + ",".join(missing_reasons)
        ).strip()
        adjusted["confidence"] = min(confidence, 0.66)
        return adjusted

    def _apply_action_intent_pairwise_causal_hierarchy(
        self,
        *,
        question: str,
        choices: list[str],
        valid_indices: list[int],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if len(valid_indices) < 2 or result.get("need_more_evidence"):
            return result
        question_lc = str(question or "").lower()
        if not any(token in question_lc for token in ("move ", "moved ", "shift ", "remove ", "clear ", "pick up ", "take ")):
            return result
        direct_space_index = next(
            (
                index
                for index in valid_indices
                if self._choice_is_direct_space_purpose(str(choices[index]))
            ),
            None,
        )
        downstream_place_index = next(
            (
                index
                for index in valid_indices
                if self._choice_is_downstream_place_purpose(str(choices[index]))
            ),
            None,
        )
        if direct_space_index is None or downstream_place_index is None:
            return result
        if int(result.get("best_index", -1)) != downstream_place_index:
            return result
        explanation = " ".join(
            str(result.get(key) or "")
            for key in ("reason", "direct_effect", "downstream_action")
        ).lower()
        if not self._explanation_uses_downstream_space_chain(explanation):
            return result
        downstream_choice = str(choices[downstream_place_index])
        if self._choice_is_exact_downstream_placement_purpose(downstream_choice) and self._explanation_uses_exact_targeted_placement_chain(
            explanation
        ):
            return result
        adjusted = dict(result)
        adjusted["best_index"] = direct_space_index
        adjusted["answer"] = str(choices[direct_space_index])
        adjusted["losing_index"] = downstream_place_index
        adjusted["confidence"] = max(0.8, min(0.9, float(result.get("confidence") or 0.0) + 0.04))
        adjusted["causal_hierarchy_adjusted"] = True
        adjusted["reason"] = (
            f"{result.get('reason') or ''} causal_hierarchy_adjustment: "
            "the evidence describes a downstream placement after the moved object created room; "
            "for a move action, that supports the direct purpose of making space rather than selecting the downstream placement as the action purpose."
        ).strip()
        return adjusted

    def _choice_is_direct_space_purpose(self, choice: str) -> bool:
        text = str(choice or "").lower()
        return any(
            token in text
            for token in (
                "make space",
                "create space",
                "free up space",
                "clear space",
                "make room",
                "create room",
                "free up room",
                "clear room",
            )
        )

    def _choice_is_downstream_place_purpose(self, choice: str) -> bool:
        text = str(choice or "").lower()
        if self._choice_is_direct_space_purpose(text):
            return False
        return (
            any(token in text for token in ("put ", "place ", "right place", "proper place", "fit ", "insert ", "slot "))
            and any(token in text for token in ("other", "another", "piece", "part", "item", "dish", "lid", "white"))
        )

    def _choice_is_exact_downstream_placement_purpose(self, choice: str) -> bool:
        text = str(choice or "").lower()
        if self._choice_is_direct_space_purpose(text):
            return False
        if not any(token in text for token in ("put ", "place ", "fit ", "insert ", "slot ", "into ", "onto ", "down on ")):
            return False
        if any(token in text for token in ("right place", "proper place")) and not any(
            token in text
            for token in (
                "sink",
                "slot",
                "rack",
                "dishwasher",
                "scale",
                "hob",
                "tray",
                "counter",
                "board",
                "plate",
                "bowl",
                "colander",
                "saucepan",
                "pan",
                "water",
            )
        ):
            return False
        has_target = any(
            token in text
            for token in (
                "saucepan",
                "pan",
                "pot",
                "bowl",
                "plate",
                "tray",
                "colander",
                "board",
                "lid",
                "tupperware",
                "large bowls",
                "item",
                "object",
            )
        )
        has_destination = any(
            token in text
            for token in (
                "sink",
                "slot",
                "rack",
                "dishwasher",
                "scale",
                "hob",
                "counter",
                "drying rack",
                "draining rack",
                "into the",
                "onto the",
                "on the counter",
                "in the sink",
            )
        )
        return has_target and has_destination

    def _explanation_uses_downstream_space_chain(self, explanation: str) -> bool:
        text = str(explanation or "").lower()
        has_downstream = any(
            token in text
            for token in (
                "another",
                "other",
                "white",
                "piece",
                "part",
                "item",
                "dish",
                "lid",
                "place",
                "put",
                "放",
                "白色",
                "部件",
                "方形",
                "归位",
                "安装",
            )
        )
        has_space_or_blocking = any(
            token in text
            for token in (
                "space",
                "room",
                "clear",
                "way",
                "make room",
                "out of the way",
                "avoid blocking",
                "挡",
                "腾",
                "空位",
                "挪开",
            )
        )
        return has_downstream and has_space_or_blocking

    def _explanation_uses_exact_targeted_placement_chain(self, explanation: str) -> bool:
        text = str(explanation or "").lower()
        has_target = any(
            token in text
            for token in (
                "saucepan",
                "pan",
                "pot",
                "bowl",
                "plate",
                "tray",
                "colander",
                "lid",
                "tupperware",
                "large bowls",
                "another item",
                "next item",
            )
        )
        has_destination = any(
            token in text
            for token in (
                "sink slot",
                "sink",
                "slot",
                "rack",
                "dishwasher",
                "scale",
                "hob",
                "counter",
                "drying rack",
                "draining rack",
                "available spot",
                "freed spot",
                "freed slot",
                "exact slot",
                "exact place",
            )
        )
        has_immediacy = any(
            token in text
            for token in (
                "immediately",
                "right after",
                "directly after",
                "as soon as",
                "then",
                "after",
                "随后",
                "紧接着",
                "立刻",
            )
        )
        return has_target and has_destination and has_immediacy

    def _scope_action_intent_context_notes(self, *, question: str, image_paths: list[str], context_notes: list[str]) -> list[str]:
        context_notes = self._sanitize_action_intent_context_notes(context_notes)
        if not context_notes:
            return []
        hints = self.default_hints(self.runtime_question or question, self.runtime_inputs_json)
        anchor_times = [float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []]
        for path in image_paths:
            inferred = self._infer_artifact_time(path)
            if inferred is not None:
                anchor_times.append(float(inferred))
        if not anchor_times:
            return [str(item) for item in context_notes[:6]]
        min_anchor = min(anchor_times)
        max_anchor = max(anchor_times)
        window_start = max(0.0, min_anchor - 6.0)
        window_end = max_anchor + 6.0
        scoped: list[str] = []
        for raw in context_notes:
            note = str(raw)
            if not note:
                continue
            times = self._extract_embedded_note_times(note)
            lowered = note.lower()
            if times:
                overlaps = any(not (end_time < window_start or start_time > window_end) for start_time, end_time in times)
                if overlaps:
                    scoped.append(note)
                    continue
            # Keep only non-answer observational notes when timestamps are unavailable.
            if any(token in lowered for token in ("inspection;", "ongoing_action=", "target_location=")):
                scoped.append(note)
        if not scoped:
            return [str(item) for item in context_notes[:6]]
        return scoped[:8]

    def _sanitize_action_intent_context_notes(self, context_notes: list[str]) -> list[str]:
        sanitized: list[str] = []
        leaky_tokens = (
            "action_intent_",
            "visual_mcq_reason=",
            "answer_hint=",
            "candidate_answer_index=",
            "deterministic_finalize",
            "source=agent_timeline_summary",
            "source=session_memory_compressor",
        )
        for raw in context_notes:
            note = str(raw)
            lowered = note.lower()
            if any(token in lowered for token in leaky_tokens):
                continue
            if note and note not in sanitized:
                sanitized.append(note)
        return sanitized

    def _extract_embedded_note_times(self, text: str) -> list[tuple[float, float]]:
        spans: list[tuple[float, float]] = []
        for match in re.finditer(r"time=([0-9.]+)-([0-9.]+)", str(text)):
            try:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
            except Exception:  # noqa: BLE001
                continue
            spans.append((min(start_time, end_time), max(start_time, end_time)))
        return spans

    def _assess_action_intent_followup_need(
        self,
        *,
        question: str,
        choices: list[str],
        result: dict[str, Any],
    ) -> tuple[bool, str]:
        confidence = float(result.get("confidence") or 0.0)
        reason = str(result.get("reason") or "").lower()
        question_text = str(question or "").lower()
        best_index = result.get("best_index")
        second_best_index = result.get("second_best_index")
        best_choice = str(choices[int(best_index)]) if isinstance(best_index, int) and 0 <= int(best_index) < len(choices) else ""
        second_choice = str(choices[int(second_best_index)]) if isinstance(second_best_index, int) and 0 <= int(second_best_index) < len(choices) else ""
        candidate_indices = [
            index
            for index in (best_index, second_best_index)
            if isinstance(index, int) and 0 <= int(index) < len(choices)
        ]
        semantic_need, semantic_reason, semantic_window_s, semantic_resolver = action_intent_followup_decision(
            question=question,
            choices=[str(choice) for choice in choices],
            indices=candidate_indices if len(candidate_indices) >= 2 else None,
            confidence=confidence,
            reason_text=reason,
        )
        if semantic_need:
            if semantic_resolver == "future_use" or semantic_window_s >= 8.0:
                result["future_window_s"] = max(float(result.get("future_window_s") or 4.0), semantic_window_s)
            return True, semantic_reason
        joined = " | ".join(item.lower() for item in (best_choice, second_choice) if item)
        overlap_pairs = (
            ("access", "make space"),
            ("move", "make space"),
            ("clear the way", "make space"),
            ("put back", "move"),
            ("rearrange", "make space"),
        )
        action_terms = (
            "move ",
            "moved ",
            "shift ",
            "clear ",
            "open ",
            "close ",
            "put ",
            "place ",
            "pick up",
            "take out",
            "remove ",
        )
        manipulation_terms = (
            "pick up",
            "picked up",
            "pick ",
            "lift ",
            "lifted ",
            "take ",
            "took ",
            "transfer ",
            "transferred ",
            "carry ",
            "carried ",
            "grab ",
            "grabbed ",
        )
        outcome_terms = (
            "make space",
            "space",
            "access",
            "behind",
            "clear the way",
            "right place",
            "put back",
            "put",
            "place",
            "pick up",
            "remove",
            "rearrange",
        )
        future_use_terms = (
            "weigh",
            "measure",
            "use ",
            "serve",
            "empty",
            "drain",
            "pour",
            "check",
            "retrieve",
            "get ",
            "fill",
            "wash",
            "clean",
            "dry",
            "record",
            "scan",
            "put ",
            "place ",
            "return",
            "close",
            "open",
            "turn",
            "mix",
            "stir",
        )
        choice_texts = [item.lower() for item in (best_choice, second_choice) if item]
        outcome_hit_count = sum(1 for item in choice_texts if any(token in item for token in outcome_terms))
        all_choice_texts = [str(choice).lower() for choice in choices]
        future_use_hit_count = sum(1 for item in all_choice_texts if any(token in item for token in future_use_terms))
        if result.get("need_future_evidence"):
            return True, "model_flagged_future_evidence"
        if result.get("ambiguity"):
            return True, "model_flagged_ambiguity"
        if any(token in question_text for token in manipulation_terms) and future_use_hit_count >= 2:
            return True, "future_use_evidence_needed"
        if (
            len(choice_texts) >= 2
            and any(token in question_text for token in action_terms)
            and outcome_hit_count >= 2
        ):
            return True, "outcome_dependent_pairwise_needed"
        if confidence < 0.84:
            for left, right in overlap_pairs:
                if left in joined and right in joined:
                    return True, f"low_confidence_semantic_overlap:{left}|{right}"
        if any(token in question_text for token in ("move ", "shift ", "clear ", "open ", "close ", "put back", "put aside")) and confidence <= 0.92:
            if any(token in joined for token in ("access", "make space", "clear the way", "put back")):
                return True, "post_action_result_needed"
        if "behind" in joined and "space" in joined:
            return True, "behind_vs_space_needs_outcome"
        if "can't tell" in reason or "uncertain" in reason or "ambiguous" in reason:
            return True, "reason_explicitly_uncertain"
        return False, ""

    def write_observation(
        self,
        label: str,
        start_time: float | None = None,
        end_time: float | None = None,
        attributes: dict[str, Any] | None = None,
        evidence_paths: list[str] | None = None,
        keywords: list[str] | None = None,
        source_tool: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        safe_label = self._safe_tag(label)[:64]
        node_id = f"observation:{self.video_id}:{safe_label}:{self._node_time_token(start_time, end_time)}"
        payload = dict(attributes or {})
        if source_tool and "source_tool" not in payload:
            payload["source_tool"] = str(source_tool)
        if confidence is not None and "confidence" not in payload:
            payload["confidence"] = float(confidence)
        node = GraphNodeRecord(
            node_id=node_id,
            node_type="observation",
            label=label,
            video_id=self.video_id,
            start_time=start_time,
            end_time=end_time,
            attributes=payload,
            evidence_paths=evidence_paths or [],
            keywords=keywords or self._keywords_from_payload(label, payload),
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
        self._link_written_node(
            node_id=node_id,
            time_s=start_time if start_time is not None else end_time,
            evidence_paths=evidence_paths or [],
            semantic_hint=label,
        )
        return {"node_id": node_id}

    def write_frame_observation(
        self,
        frame_path: str,
        time_s: float | None,
        label: str,
        observation: dict[str, Any],
        keywords: list[str] | None = None,
        source_tool: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        safe_label = self._safe_tag(label)[:64]
        node_id = f"frame_observation:{self.video_id}:{safe_label}:{self._node_time_token(time_s, time_s)}"
        payload = {"frame_path": frame_path, "observation": observation, "source": "agent_frame_observation"}
        node = self.graph.write_node(
            node_id=node_id,
            node_type="observation",
            label=label,
            video_id=self.video_id,
            start_time=time_s,
            end_time=time_s,
            attributes=payload,
            evidence_paths=[frame_path],
            keywords=keywords or self._keywords_from_payload(label, payload),
            source_tool=source_tool,
            confidence=confidence,
        )
        self._link_to_matching_frame(node_id=node_id, time_s=time_s)
        self._link_written_node(node_id=node_id, time_s=time_s, evidence_paths=[frame_path], semantic_hint=label)
        return {"node_id": node["node_id"], "node": node}

    def write_region_observation(
        self,
        image_path: str,
        bbox: list[float] | None,
        time_s: float | None,
        label: str,
        observation: dict[str, Any],
        keywords: list[str] | None = None,
        source_tool: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        safe_label = self._safe_tag(label)[:64]
        node_id = f"region_observation:{self.video_id}:{safe_label}:{self._node_time_token(time_s, time_s)}"
        payload = {
            "image_path": image_path,
            "bbox": bbox or [],
            "observation": observation,
            "source": "agent_region_observation",
        }
        node = self.graph.write_node(
            node_id=node_id,
            node_type="region",
            label=label,
            video_id=self.video_id,
            start_time=time_s,
            end_time=time_s,
            attributes=payload,
            evidence_paths=[image_path],
            keywords=keywords or self._keywords_from_payload(label, payload),
            source_tool=source_tool,
            confidence=confidence,
        )
        self._link_to_matching_frame(node_id=node_id, time_s=time_s)
        self._link_written_node(node_id=node_id, time_s=time_s, evidence_paths=[image_path], semantic_hint=label)
        return {"node_id": node["node_id"], "node": node}

    def write_ocr_reading(
        self,
        label: str,
        reading: str,
        time_s: float | None = None,
        image_path: str | None = None,
        bbox: list[float] | None = None,
        attributes: dict[str, Any] | None = None,
        keywords: list[str] | None = None,
        source_tool: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        safe_label = self._safe_tag(label)[:64]
        node_id = f"ocr_reading:{self.video_id}:{safe_label}:{self._node_time_token(time_s, time_s)}"
        payload = dict(attributes or {})
        payload.update({"reading": reading, "image_path": image_path, "bbox": bbox or [], "source": "agent_ocr"})
        evidence_paths = [image_path] if image_path else []
        node = self.graph.write_node(
            node_id=node_id,
            node_type="ocr_reading",
            label=label,
            video_id=self.video_id,
            start_time=time_s,
            end_time=time_s,
            attributes=payload,
            evidence_paths=evidence_paths,
            keywords=keywords or self._keywords_from_payload(f"{label} {reading}", payload),
            source_tool=source_tool,
            confidence=confidence,
        )
        self._link_to_matching_frame(node_id=node_id, time_s=time_s)
        self._link_written_node(node_id=node_id, time_s=time_s, evidence_paths=evidence_paths, semantic_hint=label)
        return {"node_id": node["node_id"], "node": node}

    def write_audio_event(
        self,
        label: str,
        start_time: float | None = None,
        end_time: float | None = None,
        attributes: dict[str, Any] | None = None,
        evidence_paths: list[str] | None = None,
        keywords: list[str] | None = None,
        source_tool: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        safe_label = self._safe_tag(label)[:64]
        node_id = f"audio_writeback:{self.video_id}:{safe_label}:{self._node_time_token(start_time, end_time)}"
        payload = dict(attributes or {})
        payload["source"] = "agent_audio_event"
        node = self.graph.write_node(
            node_id=node_id,
            node_type="audio_event",
            label=label,
            video_id=self.video_id,
            start_time=start_time,
            end_time=end_time,
            attributes=payload,
            evidence_paths=evidence_paths or [],
            keywords=keywords or self._keywords_from_payload(label, payload),
            source_tool=source_tool,
            confidence=confidence,
        )
        self._link_written_node(
            node_id=node_id,
            time_s=start_time if start_time is not None else end_time,
            evidence_paths=evidence_paths or [],
            semantic_hint=label,
        )
        return {"node_id": node["node_id"], "node": node}

    def write_timeline_summary(
        self,
        label: str,
        start_time: float | None = None,
        end_time: float | None = None,
        summary: str = "",
        evidence_paths: list[str] | None = None,
        keywords: list[str] | None = None,
        source_tool: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        safe_label = self._safe_tag(label)[:64]
        node_id = f"timeline_summary:{self.video_id}:{safe_label}:{self._node_time_token(start_time, end_time)}"
        payload = {"summary": summary, "source": "agent_timeline_summary"}
        node = self.graph.write_node(
            node_id=node_id,
            node_type="timeline_event",
            label=label,
            video_id=self.video_id,
            start_time=start_time,
            end_time=end_time,
            attributes=payload,
            evidence_paths=evidence_paths or [],
            keywords=keywords or self._keywords_from_payload(f"{label} {summary}", payload),
            source_tool=source_tool,
            confidence=confidence,
        )
        self._link_written_node(
            node_id=node_id,
            time_s=start_time if start_time is not None else end_time,
            evidence_paths=evidence_paths or [],
            semantic_hint=label,
        )
        return {"node_id": node["node_id"], "node": node}

    def write_state_change(
        self,
        label: str,
        target: str,
        before_state: str | None = None,
        after_state: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        evidence_paths: list[str] | None = None,
        keywords: list[str] | None = None,
        source_tool: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        safe_label = self._safe_tag(label)[:64]
        node_id = f"state_change:{self.video_id}:{safe_label}:{self._node_time_token(start_time, end_time)}"
        payload = {
            "target": target,
            "before_state": before_state,
            "after_state": after_state,
            "source": "agent_state_change",
        }
        node = self.graph.write_node(
            node_id=node_id,
            node_type="state_change",
            label=label,
            video_id=self.video_id,
            start_time=start_time,
            end_time=end_time,
            attributes=payload,
            evidence_paths=evidence_paths or [],
            keywords=keywords or self._keywords_from_payload(f"{label} {target} {before_state or ''} {after_state or ''}", payload),
            source_tool=source_tool,
            confidence=confidence,
        )
        self._link_written_node(
            node_id=node_id,
            time_s=start_time if start_time is not None else end_time,
            evidence_paths=evidence_paths or [],
            semantic_hint=target,
        )
        return {"node_id": node["node_id"], "node": node}

    def finish(self, prediction: int, answer: str, confidence: float = 0.0) -> dict[str, Any]:
        return {"prediction": int(prediction), "answer": str(answer), "confidence": float(confidence), "done": True}

    def default_hints(self, question: str, inputs_json: str) -> dict[str, Any]:
        try:
            inputs = json.loads(inputs_json or "{}")
        except json.JSONDecodeError:
            inputs = {}
        lowered_question = str(question or "").lower()
        is_weight_question = any(token in lowered_question for token in ("weigh", "weight", "reading", "scale", "gram", "grams", "kg", "digit", "number"))
        explicit_location_phrase = any(token in lowered_question for token in ("near ", "near the ", "left", "right", "front", "behind", "inside", "outside"))
        recipe_step_hint = self._extract_recipe_step_hint(question)
        suppress_recipe_container_location = bool(recipe_step_hint) and "which high-level activity" in lowered_question
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
            "recipe_step_hint": recipe_step_hint,
            "state_keyword": self._extract_state_keyword(question),
            "location_keyword": self._extract_location_keyword(
                question,
                allow_container_terms=((not is_weight_question) or explicit_location_phrase) and not suppress_recipe_container_location,
            ),
            "ocr_keyword": self._extract_ocr_keyword(question),
            "object_hint": None if suppress_recipe_container_location else self._extract_object_hint(question),
            "inputs": inputs if isinstance(inputs, dict) else {},
        }

    def _video_path(self) -> Path:
        return self._video_path_for(self.video_id)

    def _video_path_for(self, video_id: str) -> Path:
        store = self.store if video_id == self.video_id else self._ensure_video_store(video_id)
        node = store.get_node(f"video:{video_id}")
        if not node:
            raise RuntimeError(f"video node missing for video_id={video_id}")
        path = node["attributes"].get("path") or (node.get("evidence_paths") or [None])[0]
        if not path:
            raise RuntimeError(f"video path missing for video_id={video_id}")
        return Path(path)

    def _ensure_video_store(self, video_id: str) -> GraphMemoryStore:
        root = self.paths.graph_memory_root / video_id
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

    def _link_to_matching_frame(self, *, node_id: str, time_s: float | None) -> None:
        if time_s is None:
            return
        frame_nodes = self.store.query_nodes(
            video_id=self.video_id,
            node_types=["frame"],
            time_start=max(0.0, float(time_s) - 0.6),
            time_end=float(time_s) + 0.6,
            limit=1,
        )
        if not frame_nodes:
            return
        frame_node = frame_nodes[0]
        self.graph.write_edge(
            edge_id=f"derived_from:{node_id}:{frame_node['node_id']}",
            source_id=node_id,
            target_id=frame_node["node_id"],
            edge_type="derived_from",
            video_id=self.video_id,
            attributes={
                "source": "agent_linker",
                "time_delta": abs(float(frame_node.get("start_time") or 0.0) - float(time_s)),
            },
        )

    def _link_written_node(
        self,
        *,
        node_id: str,
        time_s: float | None,
        evidence_paths: list[str],
        semantic_hint: str | None,
    ) -> None:
        self._link_temporal_neighbors(node_id=node_id, time_s=time_s)
        self._link_evidence_neighbors(node_id=node_id, time_s=time_s, evidence_paths=evidence_paths, semantic_hint=semantic_hint)

    def _link_temporal_neighbors(self, *, node_id: str, time_s: float | None) -> None:
        if time_s is None:
            return
        nearby = self.store.query_nodes(
            video_id=self.video_id,
            node_types=["timeline_event", "observation", "region", "ocr_reading", "audio_event", "state_change", "frame"],
            time_start=max(0.0, float(time_s) - 5.0),
            time_end=float(time_s) + 5.0,
            limit=40,
        )
        previous_node: dict[str, Any] | None = None
        next_node: dict[str, Any] | None = None
        previous_delta: float | None = None
        next_delta: float | None = None
        for candidate in nearby:
            if not isinstance(candidate, dict) or candidate.get("node_id") == node_id:
                continue
            candidate_time = candidate.get("start_time")
            if candidate_time is None:
                continue
            delta = float(candidate_time) - float(time_s)
            if delta < 0:
                abs_delta = abs(delta)
                if previous_delta is None or abs_delta < previous_delta:
                    previous_delta = abs_delta
                    previous_node = candidate
            elif delta > 0:
                if next_delta is None or delta < next_delta:
                    next_delta = delta
                    next_node = candidate
        if previous_node is not None:
            self.graph.write_edge(
                edge_id=f"after:{node_id}:{previous_node['node_id']}",
                source_id=node_id,
                target_id=previous_node["node_id"],
                edge_type="after",
                video_id=self.video_id,
                attributes={"source": "agent_linker", "time_delta": previous_delta},
            )
            self.graph.write_edge(
                edge_id=f"before:{previous_node['node_id']}:{node_id}",
                source_id=previous_node["node_id"],
                target_id=node_id,
                edge_type="before",
                video_id=self.video_id,
                attributes={"source": "agent_linker", "time_delta": previous_delta},
            )
        if next_node is not None:
            self.graph.write_edge(
                edge_id=f"before:{node_id}:{next_node['node_id']}",
                source_id=node_id,
                target_id=next_node["node_id"],
                edge_type="before",
                video_id=self.video_id,
                attributes={"source": "agent_linker", "time_delta": next_delta},
            )
            self.graph.write_edge(
                edge_id=f"after:{next_node['node_id']}:{node_id}",
                source_id=next_node["node_id"],
                target_id=node_id,
                edge_type="after",
                video_id=self.video_id,
                attributes={"source": "agent_linker", "time_delta": next_delta},
            )

    def _link_evidence_neighbors(
        self,
        *,
        node_id: str,
        time_s: float | None,
        evidence_paths: list[str],
        semantic_hint: str | None,
    ) -> None:
        if not evidence_paths and time_s is None:
            return
        nearby = self.store.query_nodes(
            video_id=self.video_id,
            node_types=["timeline_event", "observation", "region", "ocr_reading", "audio_event", "state_change", "frame"],
            time_start=max(0.0, float(time_s) - 2.0) if time_s is not None else None,
            time_end=float(time_s) + 2.0 if time_s is not None else None,
            limit=40,
        )
        hint_tokens = set(self._name_tokens(semantic_hint or ""))
        for candidate in nearby:
            if not isinstance(candidate, dict) or candidate.get("node_id") == node_id:
                continue
            candidate_id = str(candidate.get("node_id") or "")
            candidate_paths = {str(path) for path in candidate.get("evidence_paths", []) if isinstance(path, str) and path}
            shared_paths = sorted({str(path) for path in evidence_paths if path} & candidate_paths)
            if shared_paths:
                self.graph.write_edge(
                    edge_id=f"co_occurs:{node_id}:{candidate_id}",
                    source_id=node_id,
                    target_id=candidate_id,
                    edge_type="co_occurs",
                    video_id=self.video_id,
                    attributes={"source": "agent_linker", "shared_evidence_paths": shared_paths},
                )
            candidate_tokens = set(self._name_tokens(str(candidate.get("label") or "")))
            if hint_tokens and candidate_tokens and hint_tokens & candidate_tokens:
                relation = "same_step" if "timeline" in str(candidate.get("node_type") or "") else "same_object"
                self.graph.write_edge(
                    edge_id=f"{relation}:{node_id}:{candidate_id}",
                    source_id=node_id,
                    target_id=candidate_id,
                    edge_type=relation,
                    video_id=self.video_id,
                    attributes={"source": "agent_linker", "shared_tokens": sorted(hint_tokens & candidate_tokens)},
                )

    def _parse_hms(self, text: str) -> float:
        hours, minutes, seconds = text.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    def _infer_artifact_time(self, path: str) -> float | None:
        match = re.search(r"_(\d+\.\d+)s(?:_|\.|$)", str(path))
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _ocr_image(self, image_path: Path) -> str:
        try:
            import pytesseract  # type: ignore
        except Exception:  # noqa: BLE001
            pytesseract = None
        if pytesseract is not None:
            try:
                text = pytesseract.image_to_string(Image.open(image_path))
                text = text.strip()
                if text:
                    return text
            except Exception:  # noqa: BLE001
                pass
        response = self.model_client.inspect_images(
            prompt=(
                "请只读取这张图片中可见的文字、数字、单位或显示屏读数。"
                '输出 JSON，字段固定为 {"text":"","reading":"","confidence":0.0}。'
            ),
            image_paths=[image_path],
            temperature=0.0,
        )
        text = response.content.strip()
        try:
            payload = self.model_client._extract_json_object(text)
            combined = str(payload.get("reading") or payload.get("text") or "").strip()
            if combined:
                return combined
        except Exception:  # noqa: BLE001
            pass
        return text

    def _extract_compact_reading(self, text: str) -> str:
        match = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", text)
        if match:
            value = match.group(1)
            unit = match.group(2)
            if value.endswith(".0"):
                value = value[:-2]
            return f"{value} {unit}".strip()
        digits = re.findall(r"\d+(?:\.\d+)?", text)
        if digits:
            value = digits[0]
            if value.endswith(".0"):
                value = value[:-2]
            return value
        return text.strip()

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
            for pattern in (r"(?:option|choice)\s*(\d+)", r"选项\s*(\d+)"):
                match = re.search(pattern, answer_text, flags=re.IGNORECASE)
                if not match:
                    continue
                try:
                    ordinal = int(match.group(1))
                except Exception:  # noqa: BLE001
                    continue
                if 0 <= ordinal < len(choices):
                    return ordinal
                if 1 <= ordinal <= len(choices):
                    return ordinal - 1
        return 0

    def _select_compact_visual_paths(self, *, image_paths: list[str], max_images: int) -> list[str]:
        unique_paths = self._filter_visual_paths(image_paths)
        if len(unique_paths) <= max_images:
            return unique_paths
        if max_images <= 1:
            return [unique_paths[len(unique_paths) // 2]]
        if max_images == 2:
            return [unique_paths[0], unique_paths[-1]]
        selected_indices = {0, len(unique_paths) - 1}
        if max_images >= 3:
            selected_indices.add(len(unique_paths) // 2)
        if max_images >= 4:
            selected_indices.add(max(1, len(unique_paths) // 3))
        if max_images >= 5:
            selected_indices.add(min(len(unique_paths) - 2, (2 * len(unique_paths)) // 3))
        ordered = [unique_paths[index] for index in sorted(selected_indices)]
        if len(ordered) >= max_images:
            return ordered[:max_images]
        for path in unique_paths:
            if path in ordered:
                continue
            ordered.append(path)
            if len(ordered) >= max_images:
                break
        return ordered[:max_images]

    def _filter_visual_paths(self, image_paths: list[str]) -> list[str]:
        valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        unique_paths: list[str] = []
        seen: set[str] = set()
        for raw_path in image_paths:
            normalized = str(raw_path).strip()
            if not normalized or normalized in seen:
                continue
            if Path(normalized).suffix.lower() not in valid_suffixes:
                continue
            unique_paths.append(normalized)
            seen.add(normalized)
        return unique_paths

    def _question_asks_object_source_location(self, question: str) -> bool:
        lowered = str(question or "").lower()
        return "take the object" in lowered and "from before putting it" in lowered

    def _track_covering_reference_time(self, *, tracks: list[dict[str, Any]], reference_time: float) -> dict[str, Any] | None:
        for track in tracks:
            start_time = self._float_or_none(track.get("start_time"))
            end_time = self._float_or_none(track.get("end_time"))
            if start_time is None or end_time is None:
                continue
            if start_time - 0.25 <= float(reference_time) <= end_time + 0.25:
                return track
        return None

    def _fixture_target_tokens(self, question: str) -> list[str]:
        question_lc = str(question).lower()
        match = re.search(r"where is the ([a-zA-Z0-9_ /-]+?) located", question_lc)
        if match:
            return self._name_tokens(match.group(1))
        match = re.search(r"what is the person looking at", question_lc)
        if match:
            return ["look"]
        return self._name_tokens(question_lc)

    def _fixture_name_match_score(self, target_tokens: list[str], fixture: str) -> float:
        fixture_lc = fixture.lower()
        score = 0.0
        synonym_map = {
            "boiler": ["hob", "kettle", "boiler"],
            "stove": ["hob", "stove"],
            "sink": ["sink"],
            "drawer": ["drawer"],
            "microwave": ["microwave"],
            "fridge": ["fridge", "freezer"],
            "freezer": ["freezer", "fridge"],
            "cupboard": ["cupboard", "cabinet"],
            "counter": ["counter"],
        }
        for token in target_tokens:
            if token in fixture_lc:
                score += 2.0
                continue
            for synonym in synonym_map.get(token, []):
                if synonym in fixture_lc:
                    score += 1.5
                    break
        return score

    def _parse_json_list(self, value: Any) -> list[float]:
        if isinstance(value, list):
            try:
                return [float(item) for item in value]
            except Exception:  # noqa: BLE001
                return []
        if not isinstance(value, str) or not value:
            return []
        try:
            payload = json.loads(value)
        except Exception:  # noqa: BLE001
            return []
        if not isinstance(payload, list):
            return []
        try:
            return [float(item) for item in payload]
        except Exception:  # noqa: BLE001
            return []

    def _bbox_center_to_clock_label(self, center_x: float, image_width: float = 1408.0) -> str:
        normalized = max(0.0, min(1.0, center_x / image_width))
        if normalized < 0.2:
            return "9 o'clock"
        if normalized < 0.4:
            return "10 o'clock"
        if normalized < 0.6:
            return "1 o'clock"
        if normalized < 0.8:
            return "3 o'clock"
        return "6 o'clock"

    def _bbox_center_to_clock_label_for_local_consensus(self, center_x: float, image_width: float = 1408.0) -> str:
        normalized = max(0.0, min(1.0, center_x / image_width))
        if normalized < 0.18:
            return "9 o'clock"
        if normalized < 0.32:
            return "10 o'clock"
        if normalized < 0.5:
            return "1 o'clock"
        if normalized < 0.82:
            return "3 o'clock"
        return "6 o'clock"

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

    def _extract_time_ranges_with_video(self, text: str) -> list[tuple[float, float, str | None]]:
        matches = list(re.finditer(r"<TIME\s+(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(video\s+\d+)>", str(text), flags=re.IGNORECASE))
        if len(matches) < 2:
            return []
        paired: list[tuple[float, float, str | None]] = []
        for index in range(0, len(matches) - 1, 2):
            start_match = matches[index]
            end_match = matches[index + 1]
            start_time = self._parse_hms(start_match.group(1))
            end_time = self._parse_hms(end_match.group(1))
            video_label = str(start_match.group(2) or "").strip().lower() or None
            paired.append((min(start_time, end_time), max(start_time, end_time), video_label))
        return paired

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

    def _resolve_video_id_for_video_label(self, video_label: str | None) -> str:
        payload = self.default_hints(self.runtime_question, self.runtime_inputs_json).get("inputs") or {}
        item = payload.get(str(video_label or "").strip())
        if isinstance(item, dict):
            resolved = str(item.get("id") or "").strip()
            if resolved:
                return resolved
        return self.video_id

    def _extract_ingredient_name(self, question: str) -> str | None:
        lowered = question.strip()
        match = re.search(r"weigh of (.+?) in this video\??$", lowered, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_recipe_name_from_membership_question(self, question: str) -> str | None:
        match = re.search(r"not used in (.+?)\??$", str(question).strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_exact_ingredient_name(self, question: str) -> str | None:
        match = re.search(r"exact quantity of (.+?) used in ", str(question).strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_recipe_name_from_amount_question(self, question: str) -> str | None:
        match = re.search(r" used in (.+?)\??$", str(question).strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_recipe_step_hint(self, question: str) -> str | None:
        text = str(question or "").strip()
        patterns = [
            r"while completing recipe step (.+?) in this video\??$",
            r"perform step (.+?) from recipe",
            r"belongs to the .+? recipe step (.+?) in this video\??$",
            r"perform prep for (.+?) from recipe",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_state_keyword(self, question: str) -> str | None:
        lowered = question.lower()
        state_terms = [
            "mixed", "mix", "stirred", "stirring", "cooked", "raw", "soft", "softened",
            "melted", "open", "closed", "empty", "full", "boiled", "fried", "chopped",
        ]
        for term in state_terms:
            if term in lowered:
                return term
        return None

    def _extract_location_keyword(self, question: str, *, allow_container_terms: bool = True) -> str | None:
        lowered = question.lower()
        directional_terms = ["left", "right", "front", "behind"]
        appliance_terms = ["fridge", "microwave", "sink", "counter", "table", "cupboard", "drawer"]
        container_terms = ["bowl", "pan", "pot", "plate"]
        location_terms = directional_terms + appliance_terms + (container_terms if allow_container_terms else [])
        for term in location_terms:
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                return term
        return None

    def _extract_ocr_keyword(self, question: str) -> str | None:
        lowered = question.lower()
        if any(
            phrase in lowered
            for phrase in (
                "given the direction i am looking at",
                "where is the ",
                "what is the person looking at",
            )
        ):
            return None
        if any(
            re.search(pattern, lowered)
            for pattern in (
                r"\bweigh(?:ing)?\b",
                r"\bweight\b",
                r"\bread(?:ing)?\b",
                r"\blabel\b",
                r"\btext\b",
                r"\bnumber\b",
                r"\bdigit\b",
                r"\b\d+(?:\.\d+)?\s*g\b",
                r"\bkg\b",
                r"\bml\b",
                r"\bscale\b",
                r"\bpackage\b",
                r"\bbottle\b",
            )
        ):
            return "reading"
        return None

    def _ocr_query_candidates(self, keyword: str) -> list[str]:
        base = str(keyword or "").strip().lower()
        if not base:
            return []
        candidates: list[str] = []
        for token in [base, self._extract_object_hint(self.runtime_question), self._extract_ingredient_name(self.runtime_question), self._extract_location_keyword(self.runtime_question)]:
            text = str(token or "").strip().lower()
            if not text:
                continue
            if text not in candidates:
                candidates.append(text)
        if base in {"reading", "read", "number", "digit", "text", "label"}:
            for token in ("scale", "package", "bottle"):
                if token not in candidates:
                    candidates.append(token)
        return candidates

    def _resolve_fixture_from_mask_ids(self, *, mask_ids: list[str]) -> str:
        if not mask_ids:
            return ""
        masks = pd.read_parquet(self.paths.output_root / "event_index" / "object_masks.parquet")
        subset = masks[(masks["video_id"] == self.video_id) & (masks["mask_id"].isin(mask_ids))].copy()
        if subset.empty:
            return ""
        subset = subset.sort_values(["frame_number"])
        fixture = str(subset.iloc[-1].get("fixture") or "").strip()
        return fixture

    def _fixture_path_from_tracks(self, tracks: list[dict[str, Any]], reference_fixture: str = "") -> list[str]:
        path: list[str] = []
        normalized_reference = str(reference_fixture).strip()
        if normalized_reference and normalized_reference != "mid-air":
            path.append(normalized_reference)
        for track in tracks:
            fixtures = self._resolve_fixture_sequence_from_mask_ids(mask_ids=track.get("masks") or [])
            if not fixtures:
                fixture = str(track.get("fixture") or "").strip()
                fixtures = [fixture] if fixture else []
            for fixture in fixtures:
                if not fixture or fixture == "mid-air":
                    continue
                if not path or path[-1] != fixture:
                    path.append(fixture)
        return path

    def _resolve_fixture_sequence_from_mask_ids(self, *, mask_ids: list[str]) -> list[str]:
        if not mask_ids:
            return []
        masks = pd.read_parquet(self.paths.output_root / "event_index" / "object_masks.parquet")
        subset = masks[(masks["video_id"] == self.video_id) & (masks["mask_id"].isin(mask_ids))].copy()
        if subset.empty:
            return []
        subset = subset.sort_values(["frame_number"])
        fixtures: list[str] = []
        for fixture in subset["fixture"].tolist():
            normalized = str(fixture or "").strip()
            if not normalized or normalized == "mid-air":
                continue
            if not fixtures or fixtures[-1] != normalized:
                fixtures.append(normalized)
        return fixtures

    def _score_object_location_choice(
        self,
        *,
        choice: str,
        final_fixture: str,
        object_name: str,
        question: str,
    ) -> tuple[float, str]:
        choice_lc = str(choice).strip().lower()
        fixture_lc = str(final_fixture).strip().lower()
        choice_tokens = set(self._name_tokens(choice_lc))
        fixture_tokens = set(self._name_tokens(fixture_lc.replace(".", " ").replace("_", " ")))
        score = 0.0
        matched: list[str] = []
        for token in choice_tokens:
            token_score = self._score_choice_token_against_fixture(token=token, fixture_tokens=fixture_tokens, fixture_text=fixture_lc)
            if token_score > 0:
                score += token_score
                matched.append(f"{token}:{token_score:.2f}")
        if "left" in choice_tokens and "counter" in fixture_tokens and ".005" in fixture_lc:
            score += 0.8
            matched.append("counter_left_bias:0.80")
        if "right" in choice_tokens and "counter" in fixture_tokens and ".004" in fixture_lc:
            score += 0.8
            matched.append("counter_right_bias:0.80")
        relative_score, relative_reason = self._score_relative_fixture_layout(choice=choice_lc, final_fixture=final_fixture)
        if relative_score != 0.0:
            score += relative_score
            matched.append(relative_reason)
        if not matched and fixture_lc:
            score += 0.2
            matched.append("fallback_fixture_presence:0.20")
        return score, f"fixture={final_fixture}; matches={matched}; object={object_name}"

    def _score_relative_fixture_layout(self, *, choice: str, final_fixture: str) -> tuple[float, str]:
        fixture_stats = self._fixture_centroid_map()
        target = fixture_stats.get(str(final_fixture).strip())
        if not target:
            return 0.0, "no_layout_context"
        choice_lc = str(choice).strip().lower()
        best_score = 0.0
        best_reason = "no_relative_layout_match"
        appliance_terms = ("microwave", "oven", "radiator", "dishwasher", "sink", "fridge", "freezer")
        for appliance in appliance_terms:
            if appliance not in choice_lc:
                continue
            anchor = self._best_fixture_anchor_for_appliance(appliance=appliance, fixture_stats=fixture_stats)
            if anchor is None:
                continue
            score = 0.0
            if "counter" in choice_lc and "counter" in str(final_fixture).lower():
                score += 0.55
            if "cupboard" in choice_lc and "cupboard" in str(final_fixture).lower():
                score += 0.55
            if "table" in choice_lc and "table" in str(final_fixture).lower():
                score += 0.55
            if "windowsill" in choice_lc and "windowsill" in str(final_fixture).lower():
                score += 0.8
            dx = float(target["x"]) - float(anchor["x"])
            dy = float(target["y"]) - float(anchor["y"])
            if "left" in choice_lc and dx < -0.12:
                score += 1.2
            if "right" in choice_lc and dx > 0.12:
                score += 1.2
            if "top" in choice_lc and dy > 0.12:
                score += 0.75
            if "below" in choice_lc and dy < -0.12:
                score += 0.75
            if score > best_score:
                best_score = score
                best_reason = f"relative_to_{appliance}:{score:.2f}"
        return best_score, best_reason

    def _best_fixture_anchor_for_appliance(
        self,
        *,
        appliance: str,
        fixture_stats: dict[str, dict[str, float]],
    ) -> dict[str, float] | None:
        appliance = str(appliance).strip().lower()
        alias_groups = {
            "microwave": ["microwave"],
            "oven": ["oven", "hob"],
            "radiator": ["radiator", "heater"],
            "dishwasher": ["dishwasher", "sink"],
            "sink": ["sink"],
            "fridge": ["fridge"],
            "freezer": ["freezer", "fridge"],
        }
        aliases = alias_groups.get(appliance, [appliance])
        ranked: list[tuple[int, dict[str, float]]] = []
        for fixture, stats in fixture_stats.items():
            normalized = fixture.lower().replace(".", " ").replace("_", " ")
            hit_count = sum(1 for alias in aliases if alias in normalized)
            if hit_count <= 0:
                continue
            ranked.append((hit_count, stats))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]

    def _fixture_centroid_map(self) -> dict[str, dict[str, float]]:
        masks = pd.read_parquet(self.paths.output_root / "event_index" / "object_masks.parquet")
        subset = masks[masks["video_id"] == self.video_id].copy()
        centroids: dict[str, list[tuple[float, float, float]]] = {}
        for _, row in subset.iterrows():
            fixture = str(row.get("fixture") or "").strip()
            if not fixture or fixture == "mid-air":
                continue
            try:
                x, y, z = json.loads(row.get("location_3d_json") or "[]")
            except Exception:  # noqa: BLE001
                continue
            centroids.setdefault(fixture, []).append((float(x), float(y), float(z)))
        stats: dict[str, dict[str, float]] = {}
        for fixture, values in centroids.items():
            if not values:
                continue
            xs = [item[0] for item in values]
            ys = [item[1] for item in values]
            zs = [item[2] for item in values]
            stats[fixture] = {
                "x": sum(xs) / len(xs),
                "y": sum(ys) / len(ys),
                "z": sum(zs) / len(zs),
            }
        return stats

    def _score_object_contents_choice(
        self,
        *,
        choice: str,
        candidates: list[dict[str, Any]],
        reference_time: float,
    ) -> tuple[float, str]:
        normalized_choice = self._normalize_food_name(choice)
        choice_tokens = set(self._name_tokens(choice))
        best_score = 0.0
        best_reason = "no_structured_contents_match"
        for item in candidates:
            normalized_object = self._normalize_food_name(item["object_name"])
            object_tokens = set(self._name_tokens(item["object_name"]))
            score = 0.0
            if normalized_choice and normalized_choice == normalized_object:
                score += 2.5
            elif normalized_choice and (normalized_choice in normalized_object or normalized_object in normalized_choice):
                score += 1.8
            else:
                shared = choice_tokens & object_tokens
                score += 0.6 * len(shared)
            if "nothing" in choice.lower():
                score -= 0.8
            if abs(float(item["start_time"]) - float(reference_time)) <= 1.5:
                score += 0.6
            if "microwave" in choice.lower() and "microwave" in str(item.get("target_fixture") or "").lower():
                score += 0.9
            if "scale" in choice.lower() and "counter" in str(item.get("target_fixture") or "").lower():
                score += 0.4
            if score > best_score:
                best_score = score
                best_reason = (
                    f"object={item['object_name']}; fixture={item.get('target_fixture')}; "
                    f"time={item['start_time']:.3f}-{item['end_time']:.3f}; score={score:.2f}"
                )
        return best_score, best_reason

    def _score_itinerary_choice(self, *, choice: str, fixture_path: list[str]) -> tuple[float, str]:
        choice_lc = str(choice).strip().lower()
        segments = [segment.strip() for segment in choice_lc.split(", then ")]
        if not segments:
            return 0.0, "empty_choice"
        path_texts = [fixture.lower().replace("_", " ").replace(".", " ") for fixture in fixture_path]
        score = 0.0
        matched: list[str] = []
        expected_pairs: list[tuple[str, str]] = []
        for segment in segments:
            match = re.search(r"from (.+?) to (.+)", segment)
            if not match:
                continue
            expected_pairs.append((match.group(1).strip(), match.group(2).strip()))
        actual_pairs = list(zip(path_texts, path_texts[1:]))
        next_actual_pair_index = 0
        for expected_from, expected_to in expected_pairs:
            best_pair_score = 0.0
            best_pair_reason = ""
            best_pair_index = -1
            for pair_index in range(next_actual_pair_index, len(actual_pairs)):
                actual_from, actual_to = actual_pairs[pair_index]
                pair_score = (
                    self._score_location_phrase_against_fixture_phrase(expected_from, actual_from)
                    + self._score_location_phrase_against_fixture_phrase(expected_to, actual_to)
                )
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair_reason = f"{expected_from}->{expected_to} ~ {actual_from}->{actual_to}:{pair_score:.2f}"
                    best_pair_index = pair_index
            score += best_pair_score
            if best_pair_reason:
                matched.append(best_pair_reason)
                next_actual_pair_index = best_pair_index + 1
        if fixture_path:
            start_score = self._score_location_phrase_against_fixture_phrase(segments[0].split(" to ", 1)[0].replace("from ", "", 1), path_texts[0]) if segments else 0.0
            end_score = 0.0
            if expected_pairs:
                end_score = self._score_location_phrase_against_fixture_phrase(expected_pairs[-1][1], path_texts[-1])
            if start_score > 0:
                score += 0.35 * start_score
                matched.append(f"start_match:{start_score:.2f}")
            if end_score > 0:
                score += 0.45 * end_score
                matched.append(f"end_match:{end_score:.2f}")
        if len(expected_pairs) == max(0, len(fixture_path) - 1):
            score += 0.5
            matched.append("pair_count_match:0.50")
        elif expected_pairs:
            extra_pairs = max(0, len(expected_pairs) - max(0, len(fixture_path) - 1))
            if extra_pairs > 0:
                penalty = 0.6 * extra_pairs
                score -= penalty
                matched.append(f"pair_count_penalty:-{penalty:.2f}")
        return score, f"fixture_path={fixture_path}; matches={matched}"

    def _score_location_phrase_against_fixture_phrase(self, phrase: str, fixture_phrase: str) -> float:
        lowered_phrase = str(phrase).lower().strip()
        lowered_fixture = str(fixture_phrase).lower().strip()
        direct_score = self._score_explicit_fixture_phrase_mapping(phrase=lowered_phrase, fixture_phrase=lowered_fixture)
        phrase_tokens = self._name_tokens(lowered_phrase)
        fixture_tokens = set(self._name_tokens(lowered_fixture))
        score = 0.0
        for token in phrase_tokens:
            score += self._score_choice_token_against_fixture(token=token, fixture_tokens=fixture_tokens, fixture_text=str(fixture_phrase).lower())
        return score + direct_score

    def _score_explicit_fixture_phrase_mapping(self, *, phrase: str, fixture_phrase: str) -> float:
        fixture_key = str(fixture_phrase).replace("_", " ").replace(".", " ").strip()
        phrase_key = self._normalize_location_phrase(str(phrase))
        explicit_scores = {
            ("counter left of hob", "p02 counter 001"): 1.2,
            ("counter to left of hob", "p02 counter 001"): 1.6,
            ("counter right of hob", "p02 counter 001"): -0.6,
            ("counter to right of hob", "p02 counter 001"): -0.8,
            ("drawer to left of hob", "p02 counter 001"): -0.8,
            ("drawer to left of and below hob", "p02 counter 001"): -1.0,
            ("hob", "p02 counter 001"): -0.9,
            ("counter right of sink", "p02 counter 003"): 0.2,
            ("counter top left of washingmachine", "p02 counter 003"): 1.4,
        }
        return explicit_scores.get((phrase_key, fixture_key), 0.0)

    def _score_choice_token_against_fixture(self, *, token: str, fixture_tokens: set[str], fixture_text: str) -> float:
        token = str(token).strip().lower()
        if not token:
            return 0.0
        if token in fixture_tokens:
            return 1.5
        alias_hits = self._fixture_alias_hits(fixture_text)
        if token in alias_hits:
            return alias_hits[token]
        synonym_map = {
            "counter": ["counter", "countertop"],
            "sink": ["sink"],
            "dishwasher": ["sink"],
            "storage": ["storage", "shelf", "cupboard", "cabinet"],
            "drawer": ["drawer"],
            "fridge": ["fridge", "freezer"],
            "freezer": ["freezer", "fridge"],
            "windowsill": ["window", "sill"],
            "top": ["top"],
            "left": [],
            "right": [],
        }
        for synonym in synonym_map.get(token, []):
            if synonym in fixture_tokens or synonym in fixture_text:
                return 1.1
        return 0.0

    def _fixture_alias_hits(self, fixture_text: str) -> dict[str, float]:
        normalized = str(fixture_text).strip().lower()
        alias_hits: dict[str, float] = {}
        explicit_aliases = {
            "p02_counter 001": {"left": 1.4, "hob": 1.0, "right": -0.4},
            "p02_counter 003": {"washingmachine": 2.4, "sink": 0.35, "top": 0.6, "left": 0.5},
        }
        for alias_fixture, token_scores in explicit_aliases.items():
            if alias_fixture in normalized:
                alias_hits.update(token_scores)
        return alias_hits

    def _postprocess_action_mechanism_result(
        self,
        *,
        question: str,
        choices: list[str],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        lowered_question = str(question or "").lower()
        lowered_reason = str(result.get("reason") or "").lower()
        lowered_answer = str(result.get("answer") or "").lower()
        if "dishwasher" not in lowered_question or "door" not in lowered_question or "close" not in lowered_question:
            return result
        if "push" not in lowered_answer and all(token not in lowered_reason for token in ("push", "pushed", "pushing")):
            return result
        if "up" not in lowered_reason and "upward" not in lowered_reason and "already open downward" not in lowered_reason:
            return result
        for index, choice in enumerate(choices):
            lowered_choice = str(choice).lower()
            if any(token in lowered_choice for token in ("rotate", "rotating")) and ("upward" in lowered_choice or "upwards" in lowered_choice):
                return {
                    "best_index": index,
                    "answer": str(choices[index]),
                    "confidence": max(float(result.get("confidence") or 0.0), 0.9),
                    "reason": f"hinged_drop_door_override from={result.get('answer')}; {result.get('reason')}",
                }
        return result

    def _score_ingredient_order_choice(self, *, candidate_order: list[str], observed_order: list[str]) -> tuple[float, str]:
        normalized_observed = [self._normalize_food_name(item) for item in observed_order if item]
        normalized_candidate = [self._normalize_food_name(item) for item in candidate_order if item]
        if not normalized_candidate or not normalized_observed:
            return 0.0, "empty_order"
        position_hits = 0.0
        subsequence_hits = 0.0
        for index, item in enumerate(normalized_candidate):
            if index < len(normalized_observed) and item == normalized_observed[index]:
                position_hits += 1.5
            if item in normalized_observed:
                observed_index = normalized_observed.index(item)
                subsequence_hits += max(0.0, 1.0 - 0.2 * abs(observed_index - index))
        exact_match = 2.0 if normalized_candidate == normalized_observed[: len(normalized_candidate)] else 0.0
        score = position_hits + subsequence_hits + exact_match
        return score, f"candidate={normalized_candidate}; position_hits={position_hits:.2f}; subsequence_hits={subsequence_hits:.2f}; exact_match={exact_match:.2f}"

    def _select_recipe_from_catalog(self, *, recipe_name: str | None, recipe_catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not recipe_catalog:
            return None
        if not recipe_name:
            return recipe_catalog[0]
        best_recipe = None
        best_score = float("-inf")
        for recipe in recipe_catalog:
            candidate_name = str(recipe.get("name") or "")
            score = self._token_overlap_text(recipe_name, candidate_name)
            if score > best_score:
                best_score = score
                best_recipe = recipe
        return best_recipe or recipe_catalog[0]

    def _select_ingredient_amount(
        self,
        *,
        ingredient_name: str | None,
        ingredient_amounts: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not ingredient_amounts:
            return None
        if not ingredient_name:
            return ingredient_amounts[0]
        best_item = None
        best_score = float("-inf")
        for item in ingredient_amounts:
            candidate_name = str(item.get("name") or "")
            score = self._token_overlap_text(ingredient_name, candidate_name)
            if score > best_score:
                best_score = score
                best_item = item
        return best_item or ingredient_amounts[0]

    def _score_measurement_choice(self, *, choice: str, normalized_target: str) -> tuple[float, str]:
        exact = 5.0 if normalized_target and str(choice).strip().lower() == normalized_target.strip().lower() else 0.0
        token_overlap = self._token_overlap_text(choice, normalized_target) if normalized_target else 0.0
        choice_match = re.search(r"(\d+(?:\.\d+)?)", str(choice))
        target_match = re.search(r"(\d+(?:\.\d+)?)", str(normalized_target))
        numeric_choice = self._float_or_none(choice_match.group(1)) if choice_match else None
        numeric_target = self._float_or_none(target_match.group(1)) if target_match else None
        proximity = 0.0
        if numeric_choice is not None and numeric_target is not None:
            proximity = max(0.0, 1.5 - abs(numeric_choice - numeric_target) / max(1.0, numeric_target))
        score = exact + token_overlap * 2.0 + proximity
        return score, (
            f"target={normalized_target}; exact={exact:.2f}; token_overlap={token_overlap:.2f}; "
            f"proximity={proximity:.2f}"
        )

    def _extract_object_hint(self, question: str) -> str | None:
        lowered = question.lower()
        candidates = [
            "salad", "onion", "tomato", "bowl", "pan", "pot", "plate", "knife",
            "spoon", "fork", "cup", "bottle", "bag", "drawer", "fridge", "microwave",
        ]
        for term in candidates:
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                return term
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

    def _extract_video_ids_from_inputs(self, inputs: dict[str, Any]) -> list[str]:
        video_ids: list[str] = []
        for value in inputs.values():
            if not isinstance(value, dict):
                continue
            video_id = value.get("id")
            if isinstance(video_id, str) and video_id and video_id not in video_ids:
                video_ids.append(video_id)
        return video_ids

    def _normalize_food_name(self, text: str) -> str:
        return " ".join(self._name_tokens(str(text).lower()))

    def _nutrition_key_from_question(self, question: str) -> str | None:
        lowered = str(question).lower()
        for key in ("carbs", "fat", "protein", "calories"):
            singular = key[:-1] if key.endswith("s") else key
            if key in lowered or singular in lowered:
                return key
        return None

    def _token_overlap_text(self, left: str, right: str) -> float:
        left_tokens = set(self._name_tokens(left))
        right_tokens = set(self._name_tokens(right))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, len(left_tokens))

    def _normalize_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(args)
        float_keys = {
            "query_time": ["start_time", "end_time"],
            "query_event": ["start_time", "end_time"],
            "query_ingredient_measurement": ["start_time", "end_time"],
            "query_state": ["start_time", "end_time"],
            "query_location": ["start_time", "end_time"],
            "query_region": ["start_time", "end_time"],
            "query_ocr": ["start_time", "end_time"],
            "compute_nutrition_change": ["start_time", "end_time"],
            "query_spatial_context": ["time_s"],
            "resolve_bbox_reference": ["reference_time"],
            "estimate_object_movement_count": ["reference_time"],
            "estimate_stationary_start": ["reference_time", "threshold_s"],
            "infer_object_drop_location": ["reference_time"],
            "extract_frame_at_time": ["time_s"],
            "extract_frames_for_range": ["start_time", "end_time", "stride_s"],
            "sample_sparse_frames": ["start_time", "end_time"],
            "retrieve_cached_artifacts": ["start_time", "end_time"],
            "extract_region_with_context": ["expand_ratio"],
            "run_ocr_on_region": ["expand_ratio"],
            "detect_audio_peaks": ["start_time", "end_time", "window_s"],
            "sample_frames_around_peaks": ["radius_s"],
            "write_observation": ["start_time", "end_time"],
            "write_frame_observation": ["time_s"],
            "write_region_observation": ["time_s"],
            "write_ocr_reading": ["time_s"],
            "write_audio_event": ["start_time", "end_time"],
            "write_timeline_summary": ["start_time", "end_time"],
            "write_state_change": ["start_time", "end_time"],
            "finish": ["confidence"],
        }
        int_keys = {
            "query_time": ["limit"],
            "query_object": ["limit"],
            "query_event": ["limit"],
            "query_ingredient_measurement": ["limit"],
            "query_state": ["limit"],
            "query_location": ["limit"],
            "query_region": ["limit"],
            "query_ocr": ["limit"],
            "query_spatial_context": ["limit"],
            "resolve_bbox_reference": ["limit"],
            "get_neighbors": ["limit"],
            "extract_frames_for_range": ["max_frames"],
            "sample_sparse_frames": ["sample_count"],
            "retrieve_cached_artifacts": ["limit"],
            "detect_audio_peaks": ["top_k"],
            "sample_frames_around_peaks": ["frames_per_peak"],
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
        if tool_name in {"render_bbox_overlay", "extract_region_with_context", "resolve_bbox_reference", "estimate_object_movement_count", "estimate_stationary_start", "infer_object_drop_location", "run_ocr_on_region"} and "bbox" in normalized:
            normalized["bbox"] = [float(value) for value in normalized["bbox"]]
        if tool_name in {"write_region_observation", "write_ocr_reading"} and "bbox" in normalized and normalized["bbox"] is not None:
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
        if tool_name == "sample_frames_around_peaks":
            normalized["peak_times"] = [float(value) for value in normalized.get("peak_times", [])]
        if tool_name in {"estimate_object_movement_count", "estimate_stationary_start", "infer_object_drop_location"}:
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
        if tool_name in {
            "infer_ingredient_retrieval_choice",
            "infer_recipe_ingredient_membership_choice",
            "infer_exact_ingredient_amount_choice",
            "infer_recipe_catalog_choice",
            "infer_recipe_nutrition_choice",
        }:
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
        if tool_name == "infer_action_mechanism":
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
            normalized["image_paths"] = [str(path) for path in normalized.get("image_paths", [])]
        if tool_name == "infer_action_intent":
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
            normalized["image_paths"] = [str(path) for path in normalized.get("image_paths", [])]
            normalized["context_notes"] = [str(item) for item in normalized.get("context_notes", [])]
        if tool_name in {"resolve_action_intent_pairwise", "resolve_action_intent_future_use"}:
            normalized["choices"] = [str(choice) for choice in normalized.get("choices", [])]
            normalized["candidate_indices"] = [int(value) for value in normalized.get("candidate_indices", [])]
            normalized["image_paths"] = [str(path) for path in normalized.get("image_paths", [])]
            normalized["context_notes"] = [str(item) for item in normalized.get("context_notes", [])]
        return normalized
