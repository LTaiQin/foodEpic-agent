"""Tool registry: define and manage all tools available to the Agent."""

from typing import Any, Callable, Dict, List, Optional


# Tool JSON Schema definitions for LLM function calling
TOOL_SCHEMAS = {
    # Perception tools
    "query_audio": {
        "description": "Query audio events in a time range. Returns sound classification results.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {"type": "number", "description": "Start time in seconds"},
                "end_time": {"type": "number", "description": "End time in seconds"},
            },
            "required": ["start_time", "end_time"],
        },
    },
    "query_video": {
        "description": "Query video frames for visual analysis. Returns detected objects with masks and scene description.",
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number", "description": "Timestamp in seconds"},
                "text_prompt": {"type": "string", "description": "What to detect (e.g. 'food ingredient', 'knife, plate')"},
                "use_scene_graph": {"type": "boolean", "description": "Also generate scene graph via LLM"},
            },
            "required": ["timestamp"],
        },
    },
    "segment_objects": {
        "description": "Segment objects in a video frame using SAM3 open-vocabulary segmentation. Returns pixel-level masks.",
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number", "description": "Timestamp in seconds"},
                "text_prompt": {"type": "string", "description": "What to segment (e.g. 'food ingredient', 'kitchen object')"},
            },
            "required": ["timestamp"],
        },
    },
    "describe_frame": {
        "description": "Describe what's visible in a video frame using vision AI. Useful for open-ended questions about the scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number", "description": "Timestamp in seconds"},
                "question": {"type": "string", "description": "Specific question about the frame"},
            },
            "required": ["timestamp"],
        },
    },
    "identify_ingredients": {
        "description": "Identify all food ingredients visible in a video frame. Returns ingredient names, locations, and states.",
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number", "description": "Timestamp in seconds"},
            },
            "required": ["timestamp"],
        },
    },
    "query_gaze": {
        "description": "Query gaze/eye-tracking data. Returns fixation targets and attention info.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {"type": "number"},
                "end_time": {"type": "number"},
            },
            "required": ["start_time", "end_time"],
        },
    },
    "query_3d": {
        "description": "Query 3D spatial information (kitchen layout, wearer pose, fixture positions).",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {"type": "string", "enum": ["layout", "wearer_pose", "nearest"]},
                "timestamp": {"type": "number"},
            },
            "required": ["query_type"],
        },
    },
    "query_hands": {
        "description": "Query hand-object interactions. Returns what the hands are doing.",
        "parameters": {
            "type": "object",
            "properties": {
                "frame_number": {"type": "integer", "description": "Frame index"},
            },
            "required": ["frame_number"],
        },
    },
    "query_nutrition": {
        "description": "Query nutritional information for ingredients.",
        "parameters": {
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"name": {"type": "string"}, "amount_g": {"type": "number"}}},
                },
            },
            "required": ["ingredients"],
        },
    },
    "query_motion": {
        "description": "Query object motion/trajectory data.",
        "parameters": {
            "type": "object",
            "properties": {
                "frame_number": {"type": "integer"},
                "video_id": {"type": "string"},
            },
            "required": ["frame_number"],
        },
    },
    # Knowledge tools
    "query_recipe": {
        "description": "Query recipe knowledge base for cooking steps and procedures.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string"},
                "step_number": {"type": "integer"},
            },
        },
    },
    "query_nutrition_kb": {
        "description": "Look up nutrition facts for a specific ingredient.",
        "parameters": {
            "type": "object",
            "properties": {
                "ingredient": {"type": "string"},
            },
            "required": ["ingredient"],
        },
    },
    "query_scene_graph": {
        "description": "Query the scene graph for objects and their relationships.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_type": {"type": "string"},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"},
            },
        },
    },
    # Control tools
    "check_evidence": {
        "description": "Check if current evidence is sufficient to answer the question.",
        "parameters": {"type": "object", "properties": {}},
    },
    "expand_search": {
        "description": "Expand the search to additional modules or time range.",
        "parameters": {
            "type": "object",
            "properties": {
                "modules": {"type": "array", "items": {"type": "string"}},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"},
            },
        },
    },
    "synthesize_answer": {
        "description": "Synthesize all collected evidence into a final answer.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class ToolRegistry:
    """Registry for all tools available to the Agent.

    Each tool has a schema (for LLM) and an implementation function.
    """

    def __init__(self):
        self._schemas: Dict[str, Dict] = dict(TOOL_SCHEMAS)
        self._implementations: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable, schema: Optional[Dict] = None) -> None:
        """Register a tool with its implementation and optional schema override."""
        self._implementations[name] = func
        if schema:
            self._schemas[name] = schema

    def get_schema(self, name: str) -> Optional[Dict]:
        """Get the JSON schema for a tool."""
        return self._schemas.get(name)

    def get_all_schemas(self) -> Dict[str, Dict]:
        """Get all tool schemas (for LLM prompt)."""
        return dict(self._schemas)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._schemas.keys())

    def call_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name.

        Args:
            name: Tool name.
            **kwargs: Tool parameters.

        Returns:
            Tool result (typically Evidence or List[Evidence]).
        """
        if name not in self._implementations:
            return {"error": f"Tool '{name}' not implemented"}

        try:
            return self._implementations[name](**kwargs)
        except Exception as e:
            return {"error": f"Tool '{name}' failed: {e}"}

    def get_tools_for_prompt(self, max_tools: int = 15) -> str:
        """Format tool schemas as a text prompt for the LLM."""
        lines = []
        for i, (name, schema) in enumerate(list(self._schemas.items())[:max_tools]):
            lines.append(f"{i+1}. {name}: {schema.get('description', '')}")
            params = schema.get("parameters", {}).get("properties", {})
            if params:
                param_str = ", ".join(f"{k}({v.get('type', '?')})" for k, v in params.items())
                lines.append(f"   Parameters: {param_str}")
        return "\n".join(lines)
