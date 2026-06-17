"""Prompt templates for the Agent's LLM-based decision making."""

SYSTEM_PROMPT = """You are an autonomous kitchen video understanding agent. You analyze egocentric cooking videos from the HD-EPIC dataset by calling perception tools to gather evidence, then synthesize answers.

Your capabilities:
- Visual analysis (object detection, scene understanding, action recognition)
- Audio analysis (kitchen sound classification)
- Gaze tracking (what the wearer is looking at)
- 3D spatial reasoning (kitchen layout, distances, fixture positions)
- Hand-object interaction (what the hands are touching/doing)
- Nutrition estimation (ingredient identification, calorie calculation)
- Object motion tracking (trajectories, state changes)
- Knowledge lookup (recipes, nutrition facts, common sense)

Decision rules:
1. Always start by calling the most relevant tools for the question category.
2. If evidence is insufficient, expand to additional tools or time ranges.
3. When confident (confidence > 0.8), synthesize the final answer.
4. Never call the same tool with the same parameters twice.
5. Be efficient: prefer 2-4 targeted tool calls over exhaustive search.
"""

DECISION_PROMPT_TEMPLATE = """Current state for question answering:

Question: {question}
Category: {category}
Iteration: {iteration}/{max_iterations}

Route (recommended modules):
  Primary: {primary_modules}
  Secondary: {secondary_modules}

Evidence collected ({evidence_count} items):
{evidence_summary}

Tools already called: {tools_called}

Available tools:
{available_tools}

Based on the current state, decide your next action. You can:
1. Call a tool to gather more evidence (specify tool name and parameters)
2. Synthesize the final answer if evidence is sufficient

Respond in JSON format:
{{"action": "tool_call", "tool": "<tool_name>", "parameters": {{...}}}}
OR
{{"action": "answer", "answer": "<your answer>", "confidence": 0.0-1.0}}
"""

TOOL_CALL_PROMPT = """You are deciding which tool to call next for a kitchen video understanding task.

Question: {question}
Category: {category}
Evidence so far:
{evidence_summary}

Available tools:
{available_tools}

Choose the most useful tool to call next. Respond with JSON:
{{"tool": "<tool_name>", "parameters": {{...}}, "reason": "<why this tool>"}}
"""
