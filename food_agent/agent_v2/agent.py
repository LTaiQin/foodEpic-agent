"""MultimodalAgent: the core autonomous tool-calling loop."""

import json
import time
from typing import Dict, List, Optional

from food_agent.perception.evidence import Evidence
from food_agent.reasoning.router import Router
from food_agent.reasoning.aggregator import Aggregator
from food_agent.reasoning.judge import Judge
from food_agent.reasoning.generator import Generator
from food_agent.reasoning.tool_registry import ToolRegistry
from .agent_state import AgentState
from .reasoning_trace import ReasoningTrace, StepRecord
from .prompts import SYSTEM_PROMPT, DECISION_PROMPT_TEMPLATE


class MultimodalAgent:
    """Autonomous agent that uses tool-calling to answer kitchen video questions.

    Core loop:
        1. Router classifies the question
        2. Loop (max N iterations):
            a. Build decision prompt from current state
            b. LLM decides next tool call OR final answer
            c. Execute tool, add evidence to state
            d. Judge evaluates sufficiency
        3. Generator produces final answer
    """

    def __init__(
        self,
        mimo_client=None,
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = 10,
        max_tool_calls: int = 20,
        timeout: float = 120.0,
    ):
        self._mimo = mimo_client
        self._router = Router()
        self._aggregator = Aggregator()
        self._judge = Judge(max_iterations=max_iterations)
        self._generator = Generator(mimo_client)
        self._tools = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.timeout = timeout

    def run(
        self,
        question: str,
        video_id: str = "",
        participant_id: str = "",
        choices: Optional[List[str]] = None,
    ) -> Dict:
        """Run the autonomous agent loop to answer a question.

        Args:
            question: The question to answer.
            video_id: HD-EPIC video ID.
            participant_id: Participant ID (e.g., "P01").
            choices: Multiple choice options (if applicable).

        Returns:
            Dict with 'answer', 'confidence', 'evidence_chain', 'reasoning_trace'.
        """
        # Step 1: Initialize state
        state = AgentState(
            question=question,
            video_id=video_id,
            participant_id=participant_id,
            choices=choices,
        )
        trace = ReasoningTrace(question=question)

        # Step 2: Route
        route = self._router.route(question)
        state.route = route
        state.category = route["category"]
        trace.category = route["category"]

        self._aggregator.clear()
        self._aggregator.set_priority(route["primary"], route["secondary"])

        # Step 3: Agent loop
        while state.iteration < self.max_iterations:
            if state.elapsed_time > self.timeout:
                break

            state.increment_iteration()

            # 3a: Build decision prompt
            prompt = self._build_decision_prompt(state)

            # 3b: LLM decides
            decision = self._get_llm_decision(prompt, state)

            if decision.get("action") == "answer" or "answer" in decision:
                # LLM wants to finalize
                answer = decision.get("answer", "")
                confidence = decision.get("confidence", 0.5)
                trace.add_step(StepRecord(
                    iteration=state.iteration,
                    action="answer_generation",
                    decision=f"LLM chose to answer: {answer[:100]}",
                    confidence_after=confidence,
                ))
                break

            # 3c: Execute tool call
            tool_name = decision.get("tool", "")
            tool_params = decision.get("parameters", {})

            if not tool_name:
                # No tool selected, try judge suggestion
                suggestion = self._judge.suggest_expansion(
                    state.evidence_list, question, route
                )
                if suggestion["modules_to_call"]:
                    tool_name = self._module_to_tool(suggestion["modules_to_call"][0])
                    tool_params = suggestion.get("parameters", {})

            if tool_name:
                # Check for duplicate calls
                call_key = f"{tool_name}:{json.dumps(tool_params, sort_keys=True)}"
                if any(
                    f"{c['tool']}:{json.dumps(c['parameters'], sort_keys=True)}" == call_key
                    for c in state.tool_call_history
                ):
                    continue

                # Execute tool
                result = self._execute_tool(tool_name, tool_params, state)
                state.add_tool_call(tool_name, tool_params, result)

                if isinstance(result, Evidence):
                    state.add_evidence(result)
                    self._aggregator.add_evidence(result)
                elif isinstance(result, list):
                    for ev in result:
                        if isinstance(ev, Evidence):
                            state.add_evidence(ev)
                            self._aggregator.add_evidence(ev)

                # Record confidence
                current_conf = self._aggregator.get_confidence()
                state.record_confidence(current_conf)

                trace.add_step(StepRecord(
                    iteration=state.iteration,
                    action="tool_call",
                    tool_name=tool_name,
                    tool_params=tool_params,
                    result_summary=str(result)[:200] if result else "empty",
                    confidence_after=current_conf,
                    decision=decision.get("reason", ""),
                ))

            # 3d: Judge evaluates sufficiency
            if self._judge.should_stop(state.evidence_list, state.iteration, question, route):
                trace.add_step(StepRecord(
                    iteration=state.iteration,
                    action="evaluation",
                    decision="Judge: evidence sufficient, stopping",
                    confidence_after=self._aggregator.get_confidence(),
                ))
                break

        # Step 4: Generate final answer
        gen_result = self._generator.generate_answer(
            state.evidence_list, question, state.category, choices
        )

        # Step 5: Finalize
        trace.finalize(gen_result["answer"], gen_result["confidence"])

        return {
            "answer": gen_result["answer"],
            "confidence": gen_result["confidence"],
            "category": state.category,
            "evidence_chain": [ev.to_dict() for ev in state.evidence_list],
            "reasoning_trace": trace.to_dict(),
            "tool_calls": state.tool_call_history,
            "iterations": state.iteration,
        }

    def _build_decision_prompt(self, state: AgentState) -> str:
        """Build the decision prompt from current state."""
        # Include choices in the question if available
        question = state.question
        if state.choices:
            choice_text = " ".join(f"{chr(65+i)}.{c}" for i, c in enumerate(state.choices))
            question = f"{question}\n\nAnswer choices: {choice_text}"

        return DECISION_PROMPT_TEMPLATE.format(
            question=question,
            category=state.category,
            iteration=state.iteration,
            max_iterations=self.max_iterations,
            primary_modules=", ".join(state.route.get("primary", [])),
            secondary_modules=", ".join(state.route.get("secondary", [])),
            evidence_count=len(state.evidence_list),
            evidence_summary=self._aggregator.get_summary() if state.evidence_list else "  None",
            tools_called=", ".join(state.unique_tools_called) or "None",
            available_tools=self._tools.get_tools_for_prompt(),
        )

    def _get_llm_decision(self, prompt: str, state: AgentState) -> Dict:
        """Get the LLM's decision on next action."""
        if self._mimo is None:
            # Fallback: use judge to suggest next tool
            suggestion = self._judge.suggest_expansion(
                state.evidence_list, state.question, state.route
            )
            if suggestion["modules_to_call"]:
                tool = self._module_to_tool(suggestion["modules_to_call"][0])
                return {"action": "tool_call", "tool": tool, "parameters": suggestion.get("parameters", {})}
            return {"action": "answer", "answer": "Insufficient evidence.", "confidence": 0.3}

        try:
            response = self._mimo.call_text(SYSTEM_PROMPT + "\n\n" + prompt)
            # Parse JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except Exception:
            pass

        # Fallback
        suggestion = self._judge.suggest_expansion(
            state.evidence_list, state.question, state.route
        )
        if suggestion["modules_to_call"]:
            tool = self._module_to_tool(suggestion["modules_to_call"][0])
            return {"action": "tool_call", "tool": tool, "parameters": suggestion.get("parameters", {})}
        return {"action": "answer", "answer": "Unable to determine.", "confidence": 0.2}

    def _execute_tool(self, tool_name: str, params: Dict, state: AgentState) -> any:
        """Execute a tool and return its result."""
        # Inject video/participant context
        params.setdefault("video_id", state.video_id)
        params.setdefault("participant_id", state.participant_id)

        # Convert module names to tool names if needed
        result = self._tools.call_tool(tool_name, **params)

        if isinstance(result, dict) and "error" in result:
            return None
        return result

    @staticmethod
    def _module_to_tool(module_name: str) -> str:
        """Convert a perception module name to its corresponding tool name."""
        mapping = {
            "AudioAnalyzer": "query_audio",
            "VisualAnalyzer": "query_video",
            "GazeTracker": "query_gaze",
            "SpatialReasoner": "query_3d",
            "HandInteractor": "query_hands",
            "NutritionEstimator": "query_nutrition",
            "MotionTracker": "query_motion",
            "RecipeKB": "query_recipe",
            "NutritionKB": "query_nutrition_kb",
            "SceneGraphKB": "query_scene_graph",
        }
        return mapping.get(module_name, module_name.lower())
