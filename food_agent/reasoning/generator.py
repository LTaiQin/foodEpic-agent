"""Answer generator: synthesize evidence into final answers."""

from typing import Dict, List, Optional

from food_agent.perception.evidence import Evidence


# Prompt templates
ANSWER_GENERATION_PROMPT = """You are a kitchen video understanding assistant. Based on the evidence collected from multiple perception modules, answer the question.

Question: {question}

Evidence Summary:
{evidence_summary}

Category: {category}

Instructions:
- Use the evidence to provide a concise, accurate answer.
- If the evidence supports multiple-choice options, select the best one.
- If evidence is insufficient, say what you can determine and acknowledge uncertainty.
- Be specific and cite evidence sources when possible.

Answer:"""


class Generator:
    """Generate final answers from collected evidence."""

    def __init__(self, mimo_client=None):
        self._mimo = mimo_client

    def format_evidence_prompt(
        self,
        evidence_list: List[Evidence],
        question: str,
        category: str = "general",
        choices: Optional[List[str]] = None,
    ) -> str:
        """Format evidence into a prompt for the LLM.

        Args:
            evidence_list: List of collected Evidence objects.
            question: The original question.
            category: Question category from Router.
            choices: Multiple choice options (if applicable).

        Returns:
            Formatted prompt string.
        """
        # Build evidence summary
        lines = []
        for i, ev in enumerate(evidence_list):
            lines.append(
                f"  [{i+1}] {ev.source_module} ({ev.evidence_type}): "
                f"confidence={ev.confidence:.2f}"
            )
            for k, v in ev.content.items():
                if isinstance(v, (str, int, float, bool)):
                    lines.append(f"      {k}: {v}")
                elif isinstance(v, list) and len(v) <= 5:
                    lines.append(f"      {k}: {v}")

        evidence_summary = "\n".join(lines) if lines else "  No evidence collected."

        prompt = ANSWER_GENERATION_PROMPT.format(
            question=question,
            evidence_summary=evidence_summary,
            category=category,
        )

        if choices:
            prompt += "\n\nOptions:\n"
            for i, choice in enumerate(choices):
                prompt += f"  {chr(65+i)}. {choice}\n"
            prompt += "\nSelect the best option (A, B, C, or D)."

        return prompt

    def generate_answer(
        self,
        evidence_list: List[Evidence],
        question: str,
        category: str = "general",
        choices: Optional[List[str]] = None,
    ) -> Dict:
        """Generate an answer from evidence.

        Returns:
            Dict with 'answer', 'confidence', 'reasoning'.
        """
        prompt = self.format_evidence_prompt(
            evidence_list, question, category, choices
        )

        if self._mimo is None:
            return {
                "answer": "Unable to generate answer (no LLM client configured).",
                "confidence": 0.0,
                "reasoning": "No LLM client available.",
            }

        try:
            response = self._mimo.call_text(prompt)
            parsed = self.parse_answer(response, choices)
            return {
                "answer": parsed["answer"],
                "confidence": parsed.get("confidence", 0.5),
                "reasoning": response[:500],
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {e}",
                "confidence": 0.0,
                "reasoning": str(e),
            }

    def parse_answer(
        self,
        response: str,
        choices: Optional[List[str]] = None,
    ) -> Dict:
        """Parse the LLM response into a structured answer.

        Args:
            response: Raw LLM response string.
            choices: Multiple choice options (if applicable).

        Returns:
            Dict with 'answer' and 'confidence'.
        """
        response = response.strip()

        if choices:
            # Try to extract a choice letter
            for letter in "ABCDEFGH"[:len(choices)]:
                if response.upper().startswith(letter) or f" {letter}." in response or f" {letter})" in response:
                    idx = ord(letter) - ord("A")
                    if idx < len(choices):
                        return {"answer": choices[idx], "confidence": 0.7}

            # Try to match choice text
            for choice in choices:
                if choice.lower() in response.lower():
                    return {"answer": choice, "confidence": 0.6}

        return {"answer": response, "confidence": 0.5}
