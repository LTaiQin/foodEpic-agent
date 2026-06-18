"""Answer generator: synthesize evidence into final answers."""

import re
from typing import Dict, List, Optional

from food_agent.perception.evidence import Evidence


# Prompt templates
ANSWER_GENERATION_PROMPT = """You are a kitchen video understanding assistant. Based on the evidence collected, answer the question.

Question: {question}

Evidence:
{evidence_summary}

{choices_block}
CRITICAL REASONING RULES:
1. For "which ingredient is NOT used" questions:
   - Look at the recipe steps in the evidence
   - Check which ingredients are mentioned in the recipe
   - The answer is the ingredient that is NOT mentioned
   - IMPORTANT: Consider ingredient types - "stilton" is a type of "blue cheese", "cheddar" is a type of "cheese"
   - If the recipe says "blue cheese", then "stilton" IS used (because stilton is a blue cheese)
   - If the recipe says "cheese", then "cheddar" IS used (because cheddar is a cheese)
   - The answer should be the ingredient that has NO connection to any ingredient in the recipe
2. For spatial questions:
   - Use the spatial data from query_3d
   - Convert directions to clock positions if needed
3. For action questions:
   - Use hand interaction and audio data
   - The answer should match what the person is actually doing

{instruction}
- Be concise and precise.

Answer:"""

CHOICES_BLOCK_MC = """Options:
{choices}

CRITICAL: Reply with ONLY the single letter of the best option (A, B, C, D, or E).
Do NOT explain. Do NOT write the full text. Just the letter.
Example: B
"""

CHOICES_BLOCK_FREE = """Instructions:
- Provide a concise, specific answer based on the evidence.
- If uncertain, state what you can determine.
"""


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
        """Format evidence into a prompt for the LLM."""
        # Build evidence summary - include key content
        lines = []
        for i, ev in enumerate(evidence_list):
            lines.append(f"[{i+1}] {ev.source_module} ({ev.evidence_type}), conf={ev.confidence:.2f}")
            for k, v in ev.content.items():
                if isinstance(v, str) and len(v) < 200:
                    lines.append(f"    {k}: {v}")
                elif isinstance(v, (int, float, bool)):
                    lines.append(f"    {k}: {v}")
                elif isinstance(v, list) and len(v) <= 8:
                    for item in v:
                        if isinstance(item, dict):
                            summary = {kk: vv for kk, vv in item.items() if isinstance(vv, (str, int, float))}
                            lines.append(f"    - {summary}")
                        else:
                            lines.append(f"    - {item}")

        evidence_summary = "\n".join(lines) if lines else "  No evidence collected."

        if choices:
            choices_block = CHOICES_BLOCK_MC.format(
                choices="\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
            )
            instruction = "Select the best option by its letter."
        else:
            choices_block = CHOICES_BLOCK_FREE
            instruction = "Provide a direct, concise answer."

        return ANSWER_GENERATION_PROMPT.format(
            question=question,
            evidence_summary=evidence_summary,
            choices_block=choices_block,
            instruction=instruction,
        )

    def generate_answer(
        self,
        evidence_list: List[Evidence],
        question: str,
        category: str = "general",
        choices: Optional[List[str]] = None,
    ) -> Dict:
        """Generate an answer from evidence."""
        prompt = self.format_evidence_prompt(evidence_list, question, category, choices)

        if self._mimo is None:
            return {"answer": "No LLM client.", "confidence": 0.0, "reasoning": ""}

        try:
            response = self._mimo.call_text(prompt)
            parsed = self.parse_answer(response, choices)

            # If parse failed and we have choices, retry with simpler prompt
            if choices and (not parsed["answer"] or parsed["answer"].strip() == ""):
                retry_prompt = (
                    f"Question: {question}\n\n"
                    + "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
                    + "\n\nReply with ONLY the letter (A, B, C, D, or E)."
                )
                response = self._mimo.call_text(retry_prompt)
                parsed = self.parse_answer(response, choices)

            return {
                "answer": parsed["answer"],
                "confidence": parsed.get("confidence", 0.5),
                "reasoning": response[:500],
            }
        except Exception as e:
            return {"answer": f"Error: {e}", "confidence": 0.0, "reasoning": str(e)}

    def parse_answer(
        self,
        response: str,
        choices: Optional[List[str]] = None,
    ) -> Dict:
        """Parse LLM response into structured answer.

        For multiple choice: returns the full choice text matching the letter.
        For free text: returns the response as-is.
        """
        response = response.strip()

        if choices:
            # Extract letter from response
            letter_idx = self._extract_choice_letter(response, len(choices))
            if letter_idx >= 0:
                return {"answer": choices[letter_idx], "confidence": 0.7, "choice_idx": letter_idx}

            # Fallback: fuzzy match choice text
            best_idx = self._fuzzy_match_choice(response, choices)
            if best_idx >= 0:
                return {"answer": choices[best_idx], "confidence": 0.5, "choice_idx": best_idx}

        return {"answer": response, "confidence": 0.5}

    def _extract_choice_letter(self, response: str, num_choices: int) -> int:
        """Extract a choice letter (A-E) from the response."""
        response = response.strip().upper()

        # Direct single letter
        if len(response) <= 3 and response and response[0] in "ABCDE"[:num_choices]:
            return ord(response[0]) - ord("A")

        # "The answer is B" or "Option B" patterns
        m = re.search(r'(?:answer|option|choice|select|choose)\s*[:=]?\s*([A-E])', response, re.IGNORECASE)
        if m and m.group(1).upper() in "ABCDE"[:num_choices]:
            return ord(m.group(1).upper()) - ord("A")

        # Letter at start followed by punctuation
        m = re.match(r'^([A-E])[.)\s]', response)
        if m and m.group(1) in "ABCDE"[:num_choices]:
            return ord(m.group(1)) - ord("A")

        # Any standalone letter in the response
        for letter in "ABCDE"[:num_choices]:
            if re.search(rf'\b{letter}\b', response):
                return ord(letter) - ord("A")

        return -1

    def _fuzzy_match_choice(self, response: str, choices: List[str]) -> int:
        """Fuzzy match response text to choice text."""
        response_lower = response.lower()
        best_idx = -1
        best_overlap = 0

        for i, choice in enumerate(choices):
            # Substring match
            if choice[:40].lower() in response_lower:
                return i

            # Word overlap
            choice_words = set(choice.lower().split()[:10])
            response_words = set(response_lower.split())
            overlap = len(choice_words & response_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        return best_idx if best_overlap >= 3 else -1
