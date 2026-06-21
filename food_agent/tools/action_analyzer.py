"""Action analysis tools for fine-grained action understanding.

Handles:
- Action recognition: what action is happening
- Action localization: when did an action occur
- How recognition: how was an action performed
- Why recognition: why was an action performed
"""

import re
from typing import Dict, List, Optional, Tuple


class ActionAnalyzer:
    """Tools for analyzing actions in kitchen videos."""

    # Common kitchen actions and their visual cues
    ACTION_CUES = {
        "chopping": {"hand_motion": "up-down", "tool": "knife", "object": "cutting board", "sound": "chopping"},
        "slicing": {"hand_motion": "forward-back", "tool": "knife", "object": "ingredient", "sound": "cutting"},
        "dicing": {"hand_motion": "up-down+forward-back", "tool": "knife", "object": "cutting board", "sound": "chopping"},
        "mincing": {"hand_motion": "rapid up-down", "tool": "knife", "object": "cutting board", "sound": "chopping"},
        "stirring": {"hand_motion": "circular", "tool": "spoon", "object": "pot/bowl", "sound": "scraping"},
        "mixing": {"hand_motion": "folding", "tool": "spatula/whisk", "object": "bowl", "sound": "mixing"},
        "whisking": {"hand_motion": "rapid circular", "tool": "whisk", "object": "bowl", "sound": "whisking"},
        "beating": {"hand_motion": "up-down", "tool": "whisk", "object": "bowl", "sound": "beating"},
        "pouring": {"hand_motion": "tilting", "tool": "container", "object": "pot/cup", "sound": "pouring"},
        "adding": {"hand_motion": "placing", "tool": "hands/spoon", "object": "pot/bowl", "sound": "adding"},
        "seasoning": {"hand_motion": "sprinkling", "tool": "shaker", "object": "dish", "sound": "sprinkling"},
        "frying": {"hand_motion": "stirring in pan", "tool": "spatula", "object": "pan", "sound": "sizzling"},
        "boiling": {"hand_motion": "waiting", "tool": "pot", "object": "water", "sound": "bubbling"},
        "baking": {"hand_motion": "placing in oven", "tool": "oven", "object": "baking sheet", "sound": "oven"},
        "roasting": {"hand_motion": "placing in oven", "tool": "oven", "object": "roasting pan", "sound": "oven"},
        "grilling": {"hand_motion": "placing on grill", "tool": "grill", "object": "grill", "sound": "sizzling"},
        "washing": {"hand_motion": "rubbing", "tool": "hands", "object": "sink", "sound": "water"},
        "peeling": {"hand_motion": "pulling", "tool": "peeler", "object": "vegetable", "sound": "peeling"},
        "grating": {"hand_motion": "back-forth", "tool": "grater", "object": "cheese", "sound": "grating"},
        "kneading": {"hand_motion": "pressing+folding", "tool": "hands", "object": "dough", "sound": "kneading"},
        "rolling": {"hand_motion": "rolling", "tool": "rolling pin", "object": "dough", "sound": "rolling"},
        "flipping": {"hand_motion": "flicking", "tool": "spatula", "object": "pan", "sound": "flipping"},
        "plating": {"hand_motion": "placing", "tool": "hands", "object": "plate", "sound": "placing"},
        "serving": {"hand_motion": "scooping", "tool": "ladle", "object": "bowl", "sound": "serving"},
        "tasting": {"hand_motion": "lifting to mouth", "tool": "spoon", "object": "mouth", "sound": "tasting"},
        "wiping": {"hand_motion": "rubbing", "tool": "cloth", "object": "counter", "sound": "wiping"},
        "opening": {"hand_motion": "pulling", "tool": "hands", "object": "door/drawer", "sound": "opening"},
        "closing": {"hand_motion": "pushing", "tool": "hands", "object": "door/drawer", "sound": "closing"},
        "picking up": {"hand_motion": "reaching+grasping", "tool": "hands", "object": "item", "sound": "picking"},
        "putting down": {"hand_motion": "releasing", "tool": "hands", "object": "surface", "sound": "placing"},
        "holding": {"hand_motion": "gripping", "tool": "hands", "object": "item", "sound": "none"},
        "reaching": {"hand_motion": "extending", "tool": "hands", "object": "item", "sound": "none"},
    }

    @staticmethod
    def build_action_recognition_prompt() -> str:
        """Build prompt for action recognition."""
        return (
            "Look at this egocentric kitchen video frame carefully. "
            "What specific action is the person performing right now? "
            "Focus on: "
            "1. What the person is doing with their hands "
            "2. What tool they are using "
            "3. What object they are interacting with "
            "4. The motion pattern (circular, up-down, forward-back, etc.) "
            "Reply with a specific action description (e.g., 'chopping onions with a knife', "
            "'stirring the pot with a spoon', 'pouring oil into the pan')."
        )

    @staticmethod
    def build_action_localization_prompt(action: str) -> str:
        """Build prompt for finding when an action occurred."""
        return (
            f"Look at this egocentric kitchen video frame. "
            f"Is the person currently performing the action '{action}'? "
            f"Look for the specific motion pattern and tool usage. "
            f"Reply with 'yes' if the action is happening now, 'no' if not."
        )

    @staticmethod
    def build_how_prompt(action: str) -> str:
        """Build prompt for understanding how an action is performed."""
        return (
            f"Look at this egocentric kitchen video frame. "
            f"The person is performing the action: {action}. "
            f"How are they performing it? Focus on: "
            f"1. What tool are they using? "
            f"2. What is the motion pattern? "
            f"3. What is the technique? "
            f"Reply with a brief description of how the action is performed."
        )

    @staticmethod
    def build_why_prompt(action: str) -> str:
        """Build prompt for understanding why an action is performed."""
        return (
            f"Look at this egocentric kitchen video frame. "
            f"The person is performing the action: {action}. "
            f"Why are they performing this action? What is the purpose? "
            f"Consider the cooking context and what comes next. "
            f"Reply with the reason/purpose of the action."
        )

    @staticmethod
    def match_action_to_choices(action: str, choices: List[str]) -> Tuple[int, float]:
        """Match recognized action to answer choices."""
        action_lower = action.lower().strip()
        best_idx = -1
        best_score = 0.0

        for i, choice in enumerate(choices):
            choice_lower = choice.lower().strip()

            # Direct substring match
            if action_lower in choice_lower or choice_lower in action_lower:
                return i, 1.0

            # Word overlap
            action_words = set(action_lower.split())
            choice_words = set(choice_lower.split())
            overlap = len(action_words & choice_words)

            if overlap > best_score:
                best_score = overlap
                best_idx = i

        return best_idx, best_score / max(len(action_lower.split()), 1)

    @staticmethod
    def match_time_to_choices(time_str: str, choices: List[str]) -> Tuple[int, float]:
        """Match time segment to answer choices."""
        # Extract time from response
        time_match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d+)', time_str)
        if not time_match:
            return -1, 0.0

        target_time = time_match.group(1)

        for i, choice in enumerate(choices):
            if target_time in choice:
                return i, 1.0

        # Fuzzy match by comparing time values
        def parse_time(t: str) -> float:
            parts = t.split(":")
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            return 0.0

        target_val = parse_time(target_time)
        best_idx = -1
        best_diff = float('inf')

        for i, choice in enumerate(choices):
            choice_match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d+)', choice)
            if choice_match:
                choice_val = parse_time(choice_match.group(1))
                diff = abs(target_val - choice_val)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

        return best_idx, 1.0 / (1.0 + best_diff)

    @staticmethod
    def get_action_knowledge(action: str) -> Dict:
        """Get knowledge about an action."""
        action_lower = action.lower().strip()

        # Direct match
        if action_lower in ActionAnalyzer.ACTION_CUES:
            return ActionAnalyzer.ACTION_CUES[action_lower]

        # Partial match
        for key in ActionAnalyzer.ACTION_CUES:
            if key in action_lower or action_lower in key:
                return ActionAnalyzer.ACTION_CUES[key]

        return {}

    @staticmethod
    def format_action_knowledge(action: str) -> str:
        """Format action knowledge as readable text."""
        knowledge = ActionAnalyzer.get_action_knowledge(action)
        if not knowledge:
            return ""

        lines = [f"Knowledge about '{action}':"]
        for key, value in knowledge.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
