"""Specialized tools for difficult question types.

These tools use creative approaches to extract information that the vision model
alone cannot reliably determine:
1. Scale reading via OCR-like prompting
2. Container contents analysis
3. Multi-frame object tracking
4. Action recognition via hand/body analysis
5. Gaze-to-object matching
6. Interaction prediction
"""

import re
from typing import Dict, List, Optional, Tuple


class ScaleReader:
    """Tool for reading weight from scale displays in video frames."""

    @staticmethod
    def build_scale_reading_prompt() -> str:
        """Build a prompt specifically for reading scale displays."""
        return (
            "Look at this kitchen scene carefully. Is there a kitchen scale visible? "
            "If yes, what weight is displayed on the scale? "
            "Look for digital displays, dial indicators, or balance beam positions. "
            "Reply with ONLY the number and unit (e.g., '150 g', '0.5 kg', '2 tbsp'). "
            "If no scale is visible, reply 'no scale visible'."
        )

    @staticmethod
    def parse_weight(response: str) -> Optional[Dict]:
        """Parse weight from vision model response."""
        response = response.lower().strip()

        # Check for no scale
        if "no scale" in response or "not visible" in response:
            return None

        # Extract number and unit
        # Patterns: "150 g", "0.5 kg", "2 tbsp", "150g", "0.5kg"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(g|gram|grams|kg|kilogram|kilograms)',
            r'(\d+(?:\.\d+)?)\s*(tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons)',
            r'(\d+(?:\.\d+)?)\s*(oz|ounce|ounces|lb|pound|pounds)',
            r'(\d+(?:\.\d+)?)\s*(ml|milliliter|milliliters|l|liter|liters)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                value = float(match.group(1))
                unit = match.group(2)

                # Convert to grams
                grams = value
                if unit in ['kg', 'kilogram', 'kilograms']:
                    grams = value * 1000
                elif unit in ['oz', 'ounce', 'ounces']:
                    grams = value * 28.35
                elif unit in ['lb', 'pound', 'pounds']:
                    grams = value * 453.6
                elif unit in ['ml', 'milliliter', 'milliliters']:
                    grams = value  # Approximate for water
                elif unit in ['l', 'liter', 'liters']:
                    grams = value * 1000

                return {
                    "value": value,
                    "unit": unit,
                    "grams": round(grams, 1),
                    "original": response,
                }

        # Try to extract just a number
        number_match = re.search(r'(\d+(?:\.\d+)?)', response)
        if number_match:
            value = float(number_match.group(1))
            return {
                "value": value,
                "unit": "unknown",
                "grams": value,  # Assume grams
                "original": response,
            }

        return None


class ContainerAnalyzer:
    """Tool for analyzing what's inside containers."""

    @staticmethod
    def build_container_prompt(container_type: str = "container") -> str:
        """Build a prompt for analyzing container contents."""
        return (
            f"Look at this kitchen scene. Focus on the {container_type}. "
            f"What objects are inside or on top of the {container_type}? "
            f"List each object you can see. "
            f"Reply with a comma-separated list of objects (e.g., 'tomatoes, onions, garlic'). "
            f"If nothing is visible inside, reply 'empty' or 'nothing visible'."
        )

    @staticmethod
    def build_tracking_prompt(container_type: str = "container") -> str:
        """Build a prompt for tracking what's put in/taken from a container."""
        return (
            f"Look at this egocentric kitchen video frame. "
            f"The person is interacting with a {container_type}. "
            f"Are they PUTTING something IN/ON the {container_type}, or TAKING something FROM it? "
            f"What specific object are they putting or taking? "
            f"Reply with: 'putting [object]' or 'taking [object]' or 'nothing'."
        )

    @staticmethod
    def parse_contents(response: str) -> List[str]:
        """Parse container contents from vision response."""
        response = response.lower().strip()

        if "empty" in response or "nothing" in response or "not visible" in response:
            return []

        # Split by commas and clean
        items = [item.strip() for item in response.split(',')]
        items = [item for item in items if item and len(item) > 1]

        return items

    @staticmethod
    def parse_tracking(response: str) -> Dict:
        """Parse tracking response."""
        response = response.lower().strip()

        if "nothing" in response:
            return {"action": "none", "object": None}

        if "putting" in response:
            # Extract object after "putting"
            obj = response.replace("putting", "").strip()
            return {"action": "putting", "object": obj}

        if "taking" in response:
            # Extract object after "taking"
            obj = response.replace("taking", "").strip()
            return {"action": "taking", "object": obj}

        return {"action": "unknown", "object": None}

    @staticmethod
    def match_to_choices(items: List[str], choices: List[str]) -> Tuple[int, float]:
        """Match identified items to answer choices."""
        if not items:
            # Look for "Nothing" in choices
            for i, choice in enumerate(choices):
                if "nothing" in choice.lower():
                    return i, 1.0
            return -1, 0.0

        best_idx = -1
        best_score = 0.0

        for i, choice in enumerate(choices):
            choice_lower = choice.lower()
            score = 0.0

            for item in items:
                if item in choice_lower or choice_lower in item:
                    score += 1.0
                elif any(word in choice_lower for word in item.split() if len(word) > 2):
                    score += 0.5

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx, best_score


class ObjectTracker:
    """Tool for tracking objects across multiple frames."""

    @staticmethod
    def build_tracking_prompt(object_name: str) -> str:
        """Build a prompt for tracking an object."""
        return (
            f"Look at this kitchen scene. Where is the {object_name}? "
            f"Is the person holding it, or has it been placed somewhere? "
            f"If placed, describe the exact location relative to kitchen fixtures. "
            f"Reply with: 'holding' or 'placed on/in [location]'."
        )

    @staticmethod
    def analyze_trajectory(locations: List[Dict]) -> Dict:
        """Analyze object movement trajectory."""
        if not locations:
            return {"status": "no_data", "final_location": "unknown"}

        # Find transitions from holding to placed
        holding_phases = []
        placed_phases = []

        for loc in locations:
            if loc.get("holding"):
                holding_phases.append(loc)
            elif loc.get("placed"):
                placed_phases.append(loc)

        # Determine final state
        if placed_phases:
            final = placed_phases[-1]
            return {
                "status": "placed",
                "final_location": final.get("location", "unknown"),
                "placement_time": final.get("timestamp"),
                "holding_duration": len(holding_phases),
            }
        elif holding_phases:
            return {
                "status": "still_holding",
                "final_location": "in hand",
                "holding_duration": len(holding_phases),
            }
        else:
            return {
                "status": "unknown",
                "final_location": "not determined",
            }


class ActionRecognizer:
    """Tool for recognizing complex actions."""

    # Common kitchen actions and their visual cues
    ACTION_CUES = {
        "chopping": {"hand_motion": "up-down", "tool": "knife", "object": "cutting board"},
        "stirring": {"hand_motion": "circular", "tool": "spoon", "object": "pot/bowl"},
        "pouring": {"hand_motion": "tilting", "tool": "container", "object": "pot/cup"},
        "washing": {"hand_motion": "rubbing", "tool": "hands", "object": "sink"},
        "peeling": {"hand_motion": "pulling", "tool": "peeler", "object": "vegetable"},
        "grating": {"hand_motion": "back-forth", "tool": "grater", "object": "cheese"},
        "mixing": {"hand_motion": "folding", "tool": "spatula", "object": "bowl"},
        "flipping": {"hand_motion": "flicking", "tool": "spatula", "object": "pan"},
        "seasoning": {"hand_motion": "sprinkling", "tool": "shaker", "object": "dish"},
        "tasting": {"hand_motion": "lifting", "tool": "spoon", "object": "mouth"},
        "plating": {"hand_motion": "placing", "tool": "hands", "object": "plate"},
        "serving": {"hand_motion": "scooping", "tool": "ladle", "object": "bowl"},
    }

    @staticmethod
    def build_action_prompt() -> str:
        """Build a prompt for action recognition."""
        return (
            "Look at this kitchen scene carefully. What specific action is the person performing? "
            "Focus on: "
            "1. What the person is doing with their hands "
            "2. What tool they are using "
            "3. What object they are interacting with "
            "Reply with a specific action (e.g., 'chopping onions', 'stirring the pot', 'pouring oil')."
        )

    @staticmethod
    def match_action_to_choices(action: str, choices: List[str]) -> Tuple[int, float]:
        """Match recognized action to answer choices."""
        action_lower = action.lower()
        best_idx = -1
        best_score = 0.0

        for i, choice in enumerate(choices):
            choice_lower = choice.lower()

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


class GazeObjectMatcher:
    """Tool for matching gaze direction to objects."""

    @staticmethod
    def build_gaze_object_prompt(gaze_direction: str) -> str:
        """Build a prompt for matching gaze to objects."""
        return (
            f"The person is looking {gaze_direction}. "
            f"What object or fixture is the person most likely looking at? "
            f"Look at what's in the direction they're facing. "
            f"Reply with the object name (e.g., 'the stove', 'the counter', 'the fridge')."
        )

    @staticmethod
    def match_gaze_to_choices(gaze_target: str, choices: List[str]) -> Tuple[int, float]:
        """Match gaze target to answer choices."""
        gaze_lower = gaze_target.lower()
        best_idx = -1
        best_score = 0.0

        for i, choice in enumerate(choices):
            choice_lower = choice.lower()

            # Direct match
            if gaze_lower in choice_lower or choice_lower in gaze_lower:
                return i, 1.0

            # Word overlap
            gaze_words = set(gaze_lower.split())
            choice_words = set(choice_lower.split())
            overlap = len(gaze_words & choice_words)

            if overlap > best_score:
                best_score = overlap
                best_idx = i

        return best_idx, best_score / max(len(gaze_lower.split()), 1)


class InteractionPredictor:
    """Tool for predicting future interactions."""

    @staticmethod
    def build_prediction_prompt() -> str:
        """Build a prompt for interaction prediction."""
        return (
            "Look at this kitchen scene. The person is currently doing something. "
            "Based on what they're doing and what's nearby, what will they likely interact with next? "
            "Consider: "
            "1. What they're holding or about to pick up "
            "2. What's in their line of sight "
            "3. What's needed for the current cooking step "
            "Reply with the object they'll likely interact with next."
        )

    @staticmethod
    def build_anticipation_prompt(current_action: str = "") -> str:
        """Build a more specific prompt for interaction anticipation."""
        action_context = f"They are currently {current_action}. " if current_action else ""
        return (
            f"Look at this egocentric kitchen video frame. {action_context}"
            f"The person is cooking. Based on the current state of the kitchen and what the person is doing, "
            f"what object will they most likely interact with NEXT? "
            f"Consider what objects are nearby and what would logically come next in the cooking process. "
            f"Reply with just the object name (1-2 words)."
        )

    @staticmethod
    def predict_next_interaction(
        current_action: str,
        gaze_target: str,
        nearby_objects: List[str],
        recipe_step: str = "",
    ) -> str:
        """Predict next interaction based on context."""
        # Simple heuristic: the person is likely to interact with what they're looking at
        if gaze_target and gaze_target.lower() not in ["unknown", "nothing"]:
            return gaze_target

        # If we know the recipe step, predict based on that
        if recipe_step:
            # Common patterns
            if "chop" in recipe_step.lower():
                return "cutting board"
            elif "stir" in recipe_step.lower():
                return "pot"
            elif "add" in recipe_step.lower():
                return "ingredient"
            elif "serve" in recipe_step.lower():
                return "plate"

        # Default to first nearby object
        if nearby_objects:
            return nearby_objects[0]

        return "unknown"

    @staticmethod
    def match_to_choices(predicted: str, choices: List[str]) -> Tuple[int, float]:
        """Match predicted object to answer choices."""
        predicted_lower = predicted.lower().strip()
        best_idx = -1
        best_score = 0.0

        for i, choice in enumerate(choices):
            choice_lower = choice.lower().strip()

            # Remove common prefixes
            choice_clean = choice_lower.replace("the ", "").strip()
            predicted_clean = predicted_lower.replace("the ", "").strip()

            # Direct match
            if predicted_clean in choice_clean or choice_clean in predicted_clean:
                return i, 1.0

            # Word overlap
            predicted_words = set(predicted_clean.split())
            choice_words = set(choice_clean.split())
            overlap = len(predicted_words & choice_words)

            if overlap > best_score:
                best_score = overlap
                best_idx = i

        return best_idx, best_score / max(len(predicted_lower.split()), 1)
