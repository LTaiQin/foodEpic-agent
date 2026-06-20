"""Question router: classify questions and determine which modules to use."""

from typing import Dict, List, Optional


# Default routing table
ROUTE_TABLE = {
    "recipe": {
        "primary": ["RecipeKB", "VisualAnalyzer"],
        "secondary": ["AudioAnalyzer", "HandInteractor"],
        "description": "Recipe identification, step matching, ingredient checking",
    },
    "ingredient": {
        "primary": ["VisualAnalyzer", "HandInteractor"],
        "secondary": ["AudioAnalyzer", "MotionTracker"],
        "description": "Ingredient identification and detection",
    },
    "nutrition": {
        "primary": ["NutritionEstimator", "VisualAnalyzer"],
        "secondary": ["HandInteractor"],
        "description": "Nutritional content estimation",
    },
    "fine_grained_action": {
        "primary": ["HandInteractor", "AudioAnalyzer"],
        "secondary": ["VisualAnalyzer", "MotionTracker"],
        "description": "Fine-grained action classification",
    },
    "3d_perception": {
        "primary": ["SpatialReasoner"],
        "secondary": ["GazeTracker", "VisualAnalyzer"],
        "description": "3D spatial queries and layout",
    },
    "object_motion": {
        "primary": ["MotionTracker", "HandInteractor"],
        "secondary": ["SpatialReasoner", "VisualAnalyzer"],
        "description": "Object tracking and state changes",
    },
    "gaze": {
        "primary": ["GazeTracker"],
        "secondary": ["VisualAnalyzer", "SpatialReasoner"],
        "description": "Gaze target and attention analysis",
    },
    "general": {
        "primary": ["VisualAnalyzer", "AudioAnalyzer"],
        "secondary": ["HandInteractor", "GazeTracker", "SpatialReasoner"],
        "description": "General kitchen scene understanding",
    },
}

# Keywords for rule-based classification
CATEGORY_KEYWORDS = {
    "recipe": ["recipe", "step", "cooking", "dish", "meal", "prepare", "cook", "which recipe", "what step", "which step", "not used in", "used in"],
    "ingredient": ["ingredient", "food item", "vegetable", "fruit", "meat", "spice", "what food", "what ingredient", "visible food", "identify"],
    "nutrition": ["calorie", "nutrition", "protein", "carb", "fat", "healthy", "diet", "how many calories", "nutritional", "meal healthy"],
    "fine_grained_action": ["doing", "action", "activity", "performing", "stirring", "cutting", "chopping", "what is.*doing", "what action", "or chopping"],
    "3d_perception": ["where is", "location", "position", "sink", "stove", "counter", "fridge", "kitchen layout", "near", "next to", "spatial"],
    "object_motion": ["move", "motion", "track", "transfer", "pick up", "put down", "place", "which object", "moved", "displaced", "where did.*go", "where did", "where.*go"],
    "gaze": ["looking", "watching", "gaze", "attention", "eye", "see", "look at", "stare", "focus", "wearer.*looking", "person attention", "person.*attention"],
}


class Router:
    """Classify questions and determine which perception modules to use."""

    def __init__(self, route_table: Optional[Dict] = None):
        self.route_table = route_table or ROUTE_TABLE

    def classify_question(self, question: str) -> str:
        """Classify a question into a category.

        Uses keyword matching as a fast heuristic.

        Args:
            question: The user's question string.

        Returns:
            Category string (e.g., 'recipe', 'ingredient', '3d_perception').
        """
        q = question.lower()
        scores = {}

        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in q)
            if score > 0:
                scores[category] = score

        if not scores:
            return "general"

        return max(scores, key=scores.get)

    def get_route(self, category: str) -> Dict:
        """Get the routing strategy for a category.

        Returns:
            Dict with 'primary' (list of module names) and 'secondary'.
        """
        return self.route_table.get(category, self.route_table["general"])

    def route(self, question: str) -> Dict:
        """Full routing: classify question and return module strategy.

        Returns:
            Dict with 'category', 'primary', 'secondary', 'description'.
        """
        category = self.classify_question(question)
        strategy = self.get_route(category)
        return {
            "category": category,
            "primary": strategy["primary"],
            "secondary": strategy["secondary"],
            "description": strategy.get("description", ""),
        }

    def route_with_llm(self, question: str, mimo_client=None) -> Dict:
        """Route using LLM for more accurate classification.

        Falls back to keyword-based if no LLM client.
        """
        if mimo_client is None:
            return self.route(question)

        categories = list(self.route_table.keys())
        prompt = (
            "Classify this question into exactly one category.\n"
            f"Categories: {', '.join(categories)}\n"
            f"Question: {question}\n"
            "Reply with only the category name."
        )

        try:
            response = mimo_client.call_text(prompt)
            if isinstance(response, list):
                response = response[0] if response else ""
            response = str(response).strip().lower()
            # Match to closest category
            for cat in categories:
                if cat in response or response in cat:
                    category = cat
                    break
            else:
                category = self.classify_question(question)
        except Exception:
            category = self.classify_question(question)

        strategy = self.get_route(category)
        return {
            "category": category,
            "primary": strategy["primary"],
            "secondary": strategy["secondary"],
            "description": strategy.get("description", ""),
        }
