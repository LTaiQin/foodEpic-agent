"""Recipe knowledge base: store and query cooking recipes and steps."""

import json
from pathlib import Path
from typing import Dict, List, Optional


class RecipeKB:
    """Knowledge base for recipes and cooking procedures.

    Loads recipe data from HD-EPIC annotations and provides
    query interfaces for the Agent's tool-calling loop.
    """

    def __init__(self, recipe_data_path: Optional[str] = None):
        self._recipes: Dict[str, Dict] = {}
        if recipe_data_path:
            self._load(recipe_data_path)

    def _load(self, path: str):
        """Load recipes from a JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for r in data:
                    name = r.get("name", r.get("recipe_name", ""))
                    if name:
                        self._recipes[name.lower()] = r
            elif isinstance(data, dict):
                self._recipes = {k.lower(): v for k, v in data.items()}
        except Exception:
            pass

    def get_recipe(self, name: str) -> Optional[Dict]:
        """Get a recipe by name.

        Returns:
            Dict with name, steps list, or None if not found.
        """
        key = name.lower()
        # Try exact match first
        if key in self._recipes:
            return self._recipes[key]
        # Try partial match
        for k, v in self._recipes.items():
            if key in k or k in key:
                return v
        return None

    def get_step(self, recipe_name: str, step_number: int) -> Optional[Dict]:
        """Get a specific step from a recipe.

        Returns:
            Dict with step_number, description, or None.
        """
        recipe = self.get_recipe(recipe_name)
        if recipe is None:
            return None
        steps = recipe.get("steps", [])
        if 0 < step_number <= len(steps):
            step = steps[step_number - 1]
            if isinstance(step, str):
                return {"step_number": step_number, "description": step}
            return step
        return None

    def search_recipes(self, ingredients: List[str]) -> List[Dict]:
        """Search recipes that use the given ingredients.

        Returns:
            List of matching recipe dicts.
        """
        results = []
        ingredients_lower = [i.lower() for i in ingredients]
        for name, recipe in self._recipes.items():
            recipe_text = json.dumps(recipe).lower()
            matches = sum(1 for ing in ingredients_lower if ing in recipe_text)
            if matches > 0:
                results.append({"name": name, "match_count": matches, "recipe": recipe})
        results.sort(key=lambda x: x["match_count"], reverse=True)
        return results

    def match_current_step(self, observations: Dict) -> Optional[Dict]:
        """Try to match observations to a recipe step.

        Args:
            observations: Dict with detected objects, actions, sounds.

        Returns:
            Best matching step dict or None.
        """
        best_match = None
        best_score = 0

        obs_text = json.dumps(observations).lower()

        for name, recipe in self._recipes.items():
            steps = recipe.get("steps", [])
            for i, step in enumerate(steps):
                step_text = step if isinstance(step, str) else json.dumps(step)
                # Simple keyword overlap scoring
                step_words = set(step_text.lower().split())
                obs_words = set(obs_text.split())
                overlap = len(step_words & obs_words)
                if overlap > best_score:
                    best_score = overlap
                    best_match = {
                        "recipe": name,
                        "step_number": i + 1,
                        "step_description": step_text,
                        "score": overlap,
                    }

        return best_match if best_score > 0 else None

    def list_recipes(self) -> List[str]:
        """List all available recipe names."""
        return list(self._recipes.keys())
