"""Nutrition estimator: ingredient identification and nutrition calculation."""

from typing import Dict, List, Optional

import numpy as np

from .evidence import Evidence

# Built-in nutrition database (per 100g)
NUTRITION_DB = {
    "tomato": {"calories_per_100g": 18, "protein_g": 0.9, "carbs_g": 3.9, "fat_g": 0.2},
    "onion": {"calories_per_100g": 40, "protein_g": 1.1, "carbs_g": 9.3, "fat_g": 0.1},
    "garlic": {"calories_per_100g": 149, "protein_g": 6.4, "carbs_g": 33.1, "fat_g": 0.5},
    "olive_oil": {"calories_per_100g": 884, "protein_g": 0, "carbs_g": 0, "fat_g": 100},
    "salt": {"calories_per_100g": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0},
    "pepper": {"calories_per_100g": 20, "protein_g": 0.9, "carbs_g": 4.6, "fat_g": 0.1},
    "chicken_breast": {"calories_per_100g": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6},
    "rice": {"calories_per_100g": 130, "protein_g": 2.7, "carbs_g": 28, "fat_g": 0.3},
    "pasta": {"calories_per_100g": 131, "protein_g": 5, "carbs_g": 25, "fat_g": 1.1},
    "egg": {"calories_per_100g": 155, "protein_g": 13, "carbs_g": 1.1, "fat_g": 11},
    "bread": {"calories_per_100g": 265, "protein_g": 9, "carbs_g": 49, "fat_g": 3.2},
    "butter": {"calories_per_100g": 717, "protein_g": 0.9, "carbs_g": 0.1, "fat_g": 81},
    "cheese": {"calories_per_100g": 402, "protein_g": 25, "carbs_g": 1.3, "fat_g": 33},
    "milk": {"calories_per_100g": 42, "protein_g": 3.4, "carbs_g": 5, "fat_g": 1},
    "potato": {"calories_per_100g": 77, "protein_g": 2, "carbs_g": 17, "fat_g": 0.1},
    "carrot": {"calories_per_100g": 41, "protein_g": 0.9, "carbs_g": 10, "fat_g": 0.2},
    "broccoli": {"calories_per_100g": 34, "protein_g": 2.8, "carbs_g": 7, "fat_g": 0.4},
    "mushroom": {"calories_per_100g": 22, "protein_g": 3.1, "carbs_g": 3.3, "fat_g": 0.3},
    "bell_pepper": {"calories_per_100g": 31, "protein_g": 1, "carbs_g": 6, "fat_g": 0.3},
    "zucchini": {"calories_per_100g": 17, "protein_g": 1.2, "carbs_g": 3.1, "fat_g": 0.3},
    "cucumber": {"calories_per_100g": 16, "protein_g": 0.7, "carbs_g": 3.6, "fat_g": 0.1},
    "lettuce": {"calories_per_100g": 15, "protein_g": 1.4, "carbs_g": 2.9, "fat_g": 0.2},
    "spinach": {"calories_per_100g": 23, "protein_g": 2.9, "carbs_g": 3.6, "fat_g": 0.4},
    "avocado": {"calories_per_100g": 160, "protein_g": 2, "carbs_g": 9, "fat_g": 15},
    "lemon": {"calories_per_100g": 29, "protein_g": 1.1, "carbs_g": 9.3, "fat_g": 0.3},
    "parsley": {"calories_per_100g": 36, "protein_g": 3, "carbs_g": 6.3, "fat_g": 0.8},
    "basil": {"calories_per_100g": 23, "protein_g": 3.2, "carbs_g": 2.7, "fat_g": 0.6},
    "oregano": {"calories_per_100g": 265, "protein_g": 9, "carbs_g": 69, "fat_g": 4.3},
    "cumin": {"calories_per_100g": 375, "protein_g": 18, "carbs_g": 44, "fat_g": 22},
    "paprika": {"calories_per_100g": 282, "protein_g": 14, "carbs_g": 54, "fat_g": 13},
    "sugar": {"calories_per_100g": 387, "protein_g": 0, "carbs_g": 100, "fat_g": 0},
    "flour": {"calories_per_100g": 364, "protein_g": 10, "carbs_g": 76, "fat_g": 1},
    "vinegar": {"calories_per_100g": 21, "protein_g": 0, "carbs_g": 0.9, "fat_g": 0},
    "soy_sauce": {"calories_per_100g": 53, "protein_g": 8.1, "carbs_g": 4.9, "fat_g": 0},
    "honey": {"calories_per_100g": 304, "protein_g": 0.3, "carbs_g": 82, "fat_g": 0},
    "yogurt": {"calories_per_100g": 59, "protein_g": 10, "carbs_g": 3.6, "fat_g": 0.4},
}


class NutritionEstimator:
    """Estimate nutrition from identified ingredients and portions.

    Uses a built-in nutrition database for lookup, and MiMo2.5 API
    for visual ingredient identification and portion estimation.
    """

    def __init__(self, nutrition_db_path: Optional[str] = None):
        self._db = dict(NUTRITION_DB)
        if nutrition_db_path:
            self._load_external_db(nutrition_db_path)

    def _load_external_db(self, path: str):
        """Load additional nutrition data from a JSON file."""
        import json
        try:
            with open(path) as f:
                data = json.load(f)
            self._db.update(data)
        except Exception:
            pass

    def lookup_nutrition(self, ingredient_name: str) -> Optional[Dict]:
        """Look up nutrition info for an ingredient.

        Returns:
            Dict with calories_per_100g, protein_g, carbs_g, fat_g, or None.
        """
        key = ingredient_name.lower().replace(" ", "_").replace("-", "_")
        return self._db.get(key)

    def estimate_ingredients(
        self, frame: np.ndarray, mimo_client=None
    ) -> List[Dict]:
        """Identify ingredients visible in a frame using MiMo2.5 API.

        Returns:
            List of dicts with name and estimated amount_g.
        """
        if mimo_client is None:
            return []

        prompt = (
            "Identify all food ingredients visible in this kitchen scene image. "
            "For each ingredient, estimate its approximate weight in grams. "
            "Return a JSON array of objects with 'name' (string) and 'amount_g' (number). "
            "Only include food ingredients, not utensils or containers."
        )

        response = mimo_client.call_vision(frame, prompt)
        try:
            import json
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except Exception:
            pass
        return []

    def estimate_portions(
        self,
        ingredients: List[Dict],
        frame: np.ndarray,
        mimo_client=None,
    ) -> List[Dict]:
        """Refine portion estimates using visual context.

        Args:
            ingredients: List of {name, amount_g} from estimate_ingredients.
            frame: BGR image for visual reference.
            mimo_client: API client.

        Returns:
            Refined list with better amount_g estimates.
        """
        if mimo_client is None or not ingredients:
            return ingredients

        ingredient_text = ", ".join(f"{i['name']} (~{i.get('amount_g', '?')}g)" for i in ingredients)
        prompt = (
            f"I identified these ingredients: {ingredient_text}. "
            "Looking at this kitchen scene, please refine the weight estimates in grams. "
            "Consider hand sizes and common portion sizes. "
            "Return a JSON array with 'name' and 'amount_g' for each."
        )

        response = mimo_client.call_vision(frame, prompt)
        try:
            import json
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except Exception:
            pass
        return ingredients

    def calculate_total(self, ingredients: List[Dict]) -> Dict:
        """Calculate total nutrition from a list of ingredients.

        Args:
            ingredients: List of dicts with 'name' and 'amount_g'.

        Returns:
            Dict with per-ingredient details and total.
        """
        details = []
        total = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}

        for ing in ingredients:
            name = ing.get("name", "")
            amount_g = float(ing.get("amount_g", 0))
            nutrition = self.lookup_nutrition(name)

            if nutrition:
                factor = amount_g / 100.0
                calories = nutrition["calories_per_100g"] * factor
                protein = nutrition["protein_g"] * factor
                carbs = nutrition["carbs_g"] * factor
                fat = nutrition["fat_g"] * factor
            else:
                calories = protein = carbs = fat = 0

            details.append({
                "name": name,
                "amount_g": amount_g,
                "calories": round(calories, 1),
                "protein_g": round(protein, 1),
                "carbs_g": round(carbs, 1),
                "fat_g": round(fat, 1),
            })
            total["calories"] += calories
            total["protein_g"] += protein
            total["carbs_g"] += carbs
            total["fat_g"] += fat

        total = {k: round(v, 1) for k, v in total.items()}
        return {"ingredients": details, "total": total}

    def get_nutrition_evidence(
        self,
        frame: np.ndarray,
        timestamp: float,
        mimo_client=None,
    ) -> Evidence:
        """Full nutrition analysis pipeline.

        Returns:
            Evidence with nutrition content.
        """
        ingredients = self.estimate_ingredients(frame, mimo_client)
        if not ingredients:
            return Evidence(
                source_module="NutritionEstimator",
                evidence_type="nutrition",
                time_range={"start": timestamp, "end": timestamp},
                content={"ingredients": [], "total": {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}},
                confidence=0.0,
            )

        result = self.calculate_total(ingredients)
        return Evidence(
            source_module="NutritionEstimator",
            evidence_type="nutrition",
            time_range={"start": timestamp, "end": timestamp},
            content=result,
            confidence=0.6 if result["total"]["calories"] > 0 else 0.2,
        )
