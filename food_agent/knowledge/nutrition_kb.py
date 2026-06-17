"""Nutrition knowledge base: ingredient-to-nutrition mapping."""

import json
from typing import Dict, List, Optional

from food_agent.perception.nutrition_estimator import NUTRITION_DB


class NutritionKB:
    """Knowledge base for nutrition facts.

    Wraps the built-in nutrition database and provides
    query interfaces for the Agent.
    """

    def __init__(self, nutrition_data_path: Optional[str] = None):
        self._db = dict(NUTRITION_DB)
        if nutrition_data_path:
            self._load(nutrition_data_path)

    def _load(self, path: str):
        try:
            with open(path) as f:
                data = json.load(f)
            self._db.update(data)
        except Exception:
            pass

    def lookup(self, ingredient: str) -> Optional[Dict]:
        """Look up nutrition facts for an ingredient.

        Returns:
            Dict with calories_per_100g, protein_g, carbs_g, fat_g, or None.
        """
        key = ingredient.lower().replace(" ", "_").replace("-", "_")
        return self._db.get(key)

    def calculate_dish(self, ingredients: List[Dict]) -> Dict:
        """Calculate total nutrition for a dish.

        Args:
            ingredients: List of {name, amount_g} dicts.

        Returns:
            Dict with per-ingredient details and total.
        """
        details = []
        total = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}

        for ing in ingredients:
            name = ing.get("name", "")
            amount_g = float(ing.get("amount_g", 0))
            nutrition = self.lookup(name)

            if nutrition:
                factor = amount_g / 100.0
                cal = nutrition["calories_per_100g"] * factor
                prot = nutrition["protein_g"] * factor
                carbs = nutrition["carbs_g"] * factor
                fat = nutrition["fat_g"] * factor
            else:
                cal = prot = carbs = fat = 0

            details.append({
                "name": name,
                "amount_g": amount_g,
                "calories": round(cal, 1),
                "protein_g": round(prot, 1),
                "carbs_g": round(carbs, 1),
                "fat_g": round(fat, 1),
            })
            total["calories"] += cal
            total["protein_g"] += prot
            total["carbs_g"] += carbs
            total["fat_g"] += fat

        total = {k: round(v, 1) for k, v in total.items()}
        return {"ingredients": details, "total": total}

    def list_ingredients(self) -> List[str]:
        """List all ingredients in the database."""
        return list(self._db.keys())
