"""Common sense knowledge base for kitchen reasoning."""

from typing import Dict, List, Optional


# Built-in kitchen common sense (no external API needed)
KITCHEN_COMMONSENSE = {
    "cooking": {
        "UsedFor": ["frying", "boiling", "baking", "grilling", "steaming", "roasting"],
        "Requires": ["heat", "ingredients", "utensils", "time"],
        "PartOf": ["meal preparation", "food processing"],
    },
    "knife": {
        "UsedFor": ["cutting", "slicing", "chopping", "dicing", "peeling"],
        "FoundIn": ["kitchen", "cutting board"],
        "UsedWith": ["cutting board", "food", "vegetables"],
    },
    "cutting_board": {
        "UsedFor": ["cutting surface", "food preparation"],
        "FoundOn": ["counter", "table"],
        "UsedWith": ["knife", "food"],
    },
    "stove": {
        "UsedFor": ["cooking", "heating", "frying", "boiling"],
        "FoundIn": ["kitchen"],
        "Produces": ["heat", "flame"],
    },
    "sink": {
        "UsedFor": ["washing", "rinsing", "draining"],
        "FoundIn": ["kitchen", "bathroom"],
        "UsedWith": ["water", "dishes", "food"],
    },
    "pan": {
        "UsedFor": ["frying", "sautéing", "cooking"],
        "UsedOn": ["stove"],
        "UsedWith": ["oil", "food", "spatula"],
    },
    "pot": {
        "UsedFor": ["boiling", "simmering", "stewing", "soup"],
        "UsedOn": ["stove"],
        "UsedWith": ["water", "ingredients", "lid"],
    },
    "plate": {
        "UsedFor": ["serving", "eating", "presenting food"],
        "FoundOn": ["table", "counter"],
    },
    "spoon": {
        "UsedFor": ["stirring", "eating", "serving", "measuring"],
        "UsedWith": ["pot", "bowl", "food"],
    },
    "fork": {
        "UsedFor": ["eating", "piercing", "holding food"],
        "UsedWith": ["plate", "food"],
    },
    "bowl": {
        "UsedFor": ["mixing", "eating", "serving", "holding"],
        "UsedWith": ["spoon", "food", "ingredients"],
    },
    "oil": {
        "UsedFor": ["cooking", "frying", "lubricating"],
        "AddedTo": ["pan", "pot", "food"],
        "Types": ["olive oil", "vegetable oil", "sesame oil"],
    },
    "water": {
        "UsedFor": ["boiling", "rinsing", "washing", "drinking"],
        "FoundIn": ["sink", "pot", "kettle"],
    },
    "tomato": {
        "IsA": ["vegetable", "fruit", "ingredient"],
        "UsedIn": ["salad", "sauce", "soup", "sandwich"],
        "PreparedBy": ["cutting", "slicing", "dicing"],
    },
    "onion": {
        "IsA": ["vegetable", "ingredient"],
        "UsedIn": ["soup", "salad", "stir-fry", "sauce"],
        "PreparedBy": ["cutting", "dicing", "slicing"],
        "CausesTears": True,
    },
    "egg": {
        "IsA": ["protein", "ingredient"],
        "UsedIn": ["omelette", "cake", "scrambled eggs"],
        "PreparedBy": ["cracking", "beating", "boiling", "frying"],
    },
}

# Action → typical next action sequences
ACTION_SEQUENCES = {
    "cutting": ["placing in pan", "stirring", "cooking"],
    "washing": ["cutting", "peeling", "placing in bowl"],
    "frying": ["stirring", "adding seasoning", "plating"],
    "boiling": ["adding ingredients", "stirring", "draining"],
    "stirring": ["adding seasoning", "tasting", "plating"],
    "peeling": ["cutting", "washing", "cooking"],
    "mixing": ["pouring", "baking", "cooking"],
    "plating": ["garnishing", "serving"],
}


class CommonSenseKB:
    """Kitchen common sense knowledge base.

    Uses a built-in knowledge graph for common kitchen concepts.
    Can optionally query ConceptNet API for additional relations.
    """

    def __init__(self, conceptnet_url: Optional[str] = None):
        self._kb = dict(KITCHEN_COMMONSENSE)
        self._conceptnet_url = conceptnet_url or "http://api.conceptnet.io"

    def query_relation(
        self, concept_a: str, relation: str, concept_b: Optional[str] = None
    ) -> List[Dict]:
        """Query relations for a concept.

        Args:
            concept_a: The concept to query.
            relation: The relation type (e.g., 'UsedFor', 'IsA').
            concept_b: Optional target concept to check.

        Returns:
            List of relation dicts.
        """
        key = concept_a.lower().replace(" ", "_")
        if key in self._kb:
            rel_data = self._kb[key].get(relation, [])
            if concept_b:
                return [{"from": concept_a, "relation": relation, "to": r}
                        for r in rel_data if concept_b.lower() in str(r).lower()]
            return [{"from": concept_a, "relation": relation, "to": r} for r in rel_data]
        return []

    def get_related_concepts(self, concept: str, relation: str) -> List:
        """Get all concepts related to the given concept via a relation.

        Returns:
            List of related concept strings.
        """
        key = concept.lower().replace(" ", "_")
        if key in self._kb:
            return self._kb[key].get(relation, [])
        return []

    def infer_cooking_purpose(self, ingredients: List[str]) -> Dict:
        """Infer what dish might be made from a list of ingredients.

        Returns:
            Dict with possible_dishes and confidence.
        """
        ingredient_set = set(i.lower() for i in ingredients)

        # Simple pattern matching
        dish_scores = {
            "salad": len(ingredient_set & {"tomato", "lettuce", "cucumber", "onion", "olive_oil"}),
            "stir_fry": len(ingredient_set & {"oil", "onion", "garlic", "vegetable", "soy_sauce"}),
            "pasta_dish": len(ingredient_set & {"pasta", "tomato", "garlic", "olive_oil", "basil"}),
            "omelette": len(ingredient_set & {"egg", "butter", "salt", "pepper"}),
            "soup": len(ingredient_set & {"water", "onion", "carrot", "potato", "salt"}),
            "sandwich": len(ingredient_set & {"bread", "cheese", "tomato", "lettuce"}),
        }

        if not any(dish_scores.values()):
            return {"possible_dishes": [], "confidence": 0}

        best = max(dish_scores, key=dish_scores.get)
        return {
            "possible_dishes": [best],
            "confidence": min(1.0, dish_scores[best] / 3.0),
        }

    def get_next_actions(self, current_action: str) -> List[str]:
        """Get typical next actions after the current action."""
        return ACTION_SEQUENCES.get(current_action.lower(), [])
