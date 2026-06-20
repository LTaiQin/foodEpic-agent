"""External knowledge tools for the food agent.

These tools provide external knowledge that cannot be inferred from video alone:
- Ingredient typical weights
- Nutrition changes during cooking
- Cooking method effects
- Common kitchen object knowledge
"""

from typing import Dict, List, Optional


class IngredientKnowledgeBase:
    """Knowledge base for ingredient properties."""

    # Typical weights for common ingredients (in grams)
    TYPICAL_WEIGHTS = {
        # Vegetables
        "onion": {"small": 100, "medium": 150, "large": 250},
        "red onion": {"small": 100, "medium": 150, "large": 250},
        "brown onion": {"small": 100, "medium": 150, "large": 250},
        "garlic": {"clove": 5, "head": 40},
        "tomato": {"small": 80, "medium": 150, "large": 250},
        "cherry tomato": {"single": 15},
        "potato": {"small": 100, "medium": 200, "large": 350},
        "carrot": {"small": 50, "medium": 100, "large": 200},
        "celery": {"stalk": 40, "bunch": 400},
        "bell pepper": {"small": 100, "medium": 175, "large": 250},
        "cucumber": {"small": 150, "medium": 300, "large": 500},
        "zucchini": {"small": 150, "medium": 250, "large": 400},
        "eggplant": {"small": 200, "medium": 350, "large": 500},
        "mushroom": {"single": 20, "cup_sliced": 70},
        "broccoli": {"head": 500, "cup_chopped": 90},
        "cauliflower": {"head": 600, "cup_chopped": 100},
        "spinach": {"cup_raw": 30, "cup_cooked": 180},
        "lettuce": {"head": 400, "cup_shredded": 50},
        "cabbage": {"head": 900, "cup_shredded": 70},
        "corn": {"ear": 200, "cup_kernels": 150},
        "peas": {"cup": 150},
        "green beans": {"cup": 125},
        "asparagus": {"bunch": 500, "spear": 20},
        "avocado": {"small": 150, "medium": 200, "large": 300},
        "lemon": {"small": 60, "medium": 100, "large": 150},
        "lime": {"small": 40, "medium": 70, "large": 100},
        "ginger": {"inch": 10, "root": 50},
        "chili": {"single": 15},
        "jalapeno": {"single": 15},

        # Proteins
        "chicken breast": {"small": 150, "medium": 200, "large": 300},
        "chicken thigh": {"small": 100, "medium": 150, "large": 200},
        "chicken leg": {"small": 150, "medium": 200, "large": 250},
        "beef steak": {"small": 150, "medium": 250, "large": 400},
        "ground beef": {"cup": 225, "pound": 450},
        "pork chop": {"small": 150, "medium": 250, "large": 350},
        "pork tenderloin": {"pound": 450},
        "salmon fillet": {"small": 150, "medium": 200, "large": 300},
        "shrimp": {"large": 20, "cup": 150},
        "egg": {"large": 60, "white": 33, "yolk": 17},
        "tofu": {"block": 400, "cup": 250},

        # Dairy
        "butter": {"tablespoon": 14, "stick": 113, "cup": 227},
        "cheese": {"cup_shredded": 113, "slice": 28, "ounce": 28},
        "cheddar": {"cup_shredded": 113, "slice": 28},
        "mozzarella": {"cup_shredded": 113, "ball": 125},
        "parmesan": {"cup_grated": 100, "tablespoon": 5},
        "cream cheese": {"tablespoon": 15, "block": 225},
        "sour cream": {"cup": 230, "tablespoon": 15},
        "yogurt": {"cup": 245, "tablespoon": 15},
        "milk": {"cup": 240, "tablespoon": 15},
        "heavy cream": {"cup": 240, "tablespoon": 15},

        # Grains & Pasta
        "rice": {"cup_dry": 185, "cup_cooked": 200},
        "pasta": {"cup_dry": 100, "cup_cooked": 200},
        "spaghetti": {"ounce": 28, "pound": 450},
        "bread": {"slice": 30, "loaf": 500},
        "flour": {"cup": 125, "tablespoon": 8},
        "sugar": {"cup": 200, "tablespoon": 12},
        "brown sugar": {"cup": 220, "tablespoon": 14},
        "oats": {"cup": 80},
        "quinoa": {"cup_dry": 170, "cup_cooked": 185},

        # Oils & Liquids
        "olive oil": {"tablespoon": 14, "cup": 216},
        "vegetable oil": {"tablespoon": 14, "cup": 218},
        "coconut oil": {"tablespoon": 14, "cup": 218},
        "sesame oil": {"tablespoon": 14},
        "soy sauce": {"tablespoon": 16, "cup": 255},
        "vinegar": {"tablespoon": 15, "cup": 239},
        "water": {"cup": 240, "tablespoon": 15},
        "broth": {"cup": 240},
        "stock": {"cup": 240},
        "wine": {"cup": 240, "glass": 150},

        # Nuts & Seeds
        "almonds": {"cup": 143, "ounce": 28},
        "walnuts": {"cup": 120, "ounce": 28},
        "peanuts": {"cup": 146, "ounce": 28},
        "cashews": {"cup": 137, "ounce": 28},
        "sesame seeds": {"tablespoon": 9, "cup": 144},
        "sunflower seeds": {"cup": 140},

        # Spices & Herbs (small quantities)
        "salt": {"teaspoon": 6, "tablespoon": 18},
        "pepper": {"teaspoon": 2},
        "cumin": {"teaspoon": 2},
        "paprika": {"teaspoon": 2},
        "cinnamon": {"teaspoon": 3},
        "oregano": {"teaspoon": 1},
        "basil": {"teaspoon": 1, "tablespoon": 3},
        "thyme": {"teaspoon": 1},
        "rosemary": {"teaspoon": 1},
        "parsley": {"tablespoon": 4, "cup": 60},
        "cilantro": {"tablespoon": 4, "cup": 60},
        "dill": {"tablespoon": 4},
        "mint": {"tablespoon": 4},
        "turmeric": {"teaspoon": 3},
        "chili powder": {"teaspoon": 3},
        "garam masala": {"teaspoon": 3},
        "curry powder": {"tablespoon": 7},
        "bay leaf": {"single": 0.5},

        # Canned/Packaged
        "canned tomatoes": {"can": 400, "cup": 240},
        "tomato paste": {"tablespoon": 16, "can": 170},
        "tomato sauce": {"cup": 245},
        "coconut milk": {"cup": 240, "can": 400},
        "chickpeas": {"cup_cooked": 164, "can": 400},
        "lentils": {"cup_cooked": 200},
        "beans": {"cup_cooked": 170, "can": 400},
    }

    # Nutrition per 100g for common ingredients
    NUTRITION_PER_100G = {
        "chicken breast": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0},
        "rice": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28},
        "pasta": {"calories": 131, "protein": 5, "fat": 1.1, "carbs": 25},
        "olive oil": {"calories": 884, "protein": 0, "fat": 100, "carbs": 0},
        "butter": {"calories": 717, "protein": 0.9, "fat": 81, "carbs": 0.1},
        "egg": {"calories": 155, "protein": 13, "fat": 11, "carbs": 1.1},
        "onion": {"calories": 40, "protein": 1.1, "fat": 0.1, "carbs": 9.3},
        "tomato": {"calories": 18, "protein": 0.9, "fat": 0.2, "carbs": 3.9},
        "potato": {"calories": 77, "protein": 2, "fat": 0.1, "carbs": 17},
        "carrot": {"calories": 41, "protein": 0.9, "fat": 0.2, "carbs": 10},
        "garlic": {"calories": 149, "protein": 6.4, "fat": 0.5, "carbs": 33},
        "cheese": {"calories": 402, "protein": 25, "fat": 33, "carbs": 1.3},
        "milk": {"calories": 42, "protein": 3.4, "fat": 1, "carbs": 5},
        "bread": {"calories": 265, "protein": 9, "fat": 3.2, "carbs": 49},
        "sugar": {"calories": 387, "protein": 0, "fat": 0, "carbs": 100},
        "flour": {"calories": 364, "protein": 10, "fat": 1, "carbs": 76},
        "banana": {"calories": 89, "protein": 1.1, "fat": 0.3, "carbs": 23},
        "apple": {"calories": 52, "protein": 0.3, "fat": 0.2, "carbs": 14},
        "avocado": {"calories": 160, "protein": 2, "fat": 15, "carbs": 9},
        "salmon": {"calories": 208, "protein": 20, "fat": 13, "carbs": 0},
        "beef": {"calories": 250, "protein": 26, "fat": 15, "carbs": 0},
        "pork": {"calories": 242, "protein": 27, "fat": 14, "carbs": 0},
        "tofu": {"calories": 76, "protein": 8, "fat": 4.8, "carbs": 1.9},
        "spinach": {"calories": 23, "protein": 2.9, "fat": 0.4, "carbs": 3.6},
        "broccoli": {"calories": 34, "protein": 2.8, "fat": 0.4, "carbs": 7},
        "mushroom": {"calories": 22, "protein": 3.1, "fat": 0.3, "carbs": 3.3},
        "corn": {"calories": 86, "protein": 3.2, "fat": 1.2, "carbs": 19},
        "peas": {"calories": 81, "protein": 5, "fat": 0.4, "carbs": 14},
        "coconut milk": {"calories": 230, "protein": 2.3, "fat": 24, "carbs": 6},
        "soy sauce": {"calories": 53, "protein": 8, "fat": 0, "carbs": 5},
        "honey": {"calories": 304, "protein": 0.3, "fat": 0, "carbs": 82},
        "lemon juice": {"calories": 22, "protein": 0.4, "fat": 0, "carbs": 7},
    }

    @classmethod
    def get_typical_weight(cls, ingredient: str, size: str = "medium") -> Optional[int]:
        """Get typical weight for an ingredient."""
        ing_lower = ingredient.lower().strip()

        # Direct match
        if ing_lower in cls.TYPICAL_WEIGHTS:
            sizes = cls.TYPICAL_WEIGHTS[ing_lower]
            if size in sizes:
                return sizes[size]
            # Return medium or first available
            return sizes.get("medium", sizes.get(list(sizes.keys())[0]))

        # Partial match
        for key in cls.TYPICAL_WEIGHTS:
            if key in ing_lower or ing_lower in key:
                sizes = cls.TYPICAL_WEIGHTS[key]
                if size in sizes:
                    return sizes[size]
                return sizes.get("medium", sizes.get(list(sizes.keys())[0]))

        return None

    @classmethod
    def get_nutrition(cls, ingredient: str, weight_g: float = 100) -> Optional[Dict]:
        """Get nutrition info for an ingredient."""
        ing_lower = ingredient.lower().strip()

        # Direct match
        if ing_lower in cls.NUTRITION_PER_100G:
            base = cls.NUTRITION_PER_100G[ing_lower]
            factor = weight_g / 100
            return {
                "calories": round(base["calories"] * factor, 1),
                "protein": round(base["protein"] * factor, 1),
                "fat": round(base["fat"] * factor, 1),
                "carbs": round(base["carbs"] * factor, 1),
            }

        # Partial match
        for key in cls.NUTRITION_PER_100G:
            if key in ing_lower or ing_lower in key:
                base = cls.NUTRITION_PER_100G[key]
                factor = weight_g / 100
                return {
                    "calories": round(base["calories"] * factor, 1),
                    "protein": round(base["protein"] * factor, 1),
                    "fat": round(base["fat"] * factor, 1),
                    "carbs": round(base["carbs"] * factor, 1),
                }

        return None

    @classmethod
    def estimate_weight_from_description(cls, description: str) -> Optional[int]:
        """Estimate weight from a description like '1 medium onion'."""
        import re

        description = description.lower().strip()

        # Extract quantity and size
        quantity_match = re.search(r'(\d+(?:\.\d+)?)', description)
        quantity = float(quantity_match.group(1)) if quantity_match else 1

        size = "medium"
        if "small" in description:
            size = "small"
        elif "large" in description:
            size = "large"
        elif "cup" in description:
            size = "cup"
        elif "tablespoon" in description or "tbsp" in description:
            size = "tablespoon"
        elif "teaspoon" in description or "tsp" in description:
            size = "teaspoon"

        # Find ingredient
        for ingredient in cls.TYPICAL_WEIGHTS:
            if ingredient in description:
                weight = cls.get_typical_weight(ingredient, size)
                if weight:
                    return int(weight * quantity)

        return None


class CookingKnowledgeBase:
    """Knowledge base for cooking methods and their effects."""

    # How cooking methods affect nutrition (percentage change)
    COOKING_EFFECTS = {
        "frying": {
            "calories": "+20-50%",  # Oil absorption
            "fat": "+30-100%",
            "vitamin_c": "-30-50%",
            "vitamin_b": "-20-40%",
            "protein": "0%",
            "carbs": "0%",
        },
        "boiling": {
            "calories": "-5-10%",
            "vitamin_c": "-40-60%",
            "vitamin_b": "-30-50%",
            "minerals": "-20-40%",  # Leaching
            "protein": "0%",
            "carbs": "0%",
        },
        "steaming": {
            "calories": "0%",
            "vitamin_c": "-10-20%",
            "vitamin_b": "-10-20%",
            "protein": "0%",
            "carbs": "0%",
        },
        "baking": {
            "calories": "0%",
            "vitamin_c": "-20-30%",
            "vitamin_b": "-20-30%",
            "protein": "0%",
            "carbs": "0%",
        },
        "roasting": {
            "calories": "0%",
            "vitamin_c": "-20-30%",
            "vitamin_b": "-20-30%",
            "protein": "0%",
            "carbs": "0%",
        },
        "grilling": {
            "calories": "-5-10%",  # Fat drips off
            "fat": "-10-20%",
            "vitamin_c": "-20-30%",
            "protein": "0%",
            "carbs": "0%",
        },
        "microwaving": {
            "calories": "0%",
            "vitamin_c": "-10-20%",
            "vitamin_b": "-10-20%",
            "protein": "0%",
            "carbs": "0%",
        },
        "stir_frying": {
            "calories": "+10-30%",
            "fat": "+20-50%",
            "vitamin_c": "-20-40%",
            "protein": "0%",
            "carbs": "0%",
        },
        "braising": {
            "calories": "+5-15%",
            "vitamin_c": "-30-50%",
            "vitamin_b": "-20-40%",
            "protein": "0%",
            "carbs": "0%",
        },
        "raw": {
            "calories": "0%",
            "vitamin_c": "0%",
            "vitamin_b": "0%",
            "protein": "0%",
            "carbs": "0%",
        },
    }

    # Common cooking actions and what they imply
    COOKING_ACTIONS = {
        "chopping": {"tool": "knife", "object": "ingredient"},
        "slicing": {"tool": "knife", "object": "ingredient"},
        "dicing": {"tool": "knife", "object": "ingredient"},
        "mincing": {"tool": "knife", "object": "ingredient"},
        "grating": {"tool": "grater", "object": "ingredient"},
        "peeling": {"tool": "peeler", "object": "ingredient"},
        "stirring": {"tool": "spoon", "object": "pot/bowl"},
        "mixing": {"tool": "spoon/whisk", "object": "bowl"},
        "whisking": {"tool": "whisk", "object": "bowl"},
        "beating": {"tool": "whisk/mixer", "object": "bowl"},
        "kneading": {"tool": "hands", "object": "dough"},
        "rolling": {"tool": "rolling pin", "object": "dough"},
        "pouring": {"tool": "container", "object": "pot/pan"},
        "adding": {"tool": "hands/spoon", "object": "ingredient"},
        "seasoning": {"tool": "shaker/hands", "object": "spice"},
        "frying": {"tool": "pan", "object": "ingredient"},
        "boiling": {"tool": "pot", "object": "water/ingredient"},
        "baking": {"tool": "oven", "object": "dish"},
        "roasting": {"tool": "oven", "object": "ingredient"},
        "grilling": {"tool": "grill", "object": "ingredient"},
        "microwaving": {"tool": "microwave", "object": "dish"},
        "plating": {"tool": "plate", "object": "food"},
        "serving": {"tool": "plate/bowl", "object": "food"},
    }

    @classmethod
    def get_cooking_effect(cls, method: str) -> Optional[Dict]:
        """Get the nutritional effect of a cooking method."""
        method_lower = method.lower().strip()

        # Direct match
        if method_lower in cls.COOKING_EFFECTS:
            return cls.COOKING_EFFECTS[method_lower]

        # Partial match
        for key in cls.COOKING_EFFECTS:
            if key in method_lower or method_lower in key:
                return cls.COOKING_EFFECTS[key]

        return None

    @classmethod
    def get_action_info(cls, action: str) -> Optional[Dict]:
        """Get information about a cooking action."""
        action_lower = action.lower().strip()

        # Direct match
        if action_lower in cls.COOKING_ACTIONS:
            return cls.COOKING_ACTIONS[action_lower]

        # Partial match
        for key in cls.COOKING_ACTIONS:
            if key in action_lower or action_lower in key:
                return cls.COOKING_ACTIONS[key]

        return None


class KitchenObjectKnowledgeBase:
    """Knowledge base for common kitchen objects and their properties."""

    OBJECT_PROPERTIES = {
        # Fixtures
        "sink": {"type": "fixture", "function": "washing", "location": "counter"},
        "stove": {"type": "fixture", "function": "cooking", "location": "counter"},
        "hob": {"type": "fixture", "function": "cooking", "location": "counter"},
        "oven": {"type": "fixture", "function": "baking", "location": "counter"},
        "microwave": {"type": "appliance", "function": "heating", "location": "counter"},
        "dishwasher": {"type": "fixture", "function": "cleaning", "location": "counter"},
        "fridge": {"type": "fixture", "function": "storage", "location": "counter"},
        "freezer": {"type": "fixture", "function": "storage", "location": "counter"},
        "counter": {"type": "surface", "function": "prep", "location": "kitchen"},
        "cupboard": {"type": "storage", "function": "storage", "location": "wall"},
        "drawer": {"type": "storage", "function": "storage", "location": "counter"},
        "shelf": {"type": "storage", "function": "storage", "location": "wall"},

        # Utensils
        "knife": {"type": "utensil", "function": "cutting", "storage": "drawer/block"},
        "spoon": {"type": "utensil", "function": "stirring", "storage": "drawer"},
        "fork": {"type": "utensil", "function": "eating", "storage": "drawer"},
        "whisk": {"type": "utensil", "function": "mixing", "storage": "drawer"},
        "spatula": {"type": "utensil", "function": "flipping", "storage": "drawer"},
        "ladle": {"type": "utensil", "function": "serving", "storage": "drawer"},
        "tongs": {"type": "utensil", "function": "gripping", "storage": "drawer"},
        "peeler": {"type": "utensil", "function": "peeling", "storage": "drawer"},
        "grater": {"type": "utensil", "function": "grating", "storage": "drawer"},
        "colander": {"type": "utensil", "function": "draining", "storage": "cupboard"},
        "measuring cup": {"type": "utensil", "function": "measuring", "storage": "cupboard"},
        "measuring spoon": {"type": "utensil", "function": "measuring", "storage": "drawer"},
        "rolling pin": {"type": "utensil", "function": "rolling", "storage": "drawer"},
        "can opener": {"type": "utensil", "function": "opening", "storage": "drawer"},
        "bottle opener": {"type": "utensil", "function": "opening", "storage": "drawer"},
        "corkscrew": {"type": "utensil", "function": "opening", "storage": "drawer"},

        # Cookware
        "pan": {"type": "cookware", "function": "frying", "storage": "cupboard"},
        "pot": {"type": "cookware", "function": "boiling", "storage": "cupboard"},
        "saucepan": {"type": "cookware", "function": "sauce", "storage": "cupboard"},
        "skillet": {"type": "cookware", "function": "frying", "storage": "cupboard"},
        "wok": {"type": "cookware", "function": "stir_frying", "storage": "cupboard"},
        "baking sheet": {"type": "cookware", "function": "baking", "storage": "cupboard"},
        "baking dish": {"type": "cookware", "function": "baking", "storage": "cupboard"},
        "roasting pan": {"type": "cookware", "function": "roasting", "storage": "cupboard"},
        "casserole": {"type": "cookware", "function": "braising", "storage": "cupboard"},
        "mixing bowl": {"type": "dishware", "function": "mixing", "storage": "cupboard"},
        "cutting board": {"type": "surface", "function": "prep", "storage": "counter"},
        "plate": {"type": "dishware", "function": "serving", "storage": "cupboard"},
        "bowl": {"type": "dishware", "function": "serving", "storage": "cupboard"},
        "cup": {"type": "dishware", "function": "drinking", "storage": "cupboard"},
        "glass": {"type": "dishware", "function": "drinking", "storage": "cupboard"},
        "mug": {"type": "dishware", "function": "drinking", "storage": "cupboard"},

        # Appliances
        "blender": {"type": "appliance", "function": "blending", "storage": "counter"},
        "food processor": {"type": "appliance", "function": "processing", "storage": "cupboard"},
        "mixer": {"type": "appliance", "function": "mixing", "storage": "cupboard"},
        "toaster": {"type": "appliance", "function": "toasting", "storage": "counter"},
        "kettle": {"type": "appliance", "function": "boiling", "storage": "counter"},
        "coffee maker": {"type": "appliance", "function": "brewing", "storage": "counter"},
        "scale": {"type": "appliance", "function": "weighing", "storage": "counter"},
        "timer": {"type": "appliance", "function": "timing", "storage": "counter"},
    }

    @classmethod
    def get_object_info(cls, object_name: str) -> Optional[Dict]:
        """Get information about a kitchen object."""
        obj_lower = object_name.lower().strip()

        # Direct match
        if obj_lower in cls.OBJECT_PROPERTIES:
            return cls.OBJECT_PROPERTIES[obj_lower]

        # Partial match
        for key in cls.OBJECT_PROPERTIES:
            if key in obj_lower or obj_lower in key:
                return cls.OBJECT_PROPERTIES[key]

        return None

    @classmethod
    def get_objects_by_type(cls, obj_type: str) -> List[str]:
        """Get all objects of a given type."""
        return [k for k, v in cls.OBJECT_PROPERTIES.items() if v.get("type") == obj_type]

    @classmethod
    def get_objects_by_function(cls, function: str) -> List[str]:
        """Get all objects with a given function."""
        return [k for k, v in cls.OBJECT_PROPERTIES.items() if v.get("function") == function]
