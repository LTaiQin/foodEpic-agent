"""Scene Graph Generator for video frames.

Inspired by SceneNet: use MLLM to generate structured scene graphs
capturing objects, attributes, spatial/temporal relationships.
"""

import json
from typing import Dict, List, Optional


class SceneGraphGenerator:
    """Generate scene graphs from video frames using MLLM."""

    SCENE_GRAPH_PROMPT = """Analyze this kitchen scene image and generate a structured scene graph.

Return a JSON object:
{
  "objects": [
    {
      "name": "object_name",
      "type": "fixture|utensil|ingredient|appliance|container",
      "attributes": ["attr1", "attr2"],
      "state": "open|closed|empty|full|raw|cooked|chopped|...",
      "position": "on counter|in drawer|on stove|..."
    }
  ],
  "spatial_relations": [
    {
      "subject": "object1",
      "relation": "on|in|next_to|above|below|left_of|right_of|in_front_of|behind",
      "object": "object2"
    }
  ],
  "actions": [
    {
      "agent": "person",
      "action": "holding|cutting|stirring|pouring|...",
      "target": "object_name",
      "tool": "tool_name"
    }
  ],
  "scene_description": "One sentence describing the overall scene"
}

Be specific and accurate. Only include objects and relations you can clearly see."""

    @staticmethod
    def generate(mimo_client, frame, context: str = "") -> Dict:
        """Generate a scene graph from a single frame."""
        prompt = SceneGraphGenerator.SCENE_GRAPH_PROMPT
        if context:
            prompt += f"\n\nContext: {context}"

        response = mimo_client.call_vision(frame, prompt)
        if isinstance(response, list):
            response = response[0] if response else ""
        response = str(response)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        return {"raw_response": response[:500]}

    @staticmethod
    def extract_objects(graph: Dict) -> List[str]:
        """Extract object names from scene graph."""
        return [obj.get("name", "") for obj in graph.get("objects", []) if obj.get("name")]

    @staticmethod
    def extract_relations(graph: Dict) -> List[Dict]:
        """Extract spatial relations."""
        return graph.get("spatial_relations", [])

    @staticmethod
    def extract_actions(graph: Dict) -> List[Dict]:
        """Extract actions."""
        return graph.get("actions", [])

    @staticmethod
    def find_object(graph: Dict, name: str) -> Optional[Dict]:
        """Find object by name."""
        for obj in graph.get("objects", []):
            if name.lower() in obj.get("name", "").lower():
                return obj
        return None

    @staticmethod
    def get_object_state(graph: Dict, name: str) -> str:
        """Get state of an object."""
        obj = SceneGraphGenerator.find_object(graph, name)
        return obj.get("state", "unknown") if obj else "unknown"

    @staticmethod
    def format_for_prompt(graph: Dict) -> str:
        """Format scene graph as readable text for LLM."""
        lines = []

        objects = graph.get("objects", [])
        if objects:
            lines.append("Objects:")
            for obj in objects:
                desc = f"  - {obj.get('name', '?')}"
                if obj.get("state"):
                    desc += f" [{obj['state']}]"
                if obj.get("position"):
                    desc += f" at {obj['position']}"
                lines.append(desc)

        relations = graph.get("spatial_relations", [])
        if relations:
            lines.append("\nSpatial relations:")
            for rel in relations:
                lines.append(f"  - {rel.get('subject', '?')} {rel.get('relation', '?')} {rel.get('object', '?')}")

        actions = graph.get("actions", [])
        if actions:
            lines.append("\nActions:")
            for act in actions:
                lines.append(f"  - {act.get('agent', 'person')} {act.get('action', '?')} {act.get('target', '?')}")

        desc = graph.get("scene_description", "")
        if desc:
            lines.append(f"\nScene: {desc}")

        return "\n".join(lines) if lines else "No scene graph available."


class ConceptNetKB:
    """ConceptNet-style knowledge base for kitchen objects."""

    KNOWLEDGE = {
        # Fixtures
        "sink": {"used_for": ["washing", "rinsing"], "has_property": ["metal", "has faucet"], "at_location": ["counter"]},
        "stove": {"used_for": ["cooking", "heating"], "has_property": ["hot", "has burners"], "at_location": ["counter"]},
        "oven": {"used_for": ["baking", "roasting"], "has_property": ["hot", "enclosed"], "at_location": ["counter"]},
        "fridge": {"used_for": ["storing food", "keeping cold"], "has_property": ["cold", "has door"], "at_location": ["kitchen"]},
        "microwave": {"used_for": ["heating", "reheating"], "has_property": ["electric", "has door"], "at_location": ["counter"]},
        "dishwasher": {"used_for": ["washing dishes"], "has_property": ["electric", "has door"], "at_location": ["counter"]},
        "counter": {"used_for": ["preparation", "placing items"], "has_property": ["flat", "hard surface"], "at_location": ["kitchen"]},
        "cupboard": {"used_for": ["storing items"], "has_property": ["has door", "has shelves"], "at_location": ["wall"]},
        "drawer": {"used_for": ["storing utensils"], "has_property": ["slides open", "has handle"], "at_location": ["counter"]},

        # Utensils
        "knife": {"used_for": ["cutting", "chopping", "slicing"], "has_property": ["sharp", "metal"], "at_location": ["drawer", "knife block"]},
        "spoon": {"used_for": ["stirring", "scooping", "tasting"], "has_property": ["round", "concave"], "at_location": ["drawer"]},
        "fork": {"used_for": ["eating", "piercing"], "has_property": ["pointed", "pronged"], "at_location": ["drawer"]},
        "spatula": {"used_for": ["flipping", "scraping"], "has_property": ["flat", "flexible"], "at_location": ["drawer"]},
        "whisk": {"used_for": ["mixing", "beating"], "has_property": ["wire loops"], "at_location": ["drawer"]},
        "ladle": {"used_for": ["serving soup", "scooping"], "has_property": ["deep bowl", "long handle"], "at_location": ["drawer"]},
        "tongs": {"used_for": ["gripping", "flipping"], "has_property": ["two arms", "spring"], "at_location": ["drawer"]},

        # Cookware
        "pot": {"used_for": ["boiling", "cooking soup"], "has_property": ["deep", "metal", "has handles"], "at_location": ["stove", "cupboard"]},
        "pan": {"used_for": ["frying", "sautéing"], "has_property": ["flat", "metal", "has handle"], "at_location": ["stove", "cupboard"]},
        "saucepan": {"used_for": ["making sauce", "boiling"], "has_property": ["deep", "has handle", "has lid"], "at_location": ["stove"]},
        "wok": {"used_for": ["stir-frying"], "has_property": ["round bottom", "large"], "at_location": ["stove"]},
        "baking sheet": {"used_for": ["baking", "roasting"], "has_property": ["flat", "metal"], "at_location": ["oven"]},

        # Ingredients
        "onion": {"used_for": ["cooking", "flavoring"], "has_property": ["vegetable", "round", "pungent"], "requires": ["peeling", "chopping"]},
        "tomato": {"used_for": ["cooking", "salad", "sauce"], "has_property": ["vegetable", "red", "round"], "requires": ["washing", "cutting"]},
        "garlic": {"used_for": ["cooking", "flavoring"], "has_property": ["vegetable", "small", "pungent"], "requires": ["peeling", "chopping"]},
        "potato": {"used_for": ["cooking", "frying", "baking"], "has_property": ["vegetable", "brown", "starchy"], "requires": ["peeling", "cutting"]},
        "carrot": {"used_for": ["cooking", "salad"], "has_property": ["vegetable", "orange", "long"], "requires": ["peeling", "cutting"]},
        "chicken": {"used_for": ["cooking", "grilling", "frying"], "has_property": ["meat", "protein"], "requires": ["washing", "cutting"]},
        "rice": {"used_for": ["cooking", "side dish"], "has_property": ["grain", "white"], "requires": ["washing", "boiling"]},
        "pasta": {"used_for": ["cooking", "main dish"], "has_property": ["grain", "dried"], "requires": ["boiling"]},
        "oil": {"used_for": ["cooking", "frying", "seasoning"], "has_property": ["liquid", "fatty"], "at_location": ["counter", "cupboard"]},
        "salt": {"used_for": ["seasoning"], "has_property": ["white", "crystalline"], "at_location": ["shaker", "counter"]},
        "pepper": {"used_for": ["seasoning"], "has_property": ["black", "powdery"], "at_location": ["shaker", "counter"]},
        "butter": {"used_for": ["cooking", "baking"], "has_property": ["dairy", "yellow", "fat"], "at_location": ["fridge", "counter"]},
        "egg": {"used_for": ["cooking", "baking"], "has_property": ["protein", "oval", "shell"], "at_location": ["fridge"]},
        "cheese": {"used_for": ["cooking", "topping"], "has_property": ["dairy", "yellow|white"], "at_location": ["fridge"]},
        "milk": {"used_for": ["cooking", "drinking"], "has_property": ["dairy", "liquid", "white"], "at_location": ["fridge"]},
        "flour": {"used_for": ["baking", "thickening"], "has_property": ["powder", "white"], "at_location": ["cupboard"]},
        "sugar": {"used_for": ["sweetening", "baking"], "has_property": ["sweet", "crystalline"], "at_location": ["cupboard"]},

        # Actions and their typical sequences
        "chopping": {"requires": ["knife", "cutting board"], "follows": ["washing"], "precedes": ["cooking", "adding to pot"]},
        "stirring": {"requires": ["spoon"], "follows": ["adding ingredients"], "precedes": ["tasting", "serving"]},
        "frying": {"requires": ["pan", "oil"], "follows": ["cutting", "seasoning"], "precedes": ["plating"]},
        "boiling": {"requires": ["pot", "water"], "follows": ["cutting"], "precedes": ["draining", "serving"]},
        "baking": {"requires": ["oven", "baking sheet"], "follows": ["preparing", "seasoning"], "precedes": ["cooling", "serving"]},
        "seasoning": {"requires": ["salt", "pepper"], "follows": ["cooking"], "precedes": ["tasting", "serving"]},
        "plating": {"requires": ["plate"], "follows": ["cooking"], "precedes": ["serving"]},
    }

    @classmethod
    def get_knowledge(cls, object_name: str) -> Dict:
        """Get knowledge for an object."""
        name_lower = object_name.lower().strip()

        if name_lower in cls.KNOWLEDGE:
            return cls.KNOWLEDGE[name_lower]

        # Partial match
        for key in cls.KNOWLEDGE:
            if key in name_lower or name_lower in key:
                return cls.KNOWLEDGE[key]

        return {}

    @classmethod
    def format_for_prompt(cls, object_name: str) -> str:
        """Format knowledge as readable text."""
        knowledge = cls.get_knowledge(object_name)
        if not knowledge:
            return ""

        lines = [f"Knowledge about {object_name}:"]
        for key, values in knowledge.items():
            if isinstance(values, list):
                lines.append(f"  {key}: {', '.join(values)}")
            else:
                lines.append(f"  {key}: {values}")
        return "\n".join(lines)

    @classmethod
    def get_action_sequence(cls, action: str) -> Dict:
        """Get typical action sequence."""
        action_lower = action.lower().strip()
        if action_lower in cls.KNOWLEDGE:
            return cls.KNOWLEDGE[action_lower]
        return {}

    @classmethod
    def get_related_actions(cls, action: str) -> List[str]:
        """Get actions that typically follow the given action."""
        action_lower = action.lower().strip()
        if action_lower in cls.KNOWLEDGE:
            return cls.KNOWLEDGE[action_lower].get("follows", [])
        return []
