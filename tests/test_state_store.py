from food_agent.state_store import _to_float, FoodStateStore


def test_to_float() -> None:
    assert _to_float("1.5") == 1.5
    assert _to_float("N/A") is None
    assert _to_float(None) is None


def test_recipe_and_ingredient_state() -> None:
    store = FoodStateStore()
    recipe = store.recipe_state("P01-20240202-110250", 16.8)
    ingredient = store.ingredient_state("P01-20240202-110250", 18.0)
    assert recipe.active_steps
    assert any(row["label"] == "nespresso capsule" for row in ingredient.added)


def test_nutrition_delta_handles_unknown_values() -> None:
    store = FoodStateStore()
    nutrition = store.nutrition_delta("P01-20240202-110250", 18.0)
    assert nutrition.evidence_ids
    assert nutrition.unknown_count > 0

