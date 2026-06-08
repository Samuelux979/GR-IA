"""
Tests de format_recipes_response() (src/vision/llava_client.py).
Verifican que el texto de presentacion tiene un formato estable.
La traduccion se mockea para no cargar el modelo MarianMT.
"""

from unittest.mock import patch

from src.vision import llava_client
from src.vision.llava_client import format_recipes_response


CLASSIFIED = {
    "main":          ["chicken"],
    "secondary":     ["onion"],
    "accompaniment": [],
    "spices":        [],
}

RECIPE = {
    "title":       "Chicken Soup",
    "ner":         ["chicken", "carrot", "onion"],
    "ingredients": ["2 chicken legs", "1 onion"],
    "chunk_text":  "Recipe: Chicken Soup. Ingredients: chicken, onion. Step 1: Boil water. Step 2: Add chicken.",
}


def test_no_recipes_returns_message():
    out = format_recipes_response(CLASSIFIED, [])
    assert out == "No se encontraron recetas para los ingredientes detectados."


def test_lists_detected_ingredients_in_intro():
    # translate_steps_to_es se mockea como identidad
    with patch("src.vision.translator.translate_steps_to_es", side_effect=lambda s: s):
        out = format_recipes_response(CLASSIFIED, [RECIPE])
    assert "chicken" in out
    assert "onion" in out


def test_shows_matches_between_detected_and_ner():
    with patch("src.vision.translator.translate_steps_to_es", side_effect=lambda s: s):
        out = format_recipes_response(CLASSIFIED, [RECIPE])
    # 'chicken' y 'onion' estan tanto en lo detectado como en el ner de la receta
    assert "Coincide con tu foto:" in out
    assert "chicken" in out and "onion" in out
    # 'carrot' esta en el ner pero NO fue detectado -> no es coincidencia
    assert "carrot" not in out.split("Coincide con tu foto:")[1].split("\n")[0]


def test_recipe_ingredients_appear():
    with patch("src.vision.translator.translate_steps_to_es", side_effect=lambda s: s):
        out = format_recipes_response(CLASSIFIED, [RECIPE])
    assert "2 chicken legs" in out
    assert "1 onion" in out


def test_steps_are_numbered_as_paso():
    with patch("src.vision.translator.translate_steps_to_es", side_effect=lambda s: s):
        out = format_recipes_response(CLASSIFIED, [RECIPE])
    assert "Paso 1." in out
    assert "Paso 2." in out
    assert "Boil water." in out


def test_translation_is_invoked_for_steps():
    # Verifica que los pasos pasan por la funcion de traduccion
    with patch("src.vision.translator.translate_steps_to_es", side_effect=lambda s: s) as mock_tr:
        format_recipes_response(CLASSIFIED, [RECIPE])
    mock_tr.assert_called_once()


def test_uses_structured_steps_when_available():
    # Si la receta trae 'steps' estructurados, se usan directamente
    recipe = {**RECIPE, "steps": ["Estructurado uno.", "Estructurado dos."],
              "chunk_text": "Recipe: X. Step 1: Del chunk."}
    with patch("src.vision.translator.translate_steps_to_es", side_effect=lambda s: s):
        out = format_recipes_response(CLASSIFIED, [recipe])
    assert "Estructurado uno." in out
    assert "Del chunk." not in out          # no se usa el fallback


def test_falls_back_to_chunk_text_without_structured_steps():
    # Sin 'steps', se parsea desde chunk_text (compatibilidad con datos antiguos)
    recipe = {k: v for k, v in RECIPE.items()}   # RECIPE no tiene 'steps'
    with patch("src.vision.translator.translate_steps_to_es", side_effect=lambda s: s):
        out = format_recipes_response(CLASSIFIED, [recipe])
    assert "Boil water." in out             # extraido de chunk_text
