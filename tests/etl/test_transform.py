"""
Tests de las transformaciones de Food.com a RecipeDocument (src/etl/transform.py).
Aseguran que los datos cargados en PostgreSQL mantienen un formato correcto.
"""

import math

import pandas as pd

from src.etl.transform import (
    _sanitize,
    _to_list,
    _build_ingredients,
    _build_chunk_text,
    transform_chunk,
    RecipeDocument,
)


def _row(**overrides) -> dict:
    """Fila valida de Food.com; los kwargs sobreescriben campos concretos."""
    base = {
        "Name":                       "Chicken Soup",
        "RecipeIngredientParts":      ["Chicken", "Onion"],
        "RecipeIngredientQuantities": ["2", "1"],
        "RecipeInstructions":         ["Boil water.", "Add chicken."],
        "Calories":                   200.0,
        "ProteinContent":             20.0,
        "FatContent":                 5.0,
        "CarbohydrateContent":        10.0,
        "FiberContent":               2.0,
        "RecipeCategory":             "Chicken",
    }
    base.update(overrides)
    return base


def _transform_one(row: dict):
    """Transforma una sola fila y devuelve (result, primer documento o None)."""
    df = pd.DataFrame([row])
    result = transform_chunk(df)
    doc = result.documents[0] if result.documents else None
    return result, doc


# --- Funciones auxiliares ---------------------------------------------------

def test_sanitize_removes_null_bytes():
    assert _sanitize("ch\x00icken") == "chicken"


def test_to_list_from_list():
    assert _to_list(["a", " b ", ""]) == ["a", "b"]


def test_to_list_invalid_returns_empty():
    assert _to_list(None) == []
    assert _to_list(42) == []


def test_build_ingredients_combines_qty_and_name():
    assert _build_ingredients(["2 cups", "1"], ["flour", "egg"]) == ["2 cups flour", "1 egg"]


def test_build_ingredients_ignores_none_quantity():
    # cantidad "None"/"nan"/"" no debe anteponerse al nombre
    assert _build_ingredients(["None", ""], ["flour", "egg"]) == ["flour", "egg"]


def test_build_chunk_text_format():
    txt = _build_chunk_text("Soup", ["chicken", "onion"], ["Boil.", "Serve."])
    assert "Recipe: Soup" in txt
    assert "Ingredients: chicken, onion" in txt
    assert "Step 1: Boil." in txt
    assert "Step 2: Serve." in txt


# --- transform_chunk: casos validos -----------------------------------------

def test_valid_row_produces_document():
    result, doc = _transform_one(_row())
    assert isinstance(doc, RecipeDocument)
    assert result.rows_discarded == 0
    assert doc.title == "Chicken Soup"


def test_ingredients_and_quantities_combined():
    _, doc = _transform_one(_row())
    assert doc.ingredients == ["2 Chicken", "1 Onion"]


def test_ner_is_lowercased_and_deduplicated_preserving_order():
    row = _row(RecipeIngredientParts=["Onion", "Chicken", "onion", "ONION"])
    _, doc = _transform_one(row)
    assert doc.ner == ["onion", "chicken"]


def test_chunk_text_contains_title_ingredients_and_steps():
    _, doc = _transform_one(_row())
    assert "Recipe: Chicken Soup" in doc.chunk_text
    assert "Step 1:" in doc.chunk_text


def test_null_byte_removed_from_title():
    _, doc = _transform_one(_row(Name="Chick\x00en Soup"))
    assert "\x00" not in doc.title
    assert doc.title == "Chicken Soup"


def test_nan_nutrition_becomes_zero():
    _, doc = _transform_one(_row(Calories=float("nan"), ProteinContent=None))
    assert doc.calories == 0.0
    assert doc.protein_g == 0.0


# --- transform_chunk: descartes ---------------------------------------------

def test_row_without_title_is_discarded():
    result, doc = _transform_one(_row(Name=""))
    assert doc is None
    assert result.rows_discarded == 1


def test_row_with_less_than_two_ingredients_is_discarded():
    result, doc = _transform_one(_row(RecipeIngredientParts=["Chicken"]))
    assert doc is None
    assert result.rows_discarded == 1


def test_row_without_steps_is_discarded():
    result, doc = _transform_one(_row(RecipeInstructions=[]))
    assert doc is None
    assert result.rows_discarded == 1


def test_mixed_chunk_keeps_valid_discards_invalid():
    df = pd.DataFrame([_row(), _row(Name=""), _row(RecipeInstructions=[])])
    result = transform_chunk(df)
    assert len(result.documents) == 1
    assert result.rows_discarded == 2


# --- steps estructurados y normalizacion de ner -----------------------------

def test_steps_stored_as_structured_list():
    _, doc = _transform_one(_row(RecipeInstructions=["Boil water.", "Add chicken.", "Serve."]))
    assert doc.steps == ["Boil water.", "Add chicken.", "Serve."]


def test_steps_preserve_order():
    pasos = ["First.", "Second.", "Third.", "Fourth."]
    _, doc = _transform_one(_row(RecipeInstructions=pasos))
    assert doc.steps == pasos


def test_ner_normalizes_plural_to_singular():
    _, doc = _transform_one(_row(RecipeIngredientParts=["Tomatoes", "Onions"]))
    assert doc.ner == ["tomato", "onion"]


def test_ner_dedup_by_morphological_variant():
    # "Tomato" y "tomatoes" colapsan al mismo termino normalizado
    _, doc = _transform_one(_row(RecipeIngredientParts=["Tomato", "tomatoes", "Onion"]))
    assert doc.ner == ["tomato", "onion"]


def test_ner_dedup_by_synonym():
    # "red pepper" y "bell pepper" comparten forma canonica
    _, doc = _transform_one(_row(RecipeIngredientParts=["red pepper", "bell pepper", "Onion"]))
    assert doc.ner == ["bell pepper", "onion"]


def test_chunk_text_includes_steps_after_normalization():
    _, doc = _transform_one(_row(RecipeInstructions=["Boil.", "Serve."]))
    assert "Step 1: Boil." in doc.chunk_text
    assert "Step 2: Serve." in doc.chunk_text
