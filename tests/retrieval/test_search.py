"""
Tests de search_recipes() (src/retrieval/search.py).

Usan un cursor falso para capturar el SQL y sus parametros, y mockean
embed_texts() para no cargar el modelo de embeddings.
"""

from unittest.mock import patch

import numpy as np

from src.retrieval import search as search_mod
from src.retrieval.search import search_recipes, _variants


# embed_texts devuelve un array (1, 384); el codigo hace [0].tolist()
_FAKE_EMBED = np.zeros((1, 384))


def _row(**overrides) -> dict:
    base = {
        "id": 1, "title": "Chicken Soup",
        "ingredients": '["2 chicken legs"]',   # JSON string (como lo devuelve la BD)
        "ner": '["chicken"]',
        "steps": None,
        "category": "Chicken", "source": "foodcom", "macros_known": True,
        "calories": 200.0, "protein_g": 20.0, "carbs_g": 10.0, "fat_g": 5.0, "fiber_g": 2.0,
        "content": "Recipe: Chicken Soup. Step 1: Boil.",
        "distance": 0.42, "score": 12,
    }
    base.update(overrides)
    return base


def test_empty_classification_returns_empty_list(fake_conn):
    conn = fake_conn(rows=[])
    classified = {"main": [], "secondary": [], "accompaniment": [], "spices": []}
    assert search_recipes(conn, classified) == []
    # no debe ejecutarse ninguna consulta
    assert conn._cursor.executed == []


def test_variants_expansion():
    assert "chicken" in _variants("chicken")
    assert "egg" in _variants("eggs")
    assert "tomato" in _variants("tomatoes")


def test_ingredients_normalized_and_passed_to_sql(fake_conn):
    conn = fake_conn(rows=[])
    classified = {"main": ["  Chicken "], "secondary": ["Onion"], "accompaniment": [], "spices": []}

    with patch.object(search_mod, "embed_texts", return_value=_FAKE_EMBED):
        search_recipes(conn, classified, top_k=5)

    sql, params = conn._cursor.executed[0]
    assert "chicken" in params["main"]       # normalizado a minusculas + variantes
    assert "onion" in params["secondary"]
    assert params["top_k"] == 5
    assert params["include_ai"] is True


def test_all_four_levels_passed_to_sql(fake_conn):
    conn = fake_conn(rows=[])
    classified = {
        "main":          ["Chicken"],
        "secondary":     ["Onion"],
        "accompaniment": ["Parsley"],
        "spices":        ["Paprika"],
    }

    with patch.object(search_mod, "embed_texts", return_value=_FAKE_EMBED):
        search_recipes(conn, classified)

    _, params = conn._cursor.executed[0]
    # las cuatro listas llegan al SQL con el contenido normalizado esperado
    assert "chicken" in params["main"]
    assert "onion" in params["secondary"]
    assert "parsley" in params["accompaniment"]
    assert "paprika" in params["spices"]


def test_normalization_preserves_level_and_weight(fake_conn):
    # 'pollo' (espanol) en main debe llegar al SQL como 'chicken' en params['main'],
    # conservando el nivel (y por tanto su peso x8 en el scoring).
    conn = fake_conn(rows=[])
    classified = {"main": ["pollo"], "secondary": ["cebolla"], "accompaniment": [], "spices": ["pimienta"]}

    with patch.object(search_mod, "embed_texts", return_value=_FAKE_EMBED):
        search_recipes(conn, classified)

    _, params = conn._cursor.executed[0]
    assert "chicken" in params["main"]        # pollo -> chicken, sigue en main (x8)
    assert "onion" in params["secondary"]     # cebolla -> onion, en secondary (x4)
    assert "pepper" in params["spices"]       # pimienta -> pepper, en spices (x1)
    # no se mezclan niveles
    assert "chicken" not in params["secondary"]


def test_synonym_normalized_in_sql(fake_conn):
    conn = fake_conn(rows=[])
    classified = {"main": ["red pepper"], "secondary": [], "accompaniment": [], "spices": []}

    with patch.object(search_mod, "embed_texts", return_value=_FAKE_EMBED):
        search_recipes(conn, classified)

    _, params = conn._cursor.executed[0]
    assert "bell pepper" in params["main"]    # red pepper -> bell pepper


def test_include_ai_flag_passed(fake_conn):
    conn = fake_conn(rows=[])
    classified = {"main": ["chicken"], "secondary": [], "accompaniment": [], "spices": []}

    with patch.object(search_mod, "embed_texts", return_value=_FAKE_EMBED):
        search_recipes(conn, classified, include_ai=False)

    _, params = conn._cursor.executed[0]
    assert params["include_ai"] is False


def test_results_parse_json_and_convert_types(fake_conn):
    conn = fake_conn(rows=[_row()])
    classified = {"main": ["chicken"], "secondary": [], "accompaniment": [], "spices": []}

    with patch.object(search_mod, "embed_texts", return_value=_FAKE_EMBED):
        results = search_recipes(conn, classified)

    r = results[0]
    assert r["ingredients"] == ["2 chicken legs"]   # parseado desde JSON string
    assert r["ner"] == ["chicken"]
    assert isinstance(r["distance"], float) and r["distance"] == 0.42
    assert isinstance(r["score"], int) and r["score"] == 12


def test_results_accept_already_parsed_lists(fake_conn):
    # Si la BD devuelve listas (no strings), tambien debe funcionar
    conn = fake_conn(rows=[_row(ingredients=["a", "b"], ner=["c"])])
    classified = {"main": ["chicken"], "secondary": [], "accompaniment": [], "spices": []}

    with patch.object(search_mod, "embed_texts", return_value=_FAKE_EMBED):
        results = search_recipes(conn, classified)

    assert results[0]["ingredients"] == ["a", "b"]
    assert results[0]["ner"] == ["c"]
