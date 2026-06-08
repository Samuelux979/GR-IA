"""
Tests unitarios de save_generated_recipe() (src/retrieval/save_generated.py).

Usan un cursor falso para capturar las consultas y mockean embed_texts()
para no cargar el modelo de embeddings. No requieren PostgreSQL.
La comprobacion de plausibilidad se desactiva para no depender del LLM.
"""

import json
from unittest.mock import patch

import numpy as np

from src.retrieval import save_generated as save_mod
from src.retrieval.save_generated import save_generated_recipe


_FAKE_EMBED = np.zeros((1, 384))


def _save(conn, recipe):
    with patch.object(save_mod, "embed_texts", return_value=_FAKE_EMBED):
        return save_generated_recipe(conn, recipe, check_plausibility=False)


def _find(executed, needle: str):
    """Devuelve (sql, params) de la primera consulta que contiene needle."""
    for sql, params in executed:
        if needle in sql:
            return sql, params
    return None, None


def test_inserts_row_into_recipes(fake_conn, sample_recipe):
    conn = fake_conn()
    _save(conn, sample_recipe)
    sql, _ = _find(conn._cursor.executed, "INSERT INTO recipes")
    assert sql is not None


def test_inserts_chunk_into_recipe_chunks(fake_conn, sample_recipe):
    conn = fake_conn()
    _save(conn, sample_recipe)
    sql, params = _find(conn._cursor.executed, "INSERT INTO recipe_chunks")
    assert sql is not None
    assert params[0] == 123          # recipe_id devuelto por la BD simulada
    assert params[1] == 0            # chunk_index


def test_etl_batch_id_is_minus_one(fake_conn, sample_recipe):
    conn = fake_conn()
    _save(conn, sample_recipe)
    _, params = _find(conn._cursor.executed, "INSERT INTO recipes")
    assert params[-1] == -1


def test_ingredients_and_ner_serialized_as_json(fake_conn, sample_recipe):
    conn = fake_conn()
    _save(conn, sample_recipe)
    _, params = _find(conn._cursor.executed, "INSERT INTO recipes")
    # params: (title, ingredients_json, ner_json, steps_json, ...)
    assert json.loads(params[1]) == sample_recipe["ingredients"]
    assert json.loads(params[2]) == sample_recipe["ner"]


def test_chunk_text_includes_title_ingredients_and_steps(fake_conn, sample_recipe):
    conn = fake_conn()
    _save(conn, sample_recipe)
    _, params = _find(conn._cursor.executed, "INSERT INTO recipe_chunks")
    chunk_text = params[2]
    assert "Recipe: Chicken and Rice Bowl" in chunk_text
    assert "rice" in chunk_text          # un ingrediente (ner)
    assert "Step 1:" in chunk_text


def test_commit_called_once(fake_conn, sample_recipe):
    conn = fake_conn()
    _save(conn, sample_recipe)
    assert conn.commits == 1


def test_returns_generated_id_and_new_flag(fake_conn, sample_recipe):
    conn = fake_conn(returning_id=777)
    recipe_id, was_new = _save(conn, sample_recipe)
    assert recipe_id == 777
    assert was_new is True
