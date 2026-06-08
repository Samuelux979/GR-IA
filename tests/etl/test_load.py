"""
Tests de carga (src/etl/load.py).

Verifican que load_batch inserta los pasos estructurados como JSON.
Se mockea psycopg2.extras.execute_values para capturar los valores sin BD.
"""

from unittest.mock import patch

import numpy as np

from src.etl import load as load_mod
from src.etl.transform import RecipeDocument


def _doc(**overrides) -> RecipeDocument:
    base = dict(
        title="Chicken Soup",
        ingredients=["2 chicken", "1 onion"],
        ner=["chicken", "onion"],
        steps=["Boil water.", "Add chicken."],
        category="Chicken",
        calories=200.0, protein_g=20.0, fat_g=5.0, carbs_g=10.0, fiber_g=2.0,
        chunk_text="Recipe: Chicken Soup. Ingredients: chicken, onion. Step 1: Boil water.",
    )
    base.update(overrides)
    return RecipeDocument(**base)


class _Cur:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Conn:
    def __init__(self): self.commits = 0
    def cursor(self): return _Cur()
    def commit(self): self.commits += 1
    def rollback(self): pass


def test_load_inserts_steps_as_json():
    captured = {}

    def fake_execute_values(cur, sql, argslist, template=None, fetch=False):
        if "INSERT INTO recipes" in sql:
            captured["recipes"] = argslist
            captured["recipe_template"] = template
            return [(1,)]                      # id devuelto (RETURNING)
        captured["chunks"] = argslist
        return None

    conn = _Conn()
    embeddings = np.zeros((1, 384))

    with patch.object(load_mod.psycopg2.extras, "execute_values", side_effect=fake_execute_values):
        n = load_mod.load_batch(conn, [_doc()], embeddings, batch_id=7)

    assert n == 1
    # La tupla de recipes incluye los steps serializados como JSON en 4a posicion
    recipe_row = captured["recipes"][0]
    assert recipe_row[3] == '["Boil water.", "Add chicken."]'
    # El template debe declarar el cast a jsonb para los steps
    assert "::jsonb" in captured["recipe_template"]
    # Se inserto el chunk asociado al id devuelto
    assert captured["chunks"][0][0] == 1
    assert conn.commits == 1


def test_load_empty_documents_returns_zero():
    conn = _Conn()
    assert load_mod.load_batch(conn, [], np.zeros((0, 384)), batch_id=1) == 0
