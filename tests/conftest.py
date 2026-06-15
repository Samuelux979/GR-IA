"""
Fixtures reutilizables para la suite de tests.

Incluye dobles de prueba (fakes) para la base de datos y datos de ejemplo,
de modo que los tests unitarios no necesiten PostgreSQL, Ollama ni modelos.
"""

import pytest


@pytest.fixture
def classified_ingredients() -> dict:
    """Ingredientes ya clasificados en los 4 niveles de prioridad."""
    return {
        "main":          ["chicken"],
        "secondary":     ["onion", "garlic"],
        "accompaniment": ["parsley"],
        "spices":        ["paprika"],
    }


@pytest.fixture
def sample_recipe() -> dict:
    """Receta generada de ejemplo, valida y plausible."""
    return {
        "title":             "Chicken and Rice Bowl",
        "ingredients":       ["2 cups rice", "1 chicken breast", "1 onion"],
        "steps":             ["Cook the rice.", "Grill the chicken.", "Combine and serve."],
        "ner":               ["rice", "chicken", "onion"],
        "category":          "AI Generated",
        "calories":          450.0,
        "protein_g":         30.0,
        "fat_g":             10.0,
        "carbs_g":           55.0,
        "fiber_g":           4.0,
        "model":             "qwen2.5:1.5b",
        "embedding_model":   "all-MiniLM-L6-v2",
        "input_ingredients": {"main": ["chicken"], "secondary": [], "accompaniment": [], "spices": []},
        "prompt_version":    "v1",
    }


class FakeCursor:
    """
    Cursor de PostgreSQL simulado: registra las consultas ejecutadas y
    devuelve filas predefinidas. Sirve para verificar el SQL y los
    parametros sin tocar una base de datos real.
    """

    def __init__(self, rows=None, returning_id=123):
        self.rows = rows or []
        self.returning_id = returning_id
        self.executed = []        # lista de (sql, params)
        self._last_sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._last_sql = sql
        self.executed.append((sql, params))

    def fetchone(self):
        # Comprobacion de duplicado por nombre previa al insert: no existe -> None
        if "SELECT id FROM recipes" in self._last_sql:
            return None
        # INSERT ... RETURNING id -> devuelve el id generado
        if "RETURNING id" in self._last_sql:
            return (self.returning_id,)
        return None

    def fetchall(self):
        return self.rows


class FakeConn:
    """Conexion simulada que entrega FakeCursor y cuenta los commits."""

    def __init__(self, rows=None, returning_id=123):
        self._cursor = FakeCursor(rows=rows, returning_id=returning_id)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, *args, **kwargs):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


@pytest.fixture
def fake_conn():
    """Factory de conexion simulada."""
    def _make(rows=None, returning_id=123):
        return FakeConn(rows=rows, returning_id=returning_id)
    return _make
