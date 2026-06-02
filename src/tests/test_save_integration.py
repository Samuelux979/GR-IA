"""
Tests de integracion para el flujo completo de guardado de recetas IA.

Estos tests requieren una base de datos PostgreSQL activa (gria_db) con
el esquema cargado. Crean recetas reales y luego las eliminan al final
de cada test para no contaminar la BD.

Ejecucion selectiva:
    pytest src/tests/test_save_integration.py -v
"""

import pytest
import psycopg2

from src.config import get_dsn
from src.retrieval.save_generated import (
    save_generated_recipe,
    RecipeValidationError,
)


@pytest.fixture(scope="module")
def conn():
    """Conexion a la BD reutilizada por todos los tests del modulo."""
    c = psycopg2.connect(get_dsn())
    yield c
    c.close()


@pytest.fixture
def recipe_factory(conn):
    """Crea recetas y las elimina al final de cada test."""
    created_ids: list[int] = []

    def make(**overrides) -> dict:
        base = {
            "title":             "Receta test " + str(id({})),
            "ingredients":       ["1 ingrediente A", "2 ingredientes B"],
            "steps":             ["Paso 1", "Paso 2"],
            "ner":               ["test_a", "test_b"],
            "category":          "AI Generated",
            "calories":          120.0,
            "protein_g":         8.5,
            "fat_g":             4.0,
            "carbs_g":           15.2,
            "fiber_g":           3.1,
            "model":             "qwen2.5:1.5b",
            "embedding_model":   "all-MiniLM-L6-v2",
            "input_ingredients": {"main": ["test"], "secondary": [], "accompaniment": [], "spices": []},
            "prompt_version":    "v1",
        }
        base.update(overrides)
        return base

    def track(recipe_id: int):
        created_ids.append(recipe_id)

    yield make, track

    if created_ids:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM recipes WHERE id = ANY(%s);", (created_ids,))
        conn.commit()


def test_save_inserts_with_provenance(conn, recipe_factory):
    make, track = recipe_factory
    recipe = make(title="Tortilla con metadatos completos")

    recipe_id, was_new = save_generated_recipe(conn, recipe)
    track(recipe_id)

    assert was_new is True

    with conn.cursor() as cur:
        cur.execute(
            "SELECT source, provenance, dedup_hash, macros_known, steps "
            "FROM recipes WHERE id = %s;",
            (recipe_id,),
        )
        source, provenance, dedup_hash, macros_known, steps = cur.fetchone()

    assert source == "ai_generated"
    assert provenance["model"] == "qwen2.5:1.5b"
    assert provenance["prompt_version"] == "v1"
    assert provenance["embedding_model"] == "all-MiniLM-L6-v2"
    assert "generated_at" in provenance
    assert dedup_hash is not None and len(dedup_hash) == 64
    assert macros_known is False           # IA -> macros siempre estimadas
    assert steps == ["Paso 1", "Paso 2"]   # pasos estructurados


def test_save_returns_existing_id_on_duplicate(conn, recipe_factory):
    make, track = recipe_factory
    recipe = make(title="Receta duplicable")

    id1, was_new1 = save_generated_recipe(conn, recipe)
    track(id1)
    id2, was_new2 = save_generated_recipe(conn, recipe)

    assert was_new1 is True
    assert was_new2 is False
    assert id1 == id2


def test_dedup_ignores_case_and_whitespace(conn, recipe_factory):
    make, track = recipe_factory

    r1 = make(title="Sopa de pollo", ner=["chicken", "celery"])
    r2 = make(title="  SOPA DE POLLO ", ner=["CELERY", "chicken"])

    id1, was_new1 = save_generated_recipe(conn, r1)
    track(id1)
    id2, was_new2 = save_generated_recipe(conn, r2)

    assert id1 == id2
    assert was_new2 is False


def test_save_marks_macros_as_estimated(conn, recipe_factory):
    make, track = recipe_factory
    recipe_id, _ = save_generated_recipe(conn, make(title="Receta con macros"))
    track(recipe_id)

    with conn.cursor() as cur:
        cur.execute("SELECT macros_known FROM recipes WHERE id = %s;", (recipe_id,))
        assert cur.fetchone()[0] is False


def test_save_converts_zero_macros_to_null(conn, recipe_factory):
    make, track = recipe_factory
    recipe = make(
        title="Receta sin macros calculables",
        calories=0, protein_g=0, fat_g=0, carbs_g=0, fiber_g=0,
    )

    recipe_id, _ = save_generated_recipe(conn, recipe)
    track(recipe_id)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT calories, protein_g, fat_g, carbs_g, fiber_g, macros_known "
            "FROM recipes WHERE id = %s;", (recipe_id,))
        row = cur.fetchone()

    # Valores que entraron como 0 deben quedar NULL
    assert row[0] is None
    assert row[1] is None
    assert row[2] is None
    assert row[3] is None
    assert row[4] is None
    assert row[5] is False


def test_save_rejects_invalid_recipe(conn):
    bad = {
        "title":       "",
        "ingredients": [],
        "steps":       [],
        "ner":         [],
    }
    with pytest.raises(RecipeValidationError) as exc:
        save_generated_recipe(conn, bad)
    assert len(exc.value.errors) >= 3


def test_save_creates_embedding_chunk(conn, recipe_factory):
    make, track = recipe_factory
    recipe_id, _ = save_generated_recipe(conn, make(title="Receta con embedding"))
    track(recipe_id)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM recipe_chunks WHERE recipe_id = %s;", (recipe_id,))
        assert cur.fetchone()[0] == 1
