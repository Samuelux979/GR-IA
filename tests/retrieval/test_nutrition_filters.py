"""
Tests del filtro nutricional determinista (src/retrieval/nutrition_filters.py).
Funciones puras: no requieren base de datos.
"""

from src.retrieval.nutrition_filters import (
    resolve_filters,
    build_sql,
    NUTRITION_PROFILES,
)


# --- resolve_filters --------------------------------------------------------

def test_profile_resolves_to_ranges():
    f = resolve_filters(profile="alta_proteina")
    assert f == {"protein_g": (25, None)}


def test_unknown_profile_is_ignored():
    assert resolve_filters(profile="inexistente") == {}


def test_ranges_only():
    f = resolve_filters(ranges={"calories": (None, 400)})
    assert f == {"calories": (None, 400)}


def test_ranges_override_profile():
    f = resolve_filters(profile="baja_caloria", ranges={"calories": (None, 300)})
    assert f["calories"] == (None, 300)


def test_ranges_ignore_unknown_macro():
    f = resolve_filters(ranges={"sodio": (None, 100)})
    assert "sodio" not in f


def test_combined_profile_has_two_macros():
    f = resolve_filters(profile="proteica_y_ligera")
    assert "protein_g" in f and "calories" in f


# --- build_sql --------------------------------------------------------------

def test_build_sql_empty():
    clause, params = build_sql({})
    assert clause == "" and params == {}


def test_build_sql_min_bound():
    clause, params = build_sql({"protein_g": (25, None)})
    assert "r.macros_known = TRUE" in clause
    assert "r.protein_g >= %(nf_protein_g_min)s" in clause
    assert params["nf_protein_g_min"] == 25.0
    assert "nf_protein_g_max" not in params


def test_build_sql_max_bound():
    clause, params = build_sql({"calories": (None, 400)})
    assert "r.calories <= %(nf_calories_max)s" in clause
    assert params["nf_calories_max"] == 400.0


def test_build_sql_both_bounds():
    clause, params = build_sql({"calories": (200, 500)})
    assert "r.calories >= %(nf_calories_min)s" in clause
    assert "r.calories <= %(nf_calories_max)s" in clause
    assert params == {"nf_calories_min": 200.0, "nf_calories_max": 500.0}


def test_build_sql_requires_macros_known():
    # Cualquier filtro activo exige valores nutricionales fiables
    clause, _ = build_sql({"fiber_g": (8, None)})
    assert clause.startswith(" AND r.macros_known = TRUE")


def test_all_profiles_build_valid_sql():
    for name in NUTRITION_PROFILES:
        clause, params = build_sql(resolve_filters(profile=name))
        assert clause and isinstance(params, dict)
