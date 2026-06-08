"""Tests unitarios para la validacion previa al guardado de recetas IA."""

from src.retrieval.validation import validate_recipe, MAX_TITLE_LEN, MAX_STEP_LEN


VALID_MINIMAL = {
    "title":       "Tortilla de patatas",
    "ingredients": ["3 patatas", "4 huevos"],
    "steps":       ["Pelar las patatas.", "Batir los huevos.", "Cuajar."],
    "ner":         ["potato", "egg"],
}


def test_minimal_valid_passes():
    assert validate_recipe(VALID_MINIMAL) == []


def test_rejects_empty_title():
    bad = {**VALID_MINIMAL, "title": ""}
    assert "Titulo vacio o no es texto." in validate_recipe(bad)


def test_rejects_whitespace_title():
    bad = {**VALID_MINIMAL, "title": "   "}
    assert "Titulo vacio o no es texto." in validate_recipe(bad)


def test_rejects_too_long_title():
    bad = {**VALID_MINIMAL, "title": "x" * (MAX_TITLE_LEN + 1)}
    assert any("demasiado largo" in e for e in validate_recipe(bad))


def test_rejects_title_with_null_byte():
    bad = {**VALID_MINIMAL, "title": "Receta\x00mala"}
    assert any("caracteres nulos" in e for e in validate_recipe(bad))


def test_rejects_no_ingredients():
    bad = {**VALID_MINIMAL, "ingredients": []}
    assert "Debe haber al menos un ingrediente." in validate_recipe(bad)


def test_rejects_no_steps():
    bad = {**VALID_MINIMAL, "steps": []}
    assert "Debe haber al menos un paso." in validate_recipe(bad)


def test_rejects_empty_ner():
    bad = {**VALID_MINIMAL, "ner": []}
    assert "Lista NER vacia." in validate_recipe(bad)


def test_rejects_too_long_step():
    bad = {**VALID_MINIMAL, "steps": ["x" * (MAX_STEP_LEN + 1)]}
    assert any("longitud maxima" in e for e in validate_recipe(bad))


def test_rejects_non_list_ingredients():
    bad = {**VALID_MINIMAL, "ingredients": "not a list"}
    assert "Debe haber al menos un ingrediente." in validate_recipe(bad)


def test_collects_multiple_errors():
    bad = {"title": "", "ingredients": [], "steps": [], "ner": []}
    errors = validate_recipe(bad)
    assert len(errors) >= 3
