"""
Tests de la normalizacion de nombres usada para detectar recetas duplicadas.
La deduplicacion compara el titulo normalizado (minusculas, espacios
colapsados), de modo que variaciones triviales cuenten como la misma receta.
"""

from src.retrieval.save_generated import _normalize_title


def test_same_title_normalizes_equal():
    assert _normalize_title("Tortilla de patata") == _normalize_title("Tortilla de patata")


def test_case_insensitive():
    assert _normalize_title("Tortilla") == _normalize_title("TORTILLA")


def test_leading_and_trailing_whitespace():
    assert _normalize_title("  Tortilla  ") == _normalize_title("Tortilla")


def test_internal_whitespace_collapses():
    assert _normalize_title("Sopa   de    pollo") == _normalize_title("Sopa de pollo")


def test_mixed_case_and_spaces():
    assert _normalize_title("  SOPA de   Pollo ") == _normalize_title("sopa de pollo")


def test_different_titles_differ():
    assert _normalize_title("Tortilla") != _normalize_title("Omelette")


def test_empty_and_none_are_safe():
    assert _normalize_title("") == ""
    assert _normalize_title(None) == ""
