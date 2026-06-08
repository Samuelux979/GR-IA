"""
Tests del hash de deduplicacion que identifica recetas equivalentes.
Cubre los casos de variacion trivial (mayusculas, espacios, orden NER).
"""

from src.retrieval.save_generated import compute_dedup_hash


def test_same_recipe_yields_same_hash():
    h1 = compute_dedup_hash("Tortilla", ["potato", "egg"])
    h2 = compute_dedup_hash("Tortilla", ["potato", "egg"])
    assert h1 == h2


def test_case_insensitive_title():
    assert compute_dedup_hash("Tortilla", ["egg"]) == compute_dedup_hash("TORTILLA", ["egg"])


def test_case_insensitive_ner():
    assert compute_dedup_hash("Tortilla", ["EGG"]) == compute_dedup_hash("Tortilla", ["egg"])


def test_whitespace_around_title():
    assert compute_dedup_hash("  Tortilla  ", ["egg"]) == compute_dedup_hash("Tortilla", ["egg"])


def test_whitespace_around_ner():
    assert compute_dedup_hash("Tortilla", [" egg ", " potato "]) == compute_dedup_hash("Tortilla", ["egg", "potato"])


def test_ner_order_does_not_matter():
    a = compute_dedup_hash("Tortilla", ["egg", "potato", "salt"])
    b = compute_dedup_hash("Tortilla", ["salt", "potato", "egg"])
    assert a == b


def test_duplicates_in_ner_collapse():
    a = compute_dedup_hash("Tortilla", ["egg", "egg", "potato"])
    b = compute_dedup_hash("Tortilla", ["egg", "potato"])
    assert a == b


def test_different_titles_yield_different_hashes():
    h1 = compute_dedup_hash("Tortilla", ["egg"])
    h2 = compute_dedup_hash("Omelette", ["egg"])
    assert h1 != h2


def test_different_ner_yields_different_hashes():
    h1 = compute_dedup_hash("Tortilla", ["egg"])
    h2 = compute_dedup_hash("Tortilla", ["egg", "potato"])
    assert h1 != h2


def test_hash_is_deterministic_across_calls():
    runs = {compute_dedup_hash("R", ["a", "b"]) for _ in range(10)}
    assert len(runs) == 1


def test_hash_is_64_chars_hex():
    h = compute_dedup_hash("R", ["a"])
    assert len(h) == 64
    int(h, 16)  # debe ser hex valido
