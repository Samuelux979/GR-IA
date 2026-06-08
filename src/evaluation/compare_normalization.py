"""
Comparativa de recuperacion CON y SIN normalizacion de ingredientes.

Ejecuta las consultas de evaluation/retrieval_queries.jsonl dos veces:
  - SIN normalizacion (solo minusculas/espacios, comportamiento antiguo)
  - CON normalizacion (sinonimos + espanol->ingles + plurales + fuzzy)

y reporta Precision@5 en ambos casos para demostrar la mejora.

Uso:
    python -m src.evaluation.compare_normalization
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import psycopg2

from src.config import get_dsn, PROJECT_ROOT
from src.retrieval import search as search_mod
from src.retrieval.search import search_recipes
from src.retrieval.ingredient_normalizer import normalize_level, normalize_ingredient

logging.basicConfig(level=logging.WARNING)


def _ner_words(recipe) -> set:
    """Conjunto de palabras normalizadas del ner de una receta."""
    words = set()
    for n in (recipe.get("ner") or []):
        words |= set(normalize_ingredient(n, fuzzy=False).split())
    return words


def _is_relevant(recipe, must_have: list) -> bool:
    """Relevante si alguna palabra requerida aparece en el ner (p.ej. 'chicken' en 'chicken breast')."""
    ner_words = _ner_words(recipe)
    return any(term in ner_words for term in must_have)

QUERIES = PROJECT_ROOT / "evaluation" / "retrieval_queries.jsonl"
K = 5


def _raw_level(ingredients, fuzzy=True):
    """Normalizacion 'antigua': solo minusculas y espacios, sin sinonimos/ES/fuzzy."""
    seen, out = set(), []
    for ing in (ingredients or []):
        n = " ".join(str(ing).lower().split())
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _precision_at_k(conn, case) -> float:
    must_have = normalize_level(case.get("relevant_if_ner_contains", []), fuzzy=False)
    recipes = search_recipes(conn, case["query_ingredients"], top_k=K)
    if not recipes:
        return 0.0
    rel = sum(1 for r in recipes if _is_relevant(r, must_have))
    return rel / len(recipes)


def run() -> None:
    cases = [json.loads(l) for l in open(QUERIES, encoding="utf-8") if l.strip()]
    conn = psycopg2.connect(get_dsn())

    print(f"{'Consulta':<45} {'SIN norm.':>10} {'CON norm.':>10}")
    print("-" * 67)

    sin_total, con_total = 0.0, 0.0
    for case in cases:
        nota = (case.get("notes") or "")[:43]

        # SIN normalizacion: parchea normalize_level por la version cruda
        with patch.object(search_mod, "normalize_level", _raw_level):
            p_sin = _precision_at_k(conn, case)
        # CON normalizacion: comportamiento real
        p_con = _precision_at_k(conn, case)

        sin_total += p_sin
        con_total += p_con
        print(f"{nota:<45} {p_sin:>10.2f} {p_con:>10.2f}")

    n = len(cases)
    print("-" * 67)
    print(f"{'MEDIA Precision@' + str(K):<45} {sin_total/n:>10.2f} {con_total/n:>10.2f}")
    conn.close()


if __name__ == "__main__":
    run()
