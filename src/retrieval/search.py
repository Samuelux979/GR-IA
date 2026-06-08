"""
Busqueda de recetas por puntuacion ponderada de ingredientes + similitud vectorial.

Pesos por nivel de ingrediente:
  principal      x 8
  secundario     x 4
  acompanamiento x 2
  especias       x 1
"""

import json
import logging

import psycopg2
import psycopg2.extras

from src.etl.embed import embed_texts
from src.retrieval.ingredient_normalizer import normalize_level

logger = logging.getLogger(__name__)

# El scoring usa dos condiciones OR:
#   1. t.value = term          -> match exacto ("avocado" = "avocado")
#   2. t.value LIKE term||' %' -> el NER empieza por el termino ("garlic" ~ "garlic cloves")
# El filtro inicial (?|) usa todas las variantes para aprovechar el indice GIN.
SQL_SEARCH = """
WITH scored AS (
    SELECT
        r.id,
        (
            (SELECT COUNT(*) FROM jsonb_array_elements_text(r.ner) t
             WHERE EXISTS (SELECT 1 FROM unnest(%(main)s::text[]) term
                           WHERE t.value = term OR t.value LIKE term || ' %%')) * 8 +
            (SELECT COUNT(*) FROM jsonb_array_elements_text(r.ner) t
             WHERE EXISTS (SELECT 1 FROM unnest(%(secondary)s::text[]) term
                           WHERE t.value = term OR t.value LIKE term || ' %%')) * 4 +
            (SELECT COUNT(*) FROM jsonb_array_elements_text(r.ner) t
             WHERE EXISTS (SELECT 1 FROM unnest(%(accompaniment)s::text[]) term
                           WHERE t.value = term OR t.value LIKE term || ' %%')) * 2 +
            (SELECT COUNT(*) FROM jsonb_array_elements_text(r.ner) t
             WHERE EXISTS (SELECT 1 FROM unnest(%(spices)s::text[]) term
                           WHERE t.value = term OR t.value LIKE term || ' %%')) * 1
        ) AS score
    FROM recipes r
    WHERE r.ner ?| %(all_variants)s
      AND (%(include_ai)s OR COALESCE(r.source, 'foodcom') <> 'ai_generated')
),
filtered AS (
    SELECT id, score FROM scored WHERE score >= 4
),
ranked AS (
    SELECT
        rc.recipe_id,
        rc.content,
        rc.embedding <=> %(query_vec)s::vector AS distance,
        f.score
    FROM recipe_chunks rc
    INNER JOIN filtered f ON f.id = rc.recipe_id
    ORDER BY f.score DESC, distance ASC
    LIMIT %(top_k)s
)
SELECT
    r.id,
    r.title,
    r.ingredients,
    r.ner,
    r.steps,
    r.category,
    r.source,
    r.macros_known,
    r.calories,
    r.protein_g,
    r.carbs_g,
    r.fat_g,
    r.fiber_g,
    rk.content,
    rk.distance,
    rk.score
FROM ranked rk
JOIN recipes r ON r.id = rk.recipe_id
WHERE r.title ~ '[A-Za-z]'
ORDER BY rk.score DESC, rk.distance ASC;
"""


def _variants(term: str) -> list[str]:
    """
    Genera variantes plural/singular de un termino para cubrir
    las distintas formas en que aparece en el dataset.
    Ej: "avocado" -> ["avocado", "avocados", "avocadoes"]
        "eggs"    -> ["eggs", "egg"]
        "tomato"  -> ["tomato", "tomatoes"]
    """
    v = {term}
    if term.endswith("oes"):
        v.add(term[:-2])           # avocadoes -> avocado
    elif term.endswith("ies"):
        v.add(term[:-3] + "y")     # berries -> berry
    elif term.endswith("es") and len(term) > 4:
        v.add(term[:-2])           # tomatoes -> tomato
    elif term.endswith("s") and len(term) > 3:
        v.add(term[:-1])           # eggs -> egg, cloves -> clove
    else:
        v.add(term + "s")          # egg -> eggs, avocado -> avocados
        if term.endswith(("o", "x", "z", "ch", "sh")):
            v.add(term + "es")     # tomato -> tomatoes
    return list(v)


def search_recipes(
    conn,
    classified: dict,
    top_k: int = 10,
    include_ai: bool = True,
) -> list[dict]:
    """
    Recupera las recetas mas relevantes usando puntuacion ponderada por nivel
    de ingrediente y similitud coseno como criterio de desempate.

    include_ai=False excluye recetas generadas por IA (source='ai_generated').

    Cada ingrediente se normaliza a su forma canonica (sinonimos, espanol->ingles,
    plurales y correccion ortografica) PRESERVANDO su nivel, de modo que el
    scoring ponderado (main x8, secondary x4, ...) se mantiene intacto.
    """
    # Normalizacion canonica por nivel (no se aplana: conserva los pesos)
    main          = normalize_level(classified.get("main", []))
    secondary     = normalize_level(classified.get("secondary", []))
    accompaniment = normalize_level(classified.get("accompaniment", []))
    spices        = normalize_level(classified.get("spices", []))

    if not any([main, secondary, accompaniment, spices]):
        return []

    main_v          = list({v for t in main          for v in _variants(t)})
    secondary_v     = list({v for t in secondary     for v in _variants(t)})
    accompaniment_v = list({v for t in accompaniment for v in _variants(t)})
    spices_v        = list({v for t in spices        for v in _variants(t)})
    all_variants    = list(set(main_v + secondary_v + accompaniment_v + spices_v))

    parts = []
    if main:
        parts.append("Main ingredient: " + ", ".join(main))
    if secondary:
        parts.append("with " + ", ".join(secondary))
    if accompaniment + spices:
        parts.append("seasoned with " + ", ".join(accompaniment + spices))
    query_text = " ".join(parts) or "Recipe with: " + ", ".join(main + secondary)
    query_vec  = embed_texts([query_text])[0].tolist()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(SQL_SEARCH, {
            "main":          main_v          or [""],
            "secondary":     secondary_v     or [""],
            "accompaniment": accompaniment_v or [""],
            "spices":        spices_v        or [""],
            "all_variants":  all_variants,
            "query_vec":     query_vec,
            "top_k":         top_k,
            "include_ai":    include_ai,
        })
        rows = cur.fetchall()

    results = []
    for row in rows:
        # Pasos: priorizar la columna estructurada; si no existe (recetas viejas),
        # se reconstruyen luego desde chunk_text en la capa de servicio.
        steps_raw = row.get("steps")
        steps     = None
        if steps_raw is not None:
            steps = steps_raw if isinstance(steps_raw, list) else json.loads(steps_raw)

        results.append({
            "id":           row["id"],
            "title":        row["title"],
            "ingredients":  row["ingredients"] if isinstance(row["ingredients"], list) else json.loads(row["ingredients"]),
            "ner":          row["ner"]         if isinstance(row["ner"],         list) else json.loads(row["ner"]),
            "steps":        steps,
            "category":     row["category"],
            "source":       row.get("source") or "foodcom",
            "macros_known": row.get("macros_known", True),
            "calories":     row["calories"],
            "protein_g":    row["protein_g"],
            "carbs_g":      row["carbs_g"],
            "fat_g":        row["fat_g"],
            "fiber_g":      row["fiber_g"],
            "chunk_text":   row["content"],
            "distance":     float(row["distance"]),
            "score":        int(row["score"]),
        })
    return results
