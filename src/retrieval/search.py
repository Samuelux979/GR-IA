"""
Retrieval — search recipes by weighted ingredient match + semantic similarity.

Phase 1: 4-level weighted ingredient scoring + vector cosine search.
Phase 2: will add nutritional metadata filters (calories, protein, etc.).

Scoring weights:
  main         × 8  — dominant ingredient, defines the dish
  secondary    × 4  — important complement
  accompaniment × 2  — garnish / small-quantity element
  spices       × 1  — dry seasonings
"""

import json
import logging

import psycopg2
import psycopg2.extras

from src.etl.embed import embed_texts  # embed.py vive en etl/ pero se usa también en inferencia

logger = logging.getLogger(__name__)

SQL_WEIGHTED_SEARCH = """
WITH scored AS (
    SELECT
        r.id,
        (
            (SELECT COUNT(*) FROM jsonb_array_elements_text(r.ner) t WHERE t.value = ANY(%(main)s::text[]))        * 8 +
            (SELECT COUNT(*) FROM jsonb_array_elements_text(r.ner) t WHERE t.value = ANY(%(secondary)s::text[]))   * 4 +
            (SELECT COUNT(*) FROM jsonb_array_elements_text(r.ner) t WHERE t.value = ANY(%(accompaniment)s::text[])) * 2 +
            (SELECT COUNT(*) FROM jsonb_array_elements_text(r.ner) t WHERE t.value = ANY(%(spices)s::text[]))      * 1
        ) AS score
    FROM recipes r
    WHERE r.ner ?| %(all_ingredients)s
      AND jsonb_array_length(r.ner) BETWEEN 3 AND 25
),
filtered AS (
    SELECT id, score FROM scored WHERE score >= 4
),
ranked_chunks AS (
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
    r.link,
    r.calories,
    r.protein_g,
    r.carbs_g,
    r.fat_g,
    rk.content,
    rk.distance,
    rk.score
FROM ranked_chunks rk
JOIN recipes r ON r.id = rk.recipe_id
ORDER BY rk.score DESC, rk.distance ASC;
"""


def search_recipes(
    conn,
    classified: dict,
    top_k: int = 10,
) -> list[dict]:
    """
    Find the most relevant recipes using a 4-level weighted ingredient score.

    classified: dict with keys main, secondary, accompaniment, spices.
    Each key maps to a list of lowercase ingredient strings.

    Scoring: main*8 + secondary*4 + accompaniment*2 + spices*1
    Then ranked by vector cosine similarity as tiebreaker.
    """
    main         = [i.lower().strip() for i in classified.get("main", [])]
    secondary    = [i.lower().strip() for i in classified.get("secondary", [])]
    accompaniment = [i.lower().strip() for i in classified.get("accompaniment", [])]
    spices       = [i.lower().strip() for i in classified.get("spices", [])]

    all_ingredients = list(set(main + secondary + accompaniment + spices))

    if not all_ingredients:
        return []

    # Build embedding query emphasising main ingredients
    query_parts = []
    if main:
        query_parts.append("Main ingredient: " + ", ".join(main))
    if secondary:
        query_parts.append("with " + ", ".join(secondary))
    if accompaniment + spices:
        query_parts.append("seasoned with " + ", ".join(accompaniment + spices))
    query_text = " ".join(query_parts) or "Recipe with: " + ", ".join(all_ingredients)

    query_vec = embed_texts([query_text])[0].tolist()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            SQL_WEIGHTED_SEARCH,
            {
                "main":          main          or [""],
                "secondary":     secondary     or [""],
                "accompaniment": accompaniment or [""],
                "spices":        spices        or [""],
                "all_ingredients": all_ingredients,
                "query_vec":     query_vec,
                "top_k":         top_k,
            },
        )
        rows = cur.fetchall()

    results = []
    for row in rows:
        results.append({
            "id":         row["id"],
            "title":      row["title"],
            "ingredients": json.loads(row["ingredients"]) if isinstance(row["ingredients"], str) else row["ingredients"],
            "ner":        json.loads(row["ner"]) if isinstance(row["ner"], str) else row["ner"],
            "link":       row["link"],
            "calories":   row["calories"],
            "protein_g":  row["protein_g"],
            "carbs_g":    row["carbs_g"],
            "fat_g":      row["fat_g"],
            "chunk_text": row["content"],
            "distance":   float(row["distance"]),
            "score":      int(row["score"]),
        })

    return results
