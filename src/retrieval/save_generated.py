"""
Guarda en la base de datos una receta generada por el modelo
si el usuario decide aprobarla.
"""

import json
import logging

import psycopg2

from src.etl.embed import embed_texts

logger = logging.getLogger(__name__)


def _build_chunk_text(title: str, ner: list, steps: list) -> str:
    ner_str   = ", ".join(ner)
    steps_str = " ".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))
    return f"Recipe: {title}. Ingredients: {ner_str}. {steps_str}"


def save_generated_recipe(conn, recipe: dict) -> int:
    """
    Inserta la receta en recipes y su embedding en recipe_chunks.
    Devuelve el id asignado. Las recetas generadas llevan etl_batch_id = -1.
    """
    title      = recipe["title"]
    steps      = recipe["steps"]
    ner        = recipe["ner"]
    chunk_text = _build_chunk_text(title, ner, steps)
    embedding  = embed_texts([chunk_text])[0].tolist()

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recipes (title, ingredients, ner, category,
                                 calories, protein_g, fat_g, carbs_g, fiber_g, etl_batch_id)
            VALUES (%s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                title,
                json.dumps(recipe["ingredients"], ensure_ascii=False),
                json.dumps(ner, ensure_ascii=False),
                recipe.get("category", "AI Generated"),
                recipe.get("calories",  0.0),
                recipe.get("protein_g", 0.0),
                recipe.get("fat_g",     0.0),
                recipe.get("carbs_g",   0.0),
                recipe.get("fiber_g",   0.0),
                -1,
            ),
        )
        recipe_id = cur.fetchone()[0]

        cur.execute(
            "INSERT INTO recipe_chunks (recipe_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s::vector)",
            (recipe_id, 0, chunk_text, embedding),
        )

    conn.commit()
    logger.info("Receta '%s' guardada con id=%d", title, recipe_id)
    return recipe_id
