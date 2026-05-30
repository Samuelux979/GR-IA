"""
Inserta recetas y chunks en PostgreSQL en una unica transaccion por batch.
"""

import json
import logging

import numpy as np
import psycopg2.extras

from src.etl.transform import RecipeDocument

logger = logging.getLogger(__name__)

SQL_INSERT_RECIPE = """
    INSERT INTO recipes (title, ingredients, ner, category,
                         calories, protein_g, fat_g, carbs_g, fiber_g, etl_batch_id)
    VALUES %s
    RETURNING id
"""

SQL_INSERT_CHUNK = """
    INSERT INTO recipe_chunks (recipe_id, chunk_index, content, embedding)
    VALUES %s
"""


def load_batch(conn, documents: list[RecipeDocument], embeddings: np.ndarray, batch_id: int) -> int:
    """
    Inserta un batch de recetas y sus chunks en una transaccion.
    Devuelve el numero de filas insertadas.
    """
    if not documents:
        return 0

    try:
        with conn.cursor() as cur:
            recipe_values = [
                (
                    doc.title,
                    json.dumps(doc.ingredients, ensure_ascii=False),
                    json.dumps(doc.ner,         ensure_ascii=False),
                    doc.category,
                    doc.calories, doc.protein_g, doc.fat_g, doc.carbs_g, doc.fiber_g,
                    batch_id,
                )
                for doc in documents
            ]

            rows = psycopg2.extras.execute_values(
                cur, SQL_INSERT_RECIPE, recipe_values,
                template="(%s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)",
                fetch=True,
            )
            recipe_ids = [row[0] for row in rows]

            chunk_values = [
                (recipe_id, 0, doc.chunk_text, embeddings[i].tolist())
                for i, (recipe_id, doc) in enumerate(zip(recipe_ids, documents))
            ]
            psycopg2.extras.execute_values(
                cur, SQL_INSERT_CHUNK, chunk_values,
                template="(%s, %s, %s, %s::vector)",
            )

        conn.commit()
        return len(documents)

    except Exception:
        conn.rollback()
        logger.exception("Batch %d fallido, rollback.", batch_id)
        raise
