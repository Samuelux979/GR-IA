"""
Load — batch insert recipes and chunks into PostgreSQL within a single transaction.
"""

import json
import logging

import numpy as np
import psycopg2
import psycopg2.extras

from src.etl.transform import RecipeDocument

logger = logging.getLogger(__name__)

SQL_INSERT_RECIPE = """
    INSERT INTO recipes (title, ingredients, ner, link, source, etl_batch_id)
    VALUES %s
    RETURNING id
"""

SQL_INSERT_CHUNK = """
    INSERT INTO recipe_chunks (recipe_id, chunk_index, content, embedding)
    VALUES %s
"""


def load_batch(
    conn,
    documents: list[RecipeDocument],
    embeddings: np.ndarray,
    batch_id: int,
) -> int:
    """
    Insert a batch of recipes + their chunks in one transaction.

    Returns the number of rows successfully inserted.
    """
    if not documents:
        return 0

    try:
        with conn.cursor() as cur:
            # ── Insert parent recipes ─────────────────────────────────
            recipe_values = [
                (
                    doc.title,
                    json.dumps(doc.ingredients),
                    json.dumps(doc.ner),
                    doc.link,
                    doc.source,
                    batch_id,
                )
                for doc in documents
            ]

            result = psycopg2.extras.execute_values(
                cur,
                SQL_INSERT_RECIPE,
                recipe_values,
                template="(%s, %s::jsonb, %s::jsonb, %s, %s, %s)",
                fetch=True,
            )
            recipe_ids = [row[0] for row in result]

            # ── Insert child chunks ───────────────────────────────────
            chunk_values = [
                (
                    recipe_id,
                    0,  # chunk_index (Phase 1: single chunk per recipe)
                    doc.chunk_text,
                    embeddings[i].tolist(),
                )
                for i, (recipe_id, doc) in enumerate(zip(recipe_ids, documents))
            ]

            psycopg2.extras.execute_values(
                cur,
                SQL_INSERT_CHUNK,
                chunk_values,
                template="(%s, %s, %s, %s::vector)",
            )

        conn.commit()
        return len(documents)

    except Exception:
        conn.rollback()
        logger.exception("Batch %d FAILED — rolled back.", batch_id)
        raise
