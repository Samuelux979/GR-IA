"""
Database schema management — creates tables, extensions, and post-ETL indices.
"""

import logging
import psycopg2
from src.config import get_dsn, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

SQL_EXTENSIONS = "CREATE EXTENSION IF NOT EXISTS vector;"

SQL_CREATE_RECIPES = f"""
CREATE TABLE IF NOT EXISTS recipes (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    ingredients     JSONB NOT NULL,
    ner             JSONB NOT NULL,
    link            TEXT,
    source          TEXT,
    -- Phase 2: nutritional metadata (NULL until enrichment)
    calories        FLOAT,
    protein_g       FLOAT,
    carbs_g         FLOAT,
    fat_g           FLOAT,
    fiber_g         FLOAT,
    -- ETL control
    created_at      TIMESTAMP DEFAULT NOW(),
    etl_batch_id    INTEGER
);
"""

SQL_CREATE_CHUNKS = f"""
CREATE TABLE IF NOT EXISTS recipe_chunks (
    id              SERIAL PRIMARY KEY,
    recipe_id       INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    chunk_index     SMALLINT NOT NULL DEFAULT 0,
    content         TEXT NOT NULL,
    embedding       VECTOR({EMBEDDING_DIMENSION}),
    created_at      TIMESTAMP DEFAULT NOW()
);
"""

# Indices created AFTER bulk load for performance
SQL_POST_ETL_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_recipes_ner ON recipes USING GIN (ner);",
    (
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON recipe_chunks "
        f"USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = 16, ef_construction = 64);"
    ),
    "CREATE INDEX IF NOT EXISTS idx_chunks_recipe_id ON recipe_chunks (recipe_id);",
]

SQL_VACUUM = [
    "VACUUM ANALYZE recipes;",
    "VACUUM ANALYZE recipe_chunks;",
]


def ensure_schema(conn) -> None:
    """Create extensions and tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(SQL_EXTENSIONS)
        cur.execute(SQL_CREATE_RECIPES)
        cur.execute(SQL_CREATE_CHUNKS)
    conn.commit()
    logger.info("Schema verified/created.")


def create_post_etl_indices(conn) -> None:
    """Create heavy indices after the bulk load is complete."""
    with conn.cursor() as cur:
        for sql in SQL_POST_ETL_INDICES:
            logger.info("Creating index: %s", sql[:80])
            cur.execute(sql)
    conn.commit()
    logger.info("All post-ETL indices created.")


def run_vacuum(dsn: str) -> None:
    """VACUUM ANALYZE requires autocommit."""
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        for sql in SQL_VACUUM:
            logger.info("Running: %s", sql)
            cur.execute(sql)
    conn.close()
    logger.info("VACUUM ANALYZE complete.")
