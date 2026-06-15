"""
Gestion del schema de la base de datos: tablas, extension pgvector e indices.
"""

import logging
import psycopg2
from src.config import get_dsn, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

SQL_CREATE_RECIPES = f"""
CREATE TABLE IF NOT EXISTS recipes (
    id            SERIAL PRIMARY KEY,
    title         TEXT NOT NULL,
    ingredients   JSONB NOT NULL,
    ner           JSONB NOT NULL,
    steps         JSONB,
    category      TEXT,
    -- Procedencia
    source        TEXT DEFAULT 'foodcom',  -- 'foodcom' | 'ai_generated'
    provenance    JSONB,                   -- metadatos de generacion (solo IA)
    -- Nutricion (NULL si se desconoce)
    calories      FLOAT,
    protein_g     FLOAT,
    fat_g         FLOAT,
    carbs_g       FLOAT,
    fiber_g       FLOAT,
    macros_known  BOOLEAN DEFAULT TRUE,    -- false = valores estimados con USDA
    -- Control ETL
    created_at    TIMESTAMP DEFAULT NOW(),
    etl_batch_id  INTEGER
);
"""

SQL_CREATE_CHUNKS = f"""
CREATE TABLE IF NOT EXISTS recipe_chunks (
    id          SERIAL PRIMARY KEY,
    recipe_id   INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    chunk_index SMALLINT NOT NULL DEFAULT 0,
    content     TEXT NOT NULL,
    embedding   VECTOR({EMBEDDING_DIMENSION}),
    created_at  TIMESTAMP DEFAULT NOW()
);
"""

SQL_POST_ETL_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_recipes_ner ON recipes USING GIN (ner);",
    "CREATE INDEX IF NOT EXISTS idx_recipes_source ON recipes (source);",
    (
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON recipe_chunks "
        f"USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
    ),
    "CREATE INDEX IF NOT EXISTS idx_chunks_recipe_id ON recipe_chunks (recipe_id);",
]

SQL_VACUUM = [
    "VACUUM ANALYZE recipes;",
    "VACUUM ANALYZE recipe_chunks;",
]


def ensure_schema(conn) -> None:
    """Crea la extension pgvector y las tablas si no existen."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(SQL_CREATE_RECIPES)
        cur.execute(SQL_CREATE_CHUNKS)
    conn.commit()
    logger.info("Schema verificado.")


def create_post_etl_indices(conn) -> None:
    """Crea los indices tras la carga masiva de datos."""
    with conn.cursor() as cur:
        for sql in SQL_POST_ETL_INDICES:
            logger.info("Creando indice: %s", sql[:80])
            cur.execute(sql)
    conn.commit()
    logger.info("Indices creados.")


def run_vacuum(dsn: str) -> None:
    """Ejecuta VACUUM ANALYZE (requiere autocommit)."""
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        for sql in SQL_VACUUM:
            logger.info("Ejecutando: %s", sql)
            cur.execute(sql)
    conn.close()
    logger.info("VACUUM ANALYZE completado.")
