"""
Migracion idempotente para anadir metadatos de procedencia a la tabla recipes.

Anade las columnas:
  - source        : 'foodcom' o 'ai_generated'
  - provenance    : JSONB con metadatos de generacion
  - steps         : JSONB con pasos estructurados
  - macros_known  : BOOLEAN, false si los valores nutricionales son estimados

Y migra las recetas existentes:
  - Recetas con etl_batch_id = -1 se marcan como source = 'ai_generated'
  - Sus macronutrientes pasan a NULL (los originales eran 0.0 inventados)
  - Las recetas IA reciben macros_known = FALSE

El script es idempotente: si las columnas ya existen, no falla; si los
datos ya estan migrados, no los modifica.

Uso:
    python -m src.etl.migrate_provenance
"""

import logging

import psycopg2

from src.config import get_dsn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALTER_STATEMENTS = [
    "ALTER TABLE recipes ADD COLUMN IF NOT EXISTS source       TEXT DEFAULT 'foodcom';",
    "ALTER TABLE recipes ADD COLUMN IF NOT EXISTS provenance   JSONB;",
    "ALTER TABLE recipes ADD COLUMN IF NOT EXISTS steps        JSONB;",
    "ALTER TABLE recipes ADD COLUMN IF NOT EXISTS macros_known BOOLEAN DEFAULT TRUE;",
    # La deduplicacion por hash se sustituyo por comprobacion de nombre:
    # se elimina la columna y su indice si existieran de versiones previas.
    "DROP INDEX IF EXISTS idx_recipes_dedup_hash;",
    "ALTER TABLE recipes DROP COLUMN IF EXISTS dedup_hash;",
]


def run() -> None:
    conn = psycopg2.connect(get_dsn())
    conn.autocommit = False
    cur = conn.cursor()

    # 1. Anadir columnas nuevas
    logger.info("Anadiendo columnas de procedencia a la tabla recipes...")
    for sql in ALTER_STATEMENTS:
        cur.execute(sql)
    conn.commit()

    # 2. Marcar como ai_generated las recetas con etl_batch_id = -1
    cur.execute("""
        UPDATE recipes
        SET source = 'ai_generated'
        WHERE etl_batch_id = -1 AND (source IS NULL OR source = 'foodcom');
    """)
    n_ai = cur.rowcount
    logger.info("Recetas marcadas como 'ai_generated': %d", n_ai)

    # 3. Marcar el resto explicitamente como foodcom
    cur.execute("""
        UPDATE recipes
        SET source = 'foodcom'
        WHERE source IS NULL;
    """)
    n_food = cur.rowcount
    logger.info("Recetas marcadas como 'foodcom': %d", n_food)

    # 4a. Las recetas IA con macros 0.0 inventadas: convertirlas a NULL
    cur.execute("""
        UPDATE recipes
        SET calories  = NULL, protein_g = NULL, fat_g = NULL,
            carbs_g   = NULL, fiber_g   = NULL
        WHERE source = 'ai_generated'
          AND calories = 0 AND protein_g = 0 AND fat_g = 0
          AND carbs_g = 0  AND fiber_g = 0;
    """)
    logger.info("Recetas IA con macros 0.0 convertidas a NULL: %d", cur.rowcount)

    # 4b. Todas las recetas IA tienen macros estimadas (nunca reales)
    cur.execute("""
        UPDATE recipes
        SET macros_known = FALSE
        WHERE source = 'ai_generated' AND macros_known IS DISTINCT FROM FALSE;
    """)
    logger.info("Recetas IA marcadas con macros_known=FALSE: %d", cur.rowcount)

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Migracion completada.")


if __name__ == "__main__":
    run()
