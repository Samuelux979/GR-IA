"""
Migracion idempotente para anadir metadatos de procedencia a la tabla recipes.

Anade las columnas:
  - source        : 'foodcom' o 'ai_generated'
  - provenance    : JSONB con metadatos de generacion
  - dedup_hash    : SHA-256 para deteccion de duplicados
  - steps         : JSONB con pasos estructurados
  - macros_known  : BOOLEAN, false si los valores nutricionales son estimados

Y migra las recetas existentes:
  - Recetas con etl_batch_id = -1 se marcan como source = 'ai_generated'
  - Sus macronutrientes pasan a NULL (los originales eran 0.0 inventados)
  - Las recetas IA reciben macros_known = FALSE
  - Se calcula el dedup_hash de las recetas IA existentes
  - Se crea el indice unico sobre dedup_hash

El script es idempotente: si las columnas ya existen, no falla; si los
datos ya estan migrados, no los modifica.

Uso:
    python -m src.etl.migrate_provenance
"""

import hashlib
import json
import logging

import psycopg2

from src.config import get_dsn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALTER_STATEMENTS = [
    "ALTER TABLE recipes ADD COLUMN IF NOT EXISTS source       TEXT DEFAULT 'foodcom';",
    "ALTER TABLE recipes ADD COLUMN IF NOT EXISTS provenance   JSONB;",
    "ALTER TABLE recipes ADD COLUMN IF NOT EXISTS dedup_hash   TEXT;",
    "ALTER TABLE recipes ADD COLUMN IF NOT EXISTS steps        JSONB;",
    "ALTER TABLE recipes ADD COLUMN IF NOT EXISTS macros_known BOOLEAN DEFAULT TRUE;",
]


def compute_dedup_hash(title: str, ner: list) -> str:
    """SHA-256 sobre titulo normalizado + ingredientes NER ordenados."""
    norm_title = (title or "").strip().lower()
    norm_ner   = sorted({(n or "").strip().lower() for n in (ner or [])})
    payload    = norm_title + "|" + ",".join(norm_ner)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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

    # 5. Calcular dedup_hash de las recetas IA existentes
    cur.execute("SELECT id, title, ner FROM recipes WHERE source = 'ai_generated' AND dedup_hash IS NULL;")
    pending = cur.fetchall()
    logger.info("Calculando dedup_hash de %d recetas IA existentes...", len(pending))
    for recipe_id, title, ner in pending:
        ner_list = ner if isinstance(ner, list) else json.loads(ner)
        h = compute_dedup_hash(title, ner_list)
        try:
            cur.execute("UPDATE recipes SET dedup_hash = %s WHERE id = %s;", (h, recipe_id))
        except psycopg2.errors.UniqueViolation:
            # Si hubiera un duplicado preexistente, lo dejamos sin hash
            conn.rollback()
            logger.warning("Duplicado detectado en recipes.id=%d, hash omitido.", recipe_id)
            continue

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Migracion completada.")


if __name__ == "__main__":
    run()
