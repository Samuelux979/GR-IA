"""
Migracion idempotente de calidad de datos para recetas ya cargadas.

Aplica a las recetas existentes las mejoras del ETL sin re-ejecutar la carga
completa (no re-embebe, que es lo costoso):
  - Normaliza y deduplica 'ner' con el normalizador de la busqueda.
  - Rellena la columna 'steps' parseando los pasos desde chunk_text.

Solo toca recetas del dataset (source = 'foodcom'); las generadas por IA ya
guardan 'steps' de forma estructurada.

Uso:
    python -m src.etl.migrate_etl_quality
"""

import json
import logging

import psycopg2

from src.config import get_dsn
from src.retrieval.ingredient_normalizer import normalize_ingredient
from src.vision.llava_client import _parse_steps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BATCH = 2000


def _normalize_ner(ner: list) -> list:
    seen, out = set(), []
    for p in (ner or []):
        n = normalize_ingredient(p, fuzzy=False)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def run() -> None:
    conn = psycopg2.connect(get_dsn())
    cur  = conn.cursor()

    # Recetas del dataset que aun no tienen pasos estructurados
    cur.execute("""
        SELECT r.id, r.ner, rc.content
        FROM recipes r
        JOIN recipe_chunks rc ON rc.recipe_id = r.id
        WHERE COALESCE(r.source, 'foodcom') = 'foodcom'
          AND (r.steps IS NULL OR jsonb_array_length(r.steps) = 0);
    """)
    rows = cur.fetchall()
    logger.info("Recetas a migrar: %d", len(rows))

    updated = 0
    for i, (rid, ner, content) in enumerate(rows, 1):
        ner_list = ner if isinstance(ner, list) else json.loads(ner)
        new_ner  = _normalize_ner(ner_list)
        steps    = _parse_steps(content or "")

        cur.execute(
            "UPDATE recipes SET ner = %s::jsonb, steps = %s::jsonb WHERE id = %s;",
            (json.dumps(new_ner, ensure_ascii=False),
             json.dumps(steps,   ensure_ascii=False),
             rid),
        )
        updated += 1
        if i % BATCH == 0:
            conn.commit()
            logger.info("  %d / %d migradas...", i, len(rows))

    conn.commit()
    conn.close()
    logger.info("Migracion completada: %d recetas actualizadas.", updated)


if __name__ == "__main__":
    run()
