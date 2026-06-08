"""
Checks de calidad de datos posteriores al ETL.

Comprueba el estado de las recetas cargadas: pasos estructurados ausentes,
ner vacios, duplicados dentro de ner e ingredientes mas frecuentes.

Uso:
    python -m src.etl.validate_etl
"""

import logging

import psycopg2

from src.config import get_dsn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run() -> None:
    conn = psycopg2.connect(get_dsn())
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM recipes;")
    total = cur.fetchone()[0]

    # 1. Recetas sin pasos estructurados (columna steps nula o vacia)
    cur.execute("SELECT COUNT(*) FROM recipes WHERE steps IS NULL OR jsonb_array_length(steps) = 0;")
    sin_steps = cur.fetchone()[0]

    # 2. Recetas con ner vacio
    cur.execute("SELECT COUNT(*) FROM recipes WHERE ner IS NULL OR jsonb_array_length(ner) = 0;")
    ner_vacio = cur.fetchone()[0]

    # 3. Recetas con ingredientes duplicados dentro de ner
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT id FROM recipes
            WHERE jsonb_array_length(ner) <> (
                SELECT COUNT(DISTINCT value) FROM jsonb_array_elements_text(ner)
            )
        ) t;
    """)
    con_duplicados = cur.fetchone()[0]

    # 4. Ingredientes normalizados mas frecuentes
    cur.execute("""
        SELECT value, COUNT(*) AS n
        FROM recipes, jsonb_array_elements_text(ner) AS value
        GROUP BY value
        ORDER BY n DESC
        LIMIT 15;
    """)
    top = cur.fetchall()

    conn.close()

    pct = lambda x: f"{(x / total * 100):.1f}%" if total else "-"
    logger.info("=" * 55)
    logger.info("VALIDACION DE CALIDAD POST-ETL")
    logger.info("  Recetas totales            : %d", total)
    logger.info("  Sin pasos estructurados    : %d (%s)", sin_steps, pct(sin_steps))
    logger.info("  Con ner vacio              : %d (%s)", ner_vacio, pct(ner_vacio))
    logger.info("  Con duplicados en ner      : %d (%s)", con_duplicados, pct(con_duplicados))
    logger.info("-" * 55)
    logger.info("  Top 15 ingredientes (ner normalizado):")
    for value, n in top:
        logger.info("    %-22s %d", value, n)
    logger.info("=" * 55)
    logger.info("Interpretacion: 'sin pasos' y 'duplicados' deberian tender a 0")
    logger.info("tras el ETL mejorado; un ner vacio indica receta sin ingredientes.")


if __name__ == "__main__":
    run()
