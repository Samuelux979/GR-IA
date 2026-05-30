"""
Orquestador del ETL: carga el dataset Food.com en PostgreSQL.

Flujo:
  1. Lee y filtra el Parquet (150k recetas)
  2. Por cada batch: transforma -> vectoriza -> inserta -> guarda checkpoint

Uso:
    python -m src.etl.run_etl
"""

import logging
import sys
import time

import psycopg2

from src.config import get_dsn, PARQUET_PATH
from src.etl.schema     import ensure_schema, create_post_etl_indices, run_vacuum
from src.etl.extract    import load_and_filter, read_chunks
from src.etl.transform  import transform_chunk
from src.etl.embed      import embed_texts, get_model
from src.etl.load       import load_batch
from src.etl.checkpoint import load_progress, update_after_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("etl")


def run() -> None:
    if not PARQUET_PATH.exists():
        logger.error("Parquet no encontrado en %s", PARQUET_PATH)
        sys.exit(1)

    dsn  = get_dsn()
    conn = psycopg2.connect(dsn)
    ensure_schema(conn)
    get_model()

    progress    = load_progress()
    start_batch = progress["last_completed_batch"] + 1

    if start_batch > 0:
        logger.info("Reanudando desde batch %d...", start_batch)

    df             = load_and_filter()
    t_start        = time.time()
    total_loaded   = progress["total_rows_processed"]
    total_skipped  = progress["total_rows_skipped"]

    try:
        for batch_id, chunk_df in read_chunks(df, skip_batches=start_batch):
            t0     = time.time()
            result = transform_chunk(chunk_df)

            if not result.documents:
                logger.warning("Batch %d: sin documentos validos, saltando.", batch_id)
                progress = update_after_batch(progress, batch_id, 0, result.rows_discarded)
                continue

            texts      = [doc.chunk_text for doc in result.documents]
            embeddings = embed_texts(texts)
            loaded     = load_batch(conn, result.documents, embeddings, batch_id)
            progress   = update_after_batch(progress, batch_id, loaded, result.rows_discarded)

            total_loaded  += loaded
            total_skipped += result.rows_discarded

            logger.info(
                "Batch %d - %d cargadas, %d descartadas (%.1fs) | Total: %d",
                batch_id, loaded, result.rows_discarded, time.time() - t0, total_loaded,
            )

    except KeyboardInterrupt:
        logger.info("Interrumpido. El progreso se ha guardado.")
        conn.close()
        sys.exit(0)

    logger.info("Carga completa. Creando indices...")
    create_post_etl_indices(conn)
    conn.close()

    logger.info("Ejecutando VACUUM ANALYZE...")
    run_vacuum(dsn)

    elapsed = time.time() - t_start
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)

    logger.info("=" * 50)
    logger.info("ETL COMPLETADO")
    logger.info("  Recetas cargadas  : %d", total_loaded)
    logger.info("  Filas descartadas : %d", total_skipped)
    logger.info("  Tiempo total      : %dh %dm %ds", h, m, s)
    logger.info("=" * 50)


if __name__ == "__main__":
    run()
