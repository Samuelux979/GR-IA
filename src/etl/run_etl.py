"""
ETL Orchestrator — the main entry point.

Usage:
    python -m src.etl.run_etl
"""

import logging
import sys
import time

import psycopg2

from src.config import get_dsn, CSV_PATH, CHUNK_SIZE
from src.etl.schema import ensure_schema, create_post_etl_indices, run_vacuum
from src.etl.extract import read_csv_chunks
from src.etl.transform import transform_chunk
from src.etl.embed import embed_texts, get_model
from src.etl.load import load_batch
from src.etl.checkpoint import load_progress, update_after_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("etl")


def run() -> None:
    # ── Pre-flight checks ─────────────────────────────────────────────────
    if not CSV_PATH.exists():
        logger.error("CSV not found at %s. Place RecipeNLG.csv in src/data/.", CSV_PATH)
        sys.exit(1)

    dsn = get_dsn()
    logger.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(dsn)

    logger.info("Verifying schema...")
    ensure_schema(conn)

    # Pre-load embedding model so we measure its time once
    get_model()

    # ── Resume logic ──────────────────────────────────────────────────────
    progress = load_progress()
    start_batch = progress["last_completed_batch"] + 1

    if start_batch > 0:
        logger.info("Resuming from batch %d.", start_batch)

    # ── Main loop ─────────────────────────────────────────────────────────
    t_start = time.time()
    total_loaded = progress["total_rows_processed"]
    total_skipped = progress["total_rows_skipped"]

    try:
        for batch_id, chunk_df in read_csv_chunks(skip_batches=start_batch):
            t_batch = time.time()

            # 1. Transform
            result = transform_chunk(chunk_df)
            if not result.documents:
                logger.warning("Batch %d: 0 valid documents, skipping.", batch_id)
                progress = update_after_batch(
                    progress, batch_id, 0, result.rows_discarded
                )
                continue

            # 2. Embed
            texts = [doc.chunk_text for doc in result.documents]
            embeddings = embed_texts(texts)

            # 3. Load
            rows_loaded = load_batch(conn, result.documents, embeddings, batch_id)

            # 4. Checkpoint
            progress = update_after_batch(
                progress, batch_id, rows_loaded, result.rows_discarded
            )
            total_loaded += rows_loaded
            total_skipped += result.rows_discarded

            elapsed = time.time() - t_batch
            logger.info(
                "Batch %d — %d loaded, %d skipped (%.1fs) | Total: %d recipes",
                batch_id,
                rows_loaded,
                result.rows_discarded,
                elapsed,
                total_loaded,
            )

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Progress saved — safe to resume.")
        conn.close()
        sys.exit(0)

    # ── Post-ETL: indices ─────────────────────────────────────────────────
    logger.info("Bulk load complete. Creating indices (this may take a while)...")
    create_post_etl_indices(conn)
    conn.close()

    logger.info("Running VACUUM ANALYZE...")
    run_vacuum(dsn)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    hours, remainder = divmod(int(elapsed_total), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("=" * 60)
    logger.info("ETL COMPLETE")
    logger.info("  Recipes loaded  : %d", total_loaded)
    logger.info("  Rows skipped    : %d", total_skipped)
    logger.info("  Total time      : %dh %dm %ds", hours, minutes, seconds)
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
