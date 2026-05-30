"""
Control de progreso del ETL mediante un fichero JSON.
Permite reanudar la carga desde el ultimo batch completado si se interrumpe.
"""

import json
import logging
import os
from datetime import datetime, timezone

from src.config import PROGRESS_FILE, PARQUET_PATH, CHUNK_SIZE

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_progress() -> dict:
    """Lee el fichero de progreso. Si no existe, devuelve valores iniciales."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            "Progreso encontrado - ultimo batch: %d (%d filas procesadas)",
            data["last_completed_batch"],
            data["total_rows_processed"],
        )
        return data

    return {
        "parquet_path":        str(PARQUET_PATH),
        "chunk_size":          CHUNK_SIZE,
        "last_completed_batch": -1,
        "total_rows_processed": 0,
        "total_rows_skipped":   0,
        "started_at":          _now_iso(),
        "last_updated":        _now_iso(),
    }


def save_progress(progress: dict) -> None:
    """Escribe el progreso en disco de forma atomica (write + rename)."""
    progress["last_updated"] = _now_iso()
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(PROGRESS_FILE)


def update_after_batch(progress: dict, batch_id: int, rows_loaded: int, rows_skipped: int) -> dict:
    """Actualiza los contadores tras completar un batch y guarda el progreso."""
    progress["last_completed_batch"]  = batch_id
    progress["total_rows_processed"] += rows_loaded
    progress["total_rows_skipped"]   += rows_skipped
    save_progress(progress)
    return progress
