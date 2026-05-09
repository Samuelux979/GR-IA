"""
Checkpoint — fault-tolerant progress tracking via a JSON file.

If the machine crashes at batch 170, the next run resumes from 171.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROGRESS_FILE, CSV_PATH, CHUNK_SIZE

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_progress() -> dict:
    """Read the checkpoint file. Returns defaults if it doesn't exist."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            "Checkpoint found — last completed batch: %d (%d rows processed).",
            data["last_completed_batch"],
            data["total_rows_processed"],
        )
        return data

    return {
        "csv_path": str(CSV_PATH),
        "chunk_size": CHUNK_SIZE,
        "last_completed_batch": -1,
        "total_rows_processed": 0,
        "total_rows_skipped": 0,
        "started_at": _now_iso(),
        "last_updated": _now_iso(),
    }


def save_progress(progress: dict) -> None:
    """Atomically write progress to disk (write-then-rename)."""
    progress["last_updated"] = _now_iso()
    tmp_path = PROGRESS_FILE.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(PROGRESS_FILE)


def update_after_batch(
    progress: dict,
    batch_id: int,
    rows_loaded: int,
    rows_skipped: int,
) -> dict:
    """Update progress counters after a successful batch."""
    progress["last_completed_batch"] = batch_id
    progress["total_rows_processed"] += rows_loaded
    progress["total_rows_skipped"] += rows_skipped
    save_progress(progress)
    return progress
