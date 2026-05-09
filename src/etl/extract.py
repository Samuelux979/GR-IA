"""
Extract — chunked CSV reader for RecipeNLG.
Never loads the full file into RAM.
"""

import logging
from typing import Iterator

import pandas as pd

from src.config import CSV_PATH, CHUNK_SIZE

logger = logging.getLogger(__name__)

# RecipeNLG columns we care about
USE_COLS = ["title", "ingredients", "directions", "link", "source", "NER"]


def read_csv_chunks(
    skip_batches: int = 0,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """
    Yield (batch_index, DataFrame) tuples from the CSV.

    If *skip_batches* > 0 the first N chunks are skipped so the pipeline
    can resume from a checkpoint.  We use `skiprows` to avoid reading
    already-processed rows into memory at all.
    """
    skiprows = None
    if skip_batches > 0:
        # +1 because row 0 is the header and we must keep it
        skiprows = range(1, skip_batches * CHUNK_SIZE + 1)
        logger.info(
            "Skipping first %d batches (%d rows) for resume.",
            skip_batches,
            skip_batches * CHUNK_SIZE,
        )

    reader = pd.read_csv(
        CSV_PATH,
        chunksize=CHUNK_SIZE,
        usecols=USE_COLS,
        dtype=str,          # avoid costly type inference
        skiprows=skiprows,
        on_bad_lines="skip",
    )

    for i, chunk_df in enumerate(reader):
        batch_index = i + skip_batches
        logger.debug("Extracted batch %d (%d rows).", batch_index, len(chunk_df))
        yield batch_index, chunk_df
