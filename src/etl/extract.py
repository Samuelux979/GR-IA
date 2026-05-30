"""
Carga el Parquet de Food.com, aplica filtros de calidad
y devuelve una muestra de 150k recetas lista para el ETL.
"""

import logging
from typing import Iterator

import numpy as np
import pandas as pd

from src.config import PARQUET_PATH, CHUNK_SIZE, SAMPLE_SIZE

logger = logging.getLogger(__name__)

USE_COLS = [
    "Name",
    "RecipeIngredientQuantities",
    "RecipeIngredientParts",
    "RecipeInstructions",
    "Calories",
    "FatContent",
    "CarbohydrateContent",
    "FiberContent",
    "ProteinContent",
    "RecipeCategory",
]


def _to_list(val) -> list:
    if isinstance(val, (list, np.ndarray)):
        return [str(v).strip() for v in val if str(v).strip()]
    return []


def load_and_filter() -> pd.DataFrame:
    """Carga el dataset, filtra recetas de baja calidad y devuelve una muestra aleatoria."""
    logger.info("Cargando %s ...", PARQUET_PATH)
    df = pd.read_parquet(PARQUET_PATH, columns=USE_COLS)
    logger.info("Dataset completo: %d recetas", len(df))

    df["_n_ing"]   = df["RecipeIngredientParts"].apply(lambda x: len(_to_list(x)))
    df["_n_steps"] = df["RecipeInstructions"].apply(lambda x: len(_to_list(x)))

    mask = (
        df["Calories"].between(50, 2000)          &
        df["ProteinContent"].between(0, 150)      &
        df["CarbohydrateContent"].between(0, 300) &
        df["FatContent"].between(0, 150)          &
        df["_n_ing"].between(2, 20)               &
        (df["_n_steps"] >= 1)
    )
    clean = df[mask].drop(columns=["_n_ing", "_n_steps"])
    logger.info("Tras filtros: %d recetas", len(clean))

    n = min(SAMPLE_SIZE, len(clean))
    sampled = clean.sample(n, random_state=42).reset_index(drop=True)
    logger.info("Muestra final: %d recetas", len(sampled))
    return sampled


def read_chunks(df: pd.DataFrame, skip_batches: int = 0) -> Iterator[tuple[int, pd.DataFrame]]:
    """Divide el DataFrame en batches de CHUNK_SIZE y los yielda uno a uno."""
    total = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for batch_id in range(skip_batches, total):
        start = batch_id * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, len(df))
        yield batch_id, df.iloc[start:end].copy()
