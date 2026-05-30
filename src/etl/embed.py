"""
Generacion de embeddings con sentence-transformers (all-MiniLM-L6-v2).
El modelo se descarga una vez y queda cacheado en models/.
"""

import logging
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, EMBED_BATCH_SIZE, MODELS_DIR

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*get_sentence_embedding_dimension.*", category=FutureWarning)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Carga el modelo de embeddings (singleton). Usa cache local si existe."""
    global _model
    if _model is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Cargando modelo de embeddings '%s'...", EMBEDDING_MODEL_NAME)
        try:
            _model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                cache_folder=str(MODELS_DIR),
                local_files_only=True,
            )
        except Exception:
            logger.info("Cache local no encontrada, descargando modelo...")
            _model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                cache_folder=str(MODELS_DIR),
            )
        logger.info("Modelo cargado. Dimension: %d", _model.get_embedding_dimension())
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Codifica una lista de textos y devuelve un array de embeddings normalizados."""
    model = get_model()
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
