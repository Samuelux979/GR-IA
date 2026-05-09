"""
Embed — batch embedding generation using sentence-transformers.
"""

import logging
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, EMBED_BATCH_SIZE, MODELS_DIR

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None

# Suppress unrelated FutureWarning from sentence-transformers internals
warnings.filterwarnings("ignore", message=".*get_sentence_embedding_dimension.*", category=FutureWarning)


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton). Uses local cache to avoid HF requests."""
    global _model
    if _model is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
        try:
            # Load from local cache — no internet needed after first download
            _model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                cache_folder=str(MODELS_DIR),
                local_files_only=True,
            )
        except Exception:
            # First run: download and cache locally
            logger.info("Local cache not found, downloading model...")
            _model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                cache_folder=str(MODELS_DIR),
            )
        logger.info("Model loaded. Dimension: %d", _model.get_embedding_dimension())
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into embeddings.

    Returns ndarray of shape (len(texts), EMBEDDING_DIMENSION).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    )
    return embeddings
