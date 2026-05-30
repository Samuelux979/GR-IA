"""
Traduccion EN->ES usando Helsinki-NLP/opus-mt-en-es (MarianMT).
Modelo especializado en traduccion, mejor calidad que un LLM de proposito general.
Se descarga una sola vez y queda cacheado en models/.
"""

import logging
import warnings
from functools import lru_cache

from src.config import MODELS_DIR

logger  = logging.getLogger(__name__)
_MODEL  = "Helsinki-NLP/opus-mt-en-es"
_CACHE  = str(MODELS_DIR / "opus-mt-en-es")


@lru_cache(maxsize=1)
def _get_translator():
    from transformers import MarianMTModel, MarianTokenizer

    logger.info("Cargando modelo de traduccion...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tokenizer = MarianTokenizer.from_pretrained(_MODEL, cache_dir=_CACHE, local_files_only=True)
            model     = MarianMTModel.from_pretrained(_MODEL,  cache_dir=_CACHE, local_files_only=True)
            logger.info("Modelo de traduccion cargado desde cache local.")
        except Exception:
            logger.info("Descargando modelo de traduccion (primera ejecucion)...")
            tokenizer = MarianTokenizer.from_pretrained(_MODEL, cache_dir=_CACHE)
            model     = MarianMTModel.from_pretrained(_MODEL,  cache_dir=_CACHE)
    return tokenizer, model


def translate_en_to_es(texts: list[str]) -> list[str]:
    """Traduce una lista de cadenas de ingles a espanol."""
    if not texts:
        return []
    tokenizer, model = _get_translator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inputs     = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs, num_beams=4)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)


def translate_steps_to_es(steps: list[str]) -> list[str]:
    """Traduce los pasos de coccion al espanol."""
    return translate_en_to_es(steps)
