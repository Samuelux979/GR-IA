"""
Configuracion central de GR-IA: rutas, credenciales y parametros.
"""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# Rutas principales
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "src" / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

PARQUET_PATH  = DATA_DIR / "foodcom_recipes.parquet"
PROGRESS_FILE = DATA_DIR / "etl_progress.json"

# PostgreSQL
DB_HOST     = os.getenv("GRIA_DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("GRIA_DB_PORT", "5432"))
DB_NAME     = os.getenv("GRIA_DB_NAME",     "gria_db")
DB_USER     = os.getenv("GRIA_DB_USER",     "admin")
DB_PASSWORD = os.getenv("GRIA_DB_PASSWORD", "adminpassword")

# Parametros ETL
SAMPLE_SIZE      = 150_000
CHUNK_SIZE       = 2_000
EMBED_BATCH_SIZE = 256

# Modelo de embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION  = 384

# Modelos Ollama
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL",         "http://localhost:11434")
OLLAMA_VISION_MODEL     = os.getenv("OLLAMA_VISION_MODEL",     "minicpm-v")
OLLAMA_TEXT_MODEL       = os.getenv("OLLAMA_TEXT_MODEL",       "qwen2.5:1.5b")
OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "qwen2.5:1.5b")


def get_dsn() -> str:
    return (
        f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
        f"user={DB_USER} password={DB_PASSWORD}"
    )
