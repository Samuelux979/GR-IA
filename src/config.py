"""
Configuración global de GR-IA — rutas, credenciales y parámetros ajustables.
Centraliza todo para que ningún otro módulo tenga valores hardcodeados.
"""

import os
from pathlib import Path

# Silence HuggingFace Hub warnings (symlinks + unauthenticated)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent   # GR-IA/
DATA_DIR      = PROJECT_ROOT / "src" / "data"
CSV_PATH      = DATA_DIR / "RecipeNLG.csv"
PROGRESS_FILE = DATA_DIR / "etl_progress.json"
MODELS_DIR    = PROJECT_ROOT / "models"                  # caché local de modelos

# ─── PostgreSQL ───────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("GRIA_DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("GRIA_DB_PORT", "5432"))
DB_NAME     = os.getenv("GRIA_DB_NAME",     "gria_db")
DB_USER     = os.getenv("GRIA_DB_USER",     "admin")
DB_PASSWORD = os.getenv("GRIA_DB_PASSWORD", "adminpassword")

# ─── ETL Tuning ───────────────────────────────────────────────────────────────
CHUNK_SIZE       = 5_000   # filas por chunk de CSV
EMBED_BATCH_SIZE = 256     # frases por lote de embedding

# ─── Modelo de embeddings ─────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION  = 384

# ─── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL",         "http://localhost:11434")
OLLAMA_VISION_MODEL     = os.getenv("OLLAMA_VISION_MODEL",     "minicpm-v")    # detección visual
OLLAMA_TEXT_MODEL       = os.getenv("OLLAMA_TEXT_MODEL",       "qwen2.5:1.5b") # clasificación
OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "qwen2.5:1.5b") # generación


def get_dsn() -> str:
    """Devuelve el connection string de psycopg2."""
    return (
        f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
        f"user={DB_USER} password={DB_PASSWORD}"
    )
