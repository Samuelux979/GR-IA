"""
Configuracion central de GR-IA: rutas, credenciales y parametros.

Las credenciales se leen de variables de entorno (fichero .env).
Copia .env.example a .env y ajusta los valores antes de ejecutar.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# Rutas principales
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "src" / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

PARQUET_PATH  = DATA_DIR / "foodcom_recipes.parquet"
PROGRESS_FILE = DATA_DIR / "etl_progress.json"

# Cargar variables de entorno desde .env (si existe)
load_dotenv(PROJECT_ROOT / ".env")

# PostgreSQL — host/puerto/nombre/usuario no son secretos y tienen
# valores por defecto; la contrasena debe definirse en el entorno.
DB_HOST     = os.getenv("GRIA_DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("GRIA_DB_PORT", "5432"))
DB_NAME     = os.getenv("GRIA_DB_NAME",     "gria_db")
DB_USER     = os.getenv("GRIA_DB_USER",     "admin")
DB_PASSWORD = os.getenv("GRIA_DB_PASSWORD")

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
# Timeout (segundos) de las peticiones a Ollama para no quedar colgados
# si el servidor no responde o el modelo esta cargando.
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# Umbral de plausibilidad (0-10) por debajo del cual una receta generada
# se considera incoherente y no se guarda.
PLAUSIBILITY_THRESHOLD = float(os.getenv("PLAUSIBILITY_THRESHOLD", "4"))


def get_dsn() -> str:
    if not DB_PASSWORD:
        raise RuntimeError(
            "Falta la variable de entorno GRIA_DB_PASSWORD. "
            "Copia .env.example a .env y define la contrasena."
        )
    return (
        f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
        f"user={DB_USER} password={DB_PASSWORD}"
    )
