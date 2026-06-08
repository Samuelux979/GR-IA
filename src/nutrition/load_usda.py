"""
Carga la base de datos nutricional USDA SR Legacy en PostgreSQL.

Crea la tabla nutrition_usda con un fila por alimento y los 5 macronutrientes
principales (kcal, proteina, grasa, carbos, fibra) por 100 g.

Uso:
    python -m src.nutrition.load_usda             # carga si la tabla no existe / esta vacia
    python -m src.nutrition.load_usda --recreate  # recrea la tabla aunque ya tenga datos
"""

import argparse
import logging
import sys

import pandas as pd
import psycopg2

from src.config import get_dsn, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SR_DIR = DATA_DIR / "sr_legacy"

# IDs de nutrientes en USDA (verificados en nutrient.csv)
NUTRIENT_IDS = {
    1008: "calories",   # Energy (kcal)
    1003: "protein_g",  # Protein
    1004: "fat_g",      # Total lipid (fat)
    1005: "carbs_g",    # Carbohydrate, by difference
    1079: "fiber_g",    # Fiber, total dietary
}

# Categorias que nos interesan (ingredientes crudos y basicos)
# Excluimos comida procesada, fast food, marcas comerciales, dulces
EXCLUDED_CATEGORIES = {
    18,  # Baked Products
    19,  # Sweets
    21,  # Fast Foods
    22,  # Meals, Entrees, and Side Dishes
    25,  # Snacks
    35,  # American Indian/Alaska Native Foods
    36,  # Restaurant Foods
}

SQL_CREATE_TABLE = """
CREATE TABLE nutrition_usda (
    fdc_id      INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    name_lower  TEXT NOT NULL,
    category_id INTEGER,
    calories    FLOAT DEFAULT 0,
    protein_g   FLOAT DEFAULT 0,
    fat_g       FLOAT DEFAULT 0,
    carbs_g     FLOAT DEFAULT 0,
    fiber_g     FLOAT DEFAULT 0
);
CREATE INDEX idx_nutrition_name ON nutrition_usda USING GIN (to_tsvector('english', name_lower));
CREATE INDEX idx_nutrition_name_trgm ON nutrition_usda USING GIN (name_lower gin_trgm_ops);
"""


def _table_row_count(cur) -> int | None:
    """Devuelve el numero de filas de nutrition_usda, o None si no existe."""
    cur.execute("SELECT to_regclass('public.nutrition_usda');")
    if cur.fetchone()[0] is None:
        return None
    cur.execute("SELECT COUNT(*) FROM nutrition_usda;")
    return cur.fetchone()[0]


def load(recreate: bool = False) -> None:
    # Comprobacion previa: no destruir datos existentes sin --recreate
    conn = psycopg2.connect(get_dsn())
    with conn.cursor() as cur:
        existing = _table_row_count(cur)
        if existing is not None and existing > 0 and not recreate:
            conn.close()
            logger.error(
                "La tabla nutrition_usda ya contiene %d filas. "
                "Usa --recreate para recrearla (se borraran los datos actuales).",
                existing,
            )
            sys.exit(1)

    logger.info("Leyendo CSVs de USDA SR Legacy...")
    food = pd.read_csv(SR_DIR / "food.csv")
    food_nut = pd.read_csv(SR_DIR / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"])

    logger.info("Alimentos totales: %d", len(food))

    # Filtrar categorias no deseadas
    food = food[~food["food_category_id"].isin(EXCLUDED_CATEGORIES)].copy()
    logger.info("Tras filtrar categorias: %d", len(food))

    # Pivotar nutrientes (de filas a columnas)
    food_nut = food_nut[food_nut["nutrient_id"].isin(NUTRIENT_IDS)]
    pivot = food_nut.pivot_table(
        index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first"
    ).rename(columns=NUTRIENT_IDS).reset_index()

    # Unir food con sus nutrientes
    merged = food.merge(pivot, on="fdc_id", how="left").fillna(0)

    # Crear tabla (se elimina la anterior solo en este punto, ya validado)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("DROP TABLE IF EXISTS nutrition_usda;")
        cur.execute(SQL_CREATE_TABLE)

        # Inserciones por lotes
        from psycopg2.extras import execute_values
        rows = [
            (
                int(r["fdc_id"]),
                r["description"],
                r["description"].lower(),
                int(r["food_category_id"]) if pd.notna(r["food_category_id"]) else None,
                float(r.get("calories",  0)),
                float(r.get("protein_g", 0)),
                float(r.get("fat_g",     0)),
                float(r.get("carbs_g",   0)),
                float(r.get("fiber_g",   0)),
            )
            for _, r in merged.iterrows()
        ]
        execute_values(
            cur,
            "INSERT INTO nutrition_usda (fdc_id, name, name_lower, category_id, "
            "calories, protein_g, fat_g, carbs_g, fiber_g) VALUES %s",
            rows,
        )
    conn.commit()
    conn.close()
    logger.info("Tabla nutrition_usda cargada con %d alimentos.", len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carga la base nutricional USDA en PostgreSQL")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recrea la tabla aunque ya contenga datos (los borra y vuelve a cargar)",
    )
    args = parser.parse_args()
    load(recreate=args.recreate)
