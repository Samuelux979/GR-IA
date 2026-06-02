"""
Persistencia enriquecida de recetas generadas por IA.

Cada receta guardada incluye:
  - source = 'ai_generated' para distinguirla del dataset original
  - dedup_hash que identifica recetas equivalentes
  - provenance con metadatos de generacion (modelo, fecha, ingredientes input)
  - macros_known = FALSE para indicar que la nutricion es estimada via USDA
  - pasos estructurados en JSONB (no solo en chunk_text)

El insert se hace de forma transaccional y deduplicada: si una receta
equivalente ya existe se devuelve su id sin volver a insertar.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone

from src.etl.embed import embed_texts
from src.retrieval.validation import validate_recipe

logger = logging.getLogger(__name__)


class RecipeValidationError(ValueError):
    """Lanzada cuando una receta no cumple los minimos de calidad."""

    def __init__(self, errors: list[str]):
        super().__init__("Receta invalida: " + " | ".join(errors))
        self.errors = errors


def compute_dedup_hash(title: str, ner: list) -> str:
    """
    Calcula un hash SHA-256 que identifica recetas equivalentes.
    Normaliza titulo y NER (minusculas, sin espacios extra, sin orden)
    para que pequenas variaciones no generen hashes distintos.
    """
    norm_title = (title or "").strip().lower()
    norm_ner   = sorted({(n or "").strip().lower() for n in (ner or []) if n})
    payload    = norm_title + "|" + ",".join(norm_ner)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_chunk_text(title: str, ner: list, steps: list) -> str:
    """Texto semantico que se vectorizara y guardara en recipe_chunks."""
    ner_str   = ", ".join(ner)
    steps_str = " ".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))
    return f"Recipe: {title}. Ingredients: {ner_str}. {steps_str}"


def _build_provenance(recipe: dict) -> dict:
    """Construye el diccionario de metadatos de procedencia."""
    return {
        "model":            recipe.get("model", "qwen2.5:1.5b"),
        "embedding_model":  recipe.get("embedding_model", "all-MiniLM-L6-v2"),
        "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_ingredients": recipe.get("input_ingredients", {}),
        "prompt_version":   recipe.get("prompt_version", "v1"),
    }


def save_generated_recipe(conn, recipe: dict) -> tuple[int, bool]:
    """
    Persiste una receta generada por IA aplicando validacion y deduplicacion.

    Devuelve una tupla (recipe_id, was_new) donde was_new es False si la
    receta ya existia y se devuelve su id sin insertar duplicados.

    Lanza RecipeValidationError si la receta no cumple los minimos.
    """
    errors = validate_recipe(recipe)
    if errors:
        raise RecipeValidationError(errors)

    title       = recipe["title"].strip()
    ingredients = recipe["ingredients"]
    steps       = recipe["steps"]
    ner         = recipe["ner"]

    dedup_hash = compute_dedup_hash(title, ner)

    # Comprobacion de duplicado antes de insertar
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM recipes WHERE dedup_hash = %s LIMIT 1;", (dedup_hash,))
        existing = cur.fetchone()
        if existing:
            logger.info("Receta duplicada detectada (hash=%s), devolviendo id existente %d.",
                        dedup_hash[:12], existing[0])
            return existing[0], False

    # Macros estimadas (las que devolvio el calculador USDA en generate_recipe).
    # NULL si valen 0 (no se pudieron calcular).
    def _or_null(v):
        return float(v) if v is not None and float(v) > 0 else None

    macros = {
        "calories":  _or_null(recipe.get("calories")),
        "protein_g": _or_null(recipe.get("protein_g")),
        "fat_g":     _or_null(recipe.get("fat_g")),
        "carbs_g":   _or_null(recipe.get("carbs_g")),
        "fiber_g":   _or_null(recipe.get("fiber_g")),
    }
    macros_known = any(v is not None for v in macros.values())

    chunk_text = _build_chunk_text(title, ner, steps)
    embedding  = embed_texts([chunk_text])[0].tolist()
    provenance = _build_provenance(recipe)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO recipes (
                    title, ingredients, ner, steps, category,
                    source, provenance, dedup_hash,
                    calories, protein_g, fat_g, carbs_g, fiber_g,
                    macros_known, etl_batch_id
                )
                VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                        %s, %s::jsonb, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s)
                RETURNING id
                """,
                (
                    title,
                    json.dumps(ingredients, ensure_ascii=False),
                    json.dumps(ner,         ensure_ascii=False),
                    json.dumps(steps,       ensure_ascii=False),
                    "AI Generated",
                    "ai_generated",
                    json.dumps(provenance, ensure_ascii=False),
                    dedup_hash,
                    macros["calories"], macros["protein_g"], macros["fat_g"],
                    macros["carbs_g"],  macros["fiber_g"],
                    False if not macros_known else False,  # IA -> siempre estimada
                    -1,
                ),
            )
            recipe_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO recipe_chunks (recipe_id, chunk_index, content, embedding) "
                "VALUES (%s, %s, %s, %s::vector)",
                (recipe_id, 0, chunk_text, embedding),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Fallo al guardar receta IA, transaccion revertida.")
        raise

    logger.info("Receta IA '%s' guardada con id=%d (hash=%s).",
                title, recipe_id, dedup_hash[:12])
    return recipe_id, True
