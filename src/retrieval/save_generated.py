"""
Persistencia enriquecida de recetas generadas por IA.

Cada receta guardada incluye:
  - source = 'ai_generated' para distinguirla del dataset original
  - provenance con metadatos de generacion (modelo, fecha, ingredientes input)
  - macros_known = FALSE para indicar que la nutricion es estimada via USDA
  - pasos estructurados en JSONB (no solo en chunk_text)

El insert se hace de forma transaccional y deduplicada: si ya existe una
receta generada por IA con el mismo nombre se devuelve su id sin volver a
insertar.
"""

import json
import logging
from datetime import datetime, timezone

from src.config import PLAUSIBILITY_THRESHOLD
from src.etl.embed import embed_texts
from src.retrieval.validation import validate_recipe

logger = logging.getLogger(__name__)


class RecipeValidationError(ValueError):
    """Lanzada cuando una receta no cumple los minimos de calidad."""

    def __init__(self, errors: list[str]):
        super().__init__("Receta invalida: " + " | ".join(errors))
        self.errors = errors


class RecipeImplausibleError(ValueError):
    """Lanzada cuando una receta no supera el umbral de plausibilidad culinaria."""

    def __init__(self, score: float, threshold: float):
        super().__init__(
            f"Receta no plausible (puntuacion {score:.1f}/10, "
            f"minimo requerido {threshold:.1f})."
        )
        self.score = score
        self.threshold = threshold


def _normalize_title(title: str) -> str:
    """Normaliza un nombre de receta para comparar duplicados.
    Pasa a minusculas y colapsa los espacios sobrantes, de modo que
    'Sopa de Pollo' y '  sopa de pollo ' se consideren la misma receta.
    """
    return " ".join((title or "").lower().split())


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


def save_generated_recipe(conn, recipe: dict, check_plausibility: bool = True) -> tuple[int, bool]:
    """
    Persiste una receta generada por IA aplicando validacion, comprobacion
    de plausibilidad culinaria y deduplicacion.

    Devuelve una tupla (recipe_id, was_new) donde was_new es False si la
    receta ya existia y se devuelve su id sin insertar duplicados.

    Lanza RecipeValidationError si la receta no cumple los minimos de calidad.
    Lanza RecipeImplausibleError si no supera el umbral de plausibilidad.

    check_plausibility se puede desactivar en tests para no depender del LLM.
    """
    errors = validate_recipe(recipe)
    if errors:
        raise RecipeValidationError(errors)

    title       = recipe["title"].strip()
    ingredients = recipe["ingredients"]
    steps       = recipe["steps"]
    ner         = recipe["ner"]

    # Gate de plausibilidad: el LLM valora la coherencia culinaria y se
    # bloquea el guardado si la combinacion de ingredientes no es plausible.
    if check_plausibility:
        from src.vision.llava_client import assess_plausibility
        score = assess_plausibility(ner or ingredients)
        if score < PLAUSIBILITY_THRESHOLD:
            logger.info("Receta '%s' rechazada por plausibilidad (%.1f).", title, score)
            raise RecipeImplausibleError(score, PLAUSIBILITY_THRESHOLD)

    # Comprobacion de duplicado por nombre: si ya existe una receta generada
    # por IA con el mismo titulo (normalizado), se devuelve su id sin insertar.
    norm_title = _normalize_title(title)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM recipes "
            "WHERE source = 'ai_generated' "
            "AND lower(btrim(regexp_replace(title, '\\s+', ' ', 'g'))) = %s "
            "LIMIT 1;",
            (norm_title,),
        )
        existing = cur.fetchone()
        if existing:
            logger.info("Receta duplicada detectada por nombre ('%s'), "
                        "devolviendo id existente %d.", title, existing[0])
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
                    source, provenance,
                    calories, protein_g, fat_g, carbs_g, fiber_g,
                    macros_known, etl_batch_id
                )
                VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                        %s, %s::jsonb,
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

    logger.info("Receta IA '%s' guardada con id=%d.", title, recipe_id)
    return recipe_id, True
