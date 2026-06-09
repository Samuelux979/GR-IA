"""
Capa de servicio compartida entre el CLI (src/pipeline.py) y la API web.
Encapsula la orquestacion completa de una prediccion para que ambos
puntos de entrada reutilicen la misma logica sin duplicar codigo.
"""

import logging

import psycopg2

from src.config import get_dsn
from src.vision.llava_client import detect_ingredients, flatten_ingredients, generate_recipe, _parse_steps
from src.retrieval.search import search_recipes
from src.retrieval.save_generated import save_generated_recipe
from src.vision.translator import translate_steps_to_es, translate_en_to_es

logger = logging.getLogger(__name__)

# Pesos por nivel (deben coincidir con los del scoring en search.py)
_LEVEL_WEIGHTS = {"main": 8, "secondary": 4, "accompaniment": 2, "spices": 1}


def _max_possible_score(classified: dict) -> int:
    """
    Maximo de puntos alcanzable dados los ingredientes detectados: sirve de
    referencia para convertir el score absoluto en una nota sobre 10.
    """
    return sum(
        len(classified.get(level, [])) * weight
        for level, weight in _LEVEL_WEIGHTS.items()
    )


def _grade_and_level(score: int, max_score: int) -> tuple[float, str]:
    """
    Convierte el score absoluto en una nota sobre 10 y una etiqueta de nivel
    de correlacion (alta/media/baja) que la interfaz traduce a un color.
    """
    if max_score <= 0:
        return 0.0, "baja"
    grade = min(10.0, round(score / max_score * 10, 1))
    if grade >= 7:
        level = "alta"      # verde
    elif grade >= 4:
        level = "media"     # naranja
    else:
        level = "baja"      # rojo
    return grade, level


def _build_recipe_card(r: dict, detected: set, max_score: int) -> dict:
    """
    Convierte una receta cruda de la BD en la version preparada para presentar.
    Si la receta tiene 'steps' estructurados, los usa directamente; si no,
    los reconstruye desde el chunk_text (compatibilidad con recetas antiguas).
    """
    ner    = list(dict.fromkeys(r.get("ner") or []))
    shared = [ing for ing in ner if ing.lower() in detected]

    steps        = r.get("steps") or _parse_steps(r.get("chunk_text", ""))
    ingredients  = r.get("ingredients") or []
    grade, level = _grade_and_level(r["score"], max_score)

    return {
        "id":           r["id"],
        "title":        r["title"],
        "category":     r.get("category"),
        "source":       r.get("source", "foodcom"),
        "macros_known": r.get("macros_known", True),
        "ingredients":  translate_en_to_es(ingredients) if ingredients else [],
        "steps":        translate_steps_to_es(steps) if steps else [],
        "matches":      shared,
        "score":        r["score"],
        "grade":        grade,
        "match_level":  level,
        "distance":     r["distance"],
        "macros": {
            "calories":  None if r.get("calories")  is None else float(r["calories"]),
            "protein_g": None if r.get("protein_g") is None else float(r["protein_g"]),
            "fat_g":     None if r.get("fat_g")     is None else float(r["fat_g"]),
            "carbs_g":   None if r.get("carbs_g")   is None else float(r["carbs_g"]),
            "fiber_g":   None if r.get("fiber_g")   is None else float(r["fiber_g"]),
        },
    }


def run_prediction(
    image_path: str,
    top_k: int = 3,
    generate: bool = True,
    include_ai: bool = True,
    nutrition_filters: dict | None = None,
) -> dict:
    """
    Ejecuta el pipeline completo sobre una imagen y devuelve un diccionario
    estructurado con ingredientes, recetas recuperadas y receta generada.

    include_ai controla si las recetas previamente generadas por IA aparecen
    en los resultados de la busqueda.
    nutrition_filters acota las recetas recuperadas a los rangos de
    macronutrientes indicados ({macro: (min, max)}).
    """
    classified = detect_ingredients(image_path)

    if not flatten_ingredients(classified):
        return {
            "ingredients": classified,
            "recipes":     [],
            "ai_recipe":   None,
            "warnings":    ["No se detectaron ingredientes en la imagen."],
        }

    conn = psycopg2.connect(get_dsn())
    try:
        raw_recipes = search_recipes(conn, classified, top_k=top_k,
                                     include_ai=include_ai, nutrition_filters=nutrition_filters)
        detected    = {i.lower() for i in flatten_ingredients(classified)}
        max_score   = _max_possible_score(classified)
        recipes     = [_build_recipe_card(r, detected, max_score) for r in raw_recipes]

        ai_recipe = None
        warnings  = []
        if nutrition_filters and not recipes:
            warnings.append("Ninguna receta cumple los filtros nutricionales indicados.")
        if generate:
            ai_recipe = generate_recipe(classified, conn=conn)
            if ai_recipe is None:
                warnings.append("El modelo no pudo generar una receta original.")
    finally:
        conn.close()

    return {
        "ingredients": classified,
        "recipes":     recipes,
        "ai_recipe":   ai_recipe,
        "warnings":    warnings,
    }


def persist_generated_recipe(recipe: dict) -> tuple[int, bool]:
    """
    Persiste una receta generada en la base de datos.
    Devuelve (recipe_id, was_new) donde was_new=False si era duplicada.
    """
    conn = psycopg2.connect(get_dsn())
    try:
        return save_generated_recipe(conn, recipe)
    finally:
        conn.close()
