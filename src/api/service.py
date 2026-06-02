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
from src.vision.translator import translate_steps_to_es

logger = logging.getLogger(__name__)


def _build_recipe_card(r: dict, detected: set) -> dict:
    """
    Convierte una receta cruda de la BD en la version preparada para presentar.
    Si la receta tiene 'steps' estructurados, los usa directamente; si no,
    los reconstruye desde el chunk_text (compatibilidad con recetas antiguas).
    """
    ner    = list(dict.fromkeys(r.get("ner") or []))
    shared = [ing for ing in ner if ing.lower() in detected]

    steps = r.get("steps") or _parse_steps(r.get("chunk_text", ""))

    return {
        "id":           r["id"],
        "title":        r["title"],
        "category":     r.get("category"),
        "source":       r.get("source", "foodcom"),
        "macros_known": r.get("macros_known", True),
        "ingredients":  r.get("ingredients") or [],
        "steps":        translate_steps_to_es(steps) if steps else [],
        "matches":      shared,
        "score":        r["score"],
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
) -> dict:
    """
    Ejecuta el pipeline completo sobre una imagen y devuelve un diccionario
    estructurado con ingredientes, recetas recuperadas y receta generada.

    include_ai controla si las recetas previamente generadas por IA aparecen
    en los resultados de la busqueda.
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
        raw_recipes = search_recipes(conn, classified, top_k=top_k, include_ai=include_ai)
        detected    = {i.lower() for i in flatten_ingredients(classified)}
        recipes     = [_build_recipe_card(r, detected) for r in raw_recipes]

        ai_recipe = None
        warnings  = []
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
