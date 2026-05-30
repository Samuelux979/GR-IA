"""
GR-IA — punto de entrada principal.

Uso:
    python -m src.pipeline --image foto.jpg
    python -m src.pipeline --image foto.jpg --top-k 5
"""

import argparse
import logging
import sys

# Forzar UTF-8 en stdout para que la consola de Windows muestre caracteres
# como acentos y la fraccion "⁄" del dataset sin romper.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import psycopg2

from src.config import get_dsn
from src.vision.llava_client import (
    detect_ingredients,
    flatten_ingredients,
    format_recipes_response,
    generate_recipe,
)
from src.retrieval.search import search_recipes
from src.retrieval.save_generated import save_generated_recipe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(image_path: str, top_k: int = 5) -> None:
    # 1. Deteccion de ingredientes
    logger.info("Paso 1/4 - Detectando ingredientes: %s", image_path)
    classified = detect_ingredients(image_path)

    if not flatten_ingredients(classified):
        print("\n[!] No se detectaron ingredientes. Prueba con una foto mas clara.")
        sys.exit(1)

    print("\nIngredientes detectados:")
    for level, label in [
        ("main",          "  Principal    "),
        ("secondary",     "  Secundario   "),
        ("accompaniment", "  Acompanamiento"),
        ("spices",        "  Especias     "),
    ]:
        items = classified.get(level, [])
        if items:
            print(f"{label}: {', '.join(items)}")
    print()

    # 2. Busqueda de recetas
    logger.info("Paso 2/4 - Buscando recetas...")
    conn = psycopg2.connect(get_dsn())
    try:
        recipes = search_recipes(conn, classified, top_k=top_k)
    finally:
        conn.close()

    if not recipes:
        print("[!] No se encontraron recetas para estos ingredientes.")
        sys.exit(0)

    logger.info("Encontradas %d recetas (puntuacion maxima: %d)", len(recipes), recipes[0]["score"])

    # 3. Presentacion de resultados
    logger.info("Paso 3/4 - Formateando respuesta...")
    print("\n" + "=" * 60)
    print(format_recipes_response(classified, recipes))
    print("=" * 60)

    # 4. Generacion de receta propia y guardado opcional
    logger.info("Paso 4/4 - Generando receta con tus ingredientes...")
    conn = psycopg2.connect(get_dsn())
    try:
        ai_recipe = generate_recipe(classified, conn=conn)
    finally:
        conn.close()

    if ai_recipe:
        print("\n" + "=" * 60)
        print("Receta generada con tus ingredientes:")
        print("-" * 50)
        print(f"  {ai_recipe['title']}")
        print()
        print("   Ingredientes:")
        for ing in ai_recipe["ingredients"]:
            print(f"      - {ing}")
        print()
        print("   Pasos:")
        for j, step in enumerate(ai_recipe["steps_translated"], 1):
            print(f"      Paso {j}. {step}")
        print()
        print(f"   Valores nutricionales (calculados con USDA):")
        print(f"      {ai_recipe['calories']:.0f} kcal | "
              f"Proteinas {ai_recipe['protein_g']:.1f}g | "
              f"Grasas {ai_recipe['fat_g']:.1f}g | "
              f"Carbohidratos {ai_recipe['carbs_g']:.1f}g | "
              f"Fibra {ai_recipe['fiber_g']:.1f}g")
        print("=" * 60)

        try:
            answer = input("\n¿Quieres guardar esta receta? (s/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer in ("s", "si"):
            conn = psycopg2.connect(get_dsn())
            try:
                recipe_id = save_generated_recipe(conn, ai_recipe)
                print(f"\nReceta guardada con id={recipe_id}. Aparecera en futuras busquedas.")
            finally:
                conn.close()
        else:
            print("\nReceta descartada.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GR-IA: recomendacion de recetas a partir de una foto")
    parser.add_argument("--image",  required=True,       help="Ruta a la foto de ingredientes")
    parser.add_argument("--top-k",  type=int, default=5, help="Numero de recetas a recuperar")
    args = parser.parse_args()

    run_pipeline(args.image, top_k=args.top_k)


if __name__ == "__main__":
    main()
