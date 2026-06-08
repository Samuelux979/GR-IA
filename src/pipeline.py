"""
GR-IA - punto de entrada de linea de comandos.

Uso:
    python -m src.pipeline --image foto.jpg
    python -m src.pipeline --image foto.jpg --top-k 5

La orquestacion real vive en src/api/service.py para que el CLI y
la API web compartan la misma logica.
"""

import argparse
import logging
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.api.service import run_prediction, persist_generated_recipe
from src.retrieval.save_generated import RecipeImplausibleError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _print_ingredients(classified: dict) -> None:
    print("\nIngredientes detectados:")
    labels = [
        ("main",          "  Principal    "),
        ("secondary",     "  Secundario   "),
        ("accompaniment", "  Acompanamiento"),
        ("spices",        "  Especias     "),
    ]
    for level, label in labels:
        items = classified.get(level, [])
        if items:
            print(f"{label}: {', '.join(items)}")
    print()


def _print_recipe_card(i: int, recipe: dict) -> None:
    badge = " [IA]" if recipe.get("source") == "ai_generated" else ""
    nivel = {"alta": "alta", "media": "media", "baja": "baja"}.get(recipe.get("match_level", ""), "")
    print("-" * 50)
    print(f"{i}. {recipe['title']}{badge}   (afinidad: {recipe.get('grade', 0)}/10 - {nivel})")
    if recipe["matches"]:
        print(f"   Coincide con tu foto: {', '.join(recipe['matches'])}")
    print()
    print("   Ingredientes:")
    for ing in recipe["ingredients"]:
        print(f"      - {ing}")
    print()
    if recipe["steps"]:
        print("   Pasos:")
        for j, step in enumerate(recipe["steps"], 1):
            print(f"      Paso {j}. {step}")
    if recipe.get("macros", {}).get("calories") is not None and not recipe.get("macros_known", True):
        print("   (Valores nutricionales estimados via USDA)")
    print()


def _print_ai_recipe(ai: dict) -> None:
    print("\n" + "=" * 60)
    print("Receta generada con tus ingredientes:")
    print("-" * 50)
    print(f"  {ai['title']}")
    print()
    print("   Ingredientes:")
    for ing in ai["ingredients"]:
        print(f"      - {ing}")
    print()
    print("   Pasos:")
    for j, step in enumerate(ai["steps_translated"], 1):
        print(f"      Paso {j}. {step}")
    print()
    print("   Valores nutricionales (calculados con USDA):")
    print(f"      {ai['calories']:.0f} kcal | "
          f"Proteinas {ai['protein_g']:.1f}g | "
          f"Grasas {ai['fat_g']:.1f}g | "
          f"Carbohidratos {ai['carbs_g']:.1f}g | "
          f"Fibra {ai['fiber_g']:.1f}g")
    print("=" * 60)


def run_pipeline(image_path: str, top_k: int = 5, include_ai: bool = True) -> None:
    logger.info("Procesando imagen: %s", image_path)
    result = run_prediction(image_path, top_k=top_k, generate=True, include_ai=include_ai)

    if not any(result["ingredients"].values()):
        print("\n[!] No se detectaron ingredientes. Prueba con una foto mas clara.")
        sys.exit(1)

    _print_ingredients(result["ingredients"])

    if not result["recipes"]:
        print("[!] No se encontraron recetas para estos ingredientes.")
        sys.exit(0)

    logger.info("Encontradas %d recetas", len(result["recipes"]))

    print("\n" + "=" * 60)
    for i, recipe in enumerate(result["recipes"], 1):
        _print_recipe_card(i, recipe)
    print("=" * 60)

    if result["ai_recipe"]:
        _print_ai_recipe(result["ai_recipe"])

        try:
            answer = input("\n¿Quieres guardar esta receta? (s/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer in ("s", "si"):
            try:
                recipe_id, was_new = persist_generated_recipe(result["ai_recipe"])
                if was_new:
                    print(f"\nReceta guardada con id={recipe_id}.")
                else:
                    print(f"\nReceta duplicada: ya existia con id={recipe_id}.")
            except RecipeImplausibleError as e:
                print(f"\nNo se ha guardado: la combinacion no se considera "
                      f"culinariamente plausible (puntuacion {e.score:.1f}/10).")
        else:
            print("\nReceta descartada.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GR-IA: recomendacion de recetas a partir de una foto")
    parser.add_argument("--image",  required=True,       help="Ruta a la foto de ingredientes")
    parser.add_argument("--top-k",  type=int, default=5, help="Numero de recetas a recuperar")
    parser.add_argument("--no-ai",  action="store_true", help="Excluir de la busqueda recetas generadas por IA")
    args = parser.parse_args()
    run_pipeline(args.image, top_k=args.top_k, include_ai=not args.no_ai)


if __name__ == "__main__":
    main()
