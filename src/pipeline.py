"""
GR-IA Pipeline — Phase 1 MVP

Usage:
    python -m src.pipeline --image path/to/food_photo.jpg
    python -m src.pipeline --image path/to/food_photo.jpg --top-k 5
"""

import argparse
import logging
import sys

import psycopg2

from src.config import get_dsn
from src.vision.llava_client import detect_ingredients, flatten_ingredients, format_recipes_response
from src.retrieval.search import search_recipes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(image_path: str, top_k: int = 5, language: str = "Spanish") -> None:
    # ── Step 1: Vision — detect and classify ingredients ──────────────────────
    logger.info("Step 1/3 — Detecting ingredients from image: %s", image_path)
    classified = detect_ingredients(image_path)

    if not flatten_ingredients(classified):
        print("\n[!] No ingredients detected. Try a clearer photo.")
        sys.exit(1)

    print("\nDetected ingredients:")
    for level, label in [
        ("main",          "  Main         "),
        ("secondary",     "  Secondary    "),
        ("accompaniment", "  Accompaniment"),
        ("spices",        "  Spices       "),
    ]:
        items = classified.get(level, [])
        if items:
            print(f"{label}: {', '.join(items)}")
    print()

    # ── Step 2: Retrieval — weighted search ───────────────────────────────────
    logger.info("Step 2/3 — Searching recipes (weighted by ingredient level)...")
    conn = psycopg2.connect(get_dsn())
    try:
        recipes = search_recipes(conn, classified, top_k=top_k)
    finally:
        conn.close()

    if not recipes:
        print("[!] No matching recipes found in the database.")
        sys.exit(0)

    logger.info("Found %d matching recipes (top score: %d)", len(recipes), recipes[0]["score"])

    # ── Step 3: Generation — present recipes in requested language ────────────
    logger.info("Step 3/3 — Generating response in %s...", language)
    response = format_recipes_response(classified, recipes, language=language)

    print("\n" + "=" * 60)
    print(response)
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="GR-IA: recipe suggestions from a food photo")
    parser.add_argument("--image", required=True, help="Path to the food photo")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recipes to retrieve (default: 5)")
    parser.add_argument("--lang", type=str, default="Spanish", help="Response language (default: Spanish)")
    args = parser.parse_args()

    run_pipeline(args.image, top_k=args.top_k, language=args.lang)


if __name__ == "__main__":
    main()
