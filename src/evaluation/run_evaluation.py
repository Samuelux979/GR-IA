"""
Punto de entrada para evaluar GR-IA por componentes.

Uso:
    python -m src.evaluation.run_evaluation --component ingredients
    python -m src.evaluation.run_evaluation --component classification
    python -m src.evaluation.run_evaluation --component retrieval
    python -m src.evaluation.run_evaluation --component generation
    python -m src.evaluation.run_evaluation --component all

Cada componente necesita recursos distintos:
    ingredients     -> Ollama (vision)
    classification  -> Ollama (texto)
    retrieval       -> PostgreSQL + modelo de embeddings
    generation      -> Ollama (generacion)

Si un recurso no esta disponible, el componente se omite con un aviso claro.
"""

import argparse
import json
import logging
from pathlib import Path

from src.config import PROJECT_ROOT
from src.evaluation import metrics
from src.evaluation.normalize import normalize_list, normalize_ingredient
from src.evaluation.report import save_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluation")

EVAL_DIR = PROJECT_ROOT / "evaluation"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Componente: deteccion de ingredientes ----------------------------------

def eval_ingredients() -> dict:
    cases = _load_jsonl(EVAL_DIR / "ingredients_ground_truth.jsonl")
    if not cases:
        return {"error": "no hay datos en ingredients_ground_truth.jsonl"}

    from src.vision.llava_client import detect_ingredients, flatten_ingredients

    results, examples = [], []
    for case in cases:
        image = case["image"]
        path  = (EVAL_DIR / "images" / image)
        if not path.exists():
            logger.warning("Imagen no encontrada, se omite: %s", image)
            continue
        try:
            predicted = normalize_list(flatten_ingredients(detect_ingredients(str(path))))
        except Exception as e:
            return {"error": f"fallo al detectar ingredientes (Ollama?): {e}"}
        expected = normalize_list(case["expected"])
        r = metrics.prf1(predicted, expected)
        results.append(r)
        examples.append({
            "image": image,
            "false_positives": r.false_positives,
            "false_negatives": r.false_negatives,
        })

    if not results:
        return {"error": "ninguna imagen del ground truth existe en evaluation/images/"}

    agg = metrics.aggregate_prf1(results)
    agg["examples"] = examples[:5]
    return agg


# --- Componente: clasificacion ----------------------------------------------

def eval_classification() -> dict:
    cases = _load_jsonl(EVAL_DIR / "classification_ground_truth.jsonl")
    if not cases:
        return {"error": "no hay datos en classification_ground_truth.jsonl"}

    from src.vision.llava_client import _make_llm, _parse_json, CLASSIFY_PROMPT_TEMPLATE
    from src.config import OLLAMA_TEXT_MODEL
    from langchain_core.messages import HumanMessage

    per_case = []
    for case in cases:
        ingredients = case["ingredients"]
        try:
            llm = _make_llm(OLLAMA_TEXT_MODEL, temperature=0.0)
            raw = llm.invoke([HumanMessage(
                content=CLASSIFY_PROMPT_TEMPLATE.format(ingredients=", ".join(ingredients))
            )]).content.strip()
        except Exception as e:
            return {"error": f"fallo al clasificar (Ollama?): {e}"}
        data = _parse_json(raw) or {}
        predicted = {lvl: data.get(lvl, []) for lvl in metrics.LEVELS}
        per_case.append(metrics.classification_accuracy(predicted, case["expected"]))

    return metrics.aggregate_classification(per_case)


# --- Componente: recuperacion -----------------------------------------------

def eval_retrieval() -> dict:
    cases = _load_jsonl(EVAL_DIR / "retrieval_queries.jsonl")
    if not cases:
        return {"error": "no hay datos en retrieval_queries.jsonl"}

    try:
        import psycopg2
        from src.config import get_dsn
        from src.retrieval.search import search_recipes
        conn = psycopg2.connect(get_dsn(), connect_timeout=3)
    except Exception as e:
        return {"error": f"PostgreSQL no disponible: {e}"}

    ks = [3, 5, 10]
    pk_acc = {k: [] for k in ks}
    try:
        for case in cases:
            classified = case["query_ingredients"]
            must_have  = normalize_list(case.get("relevant_if_ner_contains", []))
            recipes = search_recipes(conn, classified, top_k=max(ks))
            # Relevante si alguna palabra requerida aparece en el ner de la receta
            # (p.ej. 'chicken' en 'chicken breast'), no por igualdad exacta.
            flags = []
            for r in recipes:
                ner_words = set()
                for n in (r.get("ner") or []):
                    ner_words |= set(normalize_ingredient(n).split())
                flags.append(any(term in ner_words for term in must_have))
            for k in ks:
                pk_acc[k].append(metrics.precision_at_k(flags, k))
    finally:
        conn.close()

    return {
        "precision_at_k": {
            k: round(sum(v) / len(v), 3) if v else 0.0 for k, v in pk_acc.items()
        },
        "n_queries": len(cases),
    }


# --- Componente: generacion (humana) ----------------------------------------

CRITERIA = ["uso_ingredientes", "claridad_pasos", "viabilidad", "seguridad", "utilidad"]


def eval_generation() -> dict:
    reviews_path = EVAL_DIR / "generated_recipe_reviews.jsonl"
    reviews = _load_jsonl(reviews_path)

    # Caso 1: ya hay recetas revisadas (5 criterios puntuados con 1-5) -> medias
    scored = [
        r for r in reviews
        if r.get("scores") and all(r["scores"].get(c, 0) for c in CRITERIA)
    ]
    if scored:
        per_crit, overall = {}, []
        for c in CRITERIA:
            vals = [r["scores"][c] for r in scored if c in r.get("scores", {})]
            per_crit[c] = round(sum(vals) / len(vals), 2) if vals else None
        for r in scored:
            s = r["scores"]
            overall.append(sum(s.values()) / len(s))
        return {
            "n_reviewed": len(scored),
            "mean_overall": round(sum(overall) / len(overall), 2),
            "per_criterion": per_crit,
        }

    # Caso 2: ya hay plantillas generadas pero sin puntuar -> pendiente
    if reviews:
        return {"pending": True, "n_generated": len(reviews)}

    # Caso 3: no hay nada -> generar recetas plantilla para revision humana
    queries = _load_jsonl(EVAL_DIR / "retrieval_queries.jsonl")
    if not queries:
        return {"error": "no hay retrieval_queries.jsonl para generar recetas de muestra"}

    try:
        from src.vision.llava_client import generate_recipe
    except Exception as e:
        return {"error": f"no se pudo importar la generacion: {e}"}

    templates = []
    for case in queries:
        try:
            recipe = generate_recipe(case["query_ingredients"], conn=None)
        except Exception as e:
            return {"error": f"fallo al generar receta (Ollama?): {e}"}
        if not recipe:
            continue
        templates.append({
            "title":       recipe["title"],
            "ingredients": recipe["ingredients"],
            "steps":       recipe["steps"],
            # Rellenar cada criterio con un numero del 1 al 5 (0 = sin puntuar)
            "scores":      {c: 0 for c in CRITERIA},
            "comment":     "",
        })

    with open(reviews_path, "w", encoding="utf-8") as f:
        for t in templates:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    return {"pending": True, "n_generated": len(templates)}


COMPONENTS = {
    "ingredients":    eval_ingredients,
    "classification": eval_classification,
    "retrieval":      eval_retrieval,
    "generation":     eval_generation,
}


def run(component: str) -> None:
    targets = list(COMPONENTS) if component == "all" else [component]
    results = {}
    for name in targets:
        logger.info("Evaluando: %s", name)
        results[name] = COMPONENTS[name]()
        if "error" in results[name]:
            logger.warning("  -> omitido: %s", results[name]["error"])

    path = save_report(results)
    logger.info("Reporte guardado en %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluacion de GR-IA por componentes")
    parser.add_argument(
        "--component",
        choices=[*COMPONENTS, "all"],
        default="all",
        help="Componente a evaluar (por defecto: all)",
    )
    args = parser.parse_args()
    run(args.component)


if __name__ == "__main__":
    main()
