"""
Comparativa de modelos de lenguaje para GR-IA.

Genera, con el MISMO ground truth, las respuestas de tres modelos de texto
distintos y permite comparar como cambia la calidad:

    qwen05b -> qwen2.5:0.5b   (modelo pequeno, calidad esperada baja)
    qwen15b -> qwen2.5:1.5b   (modelo en produccion, calidad media)
    opus    -> Claude Opus 4.8 (modelo frontera, generado manualmente)

Solo se comparan los componentes que dependen del modelo de texto:
clasificacion de ingredientes y generacion de recetas. La deteccion (vision)
y la recuperacion (BD + embeddings) son independientes del modelo de texto.

Fases:
    --phase generate   genera recetas y clasifica con los modelos de Ollama
    --phase report     construye el reporte comparativo a partir de lo puntuado

Uso:
    python -m src.evaluation.compare_models --phase generate
    # (luego se puntuan a mano las recetas y se anade la columna opus)
    python -m src.evaluation.compare_models --phase report
"""

import argparse
import json
import logging
from pathlib import Path

from src.config import PROJECT_ROOT, OLLAMA_TEXT_MODEL
from src.evaluation import metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_models")

EVAL_DIR = PROJECT_ROOT / "evaluation"
COMP_DIR = EVAL_DIR / "model_comparison"
CRITERIA = ["uso_ingredientes", "claridad_pasos", "viabilidad", "seguridad", "utilidad"]

# Etiqueta -> modelo de Ollama (opus se rellena a mano, no via Ollama)
OLLAMA_MODELS = {
    "qwen05b": "qwen2.5:0.5b",
    "qwen15b": "qwen2.5:1.5b",
}
ALL_LABELS = ["qwen05b", "qwen15b", "opus"]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Fase 1: generar respuestas de los modelos de Ollama --------------------

def generate(only: str | None = None) -> None:
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    queries = _load_jsonl(EVAL_DIR / "retrieval_queries.jsonl")
    class_cases = _load_jsonl(EVAL_DIR / "classification_ground_truth.jsonl")

    from src.vision.llava_client import generate_recipe, _make_llm, _parse_json, CLASSIFY_PROMPT_TEMPLATE
    from langchain_core.messages import HumanMessage

    classification = {}
    targets = {only: OLLAMA_MODELS[only]} if only else OLLAMA_MODELS

    for label, model in targets.items():
        logger.info("=== Modelo %s (%s) ===", label, model)

        # --- Generacion de recetas ---
        recipes = []
        for case in queries:
            recipe = generate_recipe(case["query_ingredients"], conn=None, model=model, translate=False)
            if not recipe:
                continue
            recipes.append({
                "title":       recipe["title"],
                "ingredients": recipe["ingredients"],
                "steps":       recipe["steps"],
                "scores":      {c: 0 for c in CRITERIA},   # a puntuar a mano
                "comment":     "",
            })
        gen_path = COMP_DIR / f"gen_{label}.jsonl"
        with open(gen_path, "w", encoding="utf-8") as f:
            for r in recipes:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("  %d recetas -> %s", len(recipes), gen_path.name)

        # --- Clasificacion de ingredientes ---
        per_case = []
        for case in class_cases:
            llm = _make_llm(model, temperature=0.0)
            raw = llm.invoke([HumanMessage(
                content=CLASSIFY_PROMPT_TEMPLATE.format(ingredients=", ".join(case["ingredients"]))
            )]).content.strip()
            data = _parse_json(raw) or {}
            predicted = {lvl: data.get(lvl, []) for lvl in metrics.LEVELS}
            per_case.append(metrics.classification_accuracy(predicted, case["expected"]))
        classification[label] = metrics.aggregate_classification(per_case)
        logger.info("  Clasificacion accuracy: %s", classification[label]["accuracy"])

    # Guardar clasificacion (opus se anade a mano luego)
    class_path = COMP_DIR / "classification.json"
    existing = json.loads(class_path.read_text(encoding="utf-8")) if class_path.exists() else {}
    existing.update(classification)
    class_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Clasificacion guardada en %s", class_path)
    logger.info("Ahora puntua las recetas de gen_*.jsonl y crea gen_opus.jsonl + opus en classification.json")


# --- Fase 2: construir el reporte comparativo -------------------------------

def _gen_means(label: str) -> dict | None:
    rows = _load_jsonl(COMP_DIR / f"gen_{label}.jsonl")
    scored = [r for r in rows if r.get("scores") and all(r["scores"].get(c, 0) for c in CRITERIA)]
    if not scored:
        return None
    per_crit = {c: round(sum(r["scores"][c] for r in scored) / len(scored), 2) for c in CRITERIA}
    overall = sum(sum(r["scores"].values()) / len(CRITERIA) for r in scored) / len(scored)
    return {"n": len(scored), "overall": round(overall, 2), "per_criterion": per_crit}


def report() -> None:
    classification = json.loads((COMP_DIR / "classification.json").read_text(encoding="utf-8")) \
        if (COMP_DIR / "classification.json").exists() else {}

    names = {"qwen05b": "qwen2.5:0.5b", "qwen15b": "qwen2.5:1.5b", "opus": "Claude Opus 4.8"}
    lines = ["# Comparativa de modelos de lenguaje - GR-IA", ""]
    lines.append("Solo se comparan los componentes dependientes del modelo de texto:")
    lines.append("clasificacion de ingredientes y generacion de recetas.")
    lines.append("(Deteccion y recuperacion son independientes del modelo de texto.)")
    lines.append("")

    # --- Generacion ---
    lines.append("## Generacion de recetas (evaluacion humana, 1-5)")
    lines.append("")
    header = "| Criterio | " + " | ".join(names[l] for l in ALL_LABELS) + " |"
    sep    = "|---|" + "---|" * len(ALL_LABELS)
    lines.append(header)
    lines.append(sep)
    means = {l: _gen_means(l) for l in ALL_LABELS}
    for c in CRITERIA:
        row = [c] + [f"{means[l]['per_criterion'][c]}" if means[l] else "-" for l in ALL_LABELS]
        lines.append("| " + " | ".join(row) + " |")
    row = ["**Media global**"] + [f"**{means[l]['overall']}**" if means[l] else "-" for l in ALL_LABELS]
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- Clasificacion ---
    lines.append("## Clasificacion de ingredientes (accuracy)")
    lines.append("")
    lines.append("| Metrica | " + " | ".join(names[l] for l in ALL_LABELS) + " |")
    lines.append(sep)
    row = ["Accuracy global"] + [
        f"{classification[l]['accuracy']}" if l in classification else "-" for l in ALL_LABELS
    ]
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    out = EVAL_DIR / "reports" / "model_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Reporte comparativo guardado en %s", out)
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparativa de 3 modelos de lenguaje")
    parser.add_argument("--phase", choices=["generate", "report"], required=True)
    parser.add_argument("--only", choices=list(OLLAMA_MODELS), default=None,
                        help="Generar solo un modelo (qwen05b o qwen15b)")
    args = parser.parse_args()
    if args.phase == "generate":
        generate(only=args.only)
    else:
        report()
