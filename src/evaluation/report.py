"""
Generacion del reporte de evaluacion en Markdown y JSON.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"


def _git_commit() -> str:
    """Devuelve el hash corto del commit actual, o 'desconocido'."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "desconocido"
    except Exception:
        return "desconocido"


def _fmt_detection(d: dict) -> list[str]:
    lines = ["## Deteccion de ingredientes", ""]
    if "error" in d:
        lines += [f"_No evaluado: {d['error']}_", ""]
        return lines
    lines += [
        "| Metrica | Valor |",
        "|---|---|",
        f"| Precision | {d['precision']} |",
        f"| Recall | {d['recall']} |",
        f"| F1 | {d['f1']} |",
        f"| Casos evaluados | {d['n_cases']} |",
        "",
    ]
    if d.get("examples"):
        lines.append("Ejemplos:")
        for ex in d["examples"]:
            lines.append(f"- **{ex['image']}** -> FP: {ex['false_positives'] or '-'} | FN: {ex['false_negatives'] or '-'}")
        lines.append("")
    return lines


def _fmt_classification(d: dict) -> list[str]:
    lines = ["## Clasificacion de ingredientes", ""]
    if "error" in d:
        lines += [f"_No evaluado: {d['error']}_", ""]
        return lines
    lines += [
        f"Accuracy global: **{d['accuracy']}** ({d['n_ingredients']} ingredientes)",
        "",
        "| Clase | Accuracy |",
        "|---|---|",
    ]
    for lvl, acc in d["per_class"].items():
        lines.append(f"| {lvl} | {acc if acc is not None else 'n/a'} |")
    lines.append("")
    return lines


def _fmt_retrieval(d: dict) -> list[str]:
    lines = ["## Recuperacion de recetas", ""]
    if "error" in d:
        lines += [f"_No evaluado: {d['error']}_", ""]
        return lines
    lines += ["| Metrica | Valor |", "|---|---|"]
    for k, v in d["precision_at_k"].items():
        lines.append(f"| Precision@{k} | {v} |")
    lines += [f"| Consultas evaluadas | {d['n_queries']} |", ""]
    return lines


def _fmt_generation(d: dict) -> list[str]:
    lines = ["## Generacion de recetas (evaluacion humana)", ""]
    if "error" in d:
        lines += [f"_No evaluado: {d['error']}_", ""]
        return lines
    if d.get("pending"):
        lines += [
            f"Se generaron **{d['n_generated']}** recetas para revisar en "
            f"`evaluation/generated_recipe_reviews.jsonl`.",
            "Rellena las puntuaciones (1-5) y vuelve a ejecutar para ver la media.",
            "",
        ]
        return lines
    lines += [
        f"Recetas revisadas: {d['n_reviewed']}",
        f"Media global: **{d['mean_overall']}**/5",
        "",
        "| Criterio | Media |",
        "|---|---|",
    ]
    for crit, val in d["per_criterion"].items():
        lines.append(f"| {crit} | {val} |")
    lines.append("")
    return lines


def build_report(results: dict) -> tuple[str, dict]:
    """Construye el reporte en Markdown y el objeto JSON equivalente."""
    now    = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit = _git_commit()

    md = [
        "# Reporte de evaluacion - GR-IA",
        "",
        f"- Fecha: {now}",
        f"- Commit: `{commit}`",
        "",
        "---",
        "",
    ]
    if "ingredients" in results:
        md += _fmt_detection(results["ingredients"])
    if "classification" in results:
        md += _fmt_classification(results["classification"])
    if "retrieval" in results:
        md += _fmt_retrieval(results["retrieval"])
    if "generation" in results:
        md += _fmt_generation(results["generation"])

    md += [
        "---",
        "",
        "## Interpretacion y limitaciones",
        "",
        "- Precision alta y recall bajo: el sistema acierta lo que detecta pero omite ingredientes.",
        "- Recall alto y precision baja: detecta de mas (falsos positivos).",
        "- La evaluacion de generacion es subjetiva y no comparable con las metricas automaticas.",
        "- Los resultados dependen de la version del dataset y de los modelos locales.",
        "",
    ]

    payload = {"generated_at": now, "commit": commit, "results": results}
    return "\n".join(md), payload


def save_report(results: dict) -> Path:
    """Escribe el reporte (.md y .json) y devuelve la ruta del Markdown."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md, payload = build_report(results)

    md_path = REPORTS_DIR / f"report_{stamp}.md"
    md_path.write_text(md, encoding="utf-8")
    (REPORTS_DIR / f"report_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return md_path
