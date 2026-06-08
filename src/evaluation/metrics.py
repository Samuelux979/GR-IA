"""
Metricas de evaluacion: precision/recall/F1 para deteccion de ingredientes,
accuracy para clasificacion y Precision@K para recuperacion.

Funciones puras: no dependen de Ollama, BD ni modelos. Reciben datos ya
producidos por el sistema y los comparan contra el ground truth.
"""

from dataclasses import dataclass, field


# --- Deteccion de ingredientes ---------------------------------------------

@dataclass
class DetectionResult:
    precision: float
    recall:    float
    f1:        float
    true_positives:  int
    false_positives: list = field(default_factory=list)
    false_negatives: list = field(default_factory=list)


def prf1(predicted: list, expected: list) -> DetectionResult:
    """
    Calcula precision, recall y F1 entre dos listas de ingredientes normalizados.
    El emparejamiento usa contencion por palabras (matcher), de modo que
    "cherry tomato" cuenta como acierto de "tomato".
    """
    from src.evaluation.normalize import matches

    matched_pred = [p for p in predicted if any(matches(p, e) for e in expected)]
    matched_exp  = [e for e in expected  if any(matches(e, p) for p in predicted)]

    fp = [p for p in predicted if p not in matched_pred]
    fn = [e for e in expected  if e not in matched_exp]

    precision = len(matched_pred) / len(predicted) if predicted else 0.0
    recall    = len(matched_exp)  / len(expected)  if expected  else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return DetectionResult(
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        true_positives=len(matched_exp),
        false_positives=sorted(fp),
        false_negatives=sorted(fn),
    )


def aggregate_prf1(results: list[DetectionResult]) -> dict:
    """Micro-average de varias evaluaciones (suma TP/FP/FN globales)."""
    tp = sum(r.true_positives for r in results)
    fp = sum(len(r.false_positives) for r in results)
    fn = sum(len(r.false_negatives) for r in results)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
        "n_cases":   len(results),
    }


# --- Clasificacion ----------------------------------------------------------

LEVELS = ("main", "secondary", "accompaniment", "spices")


def classification_accuracy(predicted: dict, expected: dict) -> dict:
    """
    Compara la clasificacion por nivel de un caso.
    Devuelve aciertos y total por clase (para agregar despues).
    """
    # invertir: ingrediente -> nivel esperado
    expected_level = {}
    for level in LEVELS:
        for ing in expected.get(level, []):
            expected_level[ing.lower().strip()] = level

    per_class = {lvl: {"correct": 0, "total": 0} for lvl in LEVELS}
    correct = 0
    total   = 0

    for level in LEVELS:
        for ing in predicted.get(level, []):
            key = ing.lower().strip()
            if key in expected_level:
                total += 1
                exp = expected_level[key]
                per_class[exp]["total"] += 1
                if exp == level:
                    correct += 1
                    per_class[exp]["correct"] += 1

    return {"correct": correct, "total": total, "per_class": per_class}


def aggregate_classification(results: list[dict]) -> dict:
    """Agrega accuracy global y por clase de varios casos."""
    correct = sum(r["correct"] for r in results)
    total   = sum(r["total"] for r in results)
    per_class = {lvl: {"correct": 0, "total": 0} for lvl in LEVELS}
    for r in results:
        for lvl in LEVELS:
            per_class[lvl]["correct"] += r["per_class"][lvl]["correct"]
            per_class[lvl]["total"]   += r["per_class"][lvl]["total"]

    return {
        "accuracy": round(correct / total, 3) if total else 0.0,
        "n_ingredients": total,
        "per_class": {
            lvl: round(pc["correct"] / pc["total"], 3) if pc["total"] else None
            for lvl, pc in per_class.items()
        },
    }


# --- Recuperacion -----------------------------------------------------------

def precision_at_k(relevant_flags: list[bool], k: int) -> float:
    """Precision@K: proporcion de relevantes entre los primeros K resultados."""
    top = relevant_flags[:k]
    return round(sum(top) / k, 3) if k else 0.0


def recall_at_k(relevant_flags: list[bool], total_relevant: int, k: int) -> float:
    """Recall@K: relevantes recuperados en top-K sobre el total de relevantes."""
    if total_relevant <= 0:
        return 0.0
    return round(sum(relevant_flags[:k]) / total_relevant, 3)
