"""
Transforma cada fila del DataFrame de Food.com en un RecipeDocument
listo para insertar en PostgreSQL.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RecipeDocument:
    title:       str
    ingredients: list
    ner:         list
    category:    str
    calories:    float
    protein_g:   float
    fat_g:       float
    carbs_g:     float
    fiber_g:     float
    chunk_text:  str


@dataclass
class TransformResult:
    documents:      list = field(default_factory=list)
    rows_discarded: int  = 0


def _sanitize(text: str) -> str:
    return text.replace("\x00", "").replace(" ", " ")


def _to_list(val) -> list:
    if isinstance(val, (list, np.ndarray)):
        return [_sanitize(str(v).strip()) for v in val if str(v).strip()]
    return []


def _build_ingredients(quantities: list, parts: list) -> list:
    """Combina cantidades y nombres: ["1 cup"] + ["flour"] -> ["1 cup flour"]."""
    empty = {"", "none", "nan"}
    result = []
    for i, name in enumerate(parts):
        qty  = quantities[i] if i < len(quantities) else ""
        qty  = qty if qty.lower() not in empty else ""
        line = f"{qty} {name}".strip() if qty else name
        result.append(line)
    return result


def _build_chunk_text(title: str, ner: list, steps: list) -> str:
    """Construye el texto semantico que se vectorizara."""
    ner_str   = ", ".join(ner)
    steps_str = " ".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))
    return f"Recipe: {title}. Ingredients: {ner_str}. {steps_str}"


def transform_chunk(df: pd.DataFrame) -> TransformResult:
    """Valida y transforma un chunk del DataFrame en RecipeDocuments."""
    result = TransformResult()

    for _, row in df.iterrows():
        title = str(row.get("Name", "")).strip()
        if not title:
            result.rows_discarded += 1
            continue

        parts      = _to_list(row.get("RecipeIngredientParts"))
        quantities = _to_list(row.get("RecipeIngredientQuantities"))
        steps      = _to_list(row.get("RecipeInstructions"))

        if len(parts) < 2 or not steps:
            result.rows_discarded += 1
            continue

        ner         = list(dict.fromkeys(p.lower() for p in parts if p))
        ingredients = _build_ingredients(quantities, parts)

        def _float(col: str) -> float:
            v = row.get(col, 0.0)
            return float(v) if v is not None and str(v) != "nan" else 0.0

        result.documents.append(RecipeDocument(
            title       = _sanitize(title),
            ingredients = ingredients,
            ner         = ner,
            category    = _sanitize(str(row.get("RecipeCategory") or "")),
            calories    = _float("Calories"),
            protein_g   = _float("ProteinContent"),
            fat_g       = _float("FatContent"),
            carbs_g     = _float("CarbohydrateContent"),
            fiber_g     = _float("FiberContent"),
            chunk_text  = _build_chunk_text(_sanitize(title), ner, steps),
        ))

    return result
