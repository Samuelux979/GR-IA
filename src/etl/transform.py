"""
Transform — validate, clean, and build parent/child documents from raw CSV rows.
"""

import json
import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RecipeDocument:
    """Validated recipe ready for loading."""
    title: str
    ingredients: list[str]
    ner: list[str]
    link: str
    source: str
    # The semantic text that will be embedded (child chunk)
    chunk_text: str


@dataclass
class TransformResult:
    """Output of one chunk transformation."""
    documents: list[RecipeDocument] = field(default_factory=list)
    rows_discarded: int = 0


def _sanitize(text: str) -> str:
    """Remove NUL bytes and other characters PostgreSQL rejects."""
    return text.replace("\x00", "").replace("\u0000", "")


def _safe_parse_json(raw: str) -> list[str] | None:
    """Parse a JSON-encoded list string, return None on failure."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [
                _sanitize(str(item)).strip()
                for item in parsed
                if _sanitize(str(item)).strip()
            ]
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def _build_chunk_text(title: str, ner: list[str], directions: list[str]) -> str:
    """
    Build a single semantic string for embedding.

    Format chosen to maximize retrieval quality: the title and normalised
    ingredient list give the bi-encoder context, and the directions give
    the procedural detail.
    """
    ner_str = ", ".join(ner)
    steps = " ".join(
        f"Step {i}: {step.strip()}" for i, step in enumerate(directions, 1)
    )
    return f"Recipe: {title}. Ingredients: {ner_str}. {steps}"


def transform_chunk(df: pd.DataFrame) -> TransformResult:
    """Validate and transform one chunk DataFrame into RecipeDocuments."""
    result = TransformResult()

    for _, row in df.iterrows():
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            result.rows_discarded += 1
            continue

        ingredients = _safe_parse_json(row.get("ingredients", ""))
        directions = _safe_parse_json(row.get("directions", ""))
        ner = _safe_parse_json(row.get("NER", ""))

        if not ingredients or len(ingredients) < 2:
            result.rows_discarded += 1
            continue
        if not directions or len(directions) < 1:
            result.rows_discarded += 1
            continue
        if not ner:
            # Fallback: use raw ingredients as NER
            ner = ingredients

        # Normalise NER entries + sanitize
        ner = [_sanitize(n).lower().strip() for n in ner]
        title_clean = _sanitize(title).strip()
        directions_clean = [_sanitize(d) for d in directions]

        chunk_text = _build_chunk_text(title_clean, ner, directions_clean)

        result.documents.append(
            RecipeDocument(
                title=title_clean,
                ingredients=[_sanitize(i) for i in ingredients],
                ner=ner,
                link=_sanitize(row.get("link", "") or ""),
                source=_sanitize(row.get("source", "") or ""),
                chunk_text=chunk_text,
            )
        )

    return result
