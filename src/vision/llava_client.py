"""
Vision module — Ollama (qwen2.5vl) for ingredient detection and classification.
Multilingual presentation via Ollama generation model.

Pipeline:
  1. Ollama vision model → flat ingredient list from image
  2. Ollama text model  → classify into 4 priority levels
  3. Ollama generation model → present recipes in requested language
"""

import base64
import json
import logging
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_VISION_MODEL,
    OLLAMA_TEXT_MODEL,
    OLLAMA_GENERATION_MODEL,
)

logger = logging.getLogger(__name__)

# ─── Prompts ──────────────────────────────────────────────────────────────────
VISION_PROMPT = """You are a precise food ingredient recognition assistant. Carefully analyze every part of this image.

Return ONLY a valid JSON object with this exact structure, no extra text:
{
  "ingredients": ["ingredient1", "ingredient2", ...]
}

Rules:
- Be SPECIFIC: write "chicken" not "meat", "celery" not "vegetable", "arugula" not "leaves"
- Identify each ingredient separately, do not group them
- Look at every area of the image including edges and background
- Include garnishes, herbs, spices and seasonings
- Do NOT use generic terms like "leaves", "herbs", "greens" — always name the specific ingredient
- Celery: pale green STRAIGHT ribbed sticks with no bulb at base
- Fennel: white/pale layered bulb at base with long feathery green fronds — write "fennel", NOT "celery" or "onion"
- Garlic: small white/beige bulb with papery skin, often in clusters of cloves — always identify it if present
- Parsley: bright green flat or curly dense leaf clusters — write "parsley", NOT "arugula"
- Arugula: small jagged dark-green leaves (also called rucola)
- For citrus fruits: lemons are yellow and oval-shaped, oranges are round and orange-colored
- Hard-boiled egg: white exterior with yellow yolk, often cut in wedges
- Radishes: small round red/pink root vegetables
- List a maximum of 6 ingredients, prioritizing the most prominent ones
- Never list the same ingredient twice
"""

CLASSIFY_PROMPT_TEMPLATE = """You are a culinary assistant. Classify these detected ingredients into 4 levels.

Ingredients: {ingredients}

Return ONLY a valid JSON object, no extra text:
{{
  "main": [...],
  "secondary": [...],
  "accompaniment": [...],
  "spices": [...]
}}

Level definitions:
- "main": dominant food items that define the dish (e.g. fish, chicken, pasta, rice, beef, shrimp)
- "secondary": important complements but not the star (e.g. tomato, onion, egg, cheese, celery, carrot, potato)
- "accompaniment": citrus fruits and garnish elements (e.g. lemon, lime, orange, basil leaves, arugula, parsley, rosemary sprigs)
- "spices": ONLY dry or ground seasonings (e.g. red chili flakes, black pepper, paprika, cumin, salt) — never put fresh fruits or vegetables here

Rules:
- Every ingredient must appear in exactly one level
- Leave a level as empty list [] if nothing belongs there
- Do not add ingredients that were not in the original list
"""

INTRO_PROMPT_TEMPLATE = """You are a cooking assistant. Write ONLY in {language}.

The user has these ingredients: {all_ingredients}.
We found {n} matching recipes with these EXACT titles (do not translate or modify them): {titles}.

Write a short friendly introduction (2-3 sentences) in {language} presenting these recipes.
Use the recipe titles EXACTLY as written above. Do NOT translate titles. Do NOT list ingredients. Do NOT add recipes.
"""

TRANSLATE_STEPS_PROMPT = """Translate the following cooking steps to {language}.
Keep the same step numbering. Translate only the text — do not add or remove steps.

{steps}

Provide only the translated steps, nothing else."""


# ─── Ollama helpers ───────────────────────────────────────────────────────────
def _make_llm(model: str, temperature: float) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=2048,
    )


def encode_image(image_path: str | Path, max_size: int = 672) -> str:
    """Resize image and return base64 encoding."""
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_json(raw_text: str) -> dict | None:
    """Extract and parse JSON from a model response, handling markdown fences."""
    json_str = raw_text
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]
    try:
        return json.loads(json_str.strip())
    except (json.JSONDecodeError, AttributeError):
        return None


# ─── Public API ───────────────────────────────────────────────────────────────
def detect_ingredients(image_path: str | Path) -> dict:
    """
    Two-step pipeline:
      1. Ollama vision model detects ingredients from image → flat list
      2. Ollama text model classifies list into 4 priority levels

    Returns a dict with keys: main, secondary, accompaniment, spices.
    """
    empty = {"main": [], "secondary": [], "accompaniment": [], "spices": []}

    # ── Step 1: vision detection ──────────────────────────────────────────────
    image_b64 = encode_image(image_path)
    llm_vision = _make_llm(OLLAMA_VISION_MODEL, temperature=0.1)

    vision_msg = HumanMessage(
        content=[
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    )

    raw_vision = llm_vision.invoke([vision_msg]).content.strip()
    logger.debug("Vision raw response: %s", raw_vision)

    vision_data = _parse_json(raw_vision)
    if not vision_data:
        logger.error("Failed to parse vision response: %s", raw_vision)
        return empty

    ingredients = [ing.lower().strip() for ing in vision_data.get("ingredients", [])]
    logger.info("Detected %d ingredients: %s", len(ingredients), ingredients)

    if not ingredients:
        return empty

    # ── Step 2: text classification into 4 levels ─────────────────────────────
    llm_text = _make_llm(OLLAMA_TEXT_MODEL, temperature=0.0)
    classify_prompt = CLASSIFY_PROMPT_TEMPLATE.format(ingredients=", ".join(ingredients))

    raw_classify = llm_text.invoke([HumanMessage(content=classify_prompt)]).content.strip()
    logger.debug("Classification raw response: %s", raw_classify)

    classify_data = _parse_json(raw_classify)
    if not classify_data:
        logger.warning("Failed to parse classification, putting all in secondary")
        return {"main": [], "secondary": ingredients, "accompaniment": [], "spices": []}

    classified = {
        level: [ing.lower().strip() for ing in classify_data.get(level, [])]
        for level in ("main", "secondary", "accompaniment", "spices")
    }
    logger.info("Classified: %s", classified)
    return classified


def flatten_ingredients(classified: dict) -> list[str]:
    """Return a flat list of all detected ingredients across all levels."""
    result = []
    for level in ("main", "secondary", "accompaniment", "spices"):
        result.extend(classified.get(level, []))
    return result


def format_recipes_response(classified: dict, recipes: list[dict], language: str = "Spanish") -> str:
    """Present retrieved recipes: LLM writes intro, Python builds recipe cards."""
    if not recipes:
        return "No se encontraron recetas para los ingredientes detectados."

    all_ingredients = flatten_ingredients(classified)
    all_ingredients_set = set(all_ingredients)
    titles = [r["title"] for r in recipes]

    # ── LLM writes only the intro (simple task → no hallucinations) ───────────
    prompt = INTRO_PROMPT_TEMPLATE.format(
        language=language,
        all_ingredients=", ".join(all_ingredients) or "various",
        n=len(recipes),
        titles=", ".join(f'"{t}"' for t in titles),
    )
    llm = _make_llm(OLLAMA_GENERATION_MODEL, temperature=0.7)
    intro = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    # ── Python builds each recipe card deterministically ─────────────────────
    llm = _make_llm(OLLAMA_GENERATION_MODEL, temperature=0.1)
    lines = [intro, ""]
    for i, r in enumerate(recipes, 1):
        ner = list(dict.fromkeys(r.get("ner") or []))
        shared = [ing for ing in ner if ing.lower() in all_ingredients_set]
        raw_ingredients = r.get("ingredients") or []
        steps = _parse_steps(r.get("chunk_text", ""))

        # Translate steps to the requested language
        translated_steps = _translate_steps(llm, steps, language) if steps else []

        lines.append(f"{'─'*50}")
        lines.append(f"📌 {i}. {r['title']}")
        if shared:
            lines.append(f"   ✓ Coincide con tu foto: {', '.join(shared)}")
        lines.append("")

        lines.append("   🧂 Ingredientes:")
        for ing in raw_ingredients:
            lines.append(f"      • {ing}")
        lines.append("")

        if translated_steps:
            lines.append("   👨‍🍳 Pasos:")
            for j, step in enumerate(translated_steps, 1):
                lines.append(f"      Paso {j}. {step}")
        lines.append("")

    return "\n".join(lines)


def _parse_steps(chunk_text: str) -> list[str]:
    """Extract steps from chunk content formatted as 'Step 1: ... Step 2: ...'"""
    import re
    steps = re.findall(r"Step \d+:\s*(.+?)(?=Step \d+:|$)", chunk_text, re.DOTALL)
    cleaned = []
    for s in steps:
        s = s.strip()
        # Skip steps that are just a lone number/label (e.g. "1." or "2.")
        if re.fullmatch(r"\d+\.?", s):
            continue
        # Strip leading lone numbers like "1.\n" at the start of a step
        s = re.sub(r"^\d+\.\s*", "", s).strip()
        if s:
            cleaned.append(s)
    return cleaned


def _translate_steps(llm: ChatOllama, steps: list[str], language: str) -> list[str]:
    """Translate a list of cooking steps to the requested language."""
    if not steps or language.lower() == "english":
        return steps
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    prompt = TRANSLATE_STEPS_PROMPT.format(language=language, steps=numbered)
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    # Parse numbered lines back into a list
    import re
    translated = re.findall(r"^\d+\.\s*(.+)", raw, re.MULTILINE)
    # Fallback: if parsing fails, return the raw text as one step
    return translated if translated else [raw]
