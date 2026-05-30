"""
Modulo de vision e interaccion con modelos de lenguaje via Ollama.

Flujo:
  1. minicpm-v detecta ingredientes en la imagen
  2. qwen2.5 los clasifica en 4 niveles de prioridad
  3. Python construye las tarjetas de receta en español
  4. MarianMT traduce los pasos del ingles al español
"""

import base64
import json
import logging
import re
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

VISION_PROMPT = """You are a precise food ingredient recognition assistant. Carefully analyze every part of this image.

Return ONLY a valid JSON object with this exact structure, no extra text:
{
  "ingredients": ["ingredient1", "ingredient2", ...]
}

Rules:
- Be SPECIFIC: write "chicken" not "meat", "celery" not "vegetable"
- Identify each ingredient separately, do not group them
- Include garnishes, herbs, spices and seasonings
- Do NOT use generic terms like "leaves", "herbs", "greens"
- Celery: pale green STRAIGHT ribbed sticks with no bulb at base
- Fennel: white/pale layered bulb at base with long feathery green fronds — write "fennel", NOT "celery"
- Garlic: small white/beige bulb with papery skin, often in clusters of cloves
- Parsley: bright green flat or curly dense leaf clusters — write "parsley", NOT "arugula"
- Arugula: small jagged dark-green leaves (also called rucola)
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
- "main": dominant food items that define the dish (e.g. fish, chicken, pasta, rice, beef)
- "secondary": important complements (e.g. tomato, onion, egg, cheese, celery, carrot)
- "accompaniment": garnish elements (e.g. lemon, lime, basil, arugula, parsley)
- "spices": ONLY dry or ground seasonings (e.g. black pepper, paprika, cumin, salt)

Rules:
- Every ingredient must appear in exactly one level
- Leave a level as empty list [] if nothing belongs there
- Do not add ingredients that were not in the original list
"""

GENERATE_RECIPE_PROMPT = """You are a creative chef. Create a simple original recipe using ONLY these ingredients plus basic pantry staples (salt, pepper, olive oil, water, garlic).

Detected ingredients: {ingredients}

Return ONLY a valid JSON object with this exact structure, no extra text:
{{
  "title": "Recipe title in English",
  "ingredients": ["amount ingredient", "amount ingredient"],
  "steps": ["Step description", "Step description"]
}}

Rules:
- Title must be concise and descriptive
- Ingredients: combine quantity and name in one string (e.g. "2 cups broccoli florets")
- Steps: 3 to 6 clear cooking steps
- Use ONLY the provided ingredients plus salt, pepper, olive oil, water, garlic
"""


def _make_llm(model: str, temperature: float) -> ChatOllama:
    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature, num_ctx=2048)


def encode_image(image_path: str | Path, max_size: int = 672) -> str:
    """Redimensiona la imagen y la devuelve en base64."""
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_json(raw: str) -> dict | None:
    """Extrae y parsea el primer bloque JSON de la respuesta del modelo."""
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start:end + 1]

    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, AttributeError):
        return None


def _parse_steps(chunk_text: str) -> list[str]:
    """Extrae los pasos del texto de un chunk (formato 'Step N: ...')."""
    steps = re.findall(r"Step \d+:\s*(.+?)(?=Step \d+:|$)", chunk_text, re.DOTALL)
    cleaned = []
    for s in steps:
        s = s.strip()
        if re.fullmatch(r"\d+\.?", s):
            continue
        s = re.sub(r"^\d+\.\s*", "", s).strip()
        if s:
            cleaned.append(s)
    return cleaned


def detect_ingredients(image_path: str | Path) -> dict:
    """
    Detecta y clasifica los ingredientes de una foto en dos pasos:
      1. minicpm-v extrae la lista de ingredientes visibles
      2. qwen2.5 los clasifica en main / secondary / accompaniment / spices
    """
    empty = {"main": [], "secondary": [], "accompaniment": [], "spices": []}

    image_b64  = encode_image(image_path)
    llm_vision = _make_llm(OLLAMA_VISION_MODEL, temperature=0.1)
    vision_msg = HumanMessage(content=[
        {"type": "text",      "text": VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ])

    vision_data = _parse_json(llm_vision.invoke([vision_msg]).content.strip())
    if not vision_data:
        logger.error("No se pudo parsear la respuesta de vision")
        return empty

    ingredients = [ing.lower().strip() for ing in vision_data.get("ingredients", [])]
    logger.info("Ingredientes detectados (%d): %s", len(ingredients), ingredients)
    if not ingredients:
        return empty

    llm_text      = _make_llm(OLLAMA_TEXT_MODEL, temperature=0.0)
    classify_data = _parse_json(
        llm_text.invoke([HumanMessage(content=CLASSIFY_PROMPT_TEMPLATE.format(
            ingredients=", ".join(ingredients)
        ))]).content.strip()
    )

    if not classify_data:
        logger.warning("Fallo al clasificar ingredientes, todos van a secundario")
        return {"main": [], "secondary": ingredients, "accompaniment": [], "spices": []}

    classified = {
        level: [ing.lower().strip() for ing in classify_data.get(level, [])]
        for level in ("main", "secondary", "accompaniment", "spices")
    }
    logger.info("Clasificacion: %s", classified)
    return classified


def flatten_ingredients(classified: dict) -> list[str]:
    """Devuelve todos los ingredientes clasificados en una lista plana."""
    result = []
    for level in ("main", "secondary", "accompaniment", "spices"):
        result.extend(classified.get(level, []))
    return result


def format_recipes_response(classified: dict, recipes: list[dict]) -> str:
    """Construye el texto de respuesta con las recetas recuperadas."""
    if not recipes:
        return "No se encontraron recetas para los ingredientes detectados."

    all_ingredients     = flatten_ingredients(classified)
    all_ingredients_set = set(all_ingredients)

    ing_str = ", ".join(all_ingredients) or "varios ingredientes"
    n       = len(recipes)
    intro   = (
        f"He detectado los siguientes ingredientes en tu foto: {ing_str}. "
        f"Aqui tienes {n} receta{'s' if n != 1 else ''} que encajan:"
    )

    from src.vision.translator import translate_steps_to_es

    lines = [intro, ""]
    for i, r in enumerate(recipes, 1):
        ner    = list(dict.fromkeys(r.get("ner") or []))
        shared = [ing for ing in ner if ing.lower() in all_ingredients_set]
        steps  = _parse_steps(r.get("chunk_text", ""))

        lines.append("-" * 50)
        lines.append(f"{i}. {r['title']}")
        if shared:
            lines.append(f"   Coincide con tu foto: {', '.join(shared)}")
        lines.append("")

        lines.append("   Ingredientes:")
        for ing in (r.get("ingredients") or []):
            lines.append(f"      - {ing}")
        lines.append("")

        translated = translate_steps_to_es(steps) if steps else []
        if translated:
            lines.append("   Pasos:")
            for j, step in enumerate(translated, 1):
                lines.append(f"      Paso {j}. {step}")
        lines.append("")

    return "\n".join(lines)


def generate_recipe(classified: dict, conn=None) -> dict | None:
    """
    Genera una receta original con los ingredientes detectados usando el LLM
    y calcula sus macronutrientes a partir de USDA si se pasa una conexion.
    """
    all_ingredients = flatten_ingredients(classified)
    if not all_ingredients:
        return None

    llm  = _make_llm(OLLAMA_GENERATION_MODEL, temperature=0.7)
    data = _parse_json(
        llm.invoke([HumanMessage(content=GENERATE_RECIPE_PROMPT.format(
            ingredients=", ".join(all_ingredients)
        ))]).content.strip()
    )

    if not data or not data.get("title") or not data.get("steps"):
        logger.warning("El modelo no devolvio una receta valida")
        return None

    title       = str(data.get("title", "")).strip()
    ingredients = [str(i).strip() for i in data.get("ingredients", []) if str(i).strip()]
    steps       = [str(s).strip() for s in data.get("steps", []) if str(s).strip()]

    if not title or not steps:
        return None

    from src.vision.translator import translate_steps_to_es

    # Calculo nutricional via USDA (si hay conexion a la BD)
    macros = {"calories": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0, "fiber_g": 0.0}
    if conn is not None:
        from src.nutrition.calculator import calculate_macros
        result = calculate_macros(conn, ingredients)
        macros = {k: result[k] for k in macros}
        logger.info("Macros calculadas: %s", macros)

    return {
        "title":            title,
        "ingredients":      ingredients,
        "steps":            steps,
        "steps_translated": translate_steps_to_es(steps),
        "ner":              list(dict.fromkeys(i.lower().strip() for i in all_ingredients)),
        "category":         "AI Generated",
        "calories":         macros["calories"],
        "protein_g":        macros["protein_g"],
        "fat_g":            macros["fat_g"],
        "carbs_g":          macros["carbs_g"],
        "fiber_g":          macros["fiber_g"],
    }
