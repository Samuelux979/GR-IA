"""
Calcula los macronutrientes totales de una receta a partir de su lista
de ingredientes en formato texto (ej: "2 cups broccoli florets").

Flujo:
  1. Parsear cada linea: cantidad + unidad + nombre del alimento
  2. Convertir la cantidad a gramos (tabla estatica)
  3. Buscar el alimento en USDA (nutrition_usda) por similitud
  4. Escalar las macros por gramos y sumar
"""

import logging
import re
from fractions import Fraction

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Conversion de unidades a gramos (valores aproximados estandar)
UNIT_TO_GRAMS = {
    # masa
    "g": 1, "gram": 1, "grams": 1,
    "kg": 1000, "kilogram": 1000, "kilograms": 1000,
    "mg": 0.001,
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.6, "lbs": 453.6, "pound": 453.6, "pounds": 453.6,
    # volumen (asumiendo densidad de agua)
    "ml": 1, "milliliter": 1, "milliliters": 1,
    "l": 1000, "liter": 1000, "liters": 1000,
    "cup": 240, "cups": 240,
    "tablespoon": 15, "tablespoons": 15, "tbsp": 15, "tbs": 15,
    "teaspoon": 5, "teaspoons": 5, "tsp": 5,
    "pinch": 0.3, "pinches": 0.3, "dash": 0.5, "dashes": 0.5,
    # piezas
    "piece": 50, "pieces": 50,
    "slice": 25, "slices": 25,
    "clove": 3, "cloves": 3,
}

# Peso medio por unidad de los ingredientes mas comunes (sin unidad explicita)
DEFAULT_PIECE_WEIGHT = {
    "avocado": 200,    "avocados": 200,
    "onion":   110,    "onions":   110,
    "tomato":  125,    "tomatoes": 125,
    "potato":  170,    "potatoes": 170,
    "carrot":  60,     "carrots":  60,
    "egg":     50,     "eggs":     50,
    "garlic clove": 3, "garlic cloves": 3,
    "chicken breast": 170, "chicken breasts": 170,
    "lemon":   65,     "lime":     45,
    "apple":   180,    "banana":   120,
    "broccoli": 90,    "cauliflower": 100,
    "fennel":  150,    "celery stalk": 40,
    "pepper":  120,    "peppers":  120,
    "cucumber": 200,
}

# Regex para parsear "1 1/2 cups broccoli florets" o "2 tablespoons olive oil"
_PARSE_RE = re.compile(
    r"^\s*"
    r"(?P<qty>[\d./\s]+?)?\s*"                       # cantidad (1, 1/2, 1 1/2, opcional)
    r"(?:(?P<unit>[a-z]+)\s+)?"                      # unidad (cup, tbsp, etc., opcional)
    r"(?P<name>[a-z][a-z\s,'-]*?)\s*$",              # nombre del alimento
    re.IGNORECASE,
)


def _parse_quantity(s: str) -> float:
    """Convierte '1', '1/2', '1 1/2' o '1.5' en float."""
    s = s.strip()
    if not s:
        return 1.0
    total = 0.0
    for part in s.split():
        try:
            total += float(Fraction(part))
        except (ValueError, ZeroDivisionError):
            pass
    return total or 1.0


def parse_ingredient(line: str) -> dict:
    """
    Parsea una linea como '2 cups broccoli florets' a
    {"qty": 2.0, "unit": "cups", "name": "broccoli florets"}.
    """
    m = _PARSE_RE.match(line.lower())
    if not m:
        return {"qty": 1.0, "unit": "", "name": line.strip().lower()}

    qty_str = (m.group("qty") or "1").strip()
    unit    = (m.group("unit") or "").strip()
    name    = (m.group("name") or "").strip()

    # Si la "unidad" no es conocida, era parte del nombre
    if unit and unit not in UNIT_TO_GRAMS:
        name = f"{unit} {name}".strip()
        unit = ""

    return {"qty": _parse_quantity(qty_str), "unit": unit, "name": name}


def to_grams(parsed: dict) -> float:
    """Convierte la cantidad parseada a gramos."""
    qty, unit, name = parsed["qty"], parsed["unit"], parsed["name"]

    if unit and unit in UNIT_TO_GRAMS:
        return qty * UNIT_TO_GRAMS[unit]

    # Sin unidad: buscar en la tabla de pesos por pieza
    for key, weight in DEFAULT_PIECE_WEIGHT.items():
        if key in name:
            return qty * weight

    # Sin unidad y sin pieza conocida: aplicar default segun el tipo de ingrediente
    # Aceites/grasas/condimentos liquidos: ~15g (1 cucharada)
    if any(k in name for k in ("oil", "vinegar", "sauce", "syrup", "honey")):
        return qty * 15
    # Especias y hierbas secas: ~2g
    if any(k in name for k in ("salt", "pepper", "paprika", "cumin", "oregano", "thyme", "basil")):
        return qty * 2

    # Por defecto, asumir una pieza mediana (~100 g)
    return qty * 100


# Condimentos / aliños con aporte calorico despreciable (los ignoramos)
NEGLIGIBLE = {"salt", "pepper", "salt and pepper", "salt to taste", "pepper to taste",
              "water", "ice", "to taste"}

# Sinonimos manuales: nombre del modelo -> termino USDA mas preciso
SYNONYMS = {
    "olive oil":    "oil, olive",
    "vegetable oil":"oil, vegetable",
    "canola oil":   "oil, canola",
    "tomato":       "tomatoes, red",
    "tomatoes":     "tomatoes, red",
    "onion":        "onions, raw",
    "potato":       "potatoes, raw",
    "garlic":       "garlic, raw",
    "chicken":      "chicken, broiler",
    "beef":         "beef, ground",
    "pork":         "pork, fresh",
    "rice":         "rice, white",
    "pasta":        "pasta, dry",
}


def lookup_nutrition(conn, name: str) -> dict | None:
    """
    Busca el alimento mas relevante en nutrition_usda.
    1. Si el nombre es despreciable (sal, agua, etc.) devuelve None.
    2. Si hay un sinonimo conocido, busca ese termino directamente.
    3. Prueba "X, raw" y "Xs, raw" (singular y plural).
    4. Fallback: similitud trigram, descartando categorias no relacionadas.
    """
    name = name.lower().strip()
    if not name or name in NEGLIGIBLE:
        return None

    # Limpiar palabras decorativas que confunden el match
    clean = re.sub(
        r"\b(florets?|chopped|sliced|diced|minced|fresh|raw|ground|whole|large|small|medium|peeled)\b",
        "", name,
    ).strip(" ,")
    if not clean:
        clean = name

    # Aplicar sinonimo si existe
    search = SYNONYMS.get(clean, clean)
    is_oil = "oil" in clean

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # 1. Match exacto sobre el termino USDA del sinonimo
        if clean in SYNONYMS:
            cur.execute("""
                SELECT name, calories, protein_g, fat_g, carbs_g, fiber_g
                FROM nutrition_usda
                WHERE name_lower LIKE %s
                ORDER BY length(name) ASC
                LIMIT 1
            """, (f"{search}%",))
            row = cur.fetchone()
            if row:
                return dict(row)

        # 2. Probar "X, raw" y plural "Xs, raw"
        for pattern in (f"{clean}, raw%", f"{clean}s, raw%"):
            cur.execute("""
                SELECT name, calories, protein_g, fat_g, carbs_g, fiber_g
                FROM nutrition_usda
                WHERE name_lower LIKE %s
                ORDER BY length(name) ASC
                LIMIT 1
            """, (pattern,))
            row = cur.fetchone()
            if row:
                return dict(row)

        # 3. Similarity trigram, excluyendo categorias no relacionadas
        exclude = "" if is_oil else (
            "AND name_lower NOT LIKE 'oil,%%' "
            "AND name_lower NOT LIKE 'spices,%%' "
            "AND name_lower NOT LIKE '%%powder%%' "
            "AND name_lower NOT LIKE '%%canned%%'"
        )
        cur.execute(f"""
            SELECT name, calories, protein_g, fat_g, carbs_g, fiber_g,
                   similarity(name_lower, %s) AS sim
            FROM nutrition_usda
            WHERE name_lower %% %s {exclude}
            ORDER BY sim DESC, length(name) ASC
            LIMIT 1
        """, (clean, clean))
        row = cur.fetchone()
        return dict(row) if row else None


def calculate_macros(conn, ingredients: list[str]) -> dict:
    """
    Calcula las macros totales sumando cada ingrediente parseado.
    Devuelve {calories, protein_g, fat_g, carbs_g, fiber_g, breakdown}
    donde breakdown contiene el detalle por ingrediente.
    """
    totals = {"calories": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0, "fiber_g": 0.0}
    breakdown = []

    for line in ingredients:
        parsed = parse_ingredient(line)
        grams  = to_grams(parsed)
        match  = lookup_nutrition(conn, parsed["name"])

        if not match:
            breakdown.append({"line": line, "matched": None, "grams": grams})
            logger.debug("No match para: %s", parsed["name"])
            continue

        # Escalar las macros por la proporcion (USDA es por 100 g)
        factor = grams / 100.0
        contrib = {k: match[k] * factor for k in totals}
        for k in totals:
            totals[k] += contrib[k]

        breakdown.append({
            "line":    line,
            "matched": match["name"],
            "grams":   round(grams, 1),
            **{k: round(v, 1) for k, v in contrib.items()},
        })

    return {**{k: round(v, 1) for k, v in totals.items()}, "breakdown": breakdown}
