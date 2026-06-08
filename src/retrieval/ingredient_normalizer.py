"""
Normalizacion canonica de ingredientes para la busqueda hibrida.

Convierte el nombre que escribe el usuario (o detecta el modelo de vision)
a la forma que usa el dataset Food.com (ingles, singular), aplicando en orden:

    1. Limpieza basica   -> minusculas, espacios colapsados, sin puntuacion
    2. Plurales          -> peppers -> pepper, tomatoes -> tomato
    3. Mapeo ES -> EN    -> pollo -> chicken, tomate -> tomato
    4. Sinonimos         -> red pepper / capsicum -> bell pepper
    5. Fuzzy matching    -> tomatoe -> tomato (solo contra vocabulario curado)

El alcance es un diccionario controlado, no traduccion automatica completa.
Para anadir un grupo nuevo basta con editar los diccionarios de este modulo.
"""

import re
import unicodedata
from difflib import get_close_matches

# ---------------------------------------------------------------------------
# 3. Mapeo espanol -> ingles (ingredientes frecuentes; lista controlada)
# ---------------------------------------------------------------------------
ES_TO_EN = {
    "pollo": "chicken",
    "tomate": "tomato",
    "tomates": "tomato",
    "cebolla": "onion",
    "cebollas": "onion",
    "ajo": "garlic",
    "arroz": "rice",
    "pasta": "pasta",
    "queso": "cheese",
    "pescado": "fish",
    "limon": "lemon",
    "lima": "lime",
    "perejil": "parsley",
    "zanahoria": "carrot",
    "zanahorias": "carrot",
    "patata": "potato",
    "patatas": "potato",
    "huevo": "egg",
    "huevos": "egg",
    "leche": "milk",
    "mantequilla": "butter",
    "harina": "flour",
    "azucar": "sugar",
    "sal": "salt",
    "pimienta": "pepper",
    "aceite": "oil",
    "carne": "beef",
    "cerdo": "pork",
    "gambas": "shrimp",
    "espinaca": "spinach",
    "espinacas": "spinach",
    "champinon": "mushroom",
    "champinones": "mushroom",
    "aguacate": "avocado",
    "manzana": "apple",
    "platano": "banana",
}

# ---------------------------------------------------------------------------
# 4. Sinonimos -> forma canonica (la que predomina en el dataset)
# ---------------------------------------------------------------------------
SYNONYMS = {
    "red pepper": "bell pepper",
    "sweet pepper": "bell pepper",
    "capsicum": "bell pepper",
    "scallion": "green onion",
    "spring onion": "green onion",
    "cilantro": "coriander",
    "garbanzo": "chickpea",
    "garbanzo bean": "chickpea",
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "rocket": "arugula",
    "rucola": "arugula",
    "beetroot": "beet",
    "prawn": "shrimp",
}

# ---------------------------------------------------------------------------
# 5. Vocabulario curado para fuzzy matching (ingredientes comunes del dataset).
#    El fuzzy solo corrige hacia terminos de esta lista, nunca a texto libre.
# ---------------------------------------------------------------------------
VOCABULARY = [
    "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp", "bacon",
    "egg", "milk", "butter", "cheese", "cream", "yogurt",
    "rice", "pasta", "flour", "bread", "noodle", "oat",
    "tomato", "onion", "garlic", "potato", "carrot", "celery", "fennel",
    "cucumber", "lettuce", "spinach", "broccoli", "cauliflower", "cabbage",
    "pepper", "bell pepper", "mushroom", "zucchini", "eggplant", "pea",
    "corn", "bean", "chickpea", "lentil", "beet", "radish", "leek", "asparagus",
    "lemon", "lime", "orange", "apple", "banana", "strawberry", "avocado",
    "grape", "pear", "peach", "mango", "pineapple",
    "parsley", "cilantro", "coriander", "basil", "mint", "dill", "rosemary",
    "thyme", "oregano", "arugula", "chive", "sage",
    "salt", "sugar", "oil", "olive oil", "vinegar", "honey", "paprika",
    "cumin", "cinnamon", "ginger", "nutmeg", "green onion", "shallot",
]

_FUZZY_THRESHOLD = 0.85   # similitud minima para aceptar una correccion
_PUNCT_RE = re.compile(r"[^\w\s]")


def _strip_accents(text: str) -> str:
    """Elimina acentos (limon <- limon con tilde) para casar con las claves ES."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _clean(name: str) -> str:
    """Limpieza basica: minusculas, sin acentos, sin puntuacion, espacios colapsados."""
    n = _strip_accents(str(name).lower())
    n = _PUNCT_RE.sub(" ", n)
    return " ".join(n.split())


def _singularize_word(w: str) -> str:
    """Singulariza una palabra con reglas seguras (no destroza la raiz)."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"          # berries -> berry
    if w.endswith("oes") and len(w) > 4:
        return w[:-2]                 # tomatoes -> tomato
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]                 # boxes -> box
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]                 # peppers -> pepper
    return w


def _singularize(name: str) -> str:
    return " ".join(_singularize_word(w) for w in name.split())


def normalize_ingredient(name: str, fuzzy: bool = True) -> str:
    """
    Devuelve la forma canonica de un ingrediente.
    fuzzy=False desactiva la correccion ortografica (util en tests).
    """
    n = _clean(name)
    if not n:
        return ""

    # Mapeo ES->EN sobre el termino completo (antes de singularizar palabras sueltas)
    if n in ES_TO_EN:
        n = ES_TO_EN[n]

    # Plurales / morfologia
    n = _singularize(n)
    n = ES_TO_EN.get(n, n)        # por si el singular coincide con una clave ES

    # Sinonimos -> forma canonica
    n = SYNONYMS.get(n, n)

    # Fuzzy matching contra el vocabulario curado (solo si sigue sin reconocerse)
    if fuzzy and n not in VOCABULARY:
        match = get_close_matches(n, VOCABULARY, n=1, cutoff=_FUZZY_THRESHOLD)
        if match:
            n = match[0]

    return n


def normalize_level(ingredients, fuzzy: bool = True) -> list:
    """
    Normaliza una lista de ingredientes de un nivel concreto, sin duplicados
    y preservando el orden. NO mezcla niveles: el scoring ponderado se mantiene.
    """
    seen, out = set(), []
    for ing in (ingredients or []):
        norm = normalize_ingredient(ing, fuzzy=fuzzy)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out
