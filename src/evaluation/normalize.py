"""
Normalizacion y emparejamiento de ingredientes para comparar la prediccion
del sistema contra el ground truth.

Antes de calcular metricas se normaliza cada nombre (minusculas, espacios,
plural/singular por palabra, sinonimos basicos). El emparejamiento admite
contencion por palabras: "cherry tomato" cuenta como acierto de "tomato" y
"garlic cloves" como acierto de "garlic".
"""

# Sinonimos / equivalencias basicas frecuentes en los datasets de comida.
_SYNONYMS = {
    "scallion":     "green onion",
    "spring onion": "green onion",
    "aubergine":    "eggplant",
    "courgette":    "zucchini",
    "capsicum":     "bell pepper",
    "coriander":    "cilantro",
    "rocket":       "arugula",
    "rucola":       "arugula",
    "beet":         "beetroot",
}

# Palabras decorativas que no aportan al nombre del ingrediente.
_STOPWORDS = {
    "fresh", "raw", "whole", "sliced", "slice", "slices", "chopped", "diced",
    "minced", "halved", "halves", "half", "wedge", "wedges", "clove", "cloves",
    "bulb", "bulbs", "frond", "fronds", "leaf", "leaves", "and", "of", "ground",
}


def _singularize_word(w: str) -> str:
    """Singulariza una palabra por reglas seguras (sin destrozar la raiz)."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"                 # berries -> berry
    if w.endswith("oes") and len(w) > 4:
        return w[:-2]                        # tomatoes -> tomato
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]                        # boxes -> box
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]                        # cloves -> clove, eggs -> egg
    return w


def normalize_ingredient(name: str) -> str:
    """Normaliza un ingrediente: minusculas, espacios, sinonimos y singular por palabra."""
    n = " ".join(str(name).lower().strip().split())
    n = _SYNONYMS.get(n, n)
    words = [_singularize_word(w) for w in n.split()]
    n = " ".join(words)
    return _SYNONYMS.get(n, n)


def _content_tokens(name: str) -> set:
    """Tokens significativos de un ingrediente (sin palabras decorativas)."""
    tokens = {w for w in normalize_ingredient(name).split() if w not in _STOPWORDS}
    return tokens or set(normalize_ingredient(name).split())


def matches(a: str, b: str) -> bool:
    """
    Dos ingredientes coinciden si comparten el nucleo de palabras:
    los tokens de uno estan contenidos en los del otro.
    Ej: "cherry tomato" ~ "tomato", "garlic cloves" ~ "garlic".
    """
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta


def normalize_list(names) -> list:
    """Normaliza una coleccion de ingredientes, sin vacios ni duplicados (mantiene orden)."""
    seen, out = set(), []
    for x in (names or []):
        n = normalize_ingredient(x)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out
