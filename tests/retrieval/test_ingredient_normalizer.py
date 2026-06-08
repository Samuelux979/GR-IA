"""
Tests de la normalizacion canonica de ingredientes (src/retrieval/ingredient_normalizer.py):
sinonimos, mapeo espanol->ingles, plurales y fuzzy matching tolerante a typos.
"""

from src.retrieval.ingredient_normalizer import normalize_ingredient, normalize_level


# --- Sinonimos --------------------------------------------------------------

def test_synonym_red_pepper_to_bell_pepper():
    assert normalize_ingredient("red pepper") == "bell pepper"


def test_synonym_capsicum_to_bell_pepper():
    assert normalize_ingredient("capsicum") == "bell pepper"


def test_synonyms_collapse_to_same_canonical():
    formas = ["red pepper", "sweet pepper", "capsicum"]
    canonicas = {normalize_ingredient(x) for x in formas}
    assert canonicas == {"bell pepper"}


def test_synonym_scallion_to_green_onion():
    assert normalize_ingredient("scallion") == "green onion"
    assert normalize_ingredient("spring onion") == "green onion"


def test_synonym_garbanzo_to_chickpea():
    assert normalize_ingredient("garbanzo") == "chickpea"


def test_synonym_aubergine_to_eggplant():
    assert normalize_ingredient("aubergine") == "eggplant"


# --- Mapeo espanol -> ingles -----------------------------------------------

def test_spanish_pollo_to_chicken():
    assert normalize_ingredient("pollo") == "chicken"


def test_spanish_with_accent():
    assert normalize_ingredient("limón") == "lemon"


def test_spanish_basic_set():
    pares = {"tomate": "tomato", "cebolla": "onion", "ajo": "garlic",
             "arroz": "rice", "queso": "cheese", "pescado": "fish",
             "perejil": "parsley"}
    for es, en in pares.items():
        assert normalize_ingredient(es) == en


def test_english_input_not_broken_by_spanish_map():
    # un ingrediente que ya llega en ingles debe quedar igual
    assert normalize_ingredient("chicken") == "chicken"
    assert normalize_ingredient("tomato") == "tomato"


# --- Plurales / morfologia --------------------------------------------------

def test_plural_peppers_to_pepper():
    assert normalize_ingredient("peppers", fuzzy=False) == "pepper"


def test_plural_tomatoes_to_tomato():
    assert normalize_ingredient("tomatoes") == "tomato"


def test_plural_berries_to_berry():
    assert normalize_ingredient("berries", fuzzy=False) == "berry"


def test_singular_and_plural_match():
    assert normalize_ingredient("carrots") == normalize_ingredient("carrot")


# --- Fuzzy matching (positivos) --------------------------------------------

def test_fuzzy_corrects_minor_typo():
    assert normalize_ingredient("brocoli") == "broccoli"
    assert normalize_ingredient("tomatoe") == "tomato"
    assert normalize_ingredient("chiken") == "chicken"


# --- Fuzzy matching (negativos: no debe corregir de forma peligrosa) -------

def test_fuzzy_does_not_correct_unknown_word():
    # palabra sin parecido suficiente: se queda igual
    assert normalize_ingredient("xyzfood") == "xyzfood"


def test_fuzzy_does_not_confuse_distinct_ingredients():
    # 'meat' no debe convertirse en 'beet' ni 'beef' (similitud insuficiente)
    assert normalize_ingredient("meat") == "meat"


def test_fuzzy_can_be_disabled():
    # con fuzzy=False, un typo no se corrige
    assert normalize_ingredient("brocoli", fuzzy=False) == "brocoli"


# --- normalize_level: preserva nivel, sin duplicados ni vacios -------------

def test_normalize_level_dedupes_and_preserves_order():
    out = normalize_level(["pollo", "chicken", "Tomate", "tomato"])
    # pollo->chicken y chicken colapsan; tomate->tomato y tomato colapsan
    assert out == ["chicken", "tomato"]


def test_normalize_level_empty():
    assert normalize_level([]) == []
