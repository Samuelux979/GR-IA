"""
Tests unitarios del modulo de calculo nutricional (calculator.py).

Cubren las funciones puras de parseo y conversion, que son la parte mas
critica y propensa a errores. No requieren base de datos.
"""

from src.nutrition.calculator import _parse_quantity, parse_ingredient, to_grams


# --- _parse_quantity --------------------------------------------------------

def test_parse_quantity_integer():
    assert _parse_quantity("2") == 2.0


def test_parse_quantity_simple_fraction():
    assert _parse_quantity("1/2") == 0.5


def test_parse_quantity_mixed_number():
    assert _parse_quantity("1 1/2") == 1.5


def test_parse_quantity_decimal():
    assert _parse_quantity("1.5") == 1.5


def test_parse_quantity_empty_defaults_to_one():
    assert _parse_quantity("") == 1.0


def test_parse_quantity_invalid_defaults_to_one():
    assert _parse_quantity("abc") == 1.0


def test_parse_quantity_ignores_division_by_zero():
    # "1/0" no debe lanzar excepcion, se ignora ese termino
    assert _parse_quantity("1/0") == 1.0


# --- parse_ingredient -------------------------------------------------------

def test_parse_ingredient_qty_unit_name():
    r = parse_ingredient("2 cups broccoli florets")
    assert r["qty"] == 2.0
    assert r["unit"] == "cups"
    assert r["name"] == "broccoli florets"


def test_parse_ingredient_no_unit():
    r = parse_ingredient("1 avocado")
    assert r["qty"] == 1.0
    assert r["unit"] == ""
    assert r["name"] == "avocado"


def test_parse_ingredient_mixed_quantity():
    r = parse_ingredient("1 1/2 cups flour")
    assert r["qty"] == 1.5
    assert r["unit"] == "cups"
    assert r["name"] == "flour"


def test_parse_ingredient_unknown_unit_folds_into_name():
    # "clove" no esta en UNIT_TO_GRAMS como unidad estandar de masa/volumen,
    # pero "cloves" si; una palabra desconocida debe quedar en el nombre
    r = parse_ingredient("2 fresh tomatoes")
    assert r["qty"] == 2.0
    assert r["unit"] == ""
    assert "tomatoes" in r["name"]


def test_parse_ingredient_no_quantity_defaults_to_one():
    r = parse_ingredient("olive oil")
    assert r["qty"] == 1.0
    assert "oil" in r["name"]


def test_parse_ingredient_is_lowercased():
    r = parse_ingredient("3 CUPS Rice")
    assert r["unit"] == "cups"
    assert r["name"] == "rice"


# --- to_grams ---------------------------------------------------------------

def test_to_grams_known_unit():
    # 2 cups = 2 * 240 g
    assert to_grams({"qty": 2.0, "unit": "cups", "name": "broccoli"}) == 480.0


def test_to_grams_tablespoon():
    assert to_grams({"qty": 1.0, "unit": "tablespoon", "name": "olive oil"}) == 15.0


def test_to_grams_piece_weight_table():
    # 1 avocado usa el peso por pieza (200 g)
    assert to_grams({"qty": 1.0, "unit": "", "name": "avocado"}) == 200.0


def test_to_grams_oil_default_without_unit():
    # aceite sin unidad ni pieza conocida -> default de 15 g
    assert to_grams({"qty": 1.0, "unit": "", "name": "sunflower oil"}) == 15.0


def test_to_grams_spice_default_without_unit():
    # especia sin unidad -> default de 2 g
    assert to_grams({"qty": 1.0, "unit": "", "name": "paprika"}) == 2.0


def test_to_grams_generic_default():
    # ingrediente desconocido sin unidad -> pieza mediana de 100 g
    assert to_grams({"qty": 2.0, "unit": "", "name": "xyzfood"}) == 200.0
