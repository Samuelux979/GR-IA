"""
Validacion previa al guardado de recetas generadas por IA.
Asegura que se cumplen los minimos de calidad antes de persistir.
"""

MAX_TITLE_LEN = 200
MAX_STEP_LEN  = 1000


def validate_recipe(recipe: dict) -> list[str]:
    """
    Comprueba que una receta cumple los minimos de calidad.
    Devuelve la lista de errores encontrados (vacia si es valida).
    """
    errors: list[str] = []

    title = recipe.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append("Titulo vacio o no es texto.")
    elif len(title) > MAX_TITLE_LEN:
        errors.append(f"Titulo demasiado largo (>{MAX_TITLE_LEN} caracteres).")
    elif "\x00" in title:
        errors.append("Titulo contiene caracteres nulos.")

    ingredients = recipe.get("ingredients")
    if not isinstance(ingredients, list) or len(ingredients) < 1:
        errors.append("Debe haber al menos un ingrediente.")
    elif any(not isinstance(i, str) or not i.strip() for i in ingredients):
        errors.append("Algun ingrediente esta vacio o no es texto.")

    steps = recipe.get("steps")
    if not isinstance(steps, list) or len(steps) < 1:
        errors.append("Debe haber al menos un paso.")
    elif any(not isinstance(s, str) or not s.strip() for s in steps):
        errors.append("Algun paso esta vacio o no es texto.")
    elif any(len(s) > MAX_STEP_LEN for s in steps):
        errors.append(f"Algun paso supera la longitud maxima ({MAX_STEP_LEN} caracteres).")

    ner = recipe.get("ner")
    if not isinstance(ner, list) or len(ner) < 1:
        errors.append("Lista NER vacia.")
    elif any(not isinstance(n, str) or not n.strip() for n in ner):
        errors.append("Algun elemento de NER esta vacio o no es texto.")

    return errors
