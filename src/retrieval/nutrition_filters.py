"""
Filtros nutricionales deterministas para la busqueda de recetas.

Permite acotar los resultados a recetas que cumplan rangos concretos de
macronutrientes (p. ej. "minimo 25 g de proteina", "maximo 400 kcal") o
perfiles predefinidos. Opera directamente sobre las columnas nutricionales
de la tabla recipes mediante condiciones SQL, sin intervencion de ningun
modelo: es un filtrado deterministo basado en metadatos.
"""

# Macronutrientes filtrables (deben coincidir con las columnas de la tabla)
MACROS = ("calories", "protein_g", "fat_g", "carbs_g", "fiber_g")

# Perfiles predefinidos: cada uno mapea a rangos (minimo, maximo) por macro.
# None indica ausencia de limite por ese extremo. Umbrales por racion.
NUTRITION_PROFILES = {
    "alta_proteina":     {"protein_g": (25, None)},
    "baja_caloria":      {"calories":  (None, 400)},
    "baja_grasa":        {"fat_g":     (None, 10)},
    "alta_fibra":        {"fiber_g":   (8, None)},
    "proteica_y_ligera": {"protein_g": (25, None), "calories": (None, 500)},
}


def resolve_filters(profile: str | None = None, ranges: dict | None = None) -> dict:
    """
    Combina un perfil predefinido y/o rangos explicitos en un unico
    diccionario {macro: (minimo, maximo)}. Los rangos explicitos tienen
    prioridad sobre el perfil si coinciden en un macro.
    """
    filters: dict = {}
    if profile and profile in NUTRITION_PROFILES:
        filters.update(NUTRITION_PROFILES[profile])
    if ranges:
        for macro, bound in ranges.items():
            if macro in MACROS and bound:
                filters[macro] = bound
    return filters


def build_sql(filters: dict) -> tuple[str, dict]:
    """
    Construye el fragmento SQL (condiciones AND) y los parametros para
    aplicar los filtros nutricionales. Devuelve ("", {}) si no hay filtros.

    Cuando hay algun filtro activo se exige macros_known = TRUE, de modo que
    solo se filtran recetas con valores nutricionales fiables (las del
    dataset), excluyendo aquellas cuya nutricion es estimada o desconocida.
    """
    if not filters:
        return "", {}

    clauses = ["r.macros_known = TRUE"]
    params: dict = {}
    for macro, (lo, hi) in filters.items():
        if macro not in MACROS:
            continue
        if lo is not None:
            clauses.append(f"r.{macro} >= %(nf_{macro}_min)s")
            params[f"nf_{macro}_min"] = float(lo)
        if hi is not None:
            clauses.append(f"r.{macro} <= %(nf_{macro}_max)s")
            params[f"nf_{macro}_max"] = float(hi)

    return " AND " + " AND ".join(clauses), params
