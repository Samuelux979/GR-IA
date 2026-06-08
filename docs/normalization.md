# Normalizacion de ingredientes en la busqueda

Este documento describe como GR-IA normaliza los nombres de ingredientes
antes de buscar recetas, para tolerar sinonimos, variantes en espanol,
plurales y pequenos errores de escritura sin alterar el scoring ponderado.

## 1. Comportamiento de la busqueda hibrida (recordatorio)

La recuperacion (`src/retrieval/search.py`) combina dos senales:

1. **Score ponderado por nivel de ingrediente** (coincidencias en el campo `ner`):
   - `main` × 8
   - `secondary` × 4
   - `accompaniment` × 2
   - `spices` × 1
2. **Similitud vectorial** (distancia coseno con pgvector) entre el embedding
   de la consulta y el de cada receta.

El score de ingredientes filtra y ordena primero; la distancia vectorial
afina el orden dentro de recetas con score parecido. El texto de la consulta
para el embedding se construye en `search_recipes()` a partir de los
ingredientes ya normalizados.

## 2. Donde vive la normalizacion

| Pieza | Ubicacion |
|---|---|
| Modulo de normalizacion | `src/retrieval/ingredient_normalizer.py` |
| Diccionario de sinonimos | constante `SYNONYMS` de ese modulo |
| Mapeo espanol -> ingles | constante `ES_TO_EN` de ese modulo |
| Vocabulario para fuzzy | constante `VOCABULARY` de ese modulo |
| Integracion en la busqueda | `search_recipes()` llama a `normalize_level()` por nivel |

## 3. Orden de normalizacion

`normalize_ingredient()` aplica los pasos en este orden:

1. **Limpieza basica**: minusculas, eliminacion de acentos y puntuacion,
   colapso de espacios. Ej: `"Limón,"` -> `"limon"`.
2. **Plurales / morfologia**: reglas seguras en ingles.
   Ej: `"peppers"` -> `"pepper"`, `"tomatoes"` -> `"tomato"`, `"berries"` -> `"berry"`.
3. **Mapeo espanol -> ingles**: diccionario controlado.
   Ej: `"pollo"` -> `"chicken"`, `"cebolla"` -> `"onion"`.
4. **Sinonimos -> forma canonica**.
   Ej: `"red pepper"` / `"capsicum"` -> `"bell pepper"`, `"cilantro"` -> `"coriander"`.
5. **Fuzzy matching** (solo si el termino aun no se reconoce): se corrige
   contra el `VOCABULARY` curado con `difflib`, umbral de similitud 0.85.
   Ej: `"brocoli"` -> `"broccoli"`, `"chiken"` -> `"chicken"`.

El fuzzy es conservador: solo corrige hacia un vocabulario conocido y con
similitud alta, para no confundir ingredientes culinariamente distintos
(ej: `"meat"` NO se convierte en `"beet"`).

## 4. Ejemplos entrada -> salida

| Entrada | Salida |
|---|---|
| `Pollo` | `chicken` |
| `limón` | `lemon` |
| `red pepper` | `bell pepper` |
| `capsicum` | `bell pepper` |
| `scallion` | `green onion` |
| `tomatoes` | `tomato` |
| `brocoli` (typo) | `broccoli` |
| `xyzfood` (desconocido) | `xyzfood` (sin cambios) |

## 5. El scoring ponderado se preserva

La normalizacion se aplica **por nivel** con `normalize_level()`, sin mezclar
los ingredientes en una lista plana. Un `"pollo"` que llega en `main` se
convierte en `"chicken"` pero **sigue en `main`**, conservando su peso ×8.
Especias y acompanamientos mantienen sus pesos menores.

## 6. Como ampliar la normalizacion

- **Nuevo grupo de sinonimos**: anade entradas a `SYNONYMS` con la forma
  canonica como valor. Ej: `"runner bean": "green bean"`.
- **Nuevo termino en espanol**: anade la pareja a `ES_TO_EN`.
- **Nuevo ingrediente reconocible por fuzzy**: anadelo a `VOCABULARY`.

## 7. Como probar los cambios

```powershell
python -m pytest tests/retrieval/test_ingredient_normalizer.py -v
python -m pytest tests/retrieval/test_search.py -v
```

Los tests cubren sinonimos, mapeo espanol, plurales, fuzzy (positivos y
negativos) y la conservacion del nivel/peso en los parametros enviados al SQL.
