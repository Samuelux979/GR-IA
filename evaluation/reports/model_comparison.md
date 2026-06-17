# Comparativa de modelos de lenguaje - GR-IA

Solo se comparan los componentes dependientes del modelo de texto:
clasificacion de ingredientes y generacion de recetas.
(Deteccion y recuperacion son independientes del modelo de texto.)

## Generacion de recetas (evaluacion humana, 1-5)

| Criterio | qwen2.5:0.5b | qwen2.5:1.5b | Claude Opus 4.8 |
|---|---|---|---|
| uso_ingredientes | 1.67 | 4.11 | 5.0 |
| claridad_pasos | 3.0 | 4.11 | 5.0 |
| viabilidad | 3.0 | 4.11 | 5.0 |
| seguridad | 3.17 | 4.33 | 5.0 |
| utilidad | 2.17 | 3.67 | 4.6 |
| **Media global** | **2.6** | **4.07** | **4.92** |

## Clasificacion de ingredientes (accuracy)

| Metrica | qwen2.5:0.5b | qwen2.5:1.5b | Claude Opus 4.8 |
|---|---|---|---|
| Accuracy global | 0.419 | 0.553 | 0.974 |
