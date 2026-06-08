# Comparativa de modelos de lenguaje

Este documento explica **de donde salen** los numeros del reporte
`evaluation/reports/model_comparison.md`: que se midio, con que modelos,
como se generaron las respuestas y como se puntuaron.

## 1. Objetivo

Justificar de forma cuantitativa la eleccion del modelo de lenguaje de
produccion (`qwen2.5:1.5b`) comparandolo con un modelo mas pequeno y con
un modelo frontera. Es decir: demostrar con datos que la calidad del
sistema depende del modelo y que el elegido es un punto de equilibrio
razonable bajo la restriccion de ejecutar **en local**.

## 2. Modelos comparados

| Etiqueta | Modelo | Rol en la comparativa |
|---|---|---|
| `qwen05b` | `qwen2.5:0.5b` | Modelo pequeno (limite inferior) |
| `qwen15b` | `qwen2.5:1.5b` | Modelo en produccion (el que usa GR-IA) |
| `opus`    | Claude Opus 4.8 | Modelo frontera (referencia, limite superior) |

Los dos primeros corren en local via Ollama. El tercero (Opus) no se
ejecuta en local: sus recetas se generaron con el modelo frontera como
**referencia de "techo" de calidad**, no como opcion desplegable.

## 3. Que se compara y que no

Solo se comparan los componentes que **dependen del modelo de texto**:

- **Clasificacion de ingredientes** (usa el modelo de texto).
- **Generacion de recetas** (usa el modelo de generacion).

La **deteccion** (modelo de vision `minicpm-v`, fijo) y la **recuperacion**
(PostgreSQL + embeddings, sin LLM) son **independientes** del modelo de
texto, por lo que no varian y no se incluyen. Esto en si mismo es un
resultado: cambiar el LLM de texto no afecta a vision ni a busqueda.

## 4. Procedimiento

Mismo `ground truth` para los tres modelos, para que la comparacion sea justa:

1. **Generacion**: se generan recetas a partir de las consultas de
   `evaluation/retrieval_queries.jsonl` (mismas entradas para los tres).
   - `qwen05b` y `qwen15b`: via `src/evaluation/compare_models.py --phase generate`.
   - `opus`: recetas redactadas por el modelo frontera con las mismas entradas.
   - En la comparativa se desactiva la traduccion de pasos (solo se evalua
     el texto original en ingles).
2. **Clasificacion**: se clasifican los ingredientes de
   `evaluation/classification_ground_truth.jsonl` con cada modelo y se
   calcula la `accuracy` frente al nivel esperado.
3. **Puntuacion de recetas** (evaluacion humana): cada receta se puntua de
   1 a 5 en cinco criterios con la rubrica del apartado 5.
4. **Reporte**: `compare_models.py --phase report` agrega las medias y
   construye `evaluation/reports/model_comparison.md`.

## 5. Rubrica de puntuacion (1-5)

| Criterio | Que valora |
|---|---|
| `uso_ingredientes` | Usa los ingredientes detectados y no inventa ingredientes no listados |
| `claridad_pasos` | Los pasos se entienden y estan completos |
| `viabilidad` | La receta es realmente cocinable |
| `seguridad` | No hay riesgos de seguridad alimentaria |
| `utilidad` | Utilidad general de la receta |

Penalizaciones aplicadas de forma consistente a los tres modelos:
ingredientes no listados que aparecen en los pasos, pasos incompletos o
incoherentes, contradicciones y, en el modelo pequeno, salidas con formato
roto (ingredientes/pasos "explotados" caracter a caracter).

## 6. Resultados

### Generacion (media 1-5)

| Criterio | qwen2.5:0.5b | qwen2.5:1.5b | Opus 4.8 |
|---|---|---|---|
| uso_ingredientes | 1.67 | 4.11 | 5.00 |
| claridad_pasos   | 3.00 | 4.11 | 5.00 |
| viabilidad       | 3.00 | 4.11 | 5.00 |
| seguridad        | 3.17 | 4.33 | 5.00 |
| utilidad         | 2.17 | 3.67 | 4.60 |
| **Media global** | **2.60** | **4.07** | **4.92** |

### Recetas validas generadas (robustez de formato)

| Modelo | Recetas validas de 10 |
|---|---|
| qwen2.5:0.5b | 6 (40% fallo al producir JSON valido) |
| qwen2.5:1.5b | 9 |
| Opus 4.8 | 10 |

### Clasificacion (accuracy global)

| qwen2.5:0.5b | qwen2.5:1.5b | Opus 4.8 |
|---|---|---|
| 0.419 | 0.553 | 0.974 |

## 7. Interpretacion

- La progresion es monotona (0.5b < 1.5b < Opus) en las tres metricas, lo
  que indica que la evaluacion es sensible y discrimina calidad real.
- El modelo de 0.5b no solo puntua bajo: **falla al producir salida
  estructurada** (ingredientes y pasos partidos caracter a caracter, 40%
  de recetas invalidas). Es un limite tecnico tangible, no subjetivo.
- El modelo de 1.5b produce recetas coherentes y cocinables, con fallos
  menores (algun ingrediente no listado). Es el **punto de equilibrio**:
  calidad aceptable ejecutando en local.
- Opus marca el techo de calidad pero requiere nube/API, lo que romperia
  el diseno local-first del proyecto (soberania tecnologica).

**Conclusion:** se elige `qwen2.5:1.5b` porque maximiza la calidad
alcanzable manteniendo la ejecucion 100% local y sin dependencias externas.

## 8. Limitaciones (honestidad metodologica)

- La puntuacion de recetas es **subjetiva**: la realiza un unico evaluador
  con una rubrica fija. No es una metrica automatica.
- Las recetas de Opus fueron generadas por el propio modelo frontera y
  puntuadas con la misma rubrica que el resto; representan una **referencia
  de techo**, no una opcion desplegada en el sistema.
- La muestra es pequena (10 consultas, 8 casos de clasificacion). Es una
  comparativa inicial, ampliable anadiendo lineas a los ficheros de
  `evaluation/` sin tocar codigo.

## 9. Como reproducir

```powershell
# 1. Generar recetas y clasificar con los modelos locales (requiere Ollama)
python -m src.evaluation.compare_models --phase generate

# 2. Puntuar a mano las recetas en evaluation/model_comparison/gen_*.jsonl
#    y anadir el set de Opus (gen_opus.jsonl) y su clasificacion.

# 3. Construir el reporte comparativo
python -m src.evaluation.compare_models --phase report
```

El reporte se guarda en `evaluation/reports/model_comparison.md`.
