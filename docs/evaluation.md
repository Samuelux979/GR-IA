# Protocolo de evaluacion de GR-IA

Este documento define como se mide la calidad de los componentes principales
de GR-IA de forma reproducible, mas alla de una demostracion puntual.

## 1. Componentes evaluados

| Componente | Que mide | Metrica | Recursos necesarios |
|---|---|---|---|
| Deteccion de ingredientes | Si el modelo de vision identifica los alimentos visibles | Precision, Recall, F1 | Ollama (vision) |
| Clasificacion | Si los ingredientes se asignan al nivel correcto | Accuracy global y por clase | Ollama (texto) |
| Recuperacion | Si las recetas recomendadas son relevantes | Precision@K (K=3,5,10) | PostgreSQL + embeddings |
| Generacion | Utilidad y calidad culinaria de las recetas IA | Rubrica humana 1-5 | Ollama (generacion) |

## 2. Estructura de datos

```
evaluation/
|-- ingredients_ground_truth.jsonl     Imagenes anotadas con ingredientes esperados
|-- classification_ground_truth.jsonl  Ingredientes de entrada + clasificacion esperada
|-- retrieval_queries.jsonl            Consultas + criterio de relevancia
|-- generated_recipe_reviews.jsonl     Recetas generadas + puntuaciones humanas
|-- images/                            Imagenes de prueba (NO versionadas)
`-- reports/                           Reportes generados (NO versionados)
```

Las imagenes no se versionan por tamano y licencia; solo se versiona el
fichero de anotaciones que las referencia por nombre.

## 3. Formatos de ground truth

### 3.1 Deteccion de ingredientes (`ingredients_ground_truth.jsonl`)
```json
{"image": "simple_01.jpg", "expected": ["tomato", "onion", "garlic"], "notes": "caso simple"}
```

### 3.2 Clasificacion (`classification_ground_truth.jsonl`)
```json
{"ingredients": ["chicken", "onion", "salt"],
 "expected": {"main": ["chicken"], "secondary": ["onion"], "accompaniment": [], "spices": ["salt"]}}
```

### 3.3 Recuperacion (`retrieval_queries.jsonl`)
```json
{"query_ingredients": {"main": ["chicken"], "secondary": ["onion"], "accompaniment": [], "spices": []},
 "relevant_if_ner_contains": ["chicken"], "notes": "principal fuerte"}
```
La relevancia se define por un **criterio verificable** (que el `ner` de la
receta contenga ciertos ingredientes) en lugar de por ids fijos, de modo que
la evaluacion siga siendo valida aunque se recargue la base de datos.

### 3.4 Generacion (`generated_recipe_reviews.jsonl`)
```json
{"title": "...", "ingredients": ["..."], "steps": ["..."],
 "scores": {"uso_ingredientes": 4, "claridad_pasos": 5, "viabilidad": 4, "seguridad": 5, "utilidad": 4},
 "comment": "..."}
```
Las plantillas se generan automaticamente con `scores: null`; el evaluador
rellena las puntuaciones a mano.

## 4. Metricas e interpretacion

### Deteccion de ingredientes
- **Precision** = aciertos / total detectado. Baja si hay muchos falsos positivos.
- **Recall** = aciertos / total esperado. Baja si se omiten ingredientes.
- **F1** = media armonica de ambas. Resume el equilibrio.
- Antes de comparar se normalizan los nombres (minusculas, espacios, plural/singular, sinonimos basicos).

### Clasificacion
- **Accuracy global** = ingredientes en el nivel correcto / total.
- **Accuracy por clase** = acierto dentro de cada nivel (main, secondary, accompaniment, spices).
- Errores tipicos: especias clasificadas como secundarios, guarniciones como principales.

### Recuperacion
- **Precision@K** = recetas relevantes entre las primeras K.
- Se evalua con K = 3, 5 y 10.
- Se registra ademas el score ponderado y la distancia vectorial cuando estan disponibles.

### Generacion (humana)
Rubrica de 1 a 5 en cinco criterios:
1. **uso_ingredientes**: usa correctamente los ingredientes detectados.
2. **claridad_pasos**: los pasos son claros y ordenados.
3. **viabilidad**: la receta es realizable en la practica.
4. **seguridad**: no propone combinaciones o cocciones inseguras.
5. **utilidad**: utilidad general de la receta.

Muestra minima recomendada: **10 recetas**. Esta evaluacion es subjetiva y
debe interpretarse por separado de las metricas automaticas.

## 5. Ejecucion

```powershell
# Evaluacion completa
python -m src.evaluation.run_evaluation --component all

# Por componente
python -m src.evaluation.run_evaluation --component ingredients
python -m src.evaluation.run_evaluation --component classification
python -m src.evaluation.run_evaluation --component retrieval
python -m src.evaluation.run_evaluation --component generation
```

Cada ejecucion genera un reporte con fecha y commit en
`evaluation/reports/report_<fecha>.md` (y su equivalente `.json`).

Si falta un recurso (Ollama, base de datos o imagenes), el componente
correspondiente se omite con un aviso claro y el resto continua.

## 6. Comparativa de modelos de lenguaje

Ademas de la evaluacion por componentes, el proyecto incluye una
comparativa entre tres modelos de lenguaje (uno pequeno, el de produccion
y un modelo frontera) para justificar la eleccion del modelo. La
metodologia, los resultados y como reproducirla estan en
[`docs/model_comparison.md`](model_comparison.md).

## 7. Limitaciones conocidas

- El dataset inicial es pequeno: los resultados son orientativos, no concluyentes.
- La deteccion y clasificacion dependen de modelos locales pequenos; su
  rendimiento varia entre ejecuciones por la naturaleza del muestreo del LLM.
- La relevancia en recuperacion usa un criterio simple (presencia del
  ingrediente principal); no captura matices de calidad culinaria.
- La evaluacion de generacion es subjetiva y depende del revisor.
- Las metricas no son comparables entre versiones si cambia el dataset de recetas.
