# GR-IA: Sistema RAG multimodal de recomendacion de recetas

**Trabajo de Fin de Grado** · Universidad San Jorge · Grado en Ingenieria Informatica
**Autor:** Samuel Ruiz
**Curso:** 2025 / 2026

---

## 1. Resumen

GR-IA es un sistema de recomendacion de recetas que parte de una fotografia
de ingredientes y devuelve recetas relevantes recuperadas de una base de
datos de 150 000 recetas (dataset Food.com), junto con una receta original
generada por un modelo de lenguaje que aprovecha los mismos ingredientes
detectados.

El sistema combina vision por computador, recuperacion semantica (RAG) y
generacion de lenguaje natural, ejecutandose de forma completamente local
sin dependencias de servicios externos en tiempo de ejecucion.

---

## 2. Arquitectura del sistema

### 2.1. Flujo principal

```
+-----------+   +---------------+   +-------------------+   +-----------+
|  Imagen   |-->|  Vision LLM   |-->|  Clasificacion    |-->|  Busqueda |
|  (.jpg)   |   |  (minicpm-v)  |   |  por niveles      |   |  hibrida  |
+-----------+   +---------------+   |  (qwen2.5)        |   +-----------+
                                    +-------------------+         |
                                                                  v
+-----------------+   +-------------------+   +------------------------+
|   Receta IA     |<--|  Generacion LLM   |   |  Top-K recetas         |
|   + macros      |   |  (qwen2.5)        |   |  (score + similitud)   |
+-----------------+   +-------------------+   +------------------------+
       |                                                |
       v                                                v
+----------------------------------------------------------------+
|       Presentacion + traduccion EN-ES (MarianMT)               |
+----------------------------------------------------------------+
```

### 2.2. Componentes

| Componente | Responsabilidad | Modelo / tecnologia |
|---|---|---|
| Deteccion visual | Identificar ingredientes en la foto | `minicpm-v` (Ollama) |
| Clasificacion | Asignar nivel de importancia a cada ingrediente | `qwen2.5:1.5b` (Ollama) |
| Embeddings | Vectorizar texto para busqueda semantica | `all-MiniLM-L6-v2` |
| Busqueda hibrida | Puntuacion ponderada + similitud coseno | PostgreSQL + pgvector |
| Generacion | Crear una receta original | `qwen2.5:1.5b` (Ollama) |
| Calculo nutricional | Macros desde base de datos USDA | USDA SR Legacy |
| Traduccion | Convertir pasos EN -> ES | `Helsinki-NLP/opus-mt-en-es` |

### 2.3. Esquema de datos (PostgreSQL)

```
recipes (150 000 filas)
  id, title, ingredients (jsonb), ner (jsonb), category,
  calories, protein_g, fat_g, carbs_g, fiber_g, etl_batch_id

recipe_chunks (150 000 filas)
  id, recipe_id, content, embedding (vector(384))

nutrition_usda (6 416 filas)
  fdc_id, name, name_lower, calories, protein_g, fat_g, carbs_g, fiber_g
```

---

## 3. Requisitos previos

### 3.1. Software

| Software | Version minima | Uso |
|---|---|---|
| Python | 3.13+ | Ejecucion del pipeline |
| Docker Desktop | 4.x | Contenedor de PostgreSQL + pgvector |
| Ollama | 0.4+ | Servidor local de LLMs |
| Git | 2.x | Clonado del repositorio |

### 3.2. Hardware recomendado

- **RAM:** 16 GB minimo (recomendable 32 GB con GPU)
- **GPU:** opcional pero recomendada (CUDA acelera embeddings y traduccion)
- **Espacio en disco:**
  - Modelos Ollama: ~6 GB
  - Modelos HuggingFace (cache local): ~700 MB
  - Dataset Food.com: ~170 MB
  - Dataset USDA SR Legacy: ~37 MB
  - Base de datos PostgreSQL: ~1.5 GB
  - **Total aproximado: 10 GB**

---

## 4. Instalacion

### 4.1. Clonar el repositorio

```powershell
git clone https://github.com/Samuelux979/GR-IA.git
cd GR-IA
```

### 4.2. Crear entorno virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4.3. Instalar dependencias Python

```powershell
pip install -r requirements.txt
```

El fichero `requirements.txt` incluye comentarios con el comando individual
de instalacion para cada paquete, util en caso de fallos puntuales.

### 4.4. Configurar variables de entorno

```powershell
copy .env.example .env
```

Los valores por defecto funcionan en una instalacion local estandar y
coinciden con `docker-compose.yml`. Solo es necesario editar `.env` si
se desea usar credenciales o puertos distintos.

### 4.5. Descargar los datasets

#### Food.com (recetas)

El fichero `src/data/foodcom_recipes.parquet` no se incluye en el
repositorio por tamano (~170 MB) y licencia.

- Fuente: https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
- Convertir a formato Parquet y colocar en: `src/data/foodcom_recipes.parquet`

Verificacion:

```powershell
python -c "from src.config import PARQUET_PATH; print('OK' if PARQUET_PATH.exists() else 'FALTA')"
```

#### USDA SR Legacy (nutricion)

- Fuente: https://fdc.nal.usda.gov/download-datasets (SR Legacy - CSV)
- Descomprimir y colocar los CSVs en: `src/data/sr_legacy/`
  - `food.csv`
  - `nutrient.csv`
  - `food_nutrient.csv`

### 4.6. Levantar servicios Docker

```powershell
docker compose up -d
```

Esto levanta un contenedor `gria_postgres` con PostgreSQL 16 y la
extension pgvector preinstalada (puerto 5432). Verificar:

```powershell
docker ps
docker exec -it gria_postgres psql -U admin -d gria_db -c "\dx"
```

### 4.7. Arrancar Ollama y descargar modelos

Asegurarse de que Ollama esta corriendo:

```powershell
ollama serve
```

En otra terminal, descargar los modelos requeridos:

```powershell
ollama pull minicpm-v
ollama pull qwen2.5:1.5b
```

Verificar:

```powershell
ollama list
```

Los modelos de HuggingFace (`all-MiniLM-L6-v2` y `opus-mt-en-es`) se
descargan automaticamente la primera vez que se ejecuta el pipeline y
quedan cacheados en `models/`.

---

## 5. Carga inicial de datos (ETL)

### 5.1. Carga del dataset Food.com

```powershell
python -m src.etl.run_etl
```

**Que hace internamente:**
1. Lee el Parquet y aplica filtros de calidad
2. Toma una muestra aleatoria reproducible de 150 000 recetas
3. Por cada batch de 2 000 recetas: transforma, vectoriza e inserta
4. Crea los indices HNSW (vector coseno) y GIN (JSONB) tras la carga
5. Ejecuta `VACUUM ANALYZE` final

**Tiempo estimado:**
- Con GPU CUDA: 30-45 minutos
- Solo CPU: 2-4 horas

**Tolerancia a fallos:** si se interrumpe, la siguiente ejecucion reanuda
desde el ultimo batch guardado en `src/data/etl_progress.json`.

**Verificacion del exito:**

```powershell
docker exec -it gria_postgres psql -U admin -d gria_db -c "SELECT COUNT(*) FROM recipes;"
```

Debe devolver 150 000.

### 5.2. Carga de la tabla nutricional USDA

```powershell
python -m src.nutrition.load_usda
```

Tiempo estimado: 5-10 segundos. Carga 6 416 alimentos filtrados de la
base USDA SR Legacy en la tabla `nutrition_usda`.

---

## 6. Ejecucion del pipeline de inferencia

### 6.1. Comando oficial

```powershell
python -m src.pipeline --image "ruta\a\foto.jpg" --top-k 3
```

**Argumentos:**

| Argumento | Tipo | Por defecto | Descripcion |
|---|---|---|---|
| `--image` | path | requerido | Ruta a la fotografia de ingredientes |
| `--top-k` | int | 5 | Numero de recetas a recuperar |

### 6.2. Salida esperada

El sistema imprime cuatro bloques:

1. **Ingredientes detectados** clasificados en 4 niveles
2. **Top-K recetas recuperadas** con titulo, ingredientes, pasos
   traducidos al espanol y puntuacion de coincidencia
3. **Receta generada por el LLM** con los ingredientes detectados,
   incluyendo el calculo nutricional desde USDA
4. **Pregunta interactiva** para guardar la receta generada en la BD

### 6.3. Requisitos previos a la ejecucion

- Contenedor Docker `gria_postgres` corriendo
- Servidor Ollama activo (`ollama serve`)
- Modelos Ollama descargados (`minicpm-v`, `qwen2.5:1.5b`)
- ETL ejecutado al menos una vez (tabla `recipes` con datos)

---

## 7. Demo Web (interfaz grafica y API)

GR-IA expone una capa web que permite usar el sistema desde el navegador
sin tener que abrir la terminal. Incluye una API REST documentada
automaticamente y una UI minima para subir imagenes y ver resultados.

### 7.1. Arranque del servidor

```powershell
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Para desarrollo, con recarga automatica al cambiar el codigo:

```powershell
python -m uvicorn src.api.main:app --reload --port 8000
```

### 7.2. Acceso a la interfaz

Una vez arrancado el servidor, abrir en el navegador:

| URL | Contenido |
|---|---|
| http://localhost:8000/ | Interfaz grafica de la demo |
| http://localhost:8000/docs | Documentacion OpenAPI/Swagger interactiva |
| http://localhost:8000/health | Estado de las dependencias en JSON |

### 7.3. Endpoints disponibles

| Metodo | Ruta | Descripcion |
|---|---|---|
| `GET` | `/` | Sirve la pagina HTML de la demo |
| `POST` | `/predict` | Recibe una imagen y devuelve ingredientes y recetas |
| `POST` | `/save` | Persiste en la BD una receta generada por la IA |
| `GET` | `/health` | Estado de PostgreSQL, Ollama y modelos |

### 7.4. Uso de la API por linea de comandos

Ejemplo de invocacion del endpoint principal con curl:

```powershell
curl -X POST -F "image=@foto.jpg" "http://localhost:8000/predict?top_k=3&generate=true"
```

Parametros disponibles:

| Parametro | Tipo | Por defecto | Descripcion |
|---|---|---|---|
| `image` | file | requerido | Imagen JPEG / PNG / WebP (max 10 MB) |
| `top_k` | int | 3 | Numero de recetas a recuperar (1-20) |
| `generate` | bool | true | Generar tambien una receta original con IA |

### 7.5. Esquema de la respuesta

```json
{
  "ingredients": {
    "main":          ["chicken"],
    "secondary":     ["celery", "egg"],
    "accompaniment": [],
    "spices":        []
  },
  "recipes": [
    {
      "id": 12345,
      "title": "Mom's Chicken Soup",
      "category": "Chicken",
      "ingredients": ["8 chicken legs", "..."],
      "steps":       ["Prepara los ingredientes.", "..."],
      "matches":     ["chicken", "celery"],
      "score":    36,
      "distance": 0.42,
      "macros":   { "calories": 450, "protein_g": 32, ... }
    }
  ],
  "ai_recipe": {
    "title":       "Chicken and Vegetable Stir-Fry",
    "ingredients": [...],
    "steps":       [...],
    "ner":         [...],
    "macros":      { "calories": 728, ... }
  },
  "warnings": []
}
```

### 7.6. Codigos de respuesta

| Codigo | Significado |
|---|---|
| `200` | Procesado correctamente |
| `400` | Formato de imagen no soportado o imagen corrupta |
| `413` | Imagen demasiado grande (mas de 10 MB) |
| `500` | Error interno inesperado |
| `503` | PostgreSQL o Ollama no disponibles |

### 7.7. Prerequisitos para la ejecucion

Antes de arrancar el servidor web, deben estar disponibles:

- Contenedor Docker `gria_postgres` corriendo (`docker compose up -d`)
- Servidor Ollama activo (`ollama serve`)
- Modelos Ollama descargados (`minicpm-v`, `qwen2.5:1.5b`)
- ETL ejecutado al menos una vez (tabla `recipes` poblada)
- Tabla nutricional USDA cargada (`python -m src.nutrition.load_usda`)

El endpoint `GET /health` permite verificar todo de un vistazo:

```powershell
curl http://localhost:8000/health
```

---

## 8. Estructura del proyecto

```
GR-IA/
|-- docker-compose.yml          Definicion de servicios Docker
|-- requirements.txt            Dependencias Python con versiones fijadas
|-- .env.example                Plantilla de variables de entorno
|-- .gitignore                  Exclusiones de Git
|-- README.md                   Este documento
|
|-- src/
|   |-- config.py               Configuracion global (rutas, credenciales)
|   |-- pipeline.py             Punto de entrada principal
|   |
|   |-- etl/                    Carga inicial del dataset
|   |   |-- run_etl.py          Orquestador del ETL
|   |   |-- extract.py          Carga y filtrado del Parquet
|   |   |-- transform.py        Validacion y normalizacion
|   |   |-- embed.py            Generacion de embeddings
|   |   |-- load.py             Insercion en PostgreSQL
|   |   |-- schema.py           Creacion de tablas e indices
|   |   `-- checkpoint.py       Control de progreso reanudable
|   |
|   |-- vision/                 Procesamiento de imagen y lenguaje
|   |   |-- llava_client.py     Cliente Ollama (vision + clasificacion + generacion)
|   |   `-- translator.py       Traduccion EN-ES con MarianMT
|   |
|   |-- retrieval/              Busqueda y guardado de recetas
|   |   |-- search.py           Busqueda hibrida ponderada + vectorial
|   |   `-- save_generated.py   Guardado de recetas generadas por IA
|   |
|   |-- nutrition/              Calculo nutricional
|   |   |-- load_usda.py        Carga de la base USDA en PostgreSQL
|   |   `-- calculator.py       Parser de ingredientes + calculo de macros
|   |
|   |-- api/                    Capa web (API REST + UI)
|   |   |-- main.py             Aplicacion FastAPI con los endpoints
|   |   |-- service.py          Logica de orquestacion compartida con el CLI
|   |   |-- schemas.py          Modelos Pydantic del contrato JSON
|   |   `-- static/
|   |       |-- index.html      Pagina de la demo
|   |       |-- style.css       Estilos
|   |       `-- app.js          Logica de la UI
|   |
|   `-- data/                   Datasets locales (no incluidos en git)
|       |-- foodcom_recipes.parquet
|       |-- sr_legacy/
|       |   |-- food.csv
|       |   |-- nutrient.csv
|       |   `-- food_nutrient.csv
|       `-- etl_progress.json
|
`-- models/                     Cache local de modelos HuggingFace
    |-- models--sentence-transformers--all-MiniLM-L6-v2/
    `-- opus-mt-en-es/
```

---

## 9. Persistencia de recetas generadas por IA

GR-IA permite que el usuario guarde en la base de datos las recetas
originales creadas por el modelo de lenguaje. Estas recetas se
distinguen explicitamente de las del dataset original, llevan
metadatos de procedencia y se deduplican para no contaminar las
busquedas posteriores.

### 9.1. Esquema de datos

Cuando una receta IA se guarda, se persiste con los siguientes campos
adicionales a los de una receta normal:

| Columna | Tipo | Uso |
|---|---|---|
| `source` | TEXT | `'foodcom'` o `'ai_generated'` |
| `provenance` | JSONB | metadatos de generacion (modelo, fecha, ingredientes de entrada, prompt) |
| `dedup_hash` | TEXT | SHA-256 normalizado para deteccion de duplicados |
| `steps` | JSONB | pasos estructurados, preservando el orden original |
| `macros_known` | BOOLEAN | `false` indica que los macros son estimados via USDA |

Las macros desconocidas se guardan como `NULL`, nunca como `0.0`,
para no contaminar analisis ni filtros nutricionales.

### 9.2. Procedencia (`provenance`)

Ejemplo del JSON guardado para cada receta generada:

```json
{
  "model":            "qwen2.5:1.5b",
  "embedding_model":  "all-MiniLM-L6-v2",
  "generated_at":     "2026-05-31T15:30:00+00:00",
  "input_ingredients": {
    "main":          ["chicken"],
    "secondary":     ["celery"],
    "accompaniment": [],
    "spices":        []
  },
  "prompt_version":   "v1"
}
```

Esta informacion permite reproducir, depurar o explicar el resultado
de cualquier receta persistida en el sistema.

### 9.3. Deduplicacion

El sistema calcula un hash SHA-256 a partir del titulo normalizado y los
ingredientes NER ordenados. Si una receta equivalente ya existe en la BD,
no se inserta de nuevo y se devuelve el `id` de la receta existente.

Se consideran equivalentes recetas que solo difieren en:
- Mayusculas / minusculas del titulo o de los ingredientes
- Espacios en blanco al inicio o al final
- Orden de los ingredientes NER
- Ingredientes duplicados dentro de la lista

### 9.4. Validacion previa al guardado

Antes de persistir, el modulo `src/retrieval/validation.py` comprueba:

- Titulo no vacio y de longitud razonable (max. 200 caracteres)
- Al menos un ingrediente y un paso
- NER no vacio
- Todos los elementos son cadenas de texto
- Ningun paso supera 1000 caracteres
- Ningun campo contiene bytes nulos

Si la validacion falla, la API devuelve `422 Unprocessable Entity`
con la lista detallada de errores y la receta no se persiste.

### 9.5. Filtrado en busquedas

Las recetas IA aparecen en los resultados de busqueda por defecto, pero
se marcan visualmente con un badge `[IA]` para que el usuario distinga
contenido original de contenido generado. El parametro `include_ai=false`
en `/predict`, o el flag `--no-ai` en el CLI, las excluye de los
resultados.

### 9.6. Migracion de datos existentes

El script `src/etl/migrate_provenance.py` aplica las nuevas columnas y
migra el contenido previo de forma idempotente. Ejecucion:

```powershell
python -m src.etl.migrate_provenance
```

Despues de ejecutarse, las recetas que estaban marcadas con
`etl_batch_id = -1` quedan etiquetadas como `source = 'ai_generated'`
con sus correspondientes `dedup_hash` y `macros_known = FALSE`.

### 9.7. Tests automatizados

Suite de pruebas con `pytest`:

```powershell
python -m pytest src/tests/ -v
```

- `test_validation.py`: 11 tests unitarios de las reglas de validacion
- `test_dedup_hash.py`: 11 tests unitarios del hash de deduplicacion
- `test_save_integration.py`: 7 tests de integracion contra PostgreSQL
  que verifican procedencia, duplicados, validacion y persistencia
  del embedding

---

## 10. Mantenimiento y limpieza

### 10.1. Detener los servicios

```powershell
docker compose stop
```

Detiene los contenedores conservando los datos.

### 10.2. Eliminar contenedores conservando datos

```powershell
docker compose down
```

### 10.3. Limpieza total (incluyendo la base de datos)

```powershell
docker compose down -v
```

**Advertencia:** la opcion `-v` elimina los volumenes Docker, lo que
**borra completamente la base de datos**. Tras ejecutar este comando
es necesario volver a lanzar el ETL completo.

### 10.4. Reiniciar el esquema de la base de datos sin perder Docker

```powershell
docker exec -it gria_postgres psql -U admin -d gria_db -c "DROP TABLE IF EXISTS recipe_chunks CASCADE; DROP TABLE IF EXISTS recipes CASCADE; DROP TABLE IF EXISTS nutrition_usda CASCADE;"
```

---

## 11. Referencias

### 11.1. Datasets

- **Food.com Recipes and Reviews**, irkaal et al. (Kaggle, 2021).
  https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
- **USDA FoodData Central - SR Legacy**, U.S. Department of Agriculture (2019).
  https://fdc.nal.usda.gov

### 11.2. Modelos

- **MiniCPM-V**, OpenBMB.
  https://github.com/OpenBMB/MiniCPM-V
- **Qwen2.5-1.5B-Instruct**, Alibaba Cloud.
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- **all-MiniLM-L6-v2**, sentence-transformers.
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Helsinki-NLP/opus-mt-en-es**, University of Helsinki.
  https://huggingface.co/Helsinki-NLP/opus-mt-en-es

### 11.3. Tecnologias

- **pgvector**: https://github.com/pgvector/pgvector
- **Ollama**: https://ollama.com
- **LangChain**: https://www.langchain.com
- **HNSW algorithm**: Malkov & Yashunin (2018), *Efficient and robust
  approximate nearest neighbor search using Hierarchical Navigable
  Small World graphs*.
