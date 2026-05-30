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

## 7. Estructura del proyecto

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

## 8. Mantenimiento y limpieza

### 8.1. Detener los servicios

```powershell
docker compose stop
```

Detiene los contenedores conservando los datos.

### 8.2. Eliminar contenedores conservando datos

```powershell
docker compose down
```

### 8.3. Limpieza total (incluyendo la base de datos)

```powershell
docker compose down -v
```

**Advertencia:** la opcion `-v` elimina los volumenes Docker, lo que
**borra completamente la base de datos**. Tras ejecutar este comando
es necesario volver a lanzar el ETL completo.

### 8.4. Reiniciar el esquema de la base de datos sin perder Docker

```powershell
docker exec -it gria_postgres psql -U admin -d gria_db -c "DROP TABLE IF EXISTS recipe_chunks CASCADE; DROP TABLE IF EXISTS recipes CASCADE; DROP TABLE IF EXISTS nutrition_usda CASCADE;"
```

---

## 9. Referencias

### 9.1. Datasets

- **Food.com Recipes and Reviews**, irkaal et al. (Kaggle, 2021).
  https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
- **USDA FoodData Central - SR Legacy**, U.S. Department of Agriculture (2019).
  https://fdc.nal.usda.gov

### 9.2. Modelos

- **MiniCPM-V**, OpenBMB.
  https://github.com/OpenBMB/MiniCPM-V
- **Qwen2.5-1.5B-Instruct**, Alibaba Cloud.
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- **all-MiniLM-L6-v2**, sentence-transformers.
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Helsinki-NLP/opus-mt-en-es**, University of Helsinki.
  https://huggingface.co/Helsinki-NLP/opus-mt-en-es

### 9.3. Tecnologias

- **pgvector**: https://github.com/pgvector/pgvector
- **Ollama**: https://ollama.com
- **LangChain**: https://www.langchain.com
- **HNSW algorithm**: Malkov & Yashunin (2018), *Efficient and robust
  approximate nearest neighbor search using Hierarchical Navigable
  Small World graphs*.
