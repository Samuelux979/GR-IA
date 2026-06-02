"""
Aplicacion FastAPI de GR-IA.

Endpoints:
  GET  /          -> Pagina HTML de la demo
  POST /predict   -> Detecta ingredientes y recomienda recetas
  POST /save      -> Guarda en BD una receta generada por la IA
  GET  /health    -> Estado de las dependencias (DB, Ollama, tabla nutricional)
  GET  /docs      -> Documentacion OpenAPI auto-generada por FastAPI

Uso:
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import tempfile
from pathlib import Path

import httpx
import psycopg2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.config import get_dsn, OLLAMA_BASE_URL, OLLAMA_VISION_MODEL, OLLAMA_TEXT_MODEL
from src.api.service import run_prediction, persist_generated_recipe
from src.api.schemas import (
    AIRecipe, HealthCheck, HealthResponse, Ingredients, Macros,
    PredictResponse, Recipe, SaveRequest, SaveResponse,
)
from src.retrieval.save_generated import RecipeValidationError

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="GR-IA",
    description="API de recomendacion de recetas a partir de una imagen de ingredientes.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    """Sirve la pagina HTML de la demo."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Evita el 404 que genera la peticion automatica del navegador."""
    return Response(status_code=204)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(..., description="Foto de los ingredientes (JPEG/PNG/WebP, max 10MB)"),
    top_k: int        = Query(3, ge=1, le=20, description="Numero de recetas a recuperar"),
    generate: bool    = Query(True, description="Generar tambien una receta original con IA"),
    include_ai: bool  = Query(True, description="Incluir recetas previamente generadas por IA en la busqueda"),
):
    """
    Procesa una imagen y devuelve los ingredientes detectados,
    las recetas mas relevantes de la base de datos y opcionalmente
    una receta original generada con LLM.
    """
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Formato no soportado. Permitidos: {', '.join(ALLOWED_TYPES)}")

    contents = await image.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"Imagen demasiado grande (max {MAX_IMAGE_BYTES // (1024*1024)} MB)")

    # Guardar a fichero temporal
    suffix  = Path(image.filename or "img.jpg").suffix.lower() or ".jpg"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)

    try:
        Path(tmp_path).write_bytes(contents)

        # Verificar que la imagen es legible
        try:
            with Image.open(tmp_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "La imagen no se puede leer o esta corrupta.")

        try:
            result = run_prediction(tmp_path, top_k=top_k, generate=generate, include_ai=include_ai)
        except psycopg2.OperationalError:
            raise HTTPException(503, "Base de datos no disponible.")
        except httpx.HTTPError:
            raise HTTPException(503, "Servidor Ollama no disponible.")
        except Exception:
            logger.exception("Error inesperado durante la prediccion")
            raise HTTPException(500, "Error interno durante la prediccion.")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return _build_predict_response(result)


def _build_predict_response(result: dict) -> PredictResponse:
    """Convierte el dict del servicio en el modelo Pydantic publico."""
    ingredients = Ingredients(**result["ingredients"])
    recipes     = [
        Recipe(
            id           = r["id"],
            title        = r["title"],
            category     = r["category"],
            source       = r.get("source", "foodcom"),
            macros_known = r.get("macros_known", True),
            ingredients  = r["ingredients"],
            steps        = r["steps"],
            matches      = r["matches"],
            score        = r["score"],
            distance     = r["distance"],
            macros       = Macros(**r["macros"]),
        )
        for r in result["recipes"]
    ]

    ai = result["ai_recipe"]
    ai_recipe = None
    if ai:
        ai_recipe = AIRecipe(
            title       = ai["title"],
            ingredients = ai["ingredients"],
            steps       = ai["steps_translated"],
            ner         = ai["ner"],
            macros      = Macros(
                calories=ai["calories"],   protein_g=ai["protein_g"], fat_g=ai["fat_g"],
                carbs_g=ai["carbs_g"],     fiber_g=ai["fiber_g"],
            ),
        )

    return PredictResponse(
        ingredients = ingredients,
        recipes     = recipes,
        ai_recipe   = ai_recipe,
        warnings    = result["warnings"],
    )


@app.post("/save", response_model=SaveResponse)
def save(req: SaveRequest):
    """
    Persiste en la base de datos una receta generada por la IA.
    Aplica validacion previa y detecta duplicados: si la receta ya existe
    devuelve su id sin volver a insertar (duplicate=True en la respuesta).
    """
    recipe = {
        "title":             req.title,
        "ingredients":       req.ingredients,
        "steps":             req.steps,
        "ner":               req.ner,
        "category":          "AI Generated",
        "calories":          req.macros.calories,
        "protein_g":         req.macros.protein_g,
        "fat_g":             req.macros.fat_g,
        "carbs_g":           req.macros.carbs_g,
        "fiber_g":           req.macros.fiber_g,
        "model":             req.model,
        "embedding_model":   req.embedding_model,
        "input_ingredients": req.input_ingredients,
        "prompt_version":    req.prompt_version,
    }
    try:
        recipe_id, was_new = persist_generated_recipe(recipe)
    except RecipeValidationError as e:
        raise HTTPException(422, {"errors": e.errors})
    except psycopg2.OperationalError:
        raise HTTPException(503, "Base de datos no disponible.")
    except Exception:
        logger.exception("Error guardando receta generada")
        raise HTTPException(500, "No se pudo guardar la receta.")

    return SaveResponse(saved=True, recipe_id=recipe_id, duplicate=not was_new)


@app.get("/health", response_model=HealthResponse)
def health():
    """Comprueba el estado de las dependencias externas."""
    checks: dict[str, HealthCheck] = {}

    # PostgreSQL
    try:
        conn = psycopg2.connect(get_dsn(), connect_timeout=3)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM recipes;")
            n_recipes = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM nutrition_usda;")
            n_nutrition = cur.fetchone()[0]
        conn.close()
        checks["database"] = HealthCheck(status="ok", detail={
            "recipes":         n_recipes,
            "nutrition_rows":  n_nutrition,
        })
    except Exception as e:
        checks["database"] = HealthCheck(status="fail", detail={"error": str(e)})

    # Ollama
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
        required  = {OLLAMA_VISION_MODEL, OLLAMA_TEXT_MODEL}
        missing   = [m for m in required if not any(m in a for a in available)]
        if missing:
            checks["ollama"] = HealthCheck(status="fail", detail={"missing_models": missing, "available": available})
        else:
            checks["ollama"] = HealthCheck(status="ok", detail={"available_models": available})
    except Exception as e:
        checks["ollama"] = HealthCheck(status="fail", detail={"error": str(e)})

    # Resumen global
    states = [c.status for c in checks.values()]
    if all(s == "ok" for s in states):
        global_status = "ok"
    elif any(s == "ok" for s in states):
        global_status = "degraded"
    else:
        global_status = "fail"

    return HealthResponse(status=global_status, checks=checks)
