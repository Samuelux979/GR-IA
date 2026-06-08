"""
Modelos Pydantic que definen el contrato JSON publico de la API.
Cualquier cambio aqui es un cambio de API y debe documentarse.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Ingredients(BaseModel):
    """Ingredientes detectados y clasificados por nivel de prioridad."""
    main:          list[str] = Field(default_factory=list)
    secondary:     list[str] = Field(default_factory=list)
    accompaniment: list[str] = Field(default_factory=list)
    spices:        list[str] = Field(default_factory=list)


class Macros(BaseModel):
    """
    Valores nutricionales de una receta.
    Los campos son opcionales: None significa que el dato es desconocido.
    """
    calories:  float | None = None
    protein_g: float | None = None
    fat_g:     float | None = None
    carbs_g:   float | None = None
    fiber_g:   float | None = None


class Recipe(BaseModel):
    """Receta recuperada de la base de datos."""
    id:           int
    title:        str
    category:     str | None = None
    source:       Literal["foodcom", "ai_generated"] = "foodcom"
    macros_known: bool = True
    ingredients:  list[str]
    steps:        list[str]
    matches:      list[str]
    score:        int
    grade:        float                                    # nota sobre 10
    match_level:  Literal["alta", "media", "baja"] = "baja"  # color de correlacion
    distance:     float
    macros:       Macros


class AIRecipe(BaseModel):
    """Receta original generada por el modelo de lenguaje."""
    title:       str
    ingredients: list[str]
    steps:       list[str]
    ner:         list[str]
    macros:      Macros


class PredictResponse(BaseModel):
    """Respuesta del endpoint /predict."""
    ingredients: Ingredients
    recipes:     list[Recipe]
    ai_recipe:   AIRecipe | None = None
    warnings:    list[str] = Field(default_factory=list)


class SaveRequest(BaseModel):
    """Cuerpo del POST /save para persistir una receta generada."""
    title:             str
    ingredients:       list[str]
    steps:             list[str]
    ner:               list[str]
    macros:            Macros = Field(default_factory=Macros)
    model:             str | None = None
    embedding_model:   str | None = None
    input_ingredients: dict | None = None
    prompt_version:    str | None = None


class SaveResponse(BaseModel):
    saved:     bool
    recipe_id: int
    duplicate: bool = False  # True si la receta ya existia y se devolvio el id


class HealthCheck(BaseModel):
    status: Literal["ok", "fail"]
    detail: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "fail"]
    checks: dict[str, HealthCheck]
