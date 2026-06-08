"""
Tests de detect_ingredients() (src/vision/llava_client.py) con mocks.

Se simulan las respuestas de los modelos de vision y clasificacion para
verificar el comportamiento sin acceder a disco, modelos ni Ollama.
"""

from unittest.mock import patch, MagicMock

from src.vision import llava_client


def _fake_llm(content: str) -> MagicMock:
    """Crea un LLM simulado cuyo invoke() devuelve el contenido dado."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


def _run(vision_response: str, classify_response: str | None = None) -> dict:
    """Ejecuta detect_ingredients con respuestas simuladas de los modelos."""
    llms = [_fake_llm(vision_response)]
    if classify_response is not None:
        llms.append(_fake_llm(classify_response))

    with patch.object(llava_client, "encode_image", return_value="ZmFrZQ=="), \
         patch.object(llava_client, "_make_llm", side_effect=llms):
        return llava_client.detect_ingredients("foto.jpg")


def test_valid_vision_and_classification():
    result = _run(
        '{"ingredients": ["Chicken", "Onion"]}',
        '{"main": ["chicken"], "secondary": ["onion"], "accompaniment": [], "spices": []}',
    )
    assert result["main"] == ["chicken"]
    assert result["secondary"] == ["onion"]


def test_classification_failure_falls_back_to_secondary():
    # La clasificacion devuelve algo no parseable -> todo a 'secondary'
    result = _run(
        '{"ingredients": ["Chicken", "Onion"]}',
        "no es json valido",
    )
    assert result["main"] == []
    assert result["secondary"] == ["chicken", "onion"]
    assert result["accompaniment"] == []
    assert result["spices"] == []


def test_vision_returns_no_ingredients_gives_empty_structure():
    result = _run('{"ingredients": []}')
    assert result == {"main": [], "secondary": [], "accompaniment": [], "spices": []}


def test_vision_invalid_json_gives_empty_structure():
    result = _run("respuesta corrupta sin json")
    assert result == {"main": [], "secondary": [], "accompaniment": [], "spices": []}


def test_ingredients_are_normalized_lowercase_trimmed():
    # Con clasificacion fallida, los ingredientes detectados van a secondary
    # ya normalizados (minusculas, sin espacios sobrantes)
    result = _run(
        '{"ingredients": ["  CHICKEN  ", "Onion"]}',
        "fallo",
    )
    assert result["secondary"] == ["chicken", "onion"]
