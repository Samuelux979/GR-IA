"""
Tests del parseo de respuestas JSON de los modelos (_parse_json en
src/vision/llava_client.py). Debe tolerar texto adicional, bloques
Markdown y devolver None ante respuestas invalidas.
"""

from src.vision.llava_client import _parse_json


def test_plain_valid_json():
    assert _parse_json('{"ingredients": ["egg"]}') == {"ingredients": ["egg"]}


def test_json_inside_markdown_fence_with_label():
    raw = '```json\n{"score": 9}\n```'
    assert _parse_json(raw) == {"score": 9}


def test_json_inside_markdown_fence_without_label():
    raw = '```\n{"score": 5}\n```'
    assert _parse_json(raw) == {"score": 5}


def test_json_with_text_before_and_after():
    raw = 'Here is the result:\n{"main": ["chicken"]}\nHope it helps!'
    assert _parse_json(raw) == {"main": ["chicken"]}


def test_text_before_json_object():
    raw = 'Based on the image: {"ingredients": ["tomato", "onion"]}'
    assert _parse_json(raw) == {"ingredients": ["tomato", "onion"]}


def test_invalid_json_returns_none():
    assert _parse_json("esto no es json") is None


def test_empty_string_returns_none():
    assert _parse_json("") is None


def test_broken_json_returns_none():
    # llaves presentes pero contenido malformado
    assert _parse_json('{"a": }') is None


def test_nested_json_object():
    raw = '{"main": ["a"], "meta": {"k": 1}}'
    assert _parse_json(raw) == {"main": ["a"], "meta": {"k": 1}}
