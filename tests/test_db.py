"""Tests para core/db.py — cliente HTTP a PostgREST."""
import pytest
from unittest.mock import patch, MagicMock
from core import db


def test_enviar_sesion_exito(monkeypatch):
    """enviar_sesion devuelve la respuesta cuando POST es exitoso."""
    mock_post = MagicMock()
    mock_post.return_value.json.return_value = {"id": 42}
    monkeypatch.setattr("core.db.requests.post", mock_post)

    resumen = {"reps": 5, "pct_correcto": 80.0}
    result = db.enviar_sesion(resumen, "pushup", usuario="alice")

    assert result == {"id": 42}
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "http://localhost:3000/rpc/crear_sesion"
    assert call_args[1]["json"]["p_usuario"] == "alice"
    assert call_args[1]["json"]["p_ejercicio"] == "pushup"
    assert call_args[1]["json"]["p_resumen"] == resumen


def test_enviar_sesion_usuario_defecto(monkeypatch):
    """enviar_sesion usa usuario 'demo' por defecto."""
    mock_post = MagicMock()
    mock_post.return_value.json.return_value = {}
    monkeypatch.setattr("core.db.requests.post", mock_post)

    db.enviar_sesion({"reps": 1}, "dom_abierta")

    call_args = mock_post.call_args
    assert call_args[1]["json"]["p_usuario"] == "demo"


def test_enviar_sesion_exception_devuelve_none(monkeypatch):
    """enviar_sesion devuelve None cuando requests.post lanza RequestException."""
    import requests
    mock_post = MagicMock(side_effect=requests.RequestException("Network error"))
    monkeypatch.setattr("core.db.requests.post", mock_post)

    result = db.enviar_sesion({"reps": 1}, "pushup")

    assert result is None


def test_enviar_sesion_timeout_devuelve_none(monkeypatch):
    """enviar_sesion devuelve None en caso de timeout."""
    import requests
    mock_post = MagicMock(side_effect=requests.Timeout("timeout"))
    monkeypatch.setattr("core.db.requests.post", mock_post)

    result = db.enviar_sesion({"reps": 1}, "pushup")

    assert result is None


def test_enviar_sesion_payload_estructura(monkeypatch):
    """enviar_sesion arma el payload con las claves exactas esperadas."""
    mock_post = MagicMock()
    mock_post.return_value.json.return_value = {}
    monkeypatch.setattr("core.db.requests.post", mock_post)

    resumen_completo = {
        "reps": 3,
        "frames_evaluados": 100,
        "frames_correctos": 80,
        "pct_correcto": 80.0,
        "frames_con_error": 20,
        "pct_frames_con_error": 20.0,
        "duracion_seg": 10.5,
        "top_errores": [["Error1", 5]],
        "errores_principales": []
    }

    db.enviar_sesion(resumen_completo, "dom_abierta", usuario="bob")

    payload = mock_post.call_args[1]["json"]
    assert set(payload.keys()) == {"p_usuario", "p_ejercicio", "p_resumen"}
    assert payload["p_usuario"] == "bob"
    assert payload["p_ejercicio"] == "dom_abierta"
    assert payload["p_resumen"] == resumen_completo
