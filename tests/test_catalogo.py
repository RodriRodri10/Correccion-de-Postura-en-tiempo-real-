"""Contrato del catalogo de ejercicios (objetivo de Ralph: rojo -> verde).

``disponible`` depende de archivos versionados en el repo: los modelos de
pushup y dom_abierta existen; el de dom_neutra no (pendiente P2).
"""
from core import catalogo

CLAVES_MVP = ("pushup", "dom_abierta")


def test_registro_contiene_ejercicios_mvp():
    for clave in CLAVES_MVP:
        assert clave in catalogo.EJERCICIOS
        entrada = catalogo.EJERCICIOS[clave]
        for campo in ("nombre", "script", "modelo", "vista"):
            assert campo in entrada


def test_disponibles_incluye_mvp():
    disp = catalogo.disponibles()
    assert "pushup" in disp
    assert "dom_abierta" in disp


def test_neutra_no_disponible():
    # Si dom_neutra esta en el registro, no debe estar disponible (falta modelo).
    if "dom_neutra" in catalogo.EJERCICIOS:
        assert catalogo.disponible("dom_neutra") is False
        assert "dom_neutra" not in catalogo.disponibles()
