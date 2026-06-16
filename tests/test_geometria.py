"""Ancla de regresion para core/geometria.py (debe pasar tal cual)."""
import pytest

from core.geometria import calcular_angulo, angulo, distancia


def test_angulo_recto():
    assert calcular_angulo([0, 1], [0, 0], [1, 0]) == pytest.approx(90.0, abs=1e-6)


def test_angulo_llano():
    assert calcular_angulo([1, 0], [0, 0], [-1, 0]) == pytest.approx(180.0, abs=1e-6)


def test_angulo_alias():
    # ``angulo`` es alias de ``calcular_angulo``.
    assert angulo([0, 1], [0, 0], [1, 0]) == pytest.approx(90.0, abs=1e-6)


def test_distancia():
    assert distancia([0, 0], [3, 4]) == pytest.approx(5.0, abs=1e-6)
