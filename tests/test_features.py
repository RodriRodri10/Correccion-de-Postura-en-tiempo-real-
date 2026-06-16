"""Ancla del contrato de features de dominada abierta (invariante de CLAUDE.md).

Bloquea el numero de features (41) para que ningun cambio rompa el orden/longitud
que comparten entrenamiento, evaluacion e inferencia.
"""
import math

from core.features import features_frame, nuevo_historial


def test_features_longitud_41(fake_landmarks):
    hist = nuevo_historial(25)
    feat = features_frame(fake_landmarks, 480, 640, 30, hist)
    assert len(feat) == 41


def test_features_finitas_con_un_frame(fake_landmarks):
    # Con historial de un solo frame (velocidades/aceleraciones = 0) no debe
    # producir NaN/inf: la ruta degenerada no rompe.
    hist = nuevo_historial(25)
    feat = features_frame(fake_landmarks, 480, 640, 30, hist)
    assert all(math.isfinite(float(v)) for v in feat)


def test_features_nombres(fake_landmarks):
    hist = nuevo_historial(25)
    feat, nombres = features_frame(fake_landmarks, 480, 640, 30, hist, con_nombres=True)
    assert len(nombres) == len(feat) == 41
