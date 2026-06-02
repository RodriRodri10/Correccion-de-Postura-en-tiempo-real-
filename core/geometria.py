"""Cálculos geométricos sobre landmarks (ángulos y distancias)."""
import numpy as np


def calcular_angulo(a, b, c):
    """Ángulo en grados [0, 180] en el vértice ``b`` entre los segmentos
    ``b->a`` y ``b->c``. Devuelve 0.0 si algún segmento es nulo."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den == 0:
        return 0.0
    cos = np.dot(ba, bc) / den
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# Alias histórico usado por los scripts de dominada abierta.
angulo = calcular_angulo


def distancia(p, q):
    """Distancia euclidiana entre dos puntos."""
    return float(np.linalg.norm(np.array(p, dtype=float) - np.array(q, dtype=float)))
