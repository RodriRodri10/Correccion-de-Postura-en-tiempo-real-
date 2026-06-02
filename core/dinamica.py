"""Helpers de dinámica temporal y etiquetado por curva (dominada neutra).

Comunes a los scripts de entrenamiento, evaluación y retroalimentación de la
dominada con agarre neutro.
"""
import numpy as np


def vel_ang(hist, fps):
    """Diferencia finita del último par de valores por fps (°/s)."""
    return (hist[-1] - hist[-2]) * fps if len(hist) >= 2 else 0.0


# La "aceleración" usa exactamente la misma diferencia finita, pero aplicada
# al historial de velocidades; por eso es un alias de vel_ang.
acc_ang = vel_ang


def mean_safe(x):
    return float(np.mean(x)) if len(x) else 0.0


def min_safe(x):
    return float(np.min(x)) if len(x) else 0.0


def max_safe(x):
    return float(np.max(x)) if len(x) else 0.0


def fase_por_curva(hist):
    """Etiqueta de fase a partir de la posición del ángulo dentro de la
    amplitud observada en la ventana: 1=abajo, 3=arriba, 2=movimiento.
    Devuelve -1 mientras la ventana no esté llena."""
    if len(hist) < hist.maxlen:
        return -1
    ang = np.array(hist)
    mn, mx = ang.min(), ang.max()
    amp = mx - mn + 1e-6
    a = ang[-1]
    if a <= mn + 0.2 * amp:
        return 1  # abajo
    if a >= mx - 0.2 * amp:
        return 3  # arriba
    return 2  # movimiento
