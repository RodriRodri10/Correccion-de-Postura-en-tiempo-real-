"""Extracción de las 41 features del ejercicio dominada con agarre abierto.

Esta lógica era idéntica en los scripts de entrenamiento, evaluación y
retroalimentación; aquí queda en un único lugar para garantizar que el vector
de inferencia coincida exactamente con el de entrenamiento.
"""
import numpy as np
from collections import deque

from .geometria import angulo, distancia

# Señales que se acumulan en el historial deslizante.
CLAVES_HIST = ["ang_l", "ang_r", "ang_h", "grip", "trunk", "hip_y"]
# Señales sobre las que se calculan estadísticas posicionales (5 c/u = 25).
CLAVES_STATS = ["ang_l", "ang_r", "ang_h", "grip", "trunk"]
# Señales sobre las que se calculan derivadas temporales (4 c/u = 16).
CLAVES_DERIV = ["ang_l", "ang_r", "ang_h", "hip_y"]


def claves_historial():
    """Claves completas del diccionario de historial (incluye _vel y _acc)."""
    return CLAVES_HIST + [k + s for k in CLAVES_DERIV for s in ("_vel", "_acc")]


def nuevo_historial(ventana=25):
    """Crea el diccionario de deques para el historial deslizante."""
    return {k: deque(maxlen=ventana) for k in claves_historial()}


def features_frame(lm, h, w, fps, hist, con_nombres=False):
    """Calcula las 41 features de un frame y actualiza ``hist`` en sitio.

    Devuelve ``np.array`` de 41 valores, o ``(array, nombres)`` si
    ``con_nombres`` es True (usado por el entrenamiento para volcar el CSV).
    """
    wri_l = np.array([lm[15].x * w, lm[15].y * h])
    wri_r = np.array([lm[16].x * w, lm[16].y * h])
    elb_l = np.array([lm[13].x * w, lm[13].y * h])
    elb_r = np.array([lm[14].x * w, lm[14].y * h])
    sho_l = np.array([lm[11].x * w, lm[11].y * h])
    sho_r = np.array([lm[12].x * w, lm[12].y * h])
    hip_l = np.array([lm[23].x * w, lm[23].y * h])
    hip_r = np.array([lm[24].x * w, lm[24].y * h])

    mid_hip = (hip_l + hip_r) / 2
    mid_sho = (sho_l + sho_r) / 2

    ang_l = angulo(sho_l, elb_l, wri_l)
    ang_r = angulo(sho_r, elb_r, wri_r)
    ang_h = angulo(elb_l, sho_l, elb_r)
    trunk = angulo(mid_hip, mid_sho, [mid_sho[0], 0])
    grip = distancia(wri_l, wri_r)

    for k, v in zip(CLAVES_HIST, [ang_l, ang_r, ang_h, grip, trunk, mid_hip[1]]):
        hist[k].append(v)

    feats, names = [], []

    for k in CLAVES_STATS:
        arr = np.array(hist[k])
        feats += [arr[-1], arr.mean(), arr.min(), arr.max(), np.std(arr)]
        names += [f"{k}_last", f"{k}_mean", f"{k}_min", f"{k}_max", f"{k}_std"]

    for k in CLAVES_DERIV:
        vel = (hist[k][-1] - hist[k][-2]) * fps if len(hist[k]) >= 2 else 0.0
        hist[k + "_vel"].append(vel)

        acc = (hist[k + "_vel"][-1] - hist[k + "_vel"][-2]) * fps if len(hist[k + "_vel"]) >= 2 else 0.0
        hist[k + "_acc"].append(acc)

        feats += [vel, np.mean(hist[k + "_vel"]), acc, np.mean(hist[k + "_acc"])]
        names += [f"{k}_vl", f"{k}_vm", f"{k}_al", f"{k}_am"]

    if con_nombres:
        return np.array(feats), names
    return np.array(feats)
