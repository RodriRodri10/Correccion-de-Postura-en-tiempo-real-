"""Wrapper de MediaPipe Pose (BlazePose) y constantes de landmarks.

Centraliza la creación del estimador de pose para que todos los scripts usen
la misma configuración (modelo Full por defecto, suavizado activado).
"""
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Índices de landmarks de MediaPipe Pose usados en el proyecto.
NARIZ = 0
HOMBRO_IZQ, HOMBRO_DER = 11, 12
CODO_IZQ, CODO_DER = 13, 14
MUNECA_IZQ, MUNECA_DER = 15, 16
CADERA_IZQ, CADERA_DER = 23, 24


def nueva_pose(deteccion=0.6, seguimiento=0.6):
    """Crea un estimador ``mp_pose.Pose`` con la configuración del proyecto.

    Al no fijar ``model_complexity`` se usa el valor por defecto 1 (Full).
    """
    return mp_pose.Pose(min_detection_confidence=deteccion,
                        min_tracking_confidence=seguimiento)


def punto(lm, idx, w, h):
    """Landmark ``idx`` desnormalizado a píxeles como ``np.array([x, y])``."""
    return np.array([lm[idx].x * w, lm[idx].y * h])
