"""Procesamiento de señales: flujo óptico y suavizado."""
import numpy as np
import cv2


def flujo_vertical(video_path):
    """Serie temporal del movimiento vertical medio del video.

    Usa flujo óptico denso de Farneback y promedia la componente Y del campo
    de velocidades en cada par de frames consecutivos. Sirve para segmentar
    repeticiones en los ejercicios de dominada.
    """
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return np.array([])

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mov = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mov.append(np.mean(flow[..., 1]))
        prev_gray = gray

    cap.release()
    return np.array(mov)


def suavizar_kalman(serie):
    """Suaviza una serie 1D con un filtro de Kalman. Devuelve la serie
    original si tiene menos de dos puntos o si el filtro falla."""
    from pykalman import KalmanFilter

    serie = np.array(serie, dtype=float)
    if len(serie) < 2:
        return serie
    try:
        kf = KalmanFilter(initial_state_mean=serie[0], n_dim_obs=1)
        estado, _ = kf.smooth(serie)
        return estado.ravel()
    except Exception:
        return serie
