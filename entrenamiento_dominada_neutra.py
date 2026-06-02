# Entrenamiento dominada agarre neutro: segmenta reps por flujo óptico y entrena el modelo de fase.
import os
import cv2
import numpy as np
import pandas as pd
from collections import deque
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

from core import config
from core.geometria import calcular_angulo
from core.pose import nueva_pose
from core.senales import flujo_vertical
from core.dinamica import vel_ang, acc_ang, fase_por_curva

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CARPETA_VIDEOS = config.VIDEOS_DOM_NEUTRA
BASE_DIR = config.DIR_DOM_NEUTRA
os.makedirs(BASE_DIR, exist_ok=True)

RUTA_MODELO = os.path.join(BASE_DIR, "modelo_fase_dominadas_rt.pkl")
RUTA_SCALER = os.path.join(BASE_DIR, "scaler_fase_dominadas_rt.pkl")
RUTA_DATASET = os.path.join(BASE_DIR, "dataset_fase_dominadas_rt.csv")


# --------------------------------------------------
# REPETICIONES (sobre la señal de flujo óptico)
# --------------------------------------------------
def detectar_repeticiones(mov):
    if len(mov) == 0:
        return []

    mov = pd.Series(mov).rolling(10, min_periods=1).mean().values
    mov = (mov - mov.min()) / (mov.max() - mov.min() + 1e-8)

    peaks, _ = find_peaks(mov, distance=30, prominence=0.15)
    valleys, _ = find_peaks(-mov, distance=30, prominence=0.15)

    reps = []
    for i in range(min(len(peaks), len(valleys))):
        reps.append((int(min(peaks[i], valleys[i])),
                     int(max(peaks[i], valleys[i]))))
    return reps


# --------------------------------------------------
# EXTRACCION DE FEATURES
# --------------------------------------------------
def procesar_video(video_path, reps, ventana=25):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ang_hist = deque(maxlen=ventana)
    vel_hist = deque(maxlen=5)

    X, y = [], []
    frame_id = 0

    with nueva_pose() as pose:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1

            if not any(ini <= frame_id <= fin for ini, fin in reps):
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            ang = calcular_angulo(
                [lm[12].x * w, lm[12].y * h],
                [lm[14].x * w, lm[14].y * h],
                [lm[16].x * w, lm[16].y * h]
            )

            ang_hist.append(ang)
            vel = vel_ang(ang_hist, fps)
            vel_hist.append(vel)
            acc = acc_ang(vel_hist, fps)

            fase = fase_por_curva(ang_hist)
            if fase == -1:
                continue

            arr = np.array(ang_hist)

            # Orden de features (referencia para inferencia/evaluación):
            # ang, vel, acc, ang_mean, ang_min, ang_max, vel_mean
            X.append([
                ang,
                vel,
                acc,
                arr.mean(),
                arr.min(),
                arr.max(),
                np.mean(vel_hist)
            ])
            y.append(fase)

    cap.release()
    return np.array(X), np.array(y)


# --------------------------------------------------
# SOBREMUESTREO
# --------------------------------------------------
def balancear_dataset(X, y):
    clases, counts = np.unique(y, return_counts=True)
    max_c = counts.max()

    Xb, yb = [], []
    for c in clases:
        Xc = X[y == c]
        yc = y[y == c]

        idx = np.random.choice(len(Xc), max_c, replace=True)
        Xb.append(Xc[idx])
        yb.append(yc[idx])

    return np.vstack(Xb), np.hstack(yb)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    X_total, y_total = [], []

    videos = [
        os.path.join(CARPETA_VIDEOS, v)
        for v in os.listdir(CARPETA_VIDEOS)
        if v.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    print("Videos encontrados:", len(videos))

    for vid in videos:
        print("\nProcesando:", os.path.basename(vid))

        mov = flujo_vertical(vid)
        reps = detectar_repeticiones(mov)

        print("  Repeticiones:", len(reps))
        if len(reps) == 0:
            continue

        X, y = procesar_video(vid, reps)
        if len(X) == 0:
            continue

        X_total.append(X)
        y_total.append(y)

    X_total = np.vstack(X_total)
    y_total = np.hstack(y_total)

    print("\nDistribucion original:", np.unique(y_total, return_counts=True))

    X_bal, y_bal = balancear_dataset(X_total, y_total)

    print("Distribucion balanceada:", np.unique(y_bal, return_counts=True))

    df = pd.DataFrame(X_bal, columns=[
        "ang", "vel", "acc",
        "ang_mean", "ang_min", "ang_max",
        "vel_mean"
    ])
    df["fase"] = y_bal
    df.to_csv(RUTA_DATASET, index=False)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_bal)

    modelo = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    modelo.fit(Xs, y_bal)

    joblib.dump(modelo, RUTA_MODELO)
    joblib.dump(scaler, RUTA_SCALER)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print("Modelo:", RUTA_MODELO)
    print("Scaler:", RUTA_SCALER)
    print("Dataset:", RUTA_DATASET)
