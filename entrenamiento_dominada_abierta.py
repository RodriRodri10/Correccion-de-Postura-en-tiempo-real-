# Entrenamiento dominada agarre abierto: segmenta reps por flujo óptico y entrena el modelo de 41 features.
import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from core import config
from core.geometria import angulo
from core.pose import nueva_pose
from core.senales import flujo_vertical
from core.features import features_frame, nuevo_historial

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CARPETA_VIDEOS = config.VIDEOS_DOM_ABIERTA
BASE_DIR = config.DIR_DOM_ABIERTA

os.makedirs(BASE_DIR, exist_ok=True)

RUTA_MODELO = os.path.join(BASE_DIR, "modelo_fases.pkl")
RUTA_SCALER = os.path.join(BASE_DIR, "scaler_fases.pkl")
RUTA_DATASET = os.path.join(BASE_DIR, "dataset_fases.csv")
RUTA_COLUMNAS = os.path.join(BASE_DIR, "columnas_fases.json")

FOTOS_DIR = os.path.join(BASE_DIR, "fotos_fases")
os.makedirs(FOTOS_DIR, exist_ok=True)


# --------------------------------------------------
# DETECTAR INICIOS DE REPETICION
# --------------------------------------------------
def detecta_inicios(mov):
    if len(mov) == 0:
        return []

    s = pd.Series(mov).rolling(10, min_periods=1).mean().values
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)

    inicios, _ = find_peaks(s, distance=40, prominence=0.2)
    return inicios.tolist()


# --------------------------------------------------
# ASIGNAR FASES BIOMECANICAS
# --------------------------------------------------
def asigna_fases(frames, angulos):
    angulos = np.array(angulos)
    ang_max = np.max(angulos)

    frame2fase = {}
    for fr, ang in zip(frames, angulos):
        if ang >= 0.75 * ang_max:
            frame2fase[fr] = 1
        elif ang >= 0.45 * ang_max:
            frame2fase[fr] = 2
        else:
            frame2fase[fr] = 3

    return frame2fase


# --------------------------------------------------
# PROCESAR VIDEO COMPLETO
# --------------------------------------------------
def procesa_video(video_path, repeticiones, ventana=25):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    hist = nuevo_historial(ventana)

    frame2fase = {}

    with nueva_pose() as pose:
        for ini, fin in repeticiones:
            frames, angs = [], []

            for fr in range(ini, fin + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, frame = cap.read()
                if not ok:
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if not res.pose_landmarks:
                    continue

                lm = res.pose_landmarks.landmark
                ang = angulo(
                    [lm[11].x * w, lm[11].y * h],
                    [lm[13].x * w, lm[13].y * h],
                    [lm[15].x * w, lm[15].y * h]
                )

                frames.append(fr)
                angs.append(ang)

            if len(frames) >= 10:
                frame2fase.update(asigna_fases(frames, angs))

    X, y = [], []
    col_names = None
    frame_id = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with nueva_pose() as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            if frame_id not in frame2fase:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            feat, col_names = features_frame(
                res.pose_landmarks.landmark, h, w, fps, hist, con_nombres=True
            )

            X.append(feat)
            y.append(frame2fase[frame_id])

    cap.release()
    return np.array(X), np.array(y), col_names


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    X_all, y_all = [], []
    col_names = None

    videos = [
        os.path.join(CARPETA_VIDEOS, v)
        for v in os.listdir(CARPETA_VIDEOS)
        if v.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    print("Videos encontrados:", len(videos))

    for vid in videos:
        print("\nProcesando:", os.path.basename(vid))
        mov = flujo_vertical(vid)
        inicios = detecta_inicios(mov)

        if len(inicios) < 2:
            print("  No hay repeticiones suficientes")
            continue

        repeticiones = []
        for i in range(len(inicios) - 1):
            repeticiones.append((inicios[i], inicios[i + 1] - 1))
        repeticiones.append((inicios[-1], len(mov) - 1))

        X, y, col_names = procesa_video(vid, repeticiones)
        if len(X) == 0:
            continue

        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    print("Distribucion:", np.unique(y_all, return_counts=True))

    df = pd.DataFrame(X_all, columns=col_names)
    df["fase"] = y_all
    df.to_csv(RUTA_DATASET, index=False)

    with open(RUTA_COLUMNAS, "w") as f:
        json.dump(col_names, f)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)

    modelo = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    modelo.fit(Xs, y_all)

    joblib.dump(modelo, RUTA_MODELO)
    joblib.dump(scaler, RUTA_SCALER)

    print("\n=== ENTRENAMIENTO FINALIZADO ===")
    print("Modelo :", RUTA_MODELO)
    print("Scaler :", RUTA_SCALER)
    print("Dataset:", RUTA_DATASET)
