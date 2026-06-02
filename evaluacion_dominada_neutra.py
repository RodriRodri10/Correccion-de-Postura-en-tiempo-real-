# Evaluación dominada agarre neutro: compara el modelo ML contra el ground truth por curva.
import os
import cv2
import numpy as np
import joblib
import pandas as pd
from collections import deque
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

from core import config
from core.geometria import calcular_angulo
from core.pose import nueva_pose
from core.dinamica import vel_ang, acc_ang, mean_safe, min_safe, max_safe, fase_por_curva

# --------------------------------------------------
# 1. RUTAS
# --------------------------------------------------
_MODELOS_DIR = config.DIR_DOM_NEUTRA
VIDEO = os.path.join(config.VIDEOS_DOM_NEUTRA, "prueba.mp4")
MODELO = os.path.join(_MODELOS_DIR, "modelo_fase_dominadas_rt.pkl")
SCALER = os.path.join(_MODELOS_DIR, "scaler_fase_dominadas_rt.pkl")
CSV_SALIDA = os.path.join(_MODELOS_DIR, "evaluacion_curva_vs_ml.csv")

clf = joblib.load(MODELO)
scaler = joblib.load(SCALER)

# --------------------------------------------------
# 2. PROCESAMIENTO FRAME A FRAME
# --------------------------------------------------
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ang_hist = deque(maxlen=20)
vel_hist = deque(maxlen=10)

datos = []
frame_id = 0

with nueva_pose() as pose:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        ang = vel = acc = 0.0
        fase_gt = -1
        fase_ml = -1

        if res.pose_landmarks:
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

            # ---------- GT POR CURVA ----------
            fase_gt = fase_por_curva(ang_hist)

            # ---------- PREDICCION ML ----------
            # Orden idéntico al entrenamiento:
            # ang, vel, acc, ang_mean, ang_min, ang_max, vel_mean
            if fase_gt != -1:
                X = np.array([[
                    ang,
                    vel,
                    acc,
                    mean_safe(ang_hist),
                    min_safe(ang_hist),
                    max_safe(ang_hist),
                    mean_safe(vel_hist)
                ]])
                Xs = scaler.transform(X)
                fase_ml = int(clf.predict(Xs)[0])

        datos.append([frame_id, ang, vel, acc, fase_gt, fase_ml])
        frame_id += 1

cap.release()

# --------------------------------------------------
# 3. MÉTRICAS Y CSV
# --------------------------------------------------
df = pd.DataFrame(datos, columns=["frame", "angulo", "vel", "acc", "fase_gt", "fase_ml"])
df = df[df["fase_gt"] != -1]
df.to_csv(CSV_SALIDA, index=False)

print("\nFrames evaluados:", len(df))
print("Coincidencia total (%):", np.mean(df["fase_gt"] == df["fase_ml"]) * 100)

cm = confusion_matrix(df["fase_gt"], df["fase_ml"], labels=[1, 2, 3])
print("\nMATRIZ DE CONFUSIÓN (GT Curva vs ML)")
print("Filas = Curva | Columnas = ML")
print(pd.DataFrame(cm, index=["Abajo", "Movimiento", "Arriba"], columns=["Abajo", "Movimiento", "Arriba"]))

print("\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(df["fase_gt"], df["fase_ml"], target_names=["Abajo", "Movimiento", "Arriba"]))

print("\nCSV generado en:")
print(CSV_SALIDA)
