# Evaluación dominada agarre abierto: compara el modelo ML contra un ground truth biomecánico.
import os
import cv2
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

from core import config
from core.geometria import angulo
from core.pose import nueva_pose
from core.features import features_frame, nuevo_historial

# --------------------------------------------------
# RUTAS
# --------------------------------------------------
_MODELOS_DIR = config.DIR_DOM_ABIERTA
VIDEO = os.path.join(config.VIDEOS_DOM_ABIERTA, "prueba.mp4")

MODELO = os.path.join(_MODELOS_DIR, "modelo_fases.pkl")
SCALER = os.path.join(_MODELOS_DIR, "scaler_fases.pkl")

CSV_SALIDA = os.path.join(_MODELOS_DIR, "evaluacion_gt_vs_ml.csv")

clf = joblib.load(MODELO)
scaler = joblib.load(SCALER)


# --------------------------------------------------
# GROUND TRUTH BIOMECANICO
# --------------------------------------------------
def fase_biomecanica(ang, vel, ang_min, ang_max,
                     vel_umbral=15,
                     ang_low=0.25,
                     ang_high=0.75):

    ang_norm = (ang - ang_min) / (ang_max - ang_min + 1e-6)

    if abs(vel) > vel_umbral:
        return 2  # Movimiento

    if ang_norm >= ang_high:
        return 1  # Arriba

    if ang_norm <= ang_low:
        return 3  # Abajo

    return 2


# --------------------------------------------------
# PREPASE PARA NORMALIZAR ANGULO
# --------------------------------------------------
cap_tmp = cv2.VideoCapture(VIDEO)
angulos = []

with nueva_pose() as pose:

    while cap_tmp.isOpened():
        ok, frame = cap_tmp.read()
        if not ok:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            ang = angulo(
                [lm[11].x * w, lm[11].y * h],
                [lm[13].x * w, lm[13].y * h],
                [lm[15].x * w, lm[15].y * h]
            )
            angulos.append(ang)

cap_tmp.release()

ang_min = np.min(angulos)
ang_max = np.max(angulos)

# --------------------------------------------------
# PROCESAMIENTO FRAME A FRAME
# --------------------------------------------------
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

hist = nuevo_historial(25)

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

        ang = vel = 0.0
        fase_gt = -1
        fase_ml = -1

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            ang = angulo(
                [lm[11].x * w, lm[11].y * h],
                [lm[13].x * w, lm[13].y * h],
                [lm[15].x * w, lm[15].y * h]
            )

            vel = (ang - hist["ang_l"][-1]) * fps if len(hist["ang_l"]) else 0.0

            fase_gt = fase_biomecanica(ang, vel, ang_min, ang_max)

            feat = features_frame(lm, h, w, fps, hist)
            Xs = scaler.transform(feat.reshape(1, -1))
            fase_ml = int(clf.predict(Xs)[0])

        datos.append([frame_id, ang, vel, fase_gt, fase_ml])
        frame_id += 1

cap.release()

# --------------------------------------------------
# METRICAS Y CSV
# --------------------------------------------------
df = pd.DataFrame(
    datos,
    columns=["frame", "angulo", "vel", "fase_gt", "fase_ml"]
)

df = df[df["fase_gt"] != -1]
df.to_csv(CSV_SALIDA, index=False)

print("\nFrames evaluados:", len(df))
print("Exactitud global (%):",
      np.mean(df["fase_gt"] == df["fase_ml"]) * 100)

cm = confusion_matrix(df["fase_gt"], df["fase_ml"], labels=[1, 2, 3])

print("\nMATRIZ DE CONFUSION (GT vs ML)")
print(pd.DataFrame(
    cm,
    index=["Arriba", "Movimiento", "Abajo"],
    columns=["Arriba", "Movimiento", "Abajo"]
))

print("\nREPORTE DE CLASIFICACION")
print(classification_report(
    df["fase_gt"],
    df["fase_ml"],
    target_names=["Arriba", "Movimiento", "Abajo"]
))

print("\nCSV generado en:")
print(CSV_SALIDA)
