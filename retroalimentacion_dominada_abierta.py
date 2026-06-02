# Retroalimentación en tiempo real para dominada con agarre abierto.
import os
import cv2
import numpy as np
import joblib

from core import config
from core.geometria import angulo
from core.pose import nueva_pose
from core.features import features_frame, nuevo_historial

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
_MODELOS_DIR = config.DIR_DOM_ABIERTA
RUTA_MODELO = os.path.join(_MODELOS_DIR, "modelo_fases.pkl")
RUTA_SCALER = os.path.join(_MODELOS_DIR, "scaler_fases.pkl")

modelo = joblib.load(RUTA_MODELO)
scaler = joblib.load(RUTA_SCALER)


# --------------------------------------------------
# Presentación / feedback
# --------------------------------------------------
def fase_txt(f):
    return {1: "Arriba", 2: "Transicion", 3: "Abajo"}.get(f, "-")


def color_fase(f):
    return {1: (0, 0, 255), 2: (0, 165, 255), 3: (0, 255, 0)}.get(f, (200, 200, 200))


def feedback_fase(fase, ang):
    if fase == 3:
        return "Extiende mas los brazos" if ang < 150 else "Buena extension"
    if fase == 2:
        return "Movimiento controlado"
    if fase == 1:
        return "Sube mas la barra" if ang > 70 else "Buena contraccion"
    return ""


# --------------------------------------------------
# MAIN TIEMPO REAL  y  MAQUINA DE ESTADOS
# --------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    hist = nuevo_historial(25)

    # ---------- MAQUINA DE ESTADOS ----------
    ABAJO, SUBE, ARRIBA, BAJA = 0, 1, 2, 3
    estado = ABAJO

    repeticiones = 0
    cont_arriba = 0
    cont_abajo = 0
    FRAMES_ESTABLES = 5

    # ---------- POSICION TEXTO ----------
    X_IZQ = 30
    Y_BASE = int(h * 0.33)
    LINEA = 35

    with nueva_pose() as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            fase = -1
            ang = 0.0
            vel_ang = 0.0
            feedback = ""

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                ang = angulo(
                    [lm[11].x * w, lm[11].y * h],
                    [lm[13].x * w, lm[13].y * h],
                    [lm[15].x * w, lm[15].y * h]
                )

                feat = features_frame(lm, h, w, fps, hist)
                Xs = scaler.transform([feat])
                fase = int(modelo.predict(Xs)[0])

                feedback = feedback_fase(fase, ang)

                if len(hist["ang_l_vel"]) > 0:
                    vel_ang = hist["ang_l_vel"][-1]

                # ---------- TRANSICIONES ----------
                if estado == ABAJO and fase == 2:
                    estado = SUBE

                elif estado == SUBE and fase == 1:
                    cont_arriba += 1
                    if cont_arriba >= FRAMES_ESTABLES:
                        estado = ARRIBA
                        cont_arriba = 0

                elif estado == ARRIBA and fase == 2:
                    estado = BAJA

                elif estado == BAJA and fase == 3:
                    cont_abajo += 1
                    if cont_abajo >= FRAMES_ESTABLES:
                        repeticiones += 1
                        estado = ABAJO
                        cont_abajo = 0

            # ---------- DRAW ----------
            cv2.putText(frame, f"REPETICIONES: {repeticiones}",
                        (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)

            cv2.putText(frame, "Fase:",
                        (X_IZQ, Y_BASE),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.putText(frame, fase_txt(fase),
                        (X_IZQ, Y_BASE + LINEA),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_fase(fase), 2)

            cv2.putText(frame, f"Angulo: {ang:.1f}",
                        (X_IZQ, Y_BASE + 2 * LINEA),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Velocidad: {vel_ang:.1f}",
                        (X_IZQ, Y_BASE + 3 * LINEA),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if feedback:
                cv2.putText(frame, feedback,
                            (X_IZQ, Y_BASE + 5 * LINEA),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Dominadas tiempo real", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------
if __name__ == "__main__":
    main()
