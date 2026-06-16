# Retroalimentación en tiempo real para dominada con agarre abierto.
import os
import time
import cv2
import numpy as np
import joblib

from core import config, sesion
from core.geometria import angulo
from core.pose import nueva_pose
from core.features import features_frame, nuevo_historial
from core.reps import FsmDominadaAbierta

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
    log = []
    t0 = time.time()
    fsm = FsmDominadaAbierta()

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
                fsm.update(fase)

            # ---- LOG DE SESION ----
            if fase == -1:
                sesion.acumular(log, fase, None, [])
            else:
                correcto = feedback.startswith("Buena")
                errores = [feedback] if not correcto else []
                sesion.acumular(log, fase, correcto, errores)

            # ---------- DRAW ----------
            cv2.putText(frame, f"REPETICIONES: {fsm.reps}",
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
    dur = time.time() - t0
    r = sesion.resumen(log, fsm.reps, dur)
    sesion.guardar(r, "dom_abierta", os.path.join(config.RAIZ, "sesiones"))


# --------------------------------------------------
if __name__ == "__main__":
    main()
