# PRD — MVP "Corrector de Postura en Tiempo Real"

## Objetivo

Convertir el proyecto (hoy un conjunto de scripts) en un producto que un usuario
ejecute en pocos pasos: abre una app, **elige un ejercicio** de una lista, lo
realiza frente a la webcam con retroalimentacion en vivo, y al terminar recibe un
**resumen** de la sesion.

## Usuario y job-to-be-done

Estudiante/usuario que quiere corregir su tecnica. "Quiero elegir mi ejercicio,
hacerlo viendo correcciones en tiempo real, y al final saber cuantas reps hice y
que tan bien las hice."

## Alcance del MVP

Solo los dos ejercicios con modelo entrenado y versionado:
- **Wall push-up** (vista lateral) — `modelos/wall_pushup/`.
- **Dominada agarre abierto** (vista posterior) — `modelos/dominada_abierta/`.

Dominada neutra queda **fuera** (falta `modelo_fase_dominadas_rt.pkl` y videos
para reentrenar). El catalogo puede listarla como "no disponible".

## Requisitos funcionales

- **FR1 — Punto de entrada unico.** `streamlit run app.py` arranca todo. Desde
  clonar el repo a ver la UI: <= 3 pasos (crear venv, instalar, correr).
- **FR2 — Seleccion de ejercicio.** La UI lista los ejercicios *disponibles*
  (catalogo). Un ejercicio sin modelo aparece deshabilitado/no seleccionable, no
  rompe la app.
- **FR3 — Iniciar sesion en vivo.** Al elegir un ejercicio y pulsar "Iniciar", se
  lanza la ventana de retroalimentacion en vivo (OpenCV, el pipeline actual) del
  ejercicio correcto, vía subprocess.
- **FR4 — Overlay en vivo (se conserva).** Fase, contador de reps, angulos y
  mensajes de tecnica sobre el video, como hoy.
- **FR5 — Resumen final.** Al cerrar la ventana (Esc), el script escribe
  `sesiones/<ejercicio>_<timestamp>.json` con el resumen, y la app muestra una
  tarjeta con: reps totales, % de postura correcta, top errores y duracion.
- **FR6 — Robustez.** Sin camara o sin modelo: mensaje claro, nunca un stacktrace.

## Definicion exacta de las metricas del resumen

Calculadas en `core/sesion.py` a partir del log por frame (lista de registros
`{"fase", "correcto", "errores"}`):
- **reps**: lo cuenta el FSM (`core/reps.py`), se pasa a `resumen()`.
- **frames_evaluados**: frames con `correcto is not None` (fase != -1).
- **frames_correctos**: frames con `correcto is True`.
- **pct_correcto**: `100 * frames_correctos / frames_evaluados`, `0.0` si no hay evaluados.
- **top_errores**: los 3 mensajes de error mas frecuentes, como `[mensaje, conteo]`.
- **duracion_seg**: tiempo de pared de la sesion.

El script de cada ejercicio decide `correcto`/`errores` con su propio feedback
(wall push-up: correcto = feedback == ["Postura correcta"], errores = mensajes
con "muy"; dominada abierta: correcto = mensajes "Buena ...", error en otro caso).

## Requisitos no funcionales

- Python 3.11; dependencias pineadas (`mediapipe==0.10.14`, `scikit-learn==1.6.1`).
- **No romper** el orden/longitud de features ni el contrato del scaler
  (invariantes de CLAUDE.md). Verificado por `tests/test_features.py`.
- **Verificable en headless**: la logica pura va en `core/` y se cubre con pytest
  sin camara ni GUI.
- Nombres ASCII (sin acentos ni espacios) en archivos/carpetas.
- Reutilizar `core/`; rutas vía `core/config.py`; no hardcodear `os.path.join`.

## Fase 2 — Persistencia (en curso)

Extension sobre el MVP: guardar el resumen de cada sesion en una base de datos
PostgreSQL expuesta por PostgREST, todo contenerizado (Docker Compose). Permite
historial y comparar progreso entre sesiones. Detalle en
`ralph/specs/08-base-datos-postgrest.md`. Decisiones: **usuarios simples** (id +
nombre, SIN autenticacion), granularidad = resumen por sesion + errores principales,
escritura via POST a PostgREST. La DB es best-effort: NO reemplaza el JSON local ni
puede tumbar la sesion de ejercicio si esta caida.

## Fuera de alcance (anti-deriva)

- Dominada neutra; reentrenar o reemplazar modelos `.pkl`.
- Embeber el video en el navegador (streamlit-webrtc) — es mejora Fase 2.
- Deploy/cloud, movil.
- **Autenticacion** (login/contrasenas/JWT): los usuarios de Fase 2 son solo
  etiquetas (id + nombre) para asociar sesiones; no hay auth.

## Criterios de aceptacion (Definition of Done)

1. `./ralph/verify.sh` en VERDE (py_compile + imports + pytest).
2. `app.py` existe; `streamlit run app.py` arranca y lista los 2 ejercicios.
3. Elegir un ejercicio e "Iniciar" lanza el script de retroalimentacion correcto.
4. Tras una sesion existe `sesiones/<ejercicio>_<ts>.json` y la app muestra la
   tarjeta de resumen leyendo el ultimo.
5. README con "Ejecucion rapida" = `streamlit run app.py`.

(Los puntos 2-4 con camara los valida el humano; el loop garantiza 1 y 5 y que el
codigo de soporte este cubierto por tests.)
