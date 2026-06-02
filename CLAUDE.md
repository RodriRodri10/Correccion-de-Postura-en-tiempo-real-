# CLAUDE.md - Correccion de Postura en Tiempo Real

## Descripcion del proyecto

Sistema de analisis de ejercicios de fuerza de tronco superior mediante vision por computadora y aprendizaje automatico. Detecta automaticamente el ejercicio que realiza el usuario y proporciona retroalimentacion en tiempo real sobre la tecnica de ejecucion. Es un trabajo terminal (TT1).

## Concepto clave: dos niveles de "modelo"

El sistema combina dos modelos muy distintos. Entenderlo es indispensable antes de tocar nada:

| | MediaPipe Pose (BlazePose) | Random Forest (de este repo) |
|---|---|---|
| Quien lo entreno | Google (pre-entrenado, caja negra) | El autor, en `entrenamiento_*.py` |
| Entrada -> salida | imagen RGB -> 33 landmarks | angulos -> fase del ejercicio (1..4) |
| Esta en el repo | No (se instala con pip) | Si, los `.pkl` en `modelos/` |

**MediaPipe NO se entrena aqui.** "Encaja perfecto" porque la vision por computadora dificil ya esta resuelta por Google. Lo unico que se entrena es el Random Forest, que opera sobre angulos limpios derivados de los landmarks.

### Detalles de MediaPipe (lo que usa el proyecto)

- API *legacy* `mp.solutions.pose` (no la Tasks API). Centralizada en `core/pose.py` (`nueva_pose()`).
- Variante **Full** (`model_complexity=1`, valor por defecto al no especificarlo); ~3.5M parametros, 6.9 MFLOPs, entrada 256x256, `smooth_landmarks=True`.
- Pipeline interno de 2 etapas: detector de persona (tipo BlazeFace) + tracker de landmarks (CNN encoder-decoder).
- `requirements.txt` fija `mediapipe==0.10.14` para garantizar que `mp.solutions.pose` siga disponible (versiones nuevas empujan `PoseLandmarker` y retiran `solutions.*`). **No actualizar sin migrar la API.**
- Indices de landmarks usados: 0 nariz; 11/12 hombros; 13/14 codos; 15/16 munecas; 23/24 caderas (constantes en `core/pose.py`).

## Stack tecnologico

- Python 3.x — en este entorno el binario es `python3` (`python` no existe).
- MediaPipe Pose (BlazePose) — deteccion de landmarks.
- OpenCV — captura/procesamiento de video y flujo optico (Farneback).
- scikit-learn — RandomForestClassifier, StandardScaler, metricas.
- scipy `find_peaks` — deteccion de repeticiones.
- pykalman — suavizado de senales (solo wall push-up).
- numpy, pandas — procesamiento numerico.
- joblib — serializacion de modelos `.pkl`.

## Estructura funcional

Nombres ASCII (sin acentos ni espacios) y logica comun centralizada en `core/`.

```text
├── deteccion_automatica.py            # punto de entrada: detecta ejercicio y lanza el script
├── entrenamiento_wall_pushup.py       # videos -> modelos/wall_pushup/
├── entrenamiento_dominada_neutra.py
├── entrenamiento_dominada_abierta.py
├── evaluacion_wall_pushup.py          # GT algoritmico vs ML (matriz de confusion)
├── evaluacion_dominada_neutra.py
├── evaluacion_dominada_abierta.py
├── retroalimentacion_wall_pushup.py   # inferencia en vivo con webcam
├── retroalimentacion_dominada_neutra.py
├── retroalimentacion_dominada_abierta.py
├── core/                              # paquete compartido (sin duplicacion)
│   ├── geometria.py       # calcular_angulo(a,b,c), angulo (alias), distancia(p,q)
│   ├── pose.py            # nueva_pose(), mp_pose, punto(lm,idx,w,h), indices de landmarks
│   ├── senales.py         # flujo_vertical(video) [optical flow], suavizar_kalman(serie)
│   ├── dinamica.py        # vel_ang, acc_ang(=alias), mean/min/max_safe, fase_por_curva (neutra)
│   ├── features.py        # features_frame(...,con_nombres=False), nuevo_historial(), claves_historial()
│   └── config.py          # RAIZ, MODELOS, VIDEOS y DIR_*/VIDEOS_* por ejercicio
├── modelos/
├── docs/
├── requirements.txt
└── .gitignore             # __pycache__/, videos/, *.mp4 de salida
```

Los scripts viven en la raiz para que `from core import ...` resuelva sin instalar el paquete (la raiz queda en `sys.path[0]` al ejecutar cada script, tambien cuando `deteccion_automatica.py` los lanza via `subprocess`).

## Modelos ML y pipeline por ejercicio

Pipeline comun de entrenamiento: video -> MediaPipe -> angulos -> deteccion de reps -> **etiquetado automatico de fase por heuristica** (no anotacion humana) -> features -> StandardScaler -> RandomForest -> `.pkl`.

| Ejercicio | Vista | Features | Fases | Deteccion de reps | Etiquetado de fase | Artefactos |
|-----------|-------|----------|-------|-------------------|--------------------|------------|
| Wall push-up | lateral | 3 (medias por fase) | 4 | senal ponderada de angulos + `find_peaks` | temporal fijo (25/45/70%) | `modelos/wall_pushup/` |
| Dominada neutra | frontal | 7 (ang+vel+acc+stats) | 3 | flujo optico vertical | umbral 20/80% sobre ventana | `modelos/dominada_neutra/` |
| Dominada abierta | posterior | 41 (5 stats x 5 senales + derivadas) | 3 | flujo optico vertical | umbral relativo al maximo de la rep | `modelos/dominada_abierta/` |

Orden EXACTO de features de dominada neutra (entrenamiento = inferencia = evaluacion): `ang, vel, acc, ang_mean, ang_min, ang_max, vel_mean`.

## Estado tecnico relevante

- Rutas centralizadas en `core/config.py` (derivadas de la ubicacion del repo).
- **Pendiente P2:** falta `modelos/dominada_neutra/modelo_fase_dominadas_rt.pkl`. Los scripts de neutra cargan el modelo a nivel de modulo, por lo que fallan al importarse hasta recuperarlo/reentrenarlo.
- **Pendiente P4 (parte 2):** el modelo de wall push-up versionado se entreno con la feature de espalda en `0.0`; conviene reentrenar para aprovechar la correccion (ver abajo).
- **Resuelto P1:** `deteccion_automatica.py` ya importa `os`.
- **Resuelto P3:** orden de features de neutra unificado en los tres pipelines.
- **Resuelto P4 (codigo):** el angulo de espalda es la inclinacion del tronco vs. la vertical (`calcular_angulo(hombro, cadera, [cadera_x, 0])`). El RandomForest actual ignora esa columna por ser constante en el entrenamiento previo, asi que la prediccion no cambia hasta reentrenar.
- `acc_ang` NO es un bug: misma diferencia finita que `vel_ang` pero aplicada al historial de velocidades (por eso es un alias en `core/dinamica.py`).
- No hay tests automatizados; la validacion se hace con `evaluacion_*.py` y un GT algoritmico (no anotacion humana).

## Ejecucion y validacion

- Entorno actual: dependencias **no instaladas** y **sin camara ni `videos/`**. La unica validacion posible aqui es estatica.
- Validacion estatica: `python3 -m py_compile core/*.py *.py`.
- Para correr de verdad: `pip install -r requirements.txt`; entrenar necesita `videos/<ejercicio>/*.mp4`; retroalimentacion necesita webcam (`cv2.VideoCapture(0)`); evaluacion necesita `videos/<ejercicio>/prueba.mp4`.
- Smoke-test de imports (tras instalar deps): `python3 -c "import core.geometria, core.pose, core.features, core.dinamica, core.senales, core.config"`.

## Convenciones al modificar codigo

- Reutilizar `core/` en vez de reintroducir helpers duplicados.
- Mantener nombres ASCII (sin acentos ni espacios) en archivos y carpetas.
- Usar `core/config.py` para rutas; no hardcodear `os.path.join` dispersos.
- Verificar que el vector de features usado en inferencia coincida EXACTAMENTE (orden incluido) con el de entrenamiento.
- No asumir que existe la carpeta `videos/`; validar su presencia antes de entrenar o evaluar.
- No reemplazar modelos `.pkl` sin indicar con que datos y script fueron generados.
- Despues de tocar scripts, ejecutar al menos `python3 -m py_compile core/*.py *.py`.

## Estructura documental

- `README.md`: entrada principal del proyecto.
- `docs/README.md`: indice de documentacion.
- `docs/ESTADO_PROYECTO.md`: estado actual, bloqueos y proximos pasos.
- `docs/DOCUMENTACION_TECNICA.md`: arquitectura, pipelines, features, MediaPipe y limitaciones.
