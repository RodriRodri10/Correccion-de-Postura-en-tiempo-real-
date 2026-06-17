# Estado del Proyecto - Correccion de Postura en Tiempo Real

**Fecha de revision:** 2026-06-16

## Resumen ejecutivo

El proyecto implementa tres pipelines de ejercicio (wall push-up, dominada con agarre
neutro y dominada con agarre abierto) sobre un paquete `core/` que centraliza la logica
comun (geometria, MediaPipe, senales, features). Sobre esa base se construyo un **MVP con
interfaz Streamlit** (Fase 1) y una capa de **persistencia contenerizada con PostgreSQL +
PostgREST** (Fase 2). Ambas fases estan cerradas y `verify.sh` pasa en verde
(py_compile + smoke de imports + 29 tests).

Estado mas importante:

- **MVP Streamlit operativo**: punto de entrada `streamlit run app.py`. Selecciona el
  ejercicio disponible, lanza la retroalimentacion por webcam (subprocess) y muestra una
  tarjeta de resumen al cerrar la sesion.
- **Persistencia opcional y no fatal**: cada sesion se guarda como JSON local
  (`sesiones/`) y, ademas, se intenta enviar a PostgREST. Si la base no esta levantada,
  el envio retorna `None` sin romper la ejecucion.
- Wall push-up y dominada abierta tienen modelo + scaler versionados.
- Dominada neutra sigue sin su modelo (`modelo_fase_dominadas_rt.pkl`); por eso se
  retiro de la interfaz principal del MVP.
- Entorno reproducible: Python 3.11, `mediapipe==0.10.14`, `scikit-learn==1.6.1`.
- `videos/` no esta versionado, por lo que entrenamiento y evaluacion no son
  reproducibles desde un clon limpio.

## Fase 1 - MVP con interfaz Streamlit (cerrada)

Punto de entrada: `streamlit run app.py`.

- `app.py` — UI web: selectbox de ejercicios disponibles, aviso de no-disponibles, boton
  Iniciar (lanza el script de retroalimentacion por `subprocess.run` bloqueante) y tarjeta
  de resumen persistida en `st.session_state` tras el rerun.
- `core/catalogo.py` — diccionario `EJERCICIOS` con metadatos y funciones
  `disponible`/`disponibles` (chequean existencia del modelo requerido).
- `core/sesion.py` — `acumular`, `resumen`, `guardar`, `cargar_ultima` para el log por
  frame y el resumen de la sesion.
- `core/reps.py` — `FsmWallPushup` y `FsmDominadaAbierta`: maquinas de estado para el
  conteo de repeticiones, extraidas de los scripts de retroalimentacion.
- `sesiones/` — JSONs de sesion generados al cerrar cada ejecucion
  (`<ejercicio>_<timestamp>.json`).

Los scripts `retroalimentacion_wall_pushup.py` y `retroalimentacion_dominada_abierta.py`
estan instrumentados: usan las FSM de `core/reps`, registran log por frame con
`core/sesion.acumular` y al salir con `Esc` escriben el resumen y lo envian a la DB.

## Fase 2 - Persistencia (PostgreSQL + PostgREST, contenerizada) (cerrada)

- `docker-compose.yml` — stack de tres servicios: `db` (postgres:16),
  `postgrest` (v12.2.3) y `swagger` (Swagger UI en `:8080`).
- `db/01_init.sql` — schema `api` con tablas `usuario`, `ejercicio`, `sesion`,
  `sesion_error`, seed del catalogo, RPC `crear_sesion` (inserta sesion + errores en una
  sola transaccion) y roles/grants (`authenticator`, `anon`).
- `.env.example` — variables de entorno; `.env` esta en `.gitignore`.
- `core/db.py` — `enviar_sesion(resumen, ejercicio, usuario=None)`: POST best-effort a
  `/rpc/crear_sesion`. No fatal: ante `RequestException` imprime el aviso y retorna `None`.
  El JSON local sigue siendo la fuente primaria.
- `requirements.txt` incluye `requests` para el cliente HTTP.

Nota de puerto: el puerto host de inspeccion de Postgres es **5433** (mapea a `db:5432`)
para no chocar con un Postgres local en 5432. PostgREST conecta por la red interna.

Validacion: el stack se probo end-to-end (seed del catalogo, RPC `crear_sesion`, upsert de
usuario y persistencia tras restart). La verificacion de Docker es manual; `verify.sh` solo
cubre `import core.db` y `tests/test_db.py` (sin red).

## Estado por componente

### Punto de entrada

| Componente | Archivo | Estado |
|------------|---------|--------|
| Interfaz MVP | `app.py` (Streamlit) | Operativa; seleccion + Iniciar + resumen. |
| Deteccion automatica (CLI) | `deteccion_automatica.py` | Usa `core/catalogo`; valida script y modelo antes de lanzar. |
| Arranque rapido | `init.sh` / `init.bat` | Scripts de inicializacion del entorno. |

### Modelos versionados

| Ejercicio | Modelo | Scaler | Otros artefactos | Estado |
|-----------|--------|--------|------------------|--------|
| Wall push-up | `modelo_fase.pkl` | `scaler_fase.pkl` | `rangos_por_fase.npy` | Presente (conviene reentrenar) |
| Dominada neutra | Falta `modelo_fase_dominadas_rt.pkl` | `scaler_fase_dominadas_rt.pkl` | `rangos_por_fase.npy` | Incompleto (fuera del MVP) |
| Dominada abierta | `modelo_fases.pkl` | `scaler_fases.pkl` | `dataset_fases.csv` | Presente |

### Paquete compartido `core/`

| Modulo | Contenido | Usado por |
|--------|-----------|-----------|
| `geometria.py` | `calcular_angulo`, `distancia` | todos |
| `pose.py` | `nueva_pose`, `mp_pose`, indices de landmarks | todos |
| `senales.py` | `flujo_vertical` (optical flow), `suavizar_kalman` | entrenamiento |
| `dinamica.py` | `vel_ang`, `acc_ang`, `*_safe`, `fase_por_curva` | dominada neutra |
| `features.py` | `features_frame` (41 features), `nuevo_historial` | dominada abierta |
| `config.py` | rutas de `modelos/` y `videos/` | todos |
| `catalogo.py` | `EJERCICIOS`, `disponible`, `disponibles` | `app.py`, deteccion |
| `sesion.py` | `acumular`, `resumen`, `guardar`, `cargar_ultima` | retroalimentacion, app |
| `reps.py` | `FsmWallPushup`, `FsmDominadaAbierta` | retroalimentacion |
| `db.py` | `enviar_sesion` (cliente PostgREST, no fatal) | retroalimentacion |

## Problemas activos

### P2 - Modelo faltante para dominada neutra

**Impacto:** `retroalimentacion_dominada_neutra.py` y `evaluacion_dominada_neutra.py`
fallan al cargar `modelos/dominada_neutra/modelo_fase_dominadas_rt.pkl` (lo cargan a nivel
de modulo). Por eso la dominada neutra se retiro de la interfaz principal del MVP.

**Solucion esperada:** recuperar el `.pkl` entrenado o reentrenar con
`entrenamiento_dominada_neutra.py` y videos de referencia.

### P4 (parcial) - Reentrenar wall push-up

**Impacto:** el calculo del angulo de espalda ya es correcto en los tres scripts de wall
push-up, pero el modelo versionado se entreno con esa feature en `0.0`. El RandomForest no
hace split en esa columna (era constante), asi que la prediccion no cambia, pero para
aprovechar la correccion hay que reentrenar.

### P5 - Videos no versionados

**Impacto:** los scripts esperan rutas como `videos/wall_pushup/prueba.mp4` y carpetas de
entrenamiento por ejercicio, pero `videos/` no existe en el repositorio (esta en
`.gitignore`). Entrenamiento y evaluacion no son reproducibles desde un clon limpio.

## Problemas resueltos

- **Fase 1 (MVP Streamlit) y Fase 2 (persistencia PostgreSQL + PostgREST):** cerradas.
- **P1 - `import os` faltante:** corregido en `deteccion_automatica.py`.
- **P3 - Orden de features en dominada neutra:** inferencia y evaluacion usan el mismo
  orden que el entrenamiento (`ang, vel, acc, ang_mean, ang_min, ang_max, vel_mean`).
- **Detector automatico robusto:** usa `core/catalogo`; no lanza retroalimentacion si falta
  el modelo requerido.
- **Contrato de features en Wall Push-Up:** retroalimentacion y evaluacion envian
  `Codo_mean`, `Hombro_mean`, `Espalda_mean`, igual que el scaler entrenado.
- **Evaluacion de dominada abierta:** el ground truth algoritmico usa el mismo mapeo de
  clases que el entrenamiento (`1=Arriba`, `2=Movimiento/Transicion`, `3=Abajo`).
- **Nombres y rutas fragiles:** archivos y carpetas en ASCII sin espacios; rutas
  centralizadas en `core/config.py`.
- **Duplicacion de codigo:** helpers comunes movidos a `core/`.

## Validacion reciente

- `verify.sh` pasa en verde: `py_compile core/*.py *.py`, smoke de imports (incluye
  `core.sesion`, `core.reps`, `core.catalogo`, `core.db`) y **29 tests** con pytest.
- Stack de Docker (db + postgrest) validado manualmente end-to-end.
- La camara local abre con `cv2.VideoCapture(0)`.

## Proximos pasos sugeridos

1. Recuperar o regenerar `modelo_fase_dominadas_rt.pkl` para reincorporar la dominada
   neutra al MVP.
2. Reentrenar el modelo de wall push-up para incorporar la feature de espalda corregida.
3. Documentar o agregar una estrategia para obtener los videos de entrenamiento y prueba.
4. (Opcional) Agregar tests sobre `core/geometria.py` y `core/features.py` adicionales y
   automatizar la verificacion del stack de Docker.
