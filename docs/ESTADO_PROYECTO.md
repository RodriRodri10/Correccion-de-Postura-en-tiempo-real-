# Estado del Proyecto - Correccion de Postura en Tiempo Real

**Fecha de revision:** 2026-06-14

## Resumen ejecutivo

El proyecto implementa tres pipelines de ejercicio: wall push-up, dominada con agarre neutro y dominada con agarre abierto. La estructura fue reorganizada: nombres ASCII (sin acentos ni espacios), carpeta `modelos/` sin espacios, y un paquete `core/` que centraliza la lógica común (geometría, MediaPipe, señales, features). Esto eliminó la duplicación que antes existía en los nueve scripts.

Estado más importante:

- Wall push-up tiene modelo y scaler versionados. El bug del ángulo de espalda fue corregido en el código (ahora es la inclinación del tronco vs. la vertical); el modelo versionado debe reentrenarse para aprovecharlo.
- Dominada con agarre abierto tiene modelo, scaler y dataset versionados.
- Dominada con agarre neutro sigue sin poder ejecutarse porque falta `modelos/dominada_neutra/modelo_fase_dominadas_rt.pkl`. El desajuste de orden de features ya fue corregido.
- El entorno local reproducible usa Python 3.11, `mediapipe==0.10.14` y `scikit-learn==1.6.1`; esta version de scikit-learn coincide con los modelos `.pkl` versionados.
- No hay carpeta `videos/` versionada, por lo que entrenamiento y evaluación no son reproducibles desde cero con solo clonar el repo.

## Estado por componente

### Punto de entrada

| Componente | Archivo | Estado |
|------------|---------|--------|
| Deteccion automatica | `deteccion_automatica.py` | Implementada; valida que exista el script y modelo antes de lanzar retroalimentación. |
| Lanzamiento de scripts | `SCRIPTS` en `deteccion_automatica.py` | Rutas relativas vía `core/config.py`, apuntan a los scripts renombrados. |

### Modelos versionados

| Ejercicio | Modelo | Scaler | Otros artefactos | Estado |
|-----------|--------|--------|------------------|--------|
| Wall push-up | `modelo_fase.pkl` | `scaler_fase.pkl` | `rangos_por_fase.npy` | Presente (conviene reentrenar) |
| Dominada neutra | Falta `modelo_fase_dominadas_rt.pkl` | `scaler_fase_dominadas_rt.pkl` | `rangos_por_fase.npy` | Incompleto |
| Dominada abierta | `modelo_fases.pkl` | `scaler_fases.pkl` | `dataset_fases.csv` | Presente |

### Scripts principales

| Tipo | Wall push-up | Dominada neutra | Dominada abierta |
|------|--------------|-----------------|------------------|
| Entrenamiento | `entrenamiento_wall_pushup.py` | `entrenamiento_dominada_neutra.py` | `entrenamiento_dominada_abierta.py` |
| Retroalimentacion | `retroalimentacion_wall_pushup.py` | `retroalimentacion_dominada_neutra.py` | `retroalimentacion_dominada_abierta.py` |
| Evaluacion | `evaluacion_wall_pushup.py` | `evaluacion_dominada_neutra.py` | `evaluacion_dominada_abierta.py` |

### Paquete compartido `core/`

| Módulo | Contenido | Usado por |
|--------|-----------|-----------|
| `geometria.py` | `calcular_angulo`, `distancia` | todos |
| `pose.py` | `nueva_pose`, `mp_pose`, índices de landmarks | todos |
| `senales.py` | `flujo_vertical` (optical flow), `suavizar_kalman` | entrenamiento |
| `dinamica.py` | `vel_ang`, `acc_ang`, `*_safe`, `fase_por_curva` | dominada neutra |
| `features.py` | `features_frame` (41 features), `nuevo_historial` | dominada abierta |
| `config.py` | rutas de `modelos/` y `videos/` | todos |

## Problemas activos

### P2 - Modelo faltante para dominada neutra

**Impacto:** `retroalimentacion_dominada_neutra.py` y `evaluacion_dominada_neutra.py` fallan al cargar `modelos/dominada_neutra/modelo_fase_dominadas_rt.pkl` (lo cargan a nivel de módulo).

**Solucion esperada:** recuperar el `.pkl` entrenado o reentrenar con `entrenamiento_dominada_neutra.py` y videos de referencia.

### P4 (parcial) - Reentrenar wall push-up

**Impacto:** el cálculo del ángulo de espalda ya es correcto en `entrenamiento_wall_pushup.py`, `retroalimentacion_wall_pushup.py` y `evaluacion_wall_pushup.py`, pero el modelo versionado se entrenó con esa feature en `0.0`. El RandomForest actual no hace split en esa columna (era constante), así que la predicción no cambia, pero para aprovechar la corrección hay que reentrenar.

### P5 - Videos no versionados

**Impacto:** los scripts esperan rutas como `videos/wall_pushup/prueba.mp4` y carpetas de entrenamiento por ejercicio, pero `videos/` no existe en el repositorio (ya está en `.gitignore`).

## Problemas resueltos

- **P1 - `import os` faltante:** corregido en `deteccion_automatica.py`.
- **P3 - Orden de features en dominada neutra:** inferencia y evaluación ahora usan el mismo orden que el entrenamiento (`ang, vel, acc, ang_mean, ang_min, ang_max, vel_mean`).
- **Detector automático robusto:** no lanza retroalimentación si falta el modelo requerido; muestra el ejercicio como no disponible.
- **Contrato de features en Wall Push-Up:** retroalimentación y evaluación envían `Codo_mean`, `Hombro_mean`, `Espalda_mean`, igual que el scaler entrenado.
- **Evaluación de dominada abierta:** el ground truth algorítmico ya usa el mismo mapeo de clases que el entrenamiento (`1=Arriba`, `2=Movimiento/Transición`, `3=Abajo`).
- **Nombres y rutas frágiles:** archivos y carpetas renombrados a ASCII sin espacios; rutas centralizadas en `core/config.py`.
- **Duplicación de código:** helpers comunes movidos a `core/`.

## Validacion reciente

- `python -m py_compile core/*.py *.py` pasa dentro de `.venv`.
- La cámara local abre con `cv2.VideoCapture(0)`.
- `retroalimentacion_wall_pushup.py` arranca con MediaPipe y procesa frames; la prueba fue detenida con `timeout`.

## Proximos pasos sugeridos

1. Recuperar o regenerar `modelo_fase_dominadas_rt.pkl`.
2. Reentrenar el modelo de wall push-up para incorporar la feature de espalda corregida.
3. Documentar o agregar una estrategia para obtener los videos de entrenamiento y prueba.
4. (Opcional) Agregar tests sobre `core/geometria.py` y `core/features.py`.
