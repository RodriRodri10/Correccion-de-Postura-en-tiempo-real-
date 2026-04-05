# CLAUDE.md — Corrección de Postura en Tiempo Real

## Descripción del proyecto

Sistema de análisis de ejercicios de fuerza de tronco superior mediante visión por computadora y aprendizaje automático. Detecta automáticamente el ejercicio que realiza el usuario (dominadas con agarre neutro, dominadas con agarre abierto, wall push-ups) y proporciona retroalimentación en tiempo real sobre su técnica de ejecución.

## Stack tecnológico

- **Python 3.x**
- **MediaPipe Pose** — detección de landmarks corporales
- **scikit-learn** — Random Forest Classifier, StandardScaler
- **OpenCV** — captura y visualización de video
- **pykalman** — suavizado de señales angulares (filtro de Kalman)
- **scipy** — detección de picos (find_peaks) y optical flow
- **joblib** — serialización/deserialización de modelos
- **pandas / numpy** — manipulación de datos

## Estructura de archivos

```
├── Detección Automatica.py                      # Punto de entrada principal
├── Entrenamiento_wall_push_up.py               # Entrena modelo wall push-ups
├── Entrenamiento_dominada_agarre_neutro.py     # Entrena modelo dominadas neutras
├── Entrenamiento_Dominada_Agarre_Abierto.py    # Entrena modelo dominadas abiertas
├── Evaluacion_wall_push_up.py                  # Valida ML vs Ground Truth
├── Evaluacion_dominada_agarre_neutro.py
├── Evaluacion_dominada_agarre_abierto.py
├── Retroalimentación_Wall__push_up.py          # Retroalimentación en tiempo real
├── Retroalimentacion_dominada_agarre_neutro.py
├── Retroalimentación_Dominada_Agarre_Abierto.py
└── Modelos/
    ├── dominadas agarre abierto/
    │   ├── modelo_fases.pkl
    │   ├── scaler_fases.pkl
    │   └── dataset_fases.csv
    ├── dominadas neutro/
    │   ├── scaler_fase_dominadas_rt.pkl
    │   └── rangos_por_fase.npy
    └── walls_push_up/
        ├── modelo_fase.pkl
        ├── scaler_fase.pkl
        └── rangos_por_fase.npy
```

## Problema crítico conocido: rutas hardcodeadas a Windows

**Todos los scripts tienen rutas absolutas de Windows (`C:\...`).** El proyecto fue desarrollado en Windows y actualmente el repositorio está en Linux. Al modificar cualquier script de retroalimentación o entrenamiento, las rutas deben actualizarse para apuntar a las ubicaciones reales de los modelos.

Rutas hardcodeadas relevantes:
- `Detección Automatica.py` → `SCRIPTS` apunta a rutas `C:\Users\Acer\...`
- `Retroalimentación_Wall__push_up.py` → `RESULTADOS_DIR = r"C:\rangos_por_fase_push"`
- `Retroalimentacion_dominada_agarre_neutro.py` → `BASE_DIR = r"C:\resultados_dominadas_uno"`
- `Retroalimentación_Dominada_Agarre_Abierto.py` → `BASE_DIR = r"C:\modelo_fases_reglas"`

Los modelos ya están en el repositorio en `/Modelos/`. Al corregir rutas, apuntar ahí.

## Modelos ML

| Ejercicio | Algoritmo | Features | Fases | Archivos |
|-----------|-----------|----------|-------|----------|
| Wall Push-Up | Random Forest (300 árboles) | 3 (codo, hombro, espalda) | 4 | `Modelos/walls_push_up/` |
| Dominada Agarre Neutro | Random Forest (400 árboles) | 7 (ángulo + vel + acc + estadísticas) | 3 | `Modelos/dominadas neutro/` |
| Dominada Agarre Abierto | Random Forest (400 árboles) | 41 (ángulos múltiples + estadísticas) | 3 | `Modelos/dominadas agarre abierto/` |

## Convenciones del código

- Las funciones de cálculo de ángulos reciben tres puntos `(a, b, c)` como arrays NumPy y retornan grados.
- Los modelos se cargan con `joblib.load()` al inicio del script de retroalimentación.
- El scaler siempre se aplica antes de predecir: `scaler.transform(features)`.
- La detección de ejercicio en `Detección Automatica.py` usa landmarks: hombro izquierdo (11), hombro derecho (12), nariz (0).

## Flujo de ejecución

1. Ejecutar `Detección Automatica.py`
2. El script detecta la postura del usuario vía MediaPipe durante 1 segundo estable
3. Lanza el script de retroalimentación correspondiente mediante `subprocess.call`
4. El script de retroalimentación carga el modelo, captura video y predice fases frame a frame

## Notas para Claude

- No existe `requirements.txt` — si se necesita, generarlo a partir de los imports de todos los scripts.
- No hay tests automatizados. La evaluación se hace manualmente con los scripts `Evaluacion_*.py`.
- Al modificar scripts de retroalimentación, verificar que las features que se construyen en tiempo real sean **idénticas** a las usadas durante el entrenamiento (mismo orden, mismas transformaciones).
- Los archivos `.pkl` con modelos están versionados en Git vía LFS (`.gitattributes` con `* text=auto`).
