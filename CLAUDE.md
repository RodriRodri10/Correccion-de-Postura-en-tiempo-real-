# CLAUDE.md - Correccion de Postura en Tiempo Real

## Descripcion del proyecto

Sistema de analisis de ejercicios de fuerza de tronco superior mediante vision por computadora y aprendizaje automatico. Detecta automaticamente el ejercicio que realiza el usuario y proporciona retroalimentacion en tiempo real sobre la tecnica de ejecucion.

## Stack tecnologico

- Python 3.x
- MediaPipe Pose para deteccion de landmarks corporales
- OpenCV para captura y procesamiento de video
- scikit-learn para Random Forest, StandardScaler y metricas
- scipy para `find_peaks`
- pykalman para suavizado de senales en wall push-up
- numpy y pandas para procesamiento numerico
- joblib para serializacion de modelos

## Estructura documental

- `README.md`: entrada principal del proyecto.
- `docs/README.md`: indice de documentacion.
- `docs/ESTADO_PROYECTO.md`: estado actual, bloqueos y proximos pasos.
- `docs/DOCUMENTACION_TECNICA.md`: arquitectura, pipelines y detalles de modelos.

## Estructura funcional

Nombres ASCII (sin acentos ni espacios) y lógica común centralizada en `core/`.

```text
├── deteccion_automatica.py
├── entrenamiento_wall_pushup.py
├── entrenamiento_dominada_neutra.py
├── entrenamiento_dominada_abierta.py
├── evaluacion_wall_pushup.py
├── evaluacion_dominada_neutra.py
├── evaluacion_dominada_abierta.py
├── retroalimentacion_wall_pushup.py
├── retroalimentacion_dominada_neutra.py
├── retroalimentacion_dominada_abierta.py
├── core/                  # paquete compartido (sin duplicación)
│   ├── geometria.py       # calcular_angulo, distancia
│   ├── pose.py            # wrapper MediaPipe (nueva_pose) e índices de landmarks
│   ├── senales.py         # flujo_vertical (optical flow), suavizar_kalman
│   ├── dinamica.py        # vel_ang, acc_ang, *_safe, fase_por_curva (neutra)
│   ├── features.py        # features_frame de 41 features (abierta)
│   └── config.py          # rutas centralizadas (modelos/, videos/)
├── modelos/
├── docs/
└── requirements.txt
```

Los scripts viven en la raíz para que `from core import ...` resuelva sin
instalar el paquete (la raíz queda en `sys.path[0]` al ejecutar cada script).

## Modelos ML

| Ejercicio | Algoritmo | Features | Fases | Artefactos |
|-----------|-----------|----------|-------|------------|
| Wall push-up | Random Forest | 3 | 4 | `modelos/wall_pushup/` |
| Dominada neutra | Random Forest | 7 | 3 | `modelos/dominada_neutra/` |
| Dominada abierta | Random Forest | 41 | 3 | `modelos/dominada_abierta/` |

## Estado tecnico relevante

- Las rutas se centralizan en `core/config.py` (derivadas de la ubicación del repo).
- Falta `modelos/dominada_neutra/modelo_fase_dominadas_rt.pkl` (requiere reentrenar con videos).
- **Resuelto:** `deteccion_automatica.py` ya importa `os`.
- **Resuelto:** el orden de features de dominada neutra coincide entre entrenamiento, inferencia y evaluación (`ang, vel, acc, ang_mean, ang_min, ang_max, vel_mean`).
- **Resuelto (código):** wall push-up calcula el ángulo de espalda como inclinación del tronco vs. la vertical. El modelo versionado `modelos/wall_pushup/modelo_fase.pkl` se entrenó con esa feature en `0.0`; conviene **reentrenar** para que la corrección tenga efecto (el RandomForest actual ignora esa columna por ser constante, así que la predicción no cambia mientras tanto).
- No hay tests automatizados; la validacion se hace con los scripts `evaluacion_*.py`.

## Convenciones al modificar codigo

- Reutilizar `core/` en vez de reintroducir helpers duplicados.
- Mantener las rutas relativas al repositorio (usar `core/config.py`).
- Verificar que el vector de features usado en inferencia coincida exactamente con el de entrenamiento.
- No asumir que existe la carpeta `videos/`; validar su presencia antes de entrenar o evaluar.
- No reemplazar modelos `.pkl` sin indicar con que datos y script fueron generados.
- Despues de tocar scripts, ejecutar al menos `python3 -m py_compile core/*.py *.py`.
