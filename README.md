# Correccion de Postura en Tiempo Real

Sistema de analisis de ejercicios de fuerza mediante vision por computadora y aprendizaje automatico. Usa MediaPipe Pose para extraer landmarks corporales y modelos Random Forest para clasificar fases de movimiento en tiempo real.

## Ejercicios soportados

| Ejercicio | Script en tiempo real | Fases |
|-----------|------------------------|-------|
| Wall push-up | `retroalimentacion_wall_pushup.py` | Inicio, descenso, abajo, subida |
| Dominada con agarre abierto | `retroalimentacion_dominada_abierta.py` | Arriba, transicion, abajo |

## Flujo del proyecto

```text
videos de referencia
    -> scripts de entrenamiento
    -> artefactos en modelos/
    -> scripts de retroalimentacion
    -> scripts de evaluacion
```

El punto de entrada general es `deteccion_automatica.py`, que detecta la postura inicial con la camara y lanza el script de retroalimentacion correspondiente. Si detecta un ejercicio cuyo modelo falta, muestra que no esta disponible en vez de lanzar un script que fallaria.

## Estructura principal

```text
.
├── deteccion_automatica.py
├── entrenamiento_*.py
├── evaluacion_*.py
├── retroalimentacion_*.py
├── core/              # lógica compartida (geometría, pose, señales, features)
├── modelos/
├── docs/
├── requirements.txt
└── CLAUDE.md
```

## Documentacion

- [Indice de documentacion](docs/README.md)
- [Documentacion tecnica](docs/DOCUMENTACION_TECNICA.md)
- [Estado del proyecto](docs/ESTADO_PROYECTO.md)
- [Notas para asistentes de desarrollo](CLAUDE.md)

## Instalacion

```bash
/usr/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Se usa Python 3.11 porque `mediapipe==0.10.14` no es compatible de forma confiable con Python 3.13. El `requirements.txt` tambien fija `scikit-learn==1.6.1`, que coincide con la version usada para serializar los modelos `.pkl` versionados.

## Ejecucion rapida

```bash
source .venv/bin/activate
streamlit run app.py
```

La app lista los ejercicios disponibles. Elige uno, pulsa **Iniciar** y la ventana de retroalimentacion en vivo (OpenCV) se abre. Realiza el ejercicio frente a la camara y cierra con `Esc`. Al terminar, la app muestra la tarjeta de resumen con reps totales, porcentaje de postura correcta y errores principales por episodios, duracion aproximada y porcentaje de tiempo evaluado.

Ejercicios disponibles en el MVP:
- **Wall push-up** (vista lateral) — modelo en `modelos/wall_pushup/`.
- **Dominada agarre abierto** (vista posterior) — modelo en `modelos/dominada_abierta/`.

La dominada con agarre neutro quedo fuera de la interfaz porque falta su modelo `modelos/dominada_neutra/modelo_fase_dominadas_rt.pkl`. Sus scripts y el scaler se conservan para reentrenar a futuro.

## Estado actual

El proyecto tiene modelos versionados para wall push-up y dominada con agarre abierto. La dominada con agarre neutro quedo fuera de la interfaz (catalogo y deteccion automatica) porque falta `modelos/dominada_neutra/modelo_fase_dominadas_rt.pkl`; sus scripts y scaler siguen versionados por si se reentrena.
