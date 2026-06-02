# Correccion de Postura en Tiempo Real

Sistema de analisis de ejercicios de fuerza mediante vision por computadora y aprendizaje automatico. Usa MediaPipe Pose para extraer landmarks corporales y modelos Random Forest para clasificar fases de movimiento en tiempo real.

## Ejercicios soportados

| Ejercicio | Script en tiempo real | Fases |
|-----------|------------------------|-------|
| Wall push-up | `retroalimentacion_wall_pushup.py` | Inicio, descenso, abajo, subida |
| Dominada con agarre neutro | `retroalimentacion_dominada_neutra.py` | Abajo, movimiento, arriba |
| Dominada con agarre abierto | `retroalimentacion_dominada_abierta.py` | Arriba, transicion, abajo |

## Flujo del proyecto

```text
videos de referencia
    -> scripts de entrenamiento
    -> artefactos en modelos/
    -> scripts de retroalimentacion
    -> scripts de evaluacion
```

El punto de entrada general es `deteccion_automatica.py`, que detecta la postura inicial con la camara y lanza el script de retroalimentacion correspondiente.

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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estado actual

El proyecto tiene modelos versionados para wall push-up y dominada con agarre abierto. La dominada con agarre neutro esta incompleta porque falta `modelos/dominada_neutra/modelo_fase_dominadas_rt.pkl`.
