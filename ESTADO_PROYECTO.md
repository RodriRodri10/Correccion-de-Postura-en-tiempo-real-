# Estado del Proyecto — Corrección de Postura en Tiempo Real

**Fecha de revisión:** 2026-03-31

---

## Resumen ejecutivo

El proyecto cuenta con tres ejercicios completamente implementados: wall push-up, dominada con agarre neutro y dominada con agarre abierto. Los modelos de Machine Learning están entrenados y versionados en el repositorio. El sistema de retroalimentación en tiempo real y la detección automática de ejercicios funcionan de forma independiente. El principal bloqueo para ejecutar el proyecto en el entorno actual (Linux) son las rutas absolutas de Windows hardcodeadas en todos los scripts.

---

## Estado por componente

### Detección automática
| Aspecto | Estado |
|---------|--------|
| Lógica de clasificación de ejercicio | Completa |
| Estabilización por movimiento de hombro | Completa |
| Countdown visual (3-2-1) | Completo |
| Rutas a scripts de retroalimentación | **Rotas** (apuntan a `C:\Users\Acer\...`) |

### Modelos entrenados
| Ejercicio | Modelo | Scaler | Dataset | Estado |
|-----------|--------|--------|---------|--------|
| Wall Push-Up | `modelo_fase.pkl` | `scaler_fase.pkl` | `rangos_por_fase.npy` | Completo |
| Dominada Neutro | *(modelo no encontrado en repo)* | `scaler_fase_dominadas_rt.pkl` | `rangos_por_fase.npy` | Incompleto — falta `modelo_fase_dominadas_rt.pkl` |
| Dominada Abierto | `modelo_fases.pkl` | `scaler_fases.pkl` | `dataset_fases.csv` | Completo |

> **Nota:** El script `Retroalimentacion_dominada_agarre_neutro.py` carga `modelo_fase_dominadas_rt.pkl` desde `C:\resultados_dominadas_uno`, pero ese archivo no está en el directorio `Modelos/dominadas neutro/` del repositorio.

### Scripts de retroalimentación en tiempo real
| Script | Lógica | Rutas | Estado |
|--------|--------|-------|--------|
| `Retroalimentación_Wall__push_up.py` | Completa | Rotas (`C:\rangos_por_fase_push`) | Necesita corrección de rutas |
| `Retroalimentacion_dominada_agarre_neutro.py` | Completa | Rotas (`C:\resultados_dominadas_uno`) | Necesita corrección de rutas + modelo faltante |
| `Retroalimentación_Dominada_Agarre_Abierto.py` | Completa | Rotas (`C:\modelo_fases_reglas`) | Necesita corrección de rutas |

### Scripts de evaluación (GT vs ML)
| Script | Estado |
|--------|--------|
| `Evaluacion_wall_push_up.py` | Implementado |
| `Evaluacion_dominada_agarre_neutro.py` | Implementado |
| `Evaluacion_dominada_agarre_abierto.py` | Implementado |

### Scripts de entrenamiento
| Script | Estado |
|--------|--------|
| `Entrenamiento_wall_push_up.py` | Implementado — requiere videos en `C:\push-up correcto` |
| `Entrenamiento_dominada_agarre_neutro.py` | Implementado — requiere videos en ruta Windows |
| `Entrenamiento_Dominada_Agarre_Abierto.py` | Implementado — requiere videos en ruta Windows |

---

## Problemas conocidos

### P1 — Rutas hardcodeadas a Windows (bloqueante)
**Impacto:** El proyecto no puede ejecutarse en Linux sin modificar manualmente las rutas en cada script.
**Archivos afectados:** Todos los scripts de retroalimentación, entrenamiento y `Detección Automatica.py`.
**Solución sugerida:** Reemplazar rutas absolutas por rutas relativas basadas en `os.path.dirname(__file__)` o `pathlib.Path`.

### P2 — Modelo faltante para dominada agarre neutro
**Impacto:** `Retroalimentacion_dominada_agarre_neutro.py` falla al cargar `modelo_fase_dominadas_rt.pkl`.
**Solución sugerida:** Verificar si el modelo existe localmente fuera del repositorio y agregarlo, o re-entrenar.

### P3 — Sin `requirements.txt`
**Impacto:** Instalación del entorno no está documentada.
**Solución sugerida:** Generar `requirements.txt` con `pip freeze` o manualmente desde los imports.

### P4 — Dependencia de cámara web en tiempo de ejecución
**Impacto:** Los scripts asumen `cv2.VideoCapture(0)` disponible. En entornos sin cámara (servidores, CI) fallarán.

---

## Próximos pasos sugeridos

1. **Corregir rutas** en todos los scripts para usar rutas relativas al repositorio.
2. **Agregar el modelo faltante** `modelo_fase_dominadas_rt.pkl` para dominada agarre neutro.
3. **Generar `requirements.txt`** con las dependencias exactas del proyecto.
4. **Probar ejecución completa** en Linux con los modelos del repositorio.

---

## Historial de cambios relevante (git log)

| Commit | Descripción |
|--------|-------------|
| `9f3a65c` | Update README.md |
| `4d934f4` | Detección automática de ejercicios |
| `29de1b3` | Corrección de archivo |
| `b8b7d75` | Modelos obtenidos tras el entrenamiento |
| `71b886f` | Archivos del ejercicio dominada con agarre neutro |
