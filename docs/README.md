# Documentacion del Proyecto

Este directorio concentra la documentacion funcional y tecnica del repositorio.

## Lectura recomendada

1. [README principal](../README.md) - resumen del proyecto, estructura y flujo general.
2. [Estado del proyecto](ESTADO_PROYECTO.md) - componentes disponibles, bloqueos y proximos pasos.
3. [Documentacion tecnica](DOCUMENTACION_TECNICA.md) - arquitectura, pipelines, features, modelos y limitaciones.
4. [Notas para asistentes](../CLAUDE.md) - contexto operativo para cambios futuros en el codigo.

## Mapa rapido

| Documento | Proposito |
|-----------|-----------|
| `../README.md` | Entrada principal para entender el repositorio. |
| `ESTADO_PROYECTO.md` | Seguimiento del estado real del proyecto. |
| `DOCUMENTACION_TECNICA.md` | Explicacion tecnica detallada de entrenamiento, inferencia y evaluacion. |
| `../CLAUDE.md` | Reglas y notas para agentes o asistentes que modifiquen el repo. |

## Convenciones

- Los scripts principales siguen en la raiz del repositorio; la lógica común está en `core/`.
- Los modelos y scalers se conservan en `modelos/` (sin espacios ni acentos).
- Los videos de entrenamiento y prueba no estan versionados; los scripts esperan una carpeta `videos/` con subcarpetas por ejercicio (`wall_pushup/`, `dominada_neutra/`, `dominada_abierta/`).
