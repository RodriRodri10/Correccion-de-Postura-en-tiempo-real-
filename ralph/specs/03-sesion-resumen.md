# Spec 03 — Sesion y resumen (core/sesion.py)

Logica pura sobre listas/dicts. Sin camara, sin OpenCV. Firmas ya fijadas en el
stub; impleméntalas sin cambiarlas. Verde: `tests/test_sesion.py`.

## Modelo de datos

`log` = lista de registros por frame. Cada registro es un dict:
`{"fase": int, "correcto": bool | None, "errores": list[str]}`.

## Funciones

- `acumular(log, fase, correcto=None, errores=())`:
  añade `{"fase": fase, "correcto": correcto, "errores": list(errores)}` a `log`
  y devuelve `log`.

- `resumen(log, reps, duracion_seg)`: devuelve dict con EXACTAMENTE estas claves:
  - `reps` = `reps` (int, viene del FSM).
  - `frames_evaluados` = nº de registros con `correcto is not None`.
  - `frames_correctos` = nº de registros con `correcto is True`.
  - `pct_correcto` = `100.0 * frames_correctos / frames_evaluados`, o `0.0` si
    `frames_evaluados == 0`.
  - `top_errores` = los 3 mensajes mas frecuentes (sobre todos los `errores` de
    todos los frames), cada uno como lista `[mensaje, conteo]`, orden descendente
    por conteo. Usa `collections.Counter(...).most_common(3)` y convierte cada
    tupla a lista (para que sobreviva el round-trip JSON).
  - `duracion_seg` = `float(duracion_seg)`.

- `guardar(resumen_dict, ejercicio, dir_sesiones)`: crea `dir_sesiones` si no
  existe (`os.makedirs(..., exist_ok=True)`), escribe JSON en
  `<ejercicio>_<timestamp>.json` (timestamp con `time.strftime("%Y%m%d_%H%M%S")`
  o `int(time.time())`), devuelve la ruta.

- `cargar_ultima(dir_sesiones, ejercicio=None)`: si el dir no existe devuelve
  None; si `ejercicio` se da, filtra archivos que empiecen por `f"{ejercicio}_"`;
  toma el `.json` mas reciente (por mtime), lo carga y lo devuelve; None si no hay.

## Nota de integracion

El JSON guardado se usa en `app.py` (tarjeta de resumen). Mantén las claves
estables: la app y el test dependen de ellas.
