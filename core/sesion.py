"""Acumulacion y resumen de una sesion de ejercicio.

Contrato compartido entre los scripts de retroalimentacion (que registran cada
frame en vivo) y la app de Streamlit (que muestra el resumen final). Es logica
pura sobre listas y diccionarios: no toca camara, OpenCV ni MediaPipe, para que
sea verificable en headless con pytest.

Estructura del log: lista de registros por frame, cada uno un dict:
    {"fase": int, "correcto": bool | None, "errores": list[str]}
donde ``correcto`` es None cuando el frame no fue evaluado (fase == -1). El
script de cada ejercicio decide ``correcto``/``errores`` con su propia logica de
feedback (p.ej. wall push-up: correcto = feedback == ["Postura correcta"];
errores = [m for m in feedback if "muy" in m]).

Implementa el contrato cubierto por tests/test_sesion.py. NO modificar las firmas.
"""
import json
import os
import time


def acumular(log, fase, correcto=None, errores=()):
    """Anade un registro de frame a ``log`` (lista) y lo devuelve.

    - ``fase``: fase predicha (int) o -1 si no hubo deteccion.
    - ``correcto``: bool si el frame fue evaluado, None si no (fase == -1).
    - ``errores``: iterable de mensajes de error de ese frame (vacio si correcto).

    Muta ``log`` en sitio (append) y devuelve ``log``.
    """
    log.append({"fase": fase, "correcto": correcto, "errores": list(errores)})
    return log


def resumen(log, reps, duracion_seg):
    """Calcula el resumen de la sesion a partir del log.

    Devuelve un dict con estas claves:
        - "reps": int (viene del FSM, se pasa como argumento)
        - "frames_evaluados": int  (frames con correcto is not None)
        - "frames_correctos": int  (frames con correcto is True)
        - "pct_correcto": float    (100*correctos/evaluados, 0.0 si evaluados==0)
        - "top_errores": list[list] (compatibilidad: errores por frame)
        - "errores_principales": list[dict] (errores por episodios)
        - "frames_con_error": int
        - "pct_frames_con_error": float
        - "duracion_seg": float
    """
    import collections
    frames_evaluados = sum(1 for r in log if r["correcto"] is not None)
    frames_correctos = sum(1 for r in log if r["correcto"] is True)
    pct_correcto = (100.0 * frames_correctos / frames_evaluados) if frames_evaluados else 0.0
    todos_errores = [e for r in log for e in r["errores"]]
    top_errores = [list(t) for t in collections.Counter(todos_errores).most_common(3)]
    frames_con_error = sum(
        1 for r in log
        if r["correcto"] is not None and len(r["errores"]) > 0
    )
    pct_frames_con_error = (100.0 * frames_con_error / frames_evaluados) if frames_evaluados else 0.0
    errores_principales = _errores_por_episodio(log, duracion_seg, frames_evaluados)
    return {
        "reps": reps,
        "frames_evaluados": frames_evaluados,
        "frames_correctos": frames_correctos,
        "pct_correcto": pct_correcto,
        "top_errores": top_errores,
        "errores_principales": errores_principales,
        "frames_con_error": frames_con_error,
        "pct_frames_con_error": pct_frames_con_error,
        "duracion_seg": float(duracion_seg),
    }


def _errores_por_episodio(log, duracion_seg, frames_evaluados, min_frames=3, limite=3):
    """Agrupa errores por rachas consecutivas para evitar conteos inflados."""
    import collections

    activos = {}
    acumulado = collections.defaultdict(lambda: {"eventos": 0, "frames": 0})

    def cerrar_ausentes(errores_actuales):
        for mensaje in list(activos):
            if mensaje not in errores_actuales:
                frames = activos.pop(mensaje)
                if frames >= min_frames:
                    acumulado[mensaje]["eventos"] += 1
                    acumulado[mensaje]["frames"] += frames

    for registro in log:
        if registro["correcto"] is None:
            cerrar_ausentes(set())
            continue

        errores_actuales = set(registro["errores"])
        cerrar_ausentes(errores_actuales)

        for mensaje in errores_actuales:
            activos[mensaje] = activos.get(mensaje, 0) + 1

    cerrar_ausentes(set())

    frame_seg = (float(duracion_seg) / len(log)) if log else 0.0
    errores = []
    for mensaje, datos in acumulado.items():
        frames = datos["frames"]
        pct = (100.0 * frames / frames_evaluados) if frames_evaluados else 0.0
        errores.append({
            "mensaje": mensaje,
            "eventos": datos["eventos"],
            "frames": frames,
            "segundos": frames * frame_seg,
            "pct_tiempo_evaluado": pct,
        })

    errores.sort(key=lambda e: (-e["frames"], -e["eventos"], e["mensaje"]))
    return errores[:limite]


def guardar(resumen_dict, ejercicio, dir_sesiones):
    """Serializa ``resumen_dict`` a JSON en ``dir_sesiones``.

    Crea el directorio si no existe. El nombre de archivo es
    ``<ejercicio>_<timestamp>.json``. Devuelve la ruta escrita.
    """
    os.makedirs(dir_sesiones, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nombre = f"{ejercicio}_{timestamp}.json"
    ruta = os.path.join(dir_sesiones, nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resumen_dict, f, ensure_ascii=False, indent=2)
    return ruta


def cargar_ultima(dir_sesiones, ejercicio=None):
    """Carga el resumen JSON mas reciente de ``dir_sesiones``.

    Si ``ejercicio`` se indica, filtra por el prefijo ``<ejercicio>_``.
    Devuelve el dict cargado o None si no hay sesiones / no existe el dir.
    """
    if not os.path.isdir(dir_sesiones):
        return None

    archivos = []
    for fname in os.listdir(dir_sesiones):
        if not fname.endswith(".json"):
            continue
        if ejercicio and not fname.startswith(f"{ejercicio}_"):
            continue
        fpath = os.path.join(dir_sesiones, fname)
        archivos.append((os.path.getmtime(fpath), fpath))

    if not archivos:
        return None

    archivos.sort(reverse=True)
    _, fpath = archivos[0]
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)
