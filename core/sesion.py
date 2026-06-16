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

Lo implementa Ralph para pasar tests/test_sesion.py. NO modificar las firmas.
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
    raise NotImplementedError


def resumen(log, reps, duracion_seg):
    """Calcula el resumen de la sesion a partir del log.

    Devuelve un dict con exactamente estas claves:
        - "reps": int (viene del FSM, se pasa como argumento)
        - "frames_evaluados": int  (frames con correcto is not None)
        - "frames_correctos": int  (frames con correcto is True)
        - "pct_correcto": float    (100*correctos/evaluados, 0.0 si evaluados==0)
        - "top_errores": list[list] (los 3 errores mas frecuentes como [msg, conteo])
        - "duracion_seg": float
    """
    raise NotImplementedError


def guardar(resumen_dict, ejercicio, dir_sesiones):
    """Serializa ``resumen_dict`` a JSON en ``dir_sesiones``.

    Crea el directorio si no existe. El nombre de archivo es
    ``<ejercicio>_<timestamp>.json``. Devuelve la ruta escrita.
    """
    raise NotImplementedError


def cargar_ultima(dir_sesiones, ejercicio=None):
    """Carga el resumen JSON mas reciente de ``dir_sesiones``.

    Si ``ejercicio`` se indica, filtra por el prefijo ``<ejercicio>_``.
    Devuelve el dict cargado o None si no hay sesiones / no existe el dir.
    """
    raise NotImplementedError
