"""Catalogo de ejercicios del MVP: nombre visible, script y modelo requerido.

Centraliza el registro que hoy esta duplicado en ``deteccion_automatica.py``
(diccionarios SCRIPTS y MODELOS_REQUERIDOS) para que tanto el detector
automatico como la app de Streamlit consulten una sola fuente de verdad.

Lo implementa Ralph para pasar tests/test_catalogo.py reutilizando las rutas de
``core/config.py``. NO duplicar os.path.join dispersos: derivar de config.

Estructura esperada de cada entrada de EJERCICIOS (clave -> dict):
    {
        "nombre":  str,   # nombre visible en la UI (puede llevar acentos)
        "script":  str,   # ruta absoluta al retroalimentacion_*.py
        "modelo":  str,   # ruta absoluta al .pkl requerido
        "vista":   str,   # "lateral" | "frontal" | "posterior"
    }

Claves del MVP: "pushup" y "dom_abierta". (Se permite incluir "dom_neutra"
en el registro, pero disponible("dom_neutra") debe devolver False mientras
falte su modelo.)
"""
import os
from core import config

EJERCICIOS = {
    "pushup": {
        "nombre": "Wall push-up",
        "script": os.path.join(config.RAIZ, "retroalimentacion_wall_pushup.py"),
        "modelo": os.path.join(config.DIR_WALL_PUSHUP, "modelo_fase.pkl"),
        "vista": "lateral",
    },
    "dom_abierta": {
        "nombre": "Dominada agarre abierto",
        "script": os.path.join(config.RAIZ, "retroalimentacion_dominada_abierta.py"),
        "modelo": os.path.join(config.DIR_DOM_ABIERTA, "modelo_fases.pkl"),
        "vista": "posterior",
    },
    "dom_neutra": {
        "nombre": "Dominada agarre neutro",
        "script": os.path.join(config.RAIZ, "retroalimentacion_dominada_neutra.py"),
        "modelo": os.path.join(config.DIR_DOM_NEUTRA, "modelo_fase_dominadas_rt.pkl"),
        "vista": "frontal",
    },
}


def disponible(clave):
    """True si existen el script y el modelo del ejercicio ``clave``."""
    if clave not in EJERCICIOS:
        return False
    entrada = EJERCICIOS[clave]
    return os.path.exists(entrada["script"]) and os.path.exists(entrada["modelo"])


def disponibles():
    """Lista de claves cuyo ejercicio esta disponible (script + modelo)."""
    return [clave for clave in EJERCICIOS if disponible(clave)]
