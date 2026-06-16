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
from core import config

# Registro de ejercicios. Ralph lo completa (placeholder vacio a proposito).
EJERCICIOS = {}


def disponible(clave):
    """True si existen el script y el modelo del ejercicio ``clave``."""
    raise NotImplementedError


def disponibles():
    """Lista de claves cuyo ejercicio esta disponible (script + modelo)."""
    raise NotImplementedError
