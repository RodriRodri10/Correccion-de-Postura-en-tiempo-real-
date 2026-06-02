"""Rutas centralizadas del repositorio.

Todas las rutas se derivan de la ubicación de este archivo, de modo que los
scripts funcionan independientemente del directorio de trabajo actual.
"""
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELOS = os.path.join(RAIZ, "modelos")
VIDEOS = os.path.join(RAIZ, "videos")

# Carpetas de artefactos por ejercicio (sin espacios ni acentos).
DIR_WALL_PUSHUP = os.path.join(MODELOS, "wall_pushup")
DIR_DOM_NEUTRA = os.path.join(MODELOS, "dominada_neutra")
DIR_DOM_ABIERTA = os.path.join(MODELOS, "dominada_abierta")

# Carpetas de videos por ejercicio (no versionadas en el repo).
VIDEOS_WALL_PUSHUP = os.path.join(VIDEOS, "wall_pushup")
VIDEOS_DOM_NEUTRA = os.path.join(VIDEOS, "dominada_neutra")
VIDEOS_DOM_ABIERTA = os.path.join(VIDEOS, "dominada_abierta")
