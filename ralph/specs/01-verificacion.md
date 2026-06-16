# Spec 01 — Verificacion (headless)

`ralph/verify.sh` es el gate del loop. Debe correr **sin camara ni GUI** y ser
determinista. Hace, en orden:

1. `py_compile` de `core/*.py` y `*.py` (raiz).
2. Smoke de imports de todos los modulos de `core/` (incluye `core.sesion`,
   `core.reps`, `core.catalogo`).
3. `pytest -q tests/`.

Sale con codigo != 0 si cualquier paso falla. El loop solo commitea cuando
`verify.sh` esta en VERDE.

## Reglas para mantenerlo verde y rapido

- Los tests NO deben abrir `cv2.VideoCapture`, ventanas ni cargar videos.
- La logica testeable vive en `core/` (pura): por eso `app.py` y los scripts de
  retroalimentacion (que usan camara/Streamlit) no se importan en los tests.
- No metas en `tests/` nada que descargue modelos o dependa de red.
- Si añades un modulo nuevo a `core/`, agrégalo al smoke de imports de `verify.sh`.

Estado inicial esperado: py_compile e imports en verde; `test_geometria.py` y
`test_features.py` en verde; `test_sesion.py`, `test_reps.py`, `test_catalogo.py`
en rojo hasta que se implementen los stubs (ese rojo->verde es el trabajo).
