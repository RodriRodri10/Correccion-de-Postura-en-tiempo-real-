# Spec 02 — Catalogo de ejercicios

## core/catalogo.py

Fuente unica de verdad del registro de ejercicios. Reutiliza `core/config.py`
(`DIR_WALL_PUSHUP`, `DIR_DOM_ABIERTA`, `DIR_DOM_NEUTRA`, `RAIZ`).

`EJERCICIOS`: dict `clave -> {"nombre", "script", "modelo", "vista"}`.
Entradas requeridas:

| clave | nombre | script | modelo | vista |
|-------|--------|--------|--------|-------|
| `pushup` | "Wall push-up" | `RAIZ/retroalimentacion_wall_pushup.py` | `DIR_WALL_PUSHUP/modelo_fase.pkl` | "lateral" |
| `dom_abierta` | "Dominada agarre abierto" | `RAIZ/retroalimentacion_dominada_abierta.py` | `DIR_DOM_ABIERTA/modelo_fases.pkl` | "posterior" |
| `dom_neutra` | "Dominada agarre neutro" | `RAIZ/retroalimentacion_dominada_neutra.py` | `DIR_DOM_NEUTRA/modelo_fase_dominadas_rt.pkl` | "frontal" |

(Incluir `dom_neutra` en el registro esta bien; su modelo no existe, asi que
`disponible("dom_neutra")` debe ser False.)

- `disponible(clave)`: True si existen el `script` y el `modelo` de esa entrada
  (`os.path.exists` en ambos).
- `disponibles()`: lista de claves con `disponible(clave) == True`.

Verde: `tests/test_catalogo.py`.

## Refactor de deteccion_automatica.py

Hoy `deteccion_automatica.py` duplica `SCRIPTS` y `MODELOS_REQUERIDOS` y define
`recursos_disponibles`. Reemplazar ese bloque por el uso de `core/catalogo`:
- las rutas de script/modelo salen de `catalogo.EJERCICIOS`;
- `recursos_disponibles(ejercicio)` pasa a delegar en `catalogo.disponible(...)`.

Mantener el comportamiento actual del detector (mensaje "no disponible" cuando
falta el modelo). No cambiar la heuristica de deteccion de pose.
Verificar con `py_compile` (no hay test de camara para este script).
