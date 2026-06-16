STATUS: IN_PROGRESS

# Progreso del MVP (Ralph marca cada ítem al cerrarlo)

Orden por dependencias. Un ítem por iteracion, de arriba hacia abajo.
Cuando todos esten en `[x]` y `ralph/verify.sh` pase, cambia la primera linea a
`STATUS: DONE`.

- [x] **core/sesion.py** — implementar `acumular`, `resumen`, `guardar`,
  `cargar_ultima` segun `ralph/specs/03-sesion-resumen.md`. Verde: `tests/test_sesion.py`.
  > Implementados los 4 metodos; 5/5 tests en verde. `collections.Counter.most_common(3)` con conversion a lista para sobrevivir round-trip JSON.
- [x] **core/reps.py** — implementar `FsmWallPushup` y `FsmDominadaAbierta` segun
  `ralph/specs/04-reps-fsm.md`. Verde: `tests/test_reps.py`.
  > FSM de wall push-up (WAIT_START/IN_REP/LOCKED) y dominada abierta (ABAJO/SUBE/ARRIBA/BAJA) extraidas de retroalimentacion_*.py. 4/4 tests en verde.
- [x] **core/catalogo.py** — implementar `EJERCICIOS`, `disponible`, `disponibles`
  segun `ralph/specs/02-catalogo-ejercicios.md`. Verde: `tests/test_catalogo.py`.
  > EJERCICIOS dict con 3 ejercicios (pushup, dom_abierta, dom_neutra). `disponible()` y `disponibles()` usan os.path.exists. 3/3 tests en verde.
- [x] **deteccion_automatica.py** — usar `core/catalogo` en vez del bloque
  duplicado `SCRIPTS`/`MODELOS_REQUERIDOS` (`ralph/specs/02-catalogo-ejercicios.md`).
  > Reemplazado SCRIPTS/MODELOS_REQUERIDOS por catalogo.EJERCICIOS. Refactor completo, py_compile verde.
- [ ] **Instrumentar retroalimentacion** — wall push-up y dominada abierta
  registran el log por frame con `core/sesion.acumular` y al salir escriben
  `sesiones/<ejercicio>_<ts>.json` (`ralph/specs/06-instrumentar-retro.md`).
- [ ] **app.py (Streamlit)** — seleccion de ejercicio + Iniciar (subprocess) +
  tarjeta de resumen (`ralph/specs/05-streamlit-app.md`).
- [ ] **Docs** — README "Ejecucion rapida" = `streamlit run app.py` y actualizar
  `docs/` (`ralph/specs/07-docs.md`).
