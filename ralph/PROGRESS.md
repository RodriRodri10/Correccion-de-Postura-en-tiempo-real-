STATUS: IN_PROGRESS

# Progreso del MVP (Ralph marca cada ítem al cerrarlo)

Orden por dependencias. Un ítem por iteracion, de arriba hacia abajo.
Cuando todos esten en `[x]` y `ralph/verify.sh` pase, cambia la primera linea a
`STATUS: DONE`.

- [ ] **core/sesion.py** — implementar `acumular`, `resumen`, `guardar`,
  `cargar_ultima` segun `ralph/specs/03-sesion-resumen.md`. Verde: `tests/test_sesion.py`.
- [ ] **core/reps.py** — implementar `FsmWallPushup` y `FsmDominadaAbierta` segun
  `ralph/specs/04-reps-fsm.md`. Verde: `tests/test_reps.py`.
- [ ] **core/catalogo.py** — implementar `EJERCICIOS`, `disponible`, `disponibles`
  segun `ralph/specs/02-catalogo-ejercicios.md`. Verde: `tests/test_catalogo.py`.
- [ ] **deteccion_automatica.py** — usar `core/catalogo` en vez del bloque
  duplicado `SCRIPTS`/`MODELOS_REQUERIDOS` (`ralph/specs/02-catalogo-ejercicios.md`).
- [ ] **Instrumentar retroalimentacion** — wall push-up y dominada abierta
  registran el log por frame con `core/sesion.acumular` y al salir escriben
  `sesiones/<ejercicio>_<ts>.json` (`ralph/specs/06-instrumentar-retro.md`).
- [ ] **app.py (Streamlit)** — seleccion de ejercicio + Iniciar (subprocess) +
  tarjeta de resumen (`ralph/specs/05-streamlit-app.md`).
- [ ] **Docs** — README "Ejecucion rapida" = `streamlit run app.py` y actualizar
  `docs/` (`ralph/specs/07-docs.md`).
