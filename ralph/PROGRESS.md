STATUS: DONE (Fase 1 MVP + Fase 2 persistencia)

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
- [x] **Instrumentar retroalimentacion** — wall push-up y dominada abierta
  registran el log por frame con `core/sesion.acumular` y al salir escriben
  `sesiones/<ejercicio>_<ts>.json` (`ralph/specs/06-instrumentar-retro.md`).
  > Ambos scripts usan FsmWallPushup/FsmDominadaAbierta de core/reps; acumular por frame; guardar al salir con claves "pushup" y "dom_abierta". py_compile + 19 tests en verde.
- [x] **app.py (Streamlit)** — seleccion de ejercicio + Iniciar (subprocess) +
  tarjeta de resumen (`ralph/specs/05-streamlit-app.md`).
  > `app.py` en raiz: selectbox con ejercicios disponibles, aviso de no-disponibles, subprocess.run bloqueante, st.session_state para persistir resumen tras rerun; 19 tests en verde.
- [x] **Docs** — README "Ejecucion rapida" = `streamlit run app.py` y actualizar
  `docs/` (`ralph/specs/07-docs.md`).
  > README actualizado con flujo app→OpenCV→resumen y lista de ejercicios del MVP. ESTADO_PROYECTO con nota del MVP Streamlit. DOCUMENTACION_TECNICA con nueva sección 3 (flujo, módulos core nuevos, métricas del resumen). verify.sh en verde.

## Fase 2 — Persistencia (PostgreSQL + PostgREST, contenerizado)

- [x] **DB infra (Parte A)** — `db/01_init.sql` (schema `api`, tablas usuario/
  ejercicio/sesion/sesion_error, seed, RPC `crear_sesion`, roles+grants),
  `docker-compose.yml` (postgres + postgrest), `.env.example`, `.env` a `.gitignore`
  (`ralph/specs/08-base-datos-postgrest.md`). Verificacion manual con docker (no en verify.sh).
  > Stack validado end-to-end: seed del catalogo, RPC crear_sesion (sesion+errores en 1 transaccion), upsert de usuario y persistencia tras restart. Fix: puerto host de db a 5433 (evita choque con Postgres local).
- [x] **DB cliente (Parte B)** — `core/db.py` (POST best-effort a `/rpc/crear_sesion`),
  integrar `db.enviar_sesion(r, "<clave>")` tras `sesion.guardar` en los 2 scripts,
  `requirements.txt` + requests, `ralph/verify.sh` + `core.db`, `tests/test_db.py`
  (`ralph/specs/08-base-datos-postgrest.md`). Verde: `ralph/verify.sh`.
  > core/db.py validado contra PostgREST vivo (enviar_sesion devuelve id) y no-fatalidad confirmada (DB caida -> None sin traceback). 29 tests en verde.
