# Spec 04 — Maquinas de estado de reps (core/reps.py)

Extraer la logica de conteo de reps a clases puras, preservando EXACTAMENTE el
comportamiento actual. Verde: `tests/test_reps.py`.

## FsmWallPushup

Reproduce `init_fsm`/`actualizar_fsm` de `retroalimentacion_wall_pushup.py`
(lineas ~75-137). Estados `WAIT_START -> IN_REP -> LOCKED`. Constantes:
`MIN_FRAMES_REP = 10`, `RESET_FRAMES = 8`, arranque tras 3 frames en fase 1.
Solo procesa fases en `{1,2,3,4}` (ignora otras).

Interfaz:
- `__init__()`: estado inicial (WAIT_START, reps=0).
- `update(fase)`: avanza la maquina con una fase; devuelve `self.reps`.
- `reps` (atributo o property): reps acumuladas.

Conteo: una rep cuando, en IN_REP, vuelve a fase 1 con `frames_rep >= 10` y
habiendo visitado todas las fases `{1,2,3,4}` (o bien fase 3 o 4 ya visitada);
pasa a LOCKED; vuelve a WAIT tras `RESET_FRAMES` frames en fase 1.

## FsmDominadaAbierta

Reproduce la maquina inline del `main()` de
`retroalimentacion_dominada_abierta.py` (lineas ~59-124). Estados
`ABAJO(0) -> SUBE(1) -> ARRIBA(2) -> BAJA(3) -> ABAJO`. `frames_estables`
(default 5) frames consecutivos confirman ARRIBA y ABAJO.

Interfaz:
- `__init__(frames_estables=5)`.
- `update(fase)`: transiciones:
  - ABAJO + fase 2 -> SUBE
  - SUBE + fase 1 (x frames_estables) -> ARRIBA
  - ARRIBA + fase 2 -> BAJA
  - BAJA + fase 3 (x frames_estables) -> ABAJO y **reps += 1**
  devuelve `self.reps`.
- `reps`: reps acumuladas.

## Importante

No cambies umbrales ni la logica de transicion: los scripts de retroalimentacion
deben poder USAR estas clases y obtener el mismo conteo que hoy. En el ítem de
instrumentacion (spec 06) los scripts pasaran a usar estas clases en vez de su
copia inline.
