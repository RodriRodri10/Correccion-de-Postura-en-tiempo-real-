# Spec 06 — Instrumentar la retroalimentacion (escribir resumen de sesion)

Hacer que `retroalimentacion_wall_pushup.py` y
`retroalimentacion_dominada_abierta.py` registren la sesion y, al salir, escriban
`sesiones/<ejercicio>_<ts>.json` usando `core/sesion`. Esto alimenta la tarjeta
de resumen de la app (FR5).

## Cambios comunes (ambos scripts)

1. Importar `from core import sesion` y `from core import config`.
2. Crear `log = []` antes del bucle y, opcionalmente, marcar `t0 = time.time()`.
3. Usar las FSM de `core/reps` en vez de la maquina inline:
   - wall push-up: `from core.reps import FsmWallPushup`.
   - dominada abierta: `from core.reps import FsmDominadaAbierta`.
   (El conteo en pantalla debe quedar igual que hoy.)
4. Por cada frame, tras calcular fase y feedback, llamar
   `sesion.acumular(log, fase, correcto, errores)` donde:
   - **wall push-up**: `correcto = (feedback == ["Postura correcta"])`;
     `errores = [m for m in feedback if "muy" in m]`. Si no hubo cuerpo
     detectado (fase == -1), `correcto=None`, `errores=[]`.
   - **dominada abierta**: el feedback es un string; `correcto` True si empieza
     con "Buena", False si es un mensaje de correccion ("Extiende...", "Sube...");
     `errores = [feedback]` cuando es correccion, `[]` si bueno. fase -1 -> None.
5. Al terminar (despues del `while`, en el cleanup): calcular
   `dur = time.time() - t0`, `r = sesion.resumen(log, fsm.reps, dur)` y
   `sesion.guardar(r, "<clave>", os.path.join(config.RAIZ, "sesiones"))`.
   Claves: `"pushup"` y `"dom_abierta"` (deben coincidir con el catalogo).

## Restricciones

- No cambiar el overlay en vivo ni los umbrales de fase/reps.
- No romper el contrato de features/scaler (no tocar el orden del vector).
- `sesiones/` esta en `.gitignore`; crear el dir si no existe (lo hace `guardar`).
- Verificar con `py_compile` (no hay test de camara).
