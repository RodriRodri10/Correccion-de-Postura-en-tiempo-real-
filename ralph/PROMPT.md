# Ralph — prompt del loop (una iteracion)

Eres un agente de codigo autonomo trabajando en este repositorio. Construyes el
MVP descrito en `ralph/PRD.md`. Trabajas **un (1) ítem por iteracion** y dejas el
repo verificable.

## Pasos de esta iteracion

1. **Lee el contexto** (en este orden): `ralph/PRD.md`, `ralph/PROGRESS.md`,
   `CLAUDE.md`, y el spec en `ralph/specs/` del ítem que vas a hacer.
2. **Elige UN solo ítem**: el primero sin marcar (`- [ ]`) en `ralph/PROGRESS.md`,
   de arriba hacia abajo. Si todos estan marcados y `ralph/verify.sh` pasa,
   escribe `STATUS: DONE` en la primera linea de `ralph/PROGRESS.md` y termina.
3. **Estudia el codigo existente antes de escribir.** Reutiliza `core/`; no
   dupliques helpers. Respeta los invariantes de `CLAUDE.md` (orden/longitud de
   features, contrato del scaler, nombres ASCII, rutas vía `core/config.py`).
4. **Implementa lo minimo** para cerrar ese ítem segun su spec. No hagas refactors
   no pedidos ni toques otros ítems.
5. **Verifica**: corre `bash ralph/verify.sh` y deja que quede en VERDE. Si el ítem
   tiene tests asociados, deben pasar. Itera hasta verde.
6. **Actualiza** `ralph/PROGRESS.md`: marca el ítem `- [x]` y agrega una nota de
   una linea (que hiciste / aprendizaje). Si quedaste bloqueado, deja el ítem sin
   marcar y escribe la razon en una linea bajo el ítem; no lo des por hecho.
7. **Termina la iteracion.** No empieces otro ítem.

## Reglas (no negociables)

- **NO** modifiques ni reemplaces los modelos `.pkl`/`.npy` de `modelos/`.
- **NO** cambies el orden ni la longitud del vector de features (lo bloquea
  `tests/test_features.py`).
- **NO** añadas dependencias sin agregarlas a `requirements.txt` y anotarlo.
- **NO** marques un ítem como hecho si `ralph/verify.sh` esta en rojo.
- Los tests son el contrato: para los ítems de `core/`, implementa hasta que el
  test correspondiente pase; **no edites los tests** para hacerlos pasar (salvo
  que el spec lo pida explicitamente).
- Mantén los cambios pequeños y enfocados en el ítem.
