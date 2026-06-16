# Spec 07 — Documentacion

Dejar claro como ejecutar el producto en pocos pasos.

## README.md

- Añadir (o actualizar) una seccion **"Ejecucion rapida"** cuyo comando principal
  sea:
  ```bash
  source .venv/bin/activate
  streamlit run app.py
  ```
- Explicar el flujo: elegir ejercicio -> "Iniciar" abre la ventana en vivo
  (OpenCV, se cierra con Esc) -> aparece la tarjeta de resumen.
- Mencionar que solo wall push-up y dominada abierta estan disponibles en el MVP
  (neutra pendiente por falta de modelo).
- Mantener la seccion de instalacion existente (venv 3.11 + requirements).

## docs/

- `docs/ESTADO_PROYECTO.md`: añadir una nota de que el MVP con interfaz Streamlit
  ya existe (punto de entrada `app.py`, resumen de sesion en `sesiones/`).
- `docs/DOCUMENTACION_TECNICA.md`: documentar brevemente el flujo
  app.py -> subprocess retroalimentacion -> JSON de sesion -> tarjeta de resumen,
  y los modulos nuevos `core/sesion.py`, `core/reps.py`, `core/catalogo.py`.

## Cierre

Cuando este ítem quede hecho y `ralph/verify.sh` este en verde con todos los
demas ítems marcados, cambia la primera linea de `ralph/PROGRESS.md` a
`STATUS: DONE`.
