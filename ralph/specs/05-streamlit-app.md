# Spec 05 — App de Streamlit (app.py)

`app.py` va en la RAIZ del repo (para que `from core import ...` resuelva igual
que los demas scripts). Es la puerta de entrada (FR1) y el tablero de resultados.

## Dependencias

`streamlit` (ya en `requirements.txt`). No usar `streamlit-webrtc` (fuera de
alcance).

## Comportamiento

1. **Titulo** y una breve descripcion.
2. **Seleccion de ejercicio**: usar `core/catalogo`. Mostrar `nombre` de cada
   ejercicio; deshabilitar/excluir los que no esten `disponible(...)` (FR2). Si
   ninguno esta disponible, mostrar un aviso claro.
3. **Boton "Iniciar sesion"**: lanza el script del ejercicio elegido por
   subprocess, igual que `deteccion_automatica.py` hace hoy:
   `subprocess.run([sys.executable, catalogo.EJERCICIOS[clave]["script"]])`.
   La llamada bloquea hasta que el usuario cierra la ventana OpenCV (Esc). Avisar
   en la UI que se abrira una ventana aparte y que se cierra con Esc.
4. **Tarjeta de resumen**: al volver del subprocess, leer
   `core/sesion.cargar_ultima(<dir sesiones>, ejercicio=clave)` y mostrar:
   reps, % postura correcta, top errores y duracion. Si no hay JSON (p.ej. el
   usuario cerro sin reps), mostrar un aviso amable.
5. **Robustez (FR6)**: si falta camara o modelo, mensaje claro, sin stacktrace.
   El directorio de sesiones es `os.path.join(config.RAIZ, "sesiones")`.

## Notas

- No bloquear el import de `app.py` con trabajo pesado; Streamlit reejecuta el
  script en cada interaccion. Usa `st.session_state` si necesitas recordar la
  ultima sesion mostrada.
- El subprocess hereda el cwd del repo; los scripts ya resuelven `core/` solos.
- No se testea con pytest (usa Streamlit/camara); se valida a mano (ver spec 07)
  y con `py_compile`.
