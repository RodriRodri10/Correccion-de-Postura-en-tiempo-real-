"""Punto de entrada del MVP — interfaz Streamlit.

Uso: streamlit run app.py
"""
import os
import sys
import subprocess

import streamlit as st

from core import catalogo, sesion, config

DIR_SESIONES = os.path.join(config.RAIZ, "sesiones")

st.title("Corrector de Postura en Tiempo Real")
st.write(
    "Selecciona un ejercicio, realizalo frente a la webcam "
    "y recibe retroalimentacion en vivo sobre tu tecnica."
)

claves_disp = catalogo.disponibles()

if not claves_disp:
    st.warning(
        "No hay ejercicios disponibles. "
        "Verifica que los modelos (.pkl) esten en `modelos/`."
    )
    st.stop()

opciones = {catalogo.EJERCICIOS[c]["nombre"]: c for c in claves_disp}

nombre_elegido = st.selectbox("Ejercicio", options=list(opciones.keys()))
clave_elegida = opciones[nombre_elegido]

no_disp = [
    catalogo.EJERCICIOS[c]["nombre"]
    for c in catalogo.EJERCICIOS
    if not catalogo.disponible(c)
]
if no_disp:
    st.info(f"No disponibles (falta modelo): {', '.join(no_disp)}")

st.write(f"Vista requerida: **{catalogo.EJERCICIOS[clave_elegida]['vista']}**")
st.caption(
    "Al pulsar Iniciar se abrira una ventana de video separada (OpenCV). "
    "Presiona **Esc** en esa ventana para finalizar la sesion y ver el resumen aqui."
)

if st.button("Iniciar sesion"):
    script = catalogo.EJERCICIOS[clave_elegida]["script"]
    aviso = st.empty()
    aviso.info("Sesion en progreso... Cierra la ventana de video con **Esc**.")
    try:
        subprocess.run([sys.executable, script], cwd=config.RAIZ)
    except Exception as e:
        aviso.error(f"No se pudo lanzar el ejercicio: {e}")
        st.stop()
    aviso.empty()
    st.session_state["ultima_sesion"] = {
        "clave": clave_elegida,
        "datos": sesion.cargar_ultima(DIR_SESIONES, ejercicio=clave_elegida),
    }
    st.rerun()

if "ultima_sesion" in st.session_state:
    info = st.session_state["ultima_sesion"]
    datos = info["datos"]
    nombre_ej = catalogo.EJERCICIOS[info["clave"]]["nombre"]
    st.divider()
    st.subheader(f"Resumen — {nombre_ej}")
    if datos is None:
        st.info(
            "No se encontro resumen de sesion "
            "(es posible que hayas cerrado antes de completar una repeticion)."
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Repeticiones", datos["reps"])
        col2.metric("Postura correcta", f"{datos['pct_correcto']:.1f}%")
        col3.metric("Duracion", f"{datos['duracion_seg']:.0f} s")
        if "errores_principales" in datos:
            if datos["errores_principales"]:
                st.write("**Errores principales:**")
                for error in datos["errores_principales"]:
                    eventos = error["eventos"]
                    etiqueta_eventos = "episodio" if eventos == 1 else "episodios"
                    st.write(
                        f"- {error['mensaje']}: {eventos} {etiqueta_eventos}, "
                        f"{error['segundos']:.1f} s, "
                        f"{error['pct_tiempo_evaluado']:.1f}% del tiempo evaluado"
                    )
            else:
                st.write("Sin errores registrados en esta sesion.")
        elif datos["top_errores"]:
            st.write("**Errores mas frecuentes:**")
            for msg, conteo in datos["top_errores"]:
                st.write(f"- {msg} ({conteo} veces)")
        else:
            st.write("Sin errores registrados en esta sesion.")
