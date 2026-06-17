"""Punto de entrada del MVP — interfaz Streamlit.

Uso: streamlit run app.py
"""
import os
import sys
import subprocess
from datetime import datetime

import streamlit as st

from core import catalogo, sesion, config

DIR_SESIONES = os.path.join(config.RAIZ, "sesiones")
VISTAS = ["Entrenar", "Consultar sesiones"]


def nombre_ejercicio(clave):
    entrada = catalogo.EJERCICIOS.get(clave)
    return entrada["nombre"] if entrada else clave


def fecha_sesion(registro):
    timestamp = registro.get("timestamp")
    if timestamp:
        try:
            return datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(registro["mtime"]).strftime("%Y-%m-%d %H:%M:%S")


def filas_sesiones(registros):
    filas = []
    for registro in registros:
        datos = registro["datos"]
        filas.append({
            "Fecha": fecha_sesion(registro),
            "Ejercicio": nombre_ejercicio(registro["ejercicio"]),
            "Reps": datos.get("reps", 0),
            "Postura correcta": f"{datos.get('pct_correcto', 0.0):.1f}%",
            "Duracion": f"{datos.get('duracion_seg', 0.0):.0f} s",
            "Archivo": registro["archivo"],
        })
    return filas


def mostrar_resumen(datos):
    if datos is None:
        st.info(
            "No se encontro resumen de sesion "
            "(es posible que hayas cerrado antes de completar una repeticion)."
        )
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Repeticiones", datos.get("reps", 0))
    col2.metric("Postura correcta", f"{datos.get('pct_correcto', 0.0):.1f}%")
    col3.metric("Duracion", f"{datos.get('duracion_seg', 0.0):.0f} s")

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
    elif datos.get("top_errores"):
        st.write("**Errores mas frecuentes:**")
        for msg, conteo in datos["top_errores"]:
            st.write(f"- {msg} ({conteo} veces)")
    else:
        st.write("Sin errores registrados en esta sesion.")


def vista_entrenar():
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
        return

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
        st.session_state["abrir_consulta"] = True
        st.rerun()

    if "ultima_sesion" in st.session_state:
        info = st.session_state["ultima_sesion"]
        st.divider()
        st.subheader(f"Resumen — {nombre_ejercicio(info['clave'])}")
        mostrar_resumen(info["datos"])


def vista_consultar_sesiones():
    st.subheader("Sesiones")
    registros = sesion.listar(DIR_SESIONES)

    if not registros:
        st.info("No hay sesiones guardadas todavia.")
        return

    ejercicios = sorted({registro["ejercicio"] for registro in registros})
    etiquetas = ["Todos"] + [nombre_ejercicio(clave) for clave in ejercicios]
    mapa_etiquetas = {nombre_ejercicio(clave): clave for clave in ejercicios}
    filtro = st.selectbox("Ejercicio", options=etiquetas)
    if filtro == "Todos":
        filtrados = registros
    else:
        filtrados = [r for r in registros if r["ejercicio"] == mapa_etiquetas[filtro]]

    st.dataframe(filas_sesiones(filtrados), width="stretch")

    opciones = {
        f"{fecha_sesion(r)} - {nombre_ejercicio(r['ejercicio'])} - {r['archivo']}": r
        for r in filtrados
    }
    etiqueta = st.selectbox("Detalle", options=list(opciones.keys()))
    registro = opciones[etiqueta]
    st.caption(registro["archivo"])
    mostrar_resumen(registro["datos"])

st.title("Corrector de Postura en Tiempo Real")

if st.session_state.get("abrir_consulta"):
    st.session_state["menu_vista"] = "Consultar sesiones"
    st.session_state["abrir_consulta"] = False

if "menu_vista" not in st.session_state:
    st.session_state["menu_vista"] = VISTAS[0]

vista = st.sidebar.selectbox("Menu", options=VISTAS, key="menu_vista")

if vista == "Entrenar":
    vista_entrenar()
else:
    vista_consultar_sesiones()
