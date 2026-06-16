"""Contrato del acumulador/resumen de sesion (objetivo de Ralph: rojo -> verde)."""
from core import sesion


def _log_ejemplo():
    log = []
    sesion.acumular(log, 1, correcto=True, errores=[])
    sesion.acumular(log, 1, correcto=True, errores=[])
    sesion.acumular(log, 1, correcto=True, errores=[])
    sesion.acumular(log, 2, correcto=False, errores=["Codo muy bajo"])
    sesion.acumular(log, 2, correcto=False, errores=["Codo muy bajo"])
    sesion.acumular(log, -1, correcto=None, errores=[])  # frame no evaluado
    return log


def test_resumen_metricas():
    r = sesion.resumen(_log_ejemplo(), reps=5, duracion_seg=12.0)
    assert r["reps"] == 5
    assert r["frames_evaluados"] == 5      # el frame con fase -1 no cuenta
    assert r["frames_correctos"] == 3
    assert r["pct_correcto"] == 60.0
    assert r["duracion_seg"] == 12.0


def test_resumen_top_errores():
    r = sesion.resumen(_log_ejemplo(), reps=5, duracion_seg=12.0)
    assert r["top_errores"][0] == ["Codo muy bajo", 2]


def test_resumen_sin_evaluar():
    log = []
    sesion.acumular(log, -1, correcto=None, errores=[])
    r = sesion.resumen(log, reps=0, duracion_seg=1.0)
    assert r["frames_evaluados"] == 0
    assert r["pct_correcto"] == 0.0


def test_guardar_y_cargar_ultima(tmp_path):
    r = sesion.resumen(_log_ejemplo(), reps=5, duracion_seg=12.0)
    ruta = sesion.guardar(r, "pushup", str(tmp_path))
    assert ruta.endswith(".json")
    cargado = sesion.cargar_ultima(str(tmp_path), ejercicio="pushup")
    assert cargado["reps"] == 5
    assert cargado["pct_correcto"] == 60.0


def test_cargar_ultima_vacio(tmp_path):
    assert sesion.cargar_ultima(str(tmp_path)) is None
