"""Contrato del acumulador/resumen de sesion (TDD: rojo -> verde)."""
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


def test_error_consecutivo_cuenta_un_episodio():
    log = []
    for _ in range(100):
        sesion.acumular(log, 2, correcto=False, errores=["Codo muy alto"])

    r = sesion.resumen(log, reps=0, duracion_seg=10.0)
    assert r["top_errores"][0] == ["Codo muy alto", 100]
    assert r["errores_principales"][0]["mensaje"] == "Codo muy alto"
    assert r["errores_principales"][0]["eventos"] == 1
    assert r["errores_principales"][0]["frames"] == 100
    assert r["errores_principales"][0]["segundos"] == 10.0
    assert r["errores_principales"][0]["pct_tiempo_evaluado"] == 100.0


def test_error_separado_cuenta_dos_episodios():
    log = []
    for _ in range(4):
        sesion.acumular(log, 2, correcto=False, errores=["Codo muy alto"])
    for _ in range(2):
        sesion.acumular(log, 1, correcto=True, errores=[])
    for _ in range(5):
        sesion.acumular(log, 2, correcto=False, errores=["Codo muy alto"])

    r = sesion.resumen(log, reps=0, duracion_seg=11.0)
    error = r["errores_principales"][0]
    assert error["mensaje"] == "Codo muy alto"
    assert error["eventos"] == 2
    assert error["frames"] == 9
    assert error["segundos"] == 9.0


def test_rachas_muy_cortas_no_son_errores_principales():
    log = []
    sesion.acumular(log, 2, correcto=False, errores=["Codo muy alto"])
    sesion.acumular(log, 2, correcto=False, errores=["Codo muy alto"])

    r = sesion.resumen(log, reps=0, duracion_seg=2.0)
    assert r["top_errores"][0] == ["Codo muy alto", 2]
    assert r["errores_principales"] == []
    assert r["frames_con_error"] == 2
    assert r["pct_frames_con_error"] == 100.0


def test_errores_simultaneos_se_agregan_por_mensaje():
    log = []
    for _ in range(3):
        sesion.acumular(log, 2, correcto=False, errores=["Codo muy alto", "Hombro muy bajo"])
    for _ in range(2):
        sesion.acumular(log, 2, correcto=False, errores=["Codo muy alto"])

    r = sesion.resumen(log, reps=0, duracion_seg=5.0)
    por_mensaje = {e["mensaje"]: e for e in r["errores_principales"]}
    assert por_mensaje["Codo muy alto"]["eventos"] == 1
    assert por_mensaje["Codo muy alto"]["frames"] == 5
    assert por_mensaje["Hombro muy bajo"]["eventos"] == 1
    assert por_mensaje["Hombro muy bajo"]["frames"] == 3
    assert r["frames_con_error"] == 5


def test_resumen_sin_errores_principales():
    log = []
    sesion.acumular(log, 1, correcto=True, errores=[])
    sesion.acumular(log, 1, correcto=True, errores=[])

    r = sesion.resumen(log, reps=0, duracion_seg=2.0)
    assert r["errores_principales"] == []
    assert r["frames_con_error"] == 0
    assert r["pct_frames_con_error"] == 0.0


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
