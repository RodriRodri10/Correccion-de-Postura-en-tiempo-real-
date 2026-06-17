"""Contrato de conteo de repeticiones (TDD: rojo -> verde).

Alimenta secuencias de fases sinteticas que simulan N repeticiones y exige que
las FSM extraidas a core/reps.py cuenten exactamente N, preservando la logica
actual de los scripts de retroalimentacion.
"""
from core.reps import FsmWallPushup, FsmDominadaAbierta


# --- Wall push-up: WAIT_START(3x fase1) -> IN_REP(>=10 frames, visita 1..4) -> LOCKED ---
def _rep_pushup():
    # 3 frames en fase 1 para arrancar, 9 frames recorriendo 2-3-4, y vuelta a 1.
    return [1, 1, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 1]


def test_pushup_una_rep():
    fsm = FsmWallPushup()
    for f in _rep_pushup():
        fsm.update(f)
    assert fsm.reps == 1


def test_pushup_dos_reps():
    fsm = FsmWallPushup()
    secuencia = _rep_pushup() + [1] * 8 + _rep_pushup()  # 8x fase1 resetea LOCKED->WAIT
    for f in secuencia:
        fsm.update(f)
    assert fsm.reps == 2


# --- Dominada abierta: ABAJO -> SUBE(2) -> ARRIBA(5x fase1) -> BAJA(2) -> ABAJO(5x fase3) ---
def _rep_dominada():
    return [2, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3]


def test_dominada_una_rep():
    fsm = FsmDominadaAbierta()
    for f in _rep_dominada():
        fsm.update(f)
    assert fsm.reps == 1


def test_dominada_dos_reps():
    fsm = FsmDominadaAbierta()
    for f in _rep_dominada() * 2:
        fsm.update(f)
    assert fsm.reps == 2
