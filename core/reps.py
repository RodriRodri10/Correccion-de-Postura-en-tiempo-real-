"""Maquinas de estado para el conteo de repeticiones.

Hoy esta logica vive incrustada en los scripts de retroalimentacion (la de wall
push-up como funciones ``init_fsm``/``actualizar_fsm`` en
``retroalimentacion_wall_pushup.py``; la de dominada abierta inline en el
``main()`` de ``retroalimentacion_dominada_abierta.py``). Aqui se extrae a clases
puras y testeables que reciben la fase por frame y exponen ``.reps``, sin
depender de OpenCV ni de la camara.

Lo implementa Ralph para pasar tests/test_reps.py preservando EXACTAMENTE el
comportamiento de conteo actual. NO cambiar las firmas ni los umbrales sin
actualizar tests/specs.
"""


class FsmWallPushup:
    """Conteo de reps de wall push-up (fases 1..4).

    Reproduce la maquina WAIT_START -> IN_REP -> LOCKED de
    ``retroalimentacion_wall_pushup.py`` (MIN_FRAMES_REP=10, RESET_FRAMES=8,
    arranque tras 3 frames en fase 1).
    """

    def __init__(self):
        raise NotImplementedError

    def update(self, fase):
        """Procesa una fase (int) de un frame. Devuelve self.reps."""
        raise NotImplementedError

    @property
    def reps(self):
        raise NotImplementedError


class FsmDominadaAbierta:
    """Conteo de reps de dominada abierta (fases 1=Arriba, 2=Transicion, 3=Abajo).

    Reproduce la maquina ABAJO -> SUBE -> ARRIBA -> BAJA -> ABAJO de
    ``retroalimentacion_dominada_abierta.py`` (una rep al volver a ABAJO tras
    ``frames_estables`` frames consecutivos en fase 3).
    """

    def __init__(self, frames_estables=5):
        raise NotImplementedError

    def update(self, fase):
        """Procesa una fase (int) de un frame. Devuelve self.reps."""
        raise NotImplementedError

    @property
    def reps(self):
        raise NotImplementedError
