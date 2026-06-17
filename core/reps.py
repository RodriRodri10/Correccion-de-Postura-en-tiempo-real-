"""Maquinas de estado para el conteo de repeticiones.

Hoy esta logica vive incrustada en los scripts de retroalimentacion (la de wall
push-up como funciones ``init_fsm``/``actualizar_fsm`` en
``retroalimentacion_wall_pushup.py``; la de dominada abierta inline en el
``main()`` de ``retroalimentacion_dominada_abierta.py``). Aqui se extrae a clases
puras y testeables que reciben la fase por frame y exponen ``.reps``, sin
depender de OpenCV ni de la camara.

Cubierto por tests/test_reps.py; preserva EXACTAMENTE el comportamiento de conteo
actual. NO cambiar las firmas ni los umbrales sin actualizar los tests.
"""


class FsmWallPushup:
    """Conteo de reps de wall push-up (fases 1..4).

    Reproduce la maquina WAIT_START -> IN_REP -> LOCKED de
    ``retroalimentacion_wall_pushup.py`` (MIN_FRAMES_REP=10, RESET_FRAMES=8,
    arranque tras 3 frames en fase 1).
    """

    def __init__(self):
        self.estado = "WAIT_START"
        self.frames_fase1 = 0
        self.frames_rep = 0
        self.visitadas = set()
        self._reps = 0

    def update(self, fase):
        """Procesa una fase (int) de un frame. Devuelve self.reps."""
        MIN_FRAMES_REP = 10
        RESET_FRAMES = 8

        if fase not in (1, 2, 3, 4):
            return self._reps

        if self.estado == "WAIT_START":
            if fase == 1:
                self.frames_fase1 += 1
                if self.frames_fase1 >= 3:
                    self.estado = "IN_REP"
                    self.frames_rep = 0
                    self.visitadas = {1}
            else:
                self.frames_fase1 = 0

        elif self.estado == "IN_REP":
            self.frames_rep += 1
            self.visitadas.add(fase)

            if self.visitadas >= {1, 2, 3, 4} and fase == 1 and self.frames_rep >= MIN_FRAMES_REP:
                self._reps += 1
                self.estado = "LOCKED"
                self.frames_fase1 = 0
            elif fase == 1 and self.frames_rep >= MIN_FRAMES_REP:
                if 3 in self.visitadas or 4 in self.visitadas:
                    self._reps += 1
                    self.estado = "LOCKED"
                    self.frames_fase1 = 0

        elif self.estado == "LOCKED":
            if fase == 1:
                self.frames_fase1 += 1
                if self.frames_fase1 >= RESET_FRAMES:
                    self.estado = "WAIT_START"
                    self.frames_fase1 = 0
            else:
                self.frames_fase1 = 0

        return self._reps

    @property
    def reps(self):
        return self._reps


class FsmDominadaAbierta:
    """Conteo de reps de dominada abierta (fases 1=Arriba, 2=Transicion, 3=Abajo).

    Reproduce la maquina ABAJO -> SUBE -> ARRIBA -> BAJA -> ABAJO de
    ``retroalimentacion_dominada_abierta.py`` (una rep al volver a ABAJO tras
    ``frames_estables`` frames consecutivos en fase 3).
    """

    def __init__(self, frames_estables=5):
        self.frames_estables = frames_estables
        self.estado = 0  # ABAJO
        self.cont_arriba = 0
        self.cont_abajo = 0
        self._reps = 0

    def update(self, fase):
        """Procesa una fase (int) de un frame. Devuelve self.reps."""
        ABAJO, SUBE, ARRIBA, BAJA = 0, 1, 2, 3

        if self.estado == ABAJO and fase == 2:
            self.estado = SUBE

        elif self.estado == SUBE and fase == 1:
            self.cont_arriba += 1
            if self.cont_arriba >= self.frames_estables:
                self.estado = ARRIBA
                self.cont_arriba = 0

        elif self.estado == ARRIBA and fase == 2:
            self.estado = BAJA

        elif self.estado == BAJA and fase == 3:
            self.cont_abajo += 1
            if self.cont_abajo >= self.frames_estables:
                self._reps += 1
                self.estado = ABAJO
                self.cont_abajo = 0

        return self._reps

    @property
    def reps(self):
        return self._reps
