"""Configuracion de pytest: deja la raiz del repo en sys.path y utilidades.

Permite ``from core import ...`` al correr pytest desde la raiz, igual que los
scripts (que resuelven core/ porque la raiz queda en sys.path[0]).
"""
import os
import sys
from types import SimpleNamespace

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)


@pytest.fixture
def fake_landmarks():
    """33 landmarks sinteticos (atributos .x/.y normalizados en [0,1]).

    Valores arbitrarios pero deterministas: solo se usan para comprobar la forma
    del vector de features, no su semantica.
    """
    lm = []
    for i in range(33):
        x = 0.3 + (i % 5) * 0.08
        y = 0.2 + (i % 7) * 0.09
        lm.append(SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0))
    return lm
