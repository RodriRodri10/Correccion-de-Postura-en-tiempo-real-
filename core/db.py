"""Cliente HTTP minimo a PostgREST. NO fatal: si la DB no esta, retorna None
sin lanzar. El JSON local (core.sesion.guardar) sigue siendo la fuente primaria."""
import os
import requests

BASE_URL = os.environ.get("TT1_DB_URL", "http://localhost:3000")
USUARIO_DEFAULT = os.environ.get("TT1_USUARIO", "demo")
TIMEOUT = 3


def enviar_sesion(resumen_dict, ejercicio, usuario=None):
    payload = {"p_usuario": usuario or USUARIO_DEFAULT,
               "p_ejercicio": ejercicio, "p_resumen": resumen_dict}
    try:
        resp = requests.post(f"{BASE_URL}/rpc/crear_sesion", json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"[db] no se pudo enviar la sesion a PostgREST: {exc}")
        return None
