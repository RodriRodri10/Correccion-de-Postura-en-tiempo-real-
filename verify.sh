#!/usr/bin/env bash
# Arnes de verificacion headless del MVP. Sin camara ni GUI.
# Corre a mano o desde el Makefile:
#   bash verify.sh      (equivalente: make verify)
set -uo pipefail

cd "$(dirname "$0")" || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "== py_compile =="
"$PY" -m py_compile core/*.py *.py || { echo "FALLO: py_compile"; exit 1; }

echo "== smoke de imports =="
"$PY" -c "import core.geometria, core.pose, core.features, core.dinamica, core.senales, core.config, core.sesion, core.reps, core.catalogo, core.db" \
  || { echo "FALLO: imports"; exit 1; }

echo "== pytest =="
"$PY" -m pytest -q tests/ || { echo "FALLO: pytest"; exit 1; }

echo "VERDE: verificacion completa"
