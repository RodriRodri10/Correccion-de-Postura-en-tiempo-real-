#!/usr/bin/env bash
# Inicializacion rapida del proyecto (Linux / macOS, y Git Bash o WSL en Windows).
#
# Uso:
#   ./init.sh            Crea el entorno virtual (.venv) e instala dependencias.
#   ./init.sh --docker   Ademas levanta la base de datos (PostgreSQL + PostgREST).
#
# Variable opcional: PYTHON=/ruta/a/python3.11 ./init.sh
set -euo pipefail
cd "$(dirname "$0")"

# 1) Localizar Python 3.11 (preferido por mediapipe==0.10.14).
PY="${PYTHON:-python3.11}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"
command -v "$PY" >/dev/null 2>&1 || {
  echo "ERROR: no se encontro Python. Instala Python 3.11 o exporta PYTHON=/ruta/a/python3.11" >&2
  exit 1
}

# 2) Entorno virtual.
echo "== Entorno virtual (.venv) con $PY =="
[ -d .venv ] || "$PY" -m venv .venv

# 3) Dependencias.
echo "== Instalando dependencias =="
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

# 4) Base de datos opcional (Docker).
if [ "${1:-}" = "--docker" ]; then
  echo "== Levantando base de datos (Docker) =="
  command -v docker >/dev/null 2>&1 || { echo "ERROR: Docker no esta instalado." >&2; exit 1; }
  [ -f .env ] || cp .env.example .env
  docker compose up -d db postgrest
fi

echo ""
echo "Listo. Siguientes pasos:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py"
if [ "${1:-}" != "--docker" ]; then
  echo ""
  echo "Persistencia opcional (base de datos): docker compose up -d db postgrest"
fi
