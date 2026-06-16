#!/usr/bin/env bash
# Loop acotado de Ralph.
#
# Uso:  bash ralph/ralph.sh [N_ITERACIONES=8] [MODELO=claude-sonnet-4-6]
#
# Cada iteracion: corre Claude en headless leyendo ralph/PROMPT.md, luego
# ralph/verify.sh; commitea SOLO si verify pasa. Corta al ver "STATUS: DONE"
# en ralph/PROGRESS.md o al agotar las iteraciones.
#
# Modelo recomendado para el loop: claude-sonnet-4-6 (mejor costo/calidad).
# Alternativa economica: claude-haiku-4-5. Opus se usa para planear, no aqui.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

N="${1:-8}"
MODELO="${2:-claude-sonnet-4-6}"
PROMPT="ralph/PROMPT.md"

command -v claude >/dev/null 2>&1 || { echo "ERROR: la CLI 'claude' no esta en PATH."; exit 1; }

for i in $(seq 1 "$N"); do
  echo "================ Ralph iteracion $i/$N (modelo: $MODELO) ================"

  if grep -q "^STATUS: DONE" ralph/PROGRESS.md; then
    echo "PRD completo (STATUS: DONE). Fin."
    exit 0
  fi

  cat "$PROMPT" | claude -p --model "$MODELO" --dangerously-skip-permissions

  if bash ralph/verify.sh; then
    echo ">> verify VERDE -> commit"
    git add -A
    git commit -m "ralph: iteracion $i" >/dev/null 2>&1 || echo "   (nada que commitear)"
  else
    echo ">> verify ROJO -> sin commit; la siguiente iteracion debe arreglarlo"
  fi
done

if grep -q "^STATUS: DONE" ralph/PROGRESS.md; then
  echo "PRD completo (STATUS: DONE)."
else
  echo "Iteraciones agotadas ($N). Revisa ralph/PROGRESS.md y el ultimo verify."
fi
