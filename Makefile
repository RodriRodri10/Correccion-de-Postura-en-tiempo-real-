# Makefile — atajos para el MVP "Corrector de Postura en Tiempo Real".
# Uso: make <target>. Sin argumentos muestra esta ayuda.
# Pensado para Linux / macOS / WSL / Git Bash. En Windows cmd.exe usa init.bat.

PY := .venv/bin/python

.DEFAULT_GOAL := help

.PHONY: help setup run verify test db-up db-down db-logs clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Crea .venv (Python 3.11) e instala dependencias
	./init.sh

run:  ## Lanza la interfaz Streamlit del MVP
	$(PY) -m streamlit run app.py

verify:  ## Arnes headless: py_compile + smoke de imports + pytest
	bash verify.sh

test:  ## Corre solo la bateria de pruebas (pytest)
	$(PY) -m pytest -q tests/

db-up:  ## Levanta la base de datos opcional (Postgres + PostgREST)
	@[ -f .env ] || cp .env.example .env
	docker compose up -d db postgrest

db-down:  ## Detiene la base de datos (conserva el volumen de datos)
	docker compose down

db-logs:  ## Sigue los logs de la base de datos y la API
	docker compose logs -f db postgrest

clean:  ## Borra caches de Python y pytest
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
