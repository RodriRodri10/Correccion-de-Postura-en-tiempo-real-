# Spec 08 — Persistencia con PostgreSQL + PostgREST (contenerizado)

Añadir una capa de persistencia **opcional y desacoplada**: PostgreSQL 16 + PostgREST
como única puerta HTTP (lectura y escritura), todo con Docker Compose. Al terminar una
sesion, la app envia el `resumen_dict` (el mismo que hoy va a `sesiones/<ej>_<ts>.json`)
con un **unico POST** a una funcion RPC que inserta sesion + errores en una transaccion.

**Invariante:** el guardado JSON actual (`core/sesion.guardar`) NO cambia. La DB es
best-effort: si PostgREST no esta arriba, la sesion de ejercicio termina normal igual.

Esto es una extension Fase 2 (persistencia/historial) sobre el MVP de `ralph/PRD.md`.
Decisiones del producto: **usuarios simples** (id + nombre, SIN auth), **granularidad =
resumen por sesion + errores principales** (sin series por frame ni por rep), **escritura
via POST a PostgREST**.

Este spec cubre dos items de `ralph/PROGRESS.md` (Parte A: infra; Parte B: cliente).

---

## Parte A — Infraestructura contenerizada

Archivos nuevos: `db/01_init.sql`, `docker-compose.yml`, `.env.example`,
`.env` (NO versionar). Añadir `.env` a `.gitignore`.

### `db/01_init.sql`

Un solo archivo (los scripts de `/docker-entrypoint-initdb.d/` corren UNA vez, solo si el
volumen esta vacio, en orden alfabetico; un archivo unico evita ambiguedad). Orden interno
obligatorio: **roles → schema → tablas → seed → RPC → grants**. Schema dedicado `api`
(no exponer `public`).

```sql
-- ROLES (antes de los GRANTs)
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'anon') THEN
    CREATE ROLE anon NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'authenticator') THEN
    CREATE ROLE authenticator NOINHERIT LOGIN PASSWORD 'authpass';
  END IF;
END $$;
GRANT anon TO authenticator;

CREATE SCHEMA IF NOT EXISTS api;

CREATE TABLE IF NOT EXISTS api.ejercicio (
    clave  text PRIMARY KEY,
    nombre text NOT NULL,
    vista  text NOT NULL
);

CREATE TABLE IF NOT EXISTS api.usuario (
    id        serial PRIMARY KEY,
    nombre    text NOT NULL,
    creado_en timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS usuario_nombre_uniq ON api.usuario (nombre);

CREATE TABLE IF NOT EXISTS api.sesion (
    id                   serial PRIMARY KEY,
    usuario_id           integer NOT NULL REFERENCES api.usuario (id),
    ejercicio            text    NOT NULL REFERENCES api.ejercicio (clave),
    reps                 integer NOT NULL,
    frames_evaluados     integer NOT NULL,
    frames_correctos     integer NOT NULL,
    pct_correcto         real    NOT NULL,
    frames_con_error     integer NOT NULL,
    pct_frames_con_error real    NOT NULL,
    duracion_seg         real    NOT NULL,
    top_errores          jsonb,
    creado_en            timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api.sesion_error (
    id                  serial PRIMARY KEY,
    sesion_id           integer NOT NULL REFERENCES api.sesion (id) ON DELETE CASCADE,
    mensaje             text    NOT NULL,
    eventos             integer NOT NULL,
    frames              integer NOT NULL,
    segundos            real    NOT NULL,
    pct_tiempo_evaluado real    NOT NULL
);
CREATE INDEX IF NOT EXISTS sesion_error_sesion_idx ON api.sesion_error (sesion_id);

-- SEED catalogo (mismas claves que core/catalogo.EJERCICIOS)
INSERT INTO api.ejercicio (clave, nombre, vista) VALUES
    ('pushup',      'Wall push-up',            'lateral'),
    ('dom_abierta', 'Dominada agarre abierto', 'posterior')
ON CONFLICT (clave) DO NOTHING;

-- RPC: una llamada, una transaccion. PostgREST la expone en POST /rpc/crear_sesion
CREATE OR REPLACE FUNCTION api.crear_sesion(
    p_usuario text, p_ejercicio text, p_resumen jsonb
) RETURNS integer
LANGUAGE plpgsql SECURITY DEFINER SET search_path = api, pg_temp
AS $$
DECLARE v_usuario_id integer; v_sesion_id integer; v_err jsonb;
BEGIN
    INSERT INTO api.usuario (nombre) VALUES (p_usuario)
      ON CONFLICT (nombre) DO UPDATE SET nombre = EXCLUDED.nombre
      RETURNING id INTO v_usuario_id;

    INSERT INTO api.sesion (
        usuario_id, ejercicio, reps, frames_evaluados, frames_correctos,
        pct_correcto, frames_con_error, pct_frames_con_error, duracion_seg, top_errores
    ) VALUES (
        v_usuario_id, p_ejercicio,
        (p_resumen->>'reps')::int,
        (p_resumen->>'frames_evaluados')::int,
        (p_resumen->>'frames_correctos')::int,
        (p_resumen->>'pct_correcto')::real,
        (p_resumen->>'frames_con_error')::int,
        (p_resumen->>'pct_frames_con_error')::real,
        (p_resumen->>'duracion_seg')::real,
        p_resumen->'top_errores'
    ) RETURNING id INTO v_sesion_id;

    FOR v_err IN SELECT * FROM jsonb_array_elements(p_resumen->'errores_principales') LOOP
        INSERT INTO api.sesion_error (sesion_id, mensaje, eventos, frames, segundos, pct_tiempo_evaluado)
        VALUES (v_sesion_id, v_err->>'mensaje', (v_err->>'eventos')::int,
                (v_err->>'frames')::int, (v_err->>'segundos')::real,
                (v_err->>'pct_tiempo_evaluado')::real);
    END LOOP;

    RETURN v_sesion_id;
END $$;

-- GRANTs (al final)
GRANT USAGE ON SCHEMA api TO anon;
GRANT SELECT ON ALL TABLES IN SCHEMA api TO anon;
GRANT INSERT ON api.usuario, api.sesion, api.sesion_error TO anon;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA api TO anon;   -- imprescindible para serial/INSERT
GRANT EXECUTE ON FUNCTION api.crear_sesion(text, text, jsonb) TO anon;
```

Las claves del `p_resumen` mapean **1:1** con lo que devuelve `core/sesion.resumen()`
(ver `ralph/specs/03-sesion-resumen.md`). `top_errores` se guarda como `jsonb` (round-trip
sin normalizar); `errores_principales` se normaliza a `api.sesion_error`.

### `docker-compose.yml` (raiz del repo)

```yaml
services:
  db:
    image: postgres:16
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
      POSTGRES_DB: ${POSTGRES_DB:-tt1}
    volumes:
      - db_data:/var/lib/postgresql/data
      - ./db:/docker-entrypoint-initdb.d:ro
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-tt1}"]
      interval: 5s
      timeout: 5s
      retries: 10
  postgrest:
    image: postgrest/postgrest:v12.2.3
    restart: unless-stopped
    depends_on:
      db: { condition: service_healthy }
    environment:
      PGRST_DB_URI: postgres://authenticator:${AUTH_PASSWORD:-authpass}@db:5432/${POSTGRES_DB:-tt1}
      PGRST_DB_SCHEMAS: api
      PGRST_DB_ANON_ROLE: anon
      PGRST_SERVER_PORT: 3000
    ports: ["3000:3000"]
  swagger:                       # opcional
    image: swaggerapi/swagger-ui
    depends_on: [postgrest]
    environment: { API_URL: "http://localhost:3000/" }
    ports: ["8080:8080"]
volumes:
  db_data:
```

`.env.example` (versionado): `POSTGRES_USER=postgres`, `POSTGRES_PASSWORD=postgres`,
`POSTGRES_DB=tt1`, `AUTH_PASSWORD=authpass`. El password de `authenticator` en el SQL
DEBE coincidir con `AUTH_PASSWORD`.

### Verificacion Parte A (manual, no en verify.sh — no hay docker en el gate)

```bash
docker compose up -d
docker compose ps                          # db healthy, postgrest up
curl -s http://localhost:3000/ejercicio    # seed: pushup + dom_abierta
```

---

## Parte B — Cliente Python e integracion

Archivos: `core/db.py` (nuevo), edicion minima de los 2 scripts de retroalimentacion,
`requirements.txt` (+ requests), `ralph/verify.sh` (+ core.db), `tests/test_db.py` (nuevo).

### `core/db.py`

Cliente HTTP minimo, **best-effort y nunca fatal**. Solo importa `os` + `requests`
(sin camara/OpenCV) para entrar en el smoke headless. URL por entorno con default local.

```python
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
```

### Integracion en los scripts (1 linea cada uno)

Justo DESPUES del `sesion.guardar(...)` existente (no antes, para que el JSON quede aunque
falle la red), añadir `db.enviar_sesion(r, "<clave>")`:

- `retroalimentacion_wall_pushup.py`: importar `db` (`from core import config, sesion, db`
  o `from core import db`) y, tras `sesion.guardar(r, "pushup", ...)`, llamar
  `db.enviar_sesion(r, "pushup")`.
- `retroalimentacion_dominada_abierta.py`: igual con clave `"dom_abierta"`.

No cambiar nada mas de esos scripts (overlay, FSM, umbrales intactos).

### `requirements.txt`

Añadir, con comentario:
```
requests             # cliente HTTP a PostgREST (core/db.py); persistencia opcional
```
Instalar en `.venv` para que el import de `core.db` pase en el gate.

### `ralph/verify.sh`

Añadir `core.db` al smoke de imports (la linea del `-c "import core...."`):
`..., core.catalogo, core.db`.

### `tests/test_db.py`

Test puro con `requests` mockeado (monkeypatch), sin red real:
- `enviar_sesion` devuelve `None` cuando `requests.post` lanza `requests.RequestException`
  (no propaga la excepcion).
- `enviar_sesion` arma el payload con claves `p_usuario`, `p_ejercicio`, `p_resumen` y
  pega a `f"{BASE_URL}/rpc/crear_sesion"` (verificar via un fake que captura args).
- usuario por defecto = `"demo"` cuando no se pasa `usuario`.

### Verificacion Parte B (gate)

`bash ralph/verify.sh` en VERDE: `py_compile`, import de `core.db`, y `tests/` (incluye
`test_db.py`). Requiere `requests` instalado en `.venv`.

---

## Verificacion end-to-end (manual, requiere docker)

```bash
docker compose up -d
curl -s -X POST http://localhost:3000/rpc/crear_sesion -H "Content-Type: application/json" \
  -d '{"p_usuario":"demo","p_ejercicio":"pushup","p_resumen":{"reps":5,"frames_evaluados":323,
       "frames_correctos":4,"pct_correcto":1.24,"frames_con_error":319,"pct_frames_con_error":98.76,
       "duracion_seg":14.41,"top_errores":[["Codo muy bajo",185]],
       "errores_principales":[{"mensaje":"Codo muy bajo","eventos":7,"frames":181,
       "segundos":7.93,"pct_tiempo_evaluado":56.04}]}}'           # -> id de sesion
curl -s "http://localhost:3000/sesion?select=*,sesion_error(*)"   # lectura con embedding
docker compose restart db postgrest && curl -s http://localhost:3000/sesion  # persiste
```

Prueba de no-fatalidad: con `docker compose stop postgrest`, correr un script de
retroalimentacion termina normal, escribe el JSON, e imprime `[db] no se pudo enviar...`
sin traceback.

---

## Gotchas (no negociables)

1. Init SQL solo corre con volumen vacio → re-aplicar DDL exige `docker compose down -v` (borra datos).
2. Orden en el SQL: roles → schema → tablas → seed → RPC → grants.
3. `GRANT USAGE ON ALL SEQUENCES ... TO anon` es imprescindible ("permission denied for sequence" si falta).
4. `PGRST_DB_ANON_ROLE=anon` definido o todo POST/GET sin token da 401/403.
5. `PGRST_DB_SCHEMAS=api` para no exponer `public`. La RPC queda en `/rpc/crear_sesion`.
6. No hace falta `NOTIFY pgrst,'reload schema'` en el primer arranque (PostgREST conecta tras el healthcheck).
7. Password de `authenticator` coincide entre SQL y `PGRST_DB_URI`/`.env`.
8. `.env` a `.gitignore`; `db/`, `docker-compose.yml`, `.env.example` SI se versionan. Nombres ASCII en todo.
9. `db.enviar_sesion` con timeout corto + try/except total: persistencia best-effort, JSON = fuente de verdad.
10. NO romper el contrato de features/scaler ni tocar `.pkl`/`.npy` (regla global del repo).
