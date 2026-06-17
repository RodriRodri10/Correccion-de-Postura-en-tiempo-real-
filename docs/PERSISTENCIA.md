# Persistencia (PostgreSQL + PostgREST)

Capa de persistencia **opcional y desacoplada** para guardar el resumen de cada sesion de
ejercicio en una base de datos, ademas del JSON local. Permite tener historial y comparar
el progreso entre sesiones. Todo corre contenerizado con Docker Compose.

> **Invariante de diseno:** el guardado local (`core/sesion.guardar` -> `sesiones/<ej>_<ts>.json`)
> es la **fuente primaria** y NO cambia. La base de datos es *best-effort*: si no esta
> levantada, la sesion de ejercicio termina normal y solo se omite el envio (sin error fatal).

La fuente de verdad del esquema es [`db/01_init.sql`](../db/01_init.sql). Este documento
explica el porque de las decisiones y como encajan las piezas.

## Decisiones de diseno

| Decision | Eleccion | Por que |
|----------|----------|---------|
| Granularidad | Resumen por sesion + errores principales | Es lo que ya calcula `core/sesion.resumen()`. No se guardan series por frame ni por rep: no aportan al objetivo (historial) y multiplicarian el almacenamiento. |
| Identidad de usuario | Usuario simple (`id` + `nombre`), **sin autenticacion** | El MVP es local y de un solo equipo; el usuario es solo una etiqueta para agrupar sesiones. Login/JWT quedan fuera de alcance. |
| Escritura | Un **unico POST** a una RPC (`crear_sesion`) | Inserta usuario + sesion + errores en **una sola transaccion**: o se guarda todo o nada. Evita estados a medias y multiples round-trips. |
| Puerta HTTP | PostgREST como unica via | El cliente (`core/db.py`) habla HTTP/JSON, no SQL. La app no necesita un driver de Postgres. |
| Schema | Dedicado `api` (no `public`) | PostgREST solo expone el schema configurado; `public` queda fuera del alcance de la API. |
| Acoplamiento | Best-effort, no fatal | La vision por computadora es el producto; la DB es un extra que no debe poder tumbar una sesion. |

## Arquitectura

```text
retroalimentacion_*.py
   |  (al salir con Esc, tras sesion.guardar -> JSON local)
   v
core/db.enviar_sesion(resumen, ejercicio)   --HTTP POST /rpc/crear_sesion-->  PostgREST  --SQL-->  PostgreSQL (schema api)
   ^                                                                              :3000                 db:5432
   |__ si PostgREST no responde: imprime aviso y retorna None (la sesion sigue)
```

Servicios definidos en [`docker-compose.yml`](../docker-compose.yml):

| Servicio | Imagen | Puerto host | Para que |
|----------|--------|-------------|----------|
| `db` | `postgres:16` | `5433` (config. con `DB_PORT`) | Base de datos. El puerto host es solo para inspeccion con `psql`. |
| `postgrest` | `postgrest/postgrest:v12.2.3` | `3000` | API REST (lectura y escritura). Conecta a la DB por la red interna (`db:5432`). |
| `swagger` | `swaggerapi/swagger-ui` | `8080` | UI opcional para explorar la API. |

> El puerto host de la DB es **5433** por defecto para no chocar con un Postgres local en
> 5432. PostgREST no usa ese puerto: conecta por la red interna de Docker.

## Modelo de datos

Cuatro tablas en el schema `api`:

```text
ejercicio (clave PK) <--- sesion (ejercicio FK) ---> usuario (id PK)
                              ^
                              | (sesion_id FK, ON DELETE CASCADE)
                          sesion_error
```

### `api.ejercicio` — catalogo (sembrado al iniciar)
| Columna | Tipo | Notas |
|---------|------|-------|
| `clave` | `text` PK | Misma clave que `core/catalogo.EJERCICIOS` (`pushup`, `dom_abierta`). |
| `nombre` | `text` | Nombre visible. |
| `vista` | `text` | Vista de camara requerida (`lateral`, `posterior`). |

Sembrado con los dos ejercicios del MVP (`ON CONFLICT DO NOTHING`, idempotente).

### `api.usuario`
| Columna | Tipo | Notas |
|---------|------|-------|
| `id` | `serial` PK | |
| `nombre` | `text` UNIQUE | Indice unico `usuario_nombre_uniq`; permite *upsert* por nombre. |
| `creado_en` | `timestamptz` | `DEFAULT now()`. |

### `api.sesion` — un registro por sesion completada
| Columna | Tipo | Origen (`resumen_dict`) |
|---------|------|--------------------------|
| `id` | `serial` PK | — |
| `usuario_id` | `int` FK -> `usuario.id` | — |
| `ejercicio` | `text` FK -> `ejercicio.clave` | argumento del RPC |
| `reps` | `int` | `reps` |
| `frames_evaluados` | `int` | `frames_evaluados` |
| `frames_correctos` | `int` | `frames_correctos` |
| `pct_correcto` | `real` | `pct_correcto` |
| `frames_con_error` | `int` | `frames_con_error` |
| `pct_frames_con_error` | `real` | `pct_frames_con_error` |
| `duracion_seg` | `real` | `duracion_seg` |
| `top_errores` | `jsonb` | `top_errores` (se guarda crudo, sin normalizar) |
| `creado_en` | `timestamptz` | `DEFAULT now()` |

### `api.sesion_error` — errores principales normalizados
Una fila por error principal de la sesion (`ON DELETE CASCADE` con la sesion).

| Columna | Tipo | Origen (cada item de `errores_principales`) |
|---------|------|----------------------------------------------|
| `id` | `serial` PK | — |
| `sesion_id` | `int` FK -> `sesion.id` | — |
| `mensaje` | `text` | `mensaje` |
| `eventos` | `int` | `eventos` |
| `frames` | `int` | `frames` |
| `segundos` | `real` | `segundos` |
| `pct_tiempo_evaluado` | `real` | `pct_tiempo_evaluado` |

## La RPC `crear_sesion`

PostgREST la expone como `POST /rpc/crear_sesion`. En una sola transaccion:

1. **Upsert del usuario** por `nombre` (lo crea si no existe, devuelve su `id`).
2. **Inserta la sesion** leyendo las claves del JSON `p_resumen`.
3. **Inserta cada error principal** recorriendo `p_resumen->'errores_principales'`.
4. Devuelve el `id` de la sesion creada.

Se declara `SECURITY DEFINER` con `search_path = api, pg_temp`, de modo que el rol `anon`
puede ejecutarla sin tener permisos directos amplios sobre las tablas.

```sql
-- firma (ver cuerpo completo en db/01_init.sql)
api.crear_sesion(p_usuario text, p_ejercicio text, p_resumen jsonb) RETURNS integer
```

### Contrato con `core/sesion.resumen()`
Las claves de `p_resumen` mapean **1:1** con lo que devuelve `core/sesion.resumen()`. Si se
agrega o renombra una metrica del resumen, hay que actualizar tambien la RPC y el esquema.

## Roles y seguridad

- `anon` — rol sin login que PostgREST usa para peticiones anonimas (`PGRST_DB_ANON_ROLE: anon`).
- `authenticator` — rol con login que PostgREST usa para conectarse; hereda `anon`.
- Grants a `anon`: `USAGE` sobre el schema, `SELECT` sobre todas las tablas, `INSERT` sobre
  `usuario`/`sesion`/`sesion_error`, `USAGE` sobre las secuencias (imprescindible para los
  `serial` al insertar) y `EXECUTE` sobre la RPC.

> **Sin autenticacion (a proposito).** Cualquiera con acceso al puerto 3000 puede leer y
> escribir. Es aceptable para un MVP local; **no** exponer PostgREST a una red no confiable
> sin agregar autenticacion (JWT) y endurecer los grants.

## Cliente: `core/db.py`

```python
db.enviar_sesion(resumen_dict, ejercicio, usuario=None)  # POST /rpc/crear_sesion
```

- Se llama en `retroalimentacion_wall_pushup.py` y `retroalimentacion_dominada_abierta.py`
  **despues** de `sesion.guardar`.
- Devuelve el `id` de la sesion si todo fue bien, o `None` ante cualquier `RequestException`
  (DB caida, timeout, etc.) imprimiendo un aviso — nunca lanza.
- Variables de entorno:
  - `TT1_DB_URL` — URL de PostgREST (por defecto `http://localhost:3000`).
  - `TT1_USUARIO` — nombre del usuario asociado (por defecto `demo`).

## Operacion (referencia rapida)

```bash
# Levantar la base de datos y la API (crea .env desde .env.example si falta)
make db-up                       # equivale a: docker compose up -d db postgrest

# Verificar que responde (catalogo sembrado)
curl -s http://localhost:3000/ejercicio

# Consultar sesiones con sus errores principales anidados
curl -s "http://localhost:3000/sesion?select=*,sesion_error(*)"

# Inspeccionar la DB directamente
psql -h localhost -p 5433 -U postgres -d tt1

make db-down                     # detener (conserva el volumen de datos)
docker compose down -v           # ademas borra el volumen (reinicia la DB desde cero)
```

El script `db/01_init.sql` solo se ejecuta **una vez**, cuando el volumen de datos esta
vacio (convencion de `/docker-entrypoint-initdb.d/` de la imagen de Postgres). Para
re-aplicar cambios al esquema durante el desarrollo hay que recrear el volumen
(`docker compose down -v`).

## Limitaciones y mejoras futuras

- Sin autenticacion ni multiusuario real (ver arriba).
- Granularidad de resumen: no hay series temporales por frame ni por repeticion.
- No hay migraciones; el esquema vive en un unico `01_init.sql` que corre en una DB vacia.
- Posible siguiente paso: una vista de historial en la app que lea de PostgREST y grafique
  el progreso por ejercicio a lo largo del tiempo.
