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
GRANT USAGE ON ALL SEQUENCES IN SCHEMA api TO anon;
GRANT EXECUTE ON FUNCTION api.crear_sesion(text, text, jsonb) TO anon;
