-- Brover spatial memory schema.
--
-- Single source of truth for the database structure. backend/db/connection.py
-- reads and applies this file on every startup; every statement is
-- `CREATE ... IF NOT EXISTS` so re-runs are safe no-ops.
--
-- Vector columns are FLOAT[1024] to match Voyage AI's voyage-multimodal-3
-- embedding model. If we ever switch embedding models, drop the *_vectors
-- virtual tables (and re-embed any stored frames) before relaunching.
--
-- Vector tables use cosine distance: smaller distance = more similar.
-- Range is 0 (identical) to 2 (opposite); similarity = 1 - distance.

-- =============================================================================
-- Places (Phase 2): named locations Brover can recognize.
-- =============================================================================

CREATE TABLE IF NOT EXISTS places (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    created_at  REAL    NOT NULL
);

-- One row per captured frame for a place. The matching embedding lives in
-- place_view_vectors with the same id (rowid). add_place_view writes both
-- inside one transaction so they cannot drift apart.
CREATE TABLE IF NOT EXISTS place_views (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id     INTEGER NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    image_path   TEXT    NOT NULL,
    heading_deg  REAL,
    distance_cm  REAL,
    captured_at  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_place_views_place_id
    ON place_views(place_id);

CREATE VIRTUAL TABLE IF NOT EXISTS place_view_vectors USING vec0(
    embedding FLOAT[1024] distance_metric=cosine
);


-- =============================================================================
-- Routes (Phase 4 — tables only, code lands later): traversals between places.
-- =============================================================================

CREATE TABLE IF NOT EXISTS routes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    from_place_id   INTEGER NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    to_place_id     INTEGER NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    created_at      REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_routes_from_to
    ON routes(from_place_id, to_place_id);

-- One frame + motor action recorded along a route. `seq` orders the steps;
-- `action` is a compact string like "forward:0.5" or "turn_left:0.3".
CREATE TABLE IF NOT EXISTS route_steps (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    route_id     INTEGER NOT NULL REFERENCES routes(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    image_path   TEXT    NOT NULL,
    action       TEXT    NOT NULL,
    distance_cm  REAL,
    captured_at  REAL    NOT NULL,
    UNIQUE(route_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_route_steps_route_id
    ON route_steps(route_id);

CREATE VIRTUAL TABLE IF NOT EXISTS route_step_vectors USING vec0(
    embedding FLOAT[1024] distance_metric=cosine
);


-- =============================================================================
-- People (Phase 7 — tables only): face memory. Local-only, opt-in.
-- =============================================================================

CREATE TABLE IF NOT EXISTS people (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    created_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS face_views (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id    INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    image_path   TEXT    NOT NULL,
    captured_at  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_face_views_person_id
    ON face_views(person_id);

CREATE VIRTUAL TABLE IF NOT EXISTS face_view_vectors USING vec0(
    embedding FLOAT[1024] distance_metric=cosine
);
