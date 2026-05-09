"""Open the SQLite database, load sqlite-vec, apply the schema.

The DB lives at `data/brover.db` on the device's SD card. The directory
is created on first run; the file is auto-created by SQLite when we
connect to a missing path. The schema in `schema.sql` is re-applied
on every connect via `executescript()` -- every CREATE in that file is
`IF NOT EXISTS`, so first-run creates everything and subsequent runs
are no-ops.

Why no migrations folder yet: at v1 the schema is small and only adds
columns/tables. When a breaking change becomes necessary we'll introduce
a `schema_version` table and a numbered migrations folder. Until then,
`schema.sql` + idempotent CREATEs is enough.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import sqlite_vec

logger = logging.getLogger(__name__)

# Resolve project paths relative to this file. Walking up three parents from
# `backend/db/connection.py` lands at the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = _PROJECT_ROOT / "data"
DB_PATH: Path = DATA_DIR / "brover.db"
CAPTURES_DIR: Path = DATA_DIR / "captures"
SCHEMA_PATH: Path = Path(__file__).resolve().parent / "schema.sql"

# Must match the FLOAT[N] declarations in schema.sql. Voyage's
# voyage-multimodal-3 model returns 1024-dim vectors.
EMBEDDING_DIM: int = 1024


def _ensure_dirs() -> None:
    """Create data/ and data/captures/ on first use; safe to re-run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    """Return a ready-to-use SQLite connection.

    On first call this creates the directory tree, the .db file, and every
    table in schema.sql. Subsequent calls are essentially free.

    Callers own the connection's lifetime and should close it when done.
    """
    _ensure_dirs()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # FK cascades only fire when this is on; off by default in SQLite.
    conn.execute("PRAGMA foreign_keys = ON")

    # sqlite-vec is loaded as an extension. Toggle the load flag around the
    # call so we don't leave the connection accepting arbitrary extensions.
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    conn.commit()

    logger.debug("connected to %s with schema applied", DB_PATH)
    return conn
