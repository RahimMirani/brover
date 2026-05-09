"""Place CRUD: named locations and their captured frames.

Phase 2 of the training pipeline. The four functions exercised here form
the read/write surface used by the upcoming teaching UI and the `localize`
tool the LLM will call:

  add_place / get_or_create_place      register a location by name
  list_places                          enumerate known places
  add_place_view                       store one (frame, embedding) sample
  find_nearest_place_views             cosine-nearest lookup for localization

Critical invariant: every row in `place_views` has exactly one matching row
in `place_view_vectors` with the same id. `add_place_view` writes both
inside a single transaction so the two cannot drift; if either insert
fails, the whole pair is rolled back.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Sequence

import sqlite_vec

from backend.db.connection import EMBEDDING_DIM


@dataclass(frozen=True)
class Place:
    id: int
    name: str
    created_at: float


@dataclass(frozen=True)
class NearestPlaceView:
    """One result from `find_nearest_place_views`.

    `distance` is cosine distance from sqlite-vec: 0 means identical, 2 means
    opposite. Convert to a similarity score with `1 - distance` if needed.
    """

    view_id: int
    place_id: int
    place_name: str
    image_path: str
    heading_deg: float | None
    distance_cm: float | None
    distance: float


def _pack(embedding: Sequence[float]) -> bytes:
    """Validate length, then pack into the byte layout sqlite-vec expects."""
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"embedding has {len(embedding)} dims, expected {EMBEDDING_DIM}"
        )
    return sqlite_vec.serialize_float32(list(embedding))


def add_place(conn: Connection, name: str) -> int:
    """Insert a new place and return its id. Raises if the name is taken."""
    cursor = conn.execute(
        "INSERT INTO places (name, created_at) VALUES (?, ?)",
        (name, time.time()),
    )
    conn.commit()
    return int(cursor.lastrowid)


def get_or_create_place(conn: Connection, name: str) -> int:
    """Return the existing id for `name`, or create one. Idempotent."""
    row = conn.execute(
        "SELECT id FROM places WHERE name = ?", (name,)
    ).fetchone()
    if row is not None:
        return int(row["id"])
    return add_place(conn, name)


def list_places(conn: Connection) -> list[Place]:
    rows = conn.execute(
        "SELECT id, name, created_at FROM places ORDER BY name"
    ).fetchall()
    return [
        Place(id=int(r["id"]), name=r["name"], created_at=float(r["created_at"]))
        for r in rows
    ]


def add_place_view(
    conn: Connection,
    place_id: int,
    image_path: str,
    embedding: Sequence[float],
    heading_deg: float | None = None,
    distance_cm: float | None = None,
) -> int:
    """Store one frame's metadata + its embedding atomically.

    The matching `place_view_vectors` row is keyed by the same id as the
    `place_views` row, so JOINs in `find_nearest_place_views` work cleanly.
    """
    blob = _pack(embedding)
    captured_at = time.time()

    try:
        conn.execute("BEGIN")
        cursor = conn.execute(
            """
            INSERT INTO place_views
                (place_id, image_path, heading_deg, distance_cm, captured_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (place_id, image_path, heading_deg, distance_cm, captured_at),
        )
        view_id = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO place_view_vectors (rowid, embedding) VALUES (?, ?)",
            (view_id, blob),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return view_id


def find_nearest_place_views(
    conn: Connection,
    embedding: Sequence[float],
    k: int = 5,
) -> list[NearestPlaceView]:
    """Return up to `k` views whose embeddings are closest to the query.

    Sorted by cosine distance ascending (most similar first).
    """
    blob = _pack(embedding)
    rows = conn.execute(
        """
        SELECT v.rowid       AS view_id,
               v.distance    AS distance,
               pv.place_id   AS place_id,
               pv.image_path AS image_path,
               pv.heading_deg AS heading_deg,
               pv.distance_cm AS distance_cm,
               p.name        AS place_name
          FROM place_view_vectors v
          JOIN place_views pv ON pv.id = v.rowid
          JOIN places     p  ON p.id = pv.place_id
         WHERE v.embedding MATCH ?
           AND k = ?
         ORDER BY v.distance
        """,
        (blob, k),
    ).fetchall()
    return [
        NearestPlaceView(
            view_id=int(r["view_id"]),
            place_id=int(r["place_id"]),
            place_name=r["place_name"],
            image_path=r["image_path"],
            heading_deg=(None if r["heading_deg"] is None else float(r["heading_deg"])),
            distance_cm=(None if r["distance_cm"] is None else float(r["distance_cm"])),
            distance=float(r["distance"]),
        )
        for r in rows
    ]
