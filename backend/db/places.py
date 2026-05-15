"""Place CRUD: named locations and their captured frames.

The read/write surface used by teaching, localization, and the live
`find_place` / `forget_place` / `/api/places` tools:

  add_place / get_or_create_place      register a location by name
  get_place_by_name                    lookup without creating
  list_places                          enumerate known places (no counts)
  list_places_with_counts              same + per-place view counts and
                                       last-taught timestamp, for the UI
  add_place_view                       store one (frame, embedding) sample
  find_nearest_place_views             cosine-nearest lookup for localization
  delete_place_by_name                 remove a place and its views/vectors

Critical invariant: every row in `place_views` has exactly one matching row
in `place_view_vectors` with the same id. `add_place_view` writes both
inside a single transaction so the two cannot drift; if either insert
fails, the whole pair is rolled back. `delete_place_by_name` deletes from
the virtual vec table explicitly because vec0 has no foreign keys -- a
cascade on the SQL side would otherwise leak orphan embeddings.
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
class PlaceSummary:
    """One row in `list_places_with_counts`. Used by the introspection API."""

    id: int
    name: str
    created_at: float
    view_count: int
    last_taught_at: float | None


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


@dataclass(frozen=True)
class PlaceView:
    """One stored frame for a place. Used by the gallery + per-view delete UI."""

    id: int
    place_id: int
    image_path: str
    heading_deg: float | None
    distance_cm: float | None
    captured_at: float


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


def get_place_by_name(conn: Connection, name: str) -> Place | None:
    """Return the place with the given name, or None if it does not exist."""
    row = conn.execute(
        "SELECT id, name, created_at FROM places WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        return None
    return Place(id=int(row["id"]), name=row["name"], created_at=float(row["created_at"]))


def list_places(conn: Connection) -> list[Place]:
    rows = conn.execute(
        "SELECT id, name, created_at FROM places ORDER BY name"
    ).fetchall()
    return [
        Place(id=int(r["id"]), name=r["name"], created_at=float(r["created_at"]))
        for r in rows
    ]


def list_places_with_counts(conn: Connection) -> list[PlaceSummary]:
    """List every place plus its view count and most recent capture timestamp.

    Powers `GET /api/places` and the `find_place(name)` tool when the user
    or the LLM wants a quick "what do you know?" answer. Places with zero
    views still appear (an empty place can happen if teaching failed
    after the row was created), with view_count=0 and last_taught_at=None.
    """
    rows = conn.execute(
        """
        SELECT p.id              AS id,
               p.name            AS name,
               p.created_at      AS created_at,
               COUNT(pv.id)      AS view_count,
               MAX(pv.captured_at) AS last_taught_at
          FROM places p
          LEFT JOIN place_views pv ON pv.place_id = p.id
         GROUP BY p.id, p.name, p.created_at
         ORDER BY p.name
        """
    ).fetchall()
    return [
        PlaceSummary(
            id=int(r["id"]),
            name=r["name"],
            created_at=float(r["created_at"]),
            view_count=int(r["view_count"]),
            last_taught_at=(
                None if r["last_taught_at"] is None else float(r["last_taught_at"])
            ),
        )
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


def list_place_views(conn: Connection, place_id: int) -> list[PlaceView]:
    """Every stored frame for one place, oldest first.

    Used by the gallery UI to render thumbnails per place. Sorted by
    `captured_at ASC` so the order matches when you taught them, which
    is the order a human is likely to want to scan ("here are the first
    angles I taught, here are the later ones").
    """
    rows = conn.execute(
        """
        SELECT id, place_id, image_path, heading_deg, distance_cm, captured_at
          FROM place_views
         WHERE place_id = ?
         ORDER BY captured_at ASC, id ASC
        """,
        (place_id,),
    ).fetchall()
    return [
        PlaceView(
            id=int(r["id"]),
            place_id=int(r["place_id"]),
            image_path=r["image_path"],
            heading_deg=(None if r["heading_deg"] is None else float(r["heading_deg"])),
            distance_cm=(None if r["distance_cm"] is None else float(r["distance_cm"])),
            captured_at=float(r["captured_at"]),
        )
        for r in rows
    ]


def get_place_view(conn: Connection, view_id: int) -> PlaceView | None:
    """Look up one stored frame by id. Returns None if it doesn't exist."""
    row = conn.execute(
        """
        SELECT id, place_id, image_path, heading_deg, distance_cm, captured_at
          FROM place_views
         WHERE id = ?
        """,
        (view_id,),
    ).fetchone()
    if row is None:
        return None
    return PlaceView(
        id=int(row["id"]),
        place_id=int(row["place_id"]),
        image_path=row["image_path"],
        heading_deg=(None if row["heading_deg"] is None else float(row["heading_deg"])),
        distance_cm=(None if row["distance_cm"] is None else float(row["distance_cm"])),
        captured_at=float(row["captured_at"]),
    )


def count_image_path_refs(conn: Connection, image_path: str) -> int:
    """Count how many place_views + route_steps rows reference `image_path`.

    Used by the per-view delete path to decide whether the JPEG on disk
    can be removed too. `captures.save_jpeg` dedups by content hash, so
    two unrelated saves of the same frame land at the same file; deleting
    one row should NOT delete the file if another row still points at it.
    """
    place_views_row = conn.execute(
        "SELECT COUNT(*) AS n FROM place_views WHERE image_path = ?",
        (image_path,),
    ).fetchone()
    route_steps_row = conn.execute(
        "SELECT COUNT(*) AS n FROM route_steps WHERE image_path = ?",
        (image_path,),
    ).fetchone()
    return int(place_views_row["n"]) + int(route_steps_row["n"])


def delete_place_view(conn: Connection, view_id: int) -> PlaceView | None:
    """Remove one stored frame (and its embedding) by id.

    Returns the deleted row (so the caller can look up `image_path` for
    on-disk cleanup) or None if no such id existed.

    Same vec0 caveat as ``delete_place_by_name``: the SQL cascade does
    not cover the virtual ``place_view_vectors`` table, so we delete
    from it explicitly inside the same transaction. Skipping that step
    would leak an orphan embedding that future similarity searches still
    consider.
    """
    view = get_place_view(conn, view_id)
    if view is None:
        return None

    try:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM place_view_vectors WHERE rowid = ?", (view_id,))
        conn.execute("DELETE FROM place_views WHERE id = ?", (view_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return view


def delete_place_by_name(conn: Connection, name: str) -> bool:
    """Remove a place along with its views and embeddings.

    Returns True if a place was deleted, False if the name was unknown.

    The SQL-side cascade on `place_views.place_id` handles the metadata,
    but `place_view_vectors` is a `vec0` virtual table with no foreign
    keys, so we delete the matching vector rows by id inside the same
    transaction. Skipping that step would leak orphan embeddings -- the
    JOIN in `find_nearest_place_views` hides them from queries but they
    still occupy index space.
    """
    row = conn.execute(
        "SELECT id FROM places WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        return False

    place_id = int(row["id"])
    try:
        conn.execute("BEGIN")
        view_ids = [
            int(r["id"])
            for r in conn.execute(
                "SELECT id FROM place_views WHERE place_id = ?", (place_id,)
            ).fetchall()
        ]
        for vid in view_ids:
            conn.execute("DELETE FROM place_view_vectors WHERE rowid = ?", (vid,))
        conn.execute("DELETE FROM places WHERE id = ?", (place_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return True
