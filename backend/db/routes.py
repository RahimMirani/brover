"""Route CRUD: recorded (frame, motor-action) sequences between places.

The read/write surface used by the route recorder, the upcoming graph-
navigation phase, and the live `find_route` / `list_routes` /
`/api/routes` tools:

  add_route                       create the route row once the destination
                                  is known (schema requires to_place_id)
  add_route_step                  append one (frame, action, embedding) row
                                  to an existing route -- caller manages
                                  the transaction so the whole route can
                                  commit atomically
  list_routes_with_step_counts    enumerate every route + step count for
                                  introspection
  find_routes_between             routes from one place to another, newest
                                  first (multiple are allowed; the graph
                                  layer picks one)
  delete_route                    remove a route and its steps/vectors,
                                  used by the recorder's auto-discard
                                  path when a recording overflows or is
                                  cancelled mid-stream

Same invariant as the place tables: every `route_steps` row has a matching
row in `route_step_vectors` with the same id, and `delete_route` deletes
from the virtual vec table explicitly (vec0 ignores SQL foreign keys, so
a cascade would otherwise leak orphan embeddings).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Sequence

import sqlite_vec

from backend.db.connection import EMBEDDING_DIM


@dataclass(frozen=True)
class Route:
    id: int
    from_place_id: int
    to_place_id: int
    created_at: float


@dataclass(frozen=True)
class RouteSummary:
    """One row in `list_routes_with_step_counts` and `find_routes_between`."""

    id: int
    from_place_name: str
    to_place_name: str
    step_count: int
    created_at: float


@dataclass(frozen=True)
class RouteStep:
    id: int
    route_id: int
    seq: int
    image_path: str
    action: str
    distance_cm: float | None
    captured_at: float


def _pack(embedding: Sequence[float]) -> bytes:
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"embedding has {len(embedding)} dims, expected {EMBEDDING_DIM}"
        )
    return sqlite_vec.serialize_float32(list(embedding))


def add_route(conn: Connection, from_place_id: int, to_place_id: int) -> int:
    """Insert a new route row and return its id."""
    cursor = conn.execute(
        "INSERT INTO routes (from_place_id, to_place_id, created_at) VALUES (?, ?, ?)",
        (from_place_id, to_place_id, time.time()),
    )
    return int(cursor.lastrowid)


def add_route_step(
    conn: Connection,
    *,
    route_id: int,
    seq: int,
    image_path: str,
    action: str,
    embedding: Sequence[float],
    distance_cm: float | None = None,
) -> int:
    """Append one route step + its embedding. Returns the step id.

    Does NOT manage a transaction. The caller (typically the route recorder's
    stop path) wraps the whole route insert in one BEGIN/COMMIT so a partial
    failure leaves no half-saved route on disk.
    """
    blob = _pack(embedding)
    captured_at = time.time()

    cursor = conn.execute(
        """
        INSERT INTO route_steps
            (route_id, seq, image_path, action, distance_cm, captured_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (route_id, seq, image_path, action, distance_cm, captured_at),
    )
    step_id = int(cursor.lastrowid)
    conn.execute(
        "INSERT INTO route_step_vectors (rowid, embedding) VALUES (?, ?)",
        (step_id, blob),
    )
    return step_id


def list_routes_with_step_counts(conn: Connection) -> list[RouteSummary]:
    """Every recorded route with how many steps it has. Powers `list_routes`
    and the upcoming `/api/routes` endpoint."""
    rows = conn.execute(
        """
        SELECT r.id              AS id,
               r.created_at      AS created_at,
               p1.name           AS from_name,
               p2.name           AS to_name,
               COUNT(rs.id)      AS step_count
          FROM routes r
          JOIN places p1 ON p1.id = r.from_place_id
          JOIN places p2 ON p2.id = r.to_place_id
          LEFT JOIN route_steps rs ON rs.route_id = r.id
         GROUP BY r.id, r.created_at, p1.name, p2.name
         ORDER BY p1.name, p2.name, r.created_at DESC
        """
    ).fetchall()
    return [
        RouteSummary(
            id=int(r["id"]),
            from_place_name=r["from_name"],
            to_place_name=r["to_name"],
            step_count=int(r["step_count"]),
            created_at=float(r["created_at"]),
        )
        for r in rows
    ]


def find_routes_between(
    conn: Connection, from_name: str, to_name: str
) -> list[RouteSummary]:
    """Routes from `from_name` to `to_name`, newest first.

    Multiple routes between the same pair are allowed -- if the user records
    the same trip twice, both rows survive. The graph layer can pick the
    most recent or use heuristics.
    """
    rows = conn.execute(
        """
        SELECT r.id              AS id,
               r.created_at      AS created_at,
               p1.name           AS from_name,
               p2.name           AS to_name,
               COUNT(rs.id)      AS step_count
          FROM routes r
          JOIN places p1 ON p1.id = r.from_place_id
          JOIN places p2 ON p2.id = r.to_place_id
          LEFT JOIN route_steps rs ON rs.route_id = r.id
         WHERE p1.name = ? AND p2.name = ?
         GROUP BY r.id, r.created_at, p1.name, p2.name
         ORDER BY r.created_at DESC
        """,
        (from_name, to_name),
    ).fetchall()
    return [
        RouteSummary(
            id=int(r["id"]),
            from_place_name=r["from_name"],
            to_place_name=r["to_name"],
            step_count=int(r["step_count"]),
            created_at=float(r["created_at"]),
        )
        for r in rows
    ]


def get_route_steps(conn: Connection, route_id: int) -> list[RouteStep]:
    """Return every step of a route in execution order."""
    rows = conn.execute(
        """
        SELECT id, route_id, seq, image_path, action, distance_cm, captured_at
          FROM route_steps
         WHERE route_id = ?
         ORDER BY seq
        """,
        (route_id,),
    ).fetchall()
    return [
        RouteStep(
            id=int(r["id"]),
            route_id=int(r["route_id"]),
            seq=int(r["seq"]),
            image_path=r["image_path"],
            action=r["action"],
            distance_cm=(None if r["distance_cm"] is None else float(r["distance_cm"])),
            captured_at=float(r["captured_at"]),
        )
        for r in rows
    ]


def delete_route(conn: Connection, route_id: int) -> bool:
    """Remove a route along with its steps and their embeddings.

    Returns True if a route was deleted, False if the id was unknown.

    `route_steps` cascades via the SQL foreign key; `route_step_vectors`
    does not (vec0 has no foreign keys) so we delete those rows by id
    inside the same transaction. Same trick as `places.delete_place_by_name`.
    """
    row = conn.execute(
        "SELECT id FROM routes WHERE id = ?", (route_id,)
    ).fetchone()
    if row is None:
        return False

    try:
        conn.execute("BEGIN")
        step_ids = [
            int(r["id"])
            for r in conn.execute(
                "SELECT id FROM route_steps WHERE route_id = ?", (route_id,)
            ).fetchall()
        ]
        for sid in step_ids:
            conn.execute("DELETE FROM route_step_vectors WHERE rowid = ?", (sid,))
        conn.execute("DELETE FROM routes WHERE id = ?", (route_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return True
