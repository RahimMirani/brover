"""People / face CRUD -- Phase 7.

Stubs only. The `people`, `face_views`, and `face_view_vectors` tables are
already created by `schema.sql`, so the database is ready for Phase 7
without a migration. Face recognition is opt-in, local-only, and never
synced; implementation lands after place memory and navigation are solid.
"""
from __future__ import annotations

from sqlite3 import Connection
from typing import Any


def add_person(conn: Connection, *args: Any, **kwargs: Any) -> int:
    raise NotImplementedError("Face CRUD lands in Phase 7")


def add_face_view(conn: Connection, *args: Any, **kwargs: Any) -> int:
    raise NotImplementedError("Face CRUD lands in Phase 7")


def find_nearest_faces(conn: Connection, *args: Any, **kwargs: Any) -> list[Any]:
    raise NotImplementedError("Face CRUD lands in Phase 7")
