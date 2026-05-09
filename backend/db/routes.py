"""Route CRUD -- Phase 4.

Stubs only. The `routes` and `route_steps` tables (plus `route_step_vectors`)
are already created by `schema.sql`, so the database is ready for Phase 4
without a migration. Implementation lands when route recording is built.
"""
from __future__ import annotations

from sqlite3 import Connection
from typing import Any


def add_route(conn: Connection, *args: Any, **kwargs: Any) -> int:
    raise NotImplementedError("Route CRUD lands in Phase 4")


def add_route_step(conn: Connection, *args: Any, **kwargs: Any) -> int:
    raise NotImplementedError("Route CRUD lands in Phase 4")


def find_nearest_route_steps(conn: Connection, *args: Any, **kwargs: Any) -> list[Any]:
    raise NotImplementedError("Route CRUD lands in Phase 4")
