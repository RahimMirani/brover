"""Tests for backend.db.routes (the real CRUD layer).

Focused on the SQL itself: aggregation, ordering, vector-row lifecycle.
The recorder's higher-level orchestration is covered separately in
test_route_recording.py.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.db import connection, places, routes as db_routes
from backend.db.connection import EMBEDDING_DIM


def _vector(seed: float) -> list[float]:
    return [seed + 0.001 * i for i in range(EMBEDDING_DIM)]


class RoutesDbTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = Path(self._tmp.name) / "test.db"
        self.conn = connection.connect(db_path=db_path)
        self.addCleanup(self.conn.close)

        self.kitchen_id = places.add_place(self.conn, "kitchen")
        self.bedroom_id = places.add_place(self.conn, "bedroom")
        self.hallway_id = places.add_place(self.conn, "hallway")

    def _record(self, from_id: int, to_id: int, n_steps: int) -> int:
        """Insert a route + n_steps in one transaction, returning the route id."""
        try:
            self.conn.execute("BEGIN")
            route_id = db_routes.add_route(self.conn, from_id, to_id)
            for i in range(n_steps):
                db_routes.add_route_step(
                    self.conn,
                    route_id=route_id,
                    seq=i,
                    image_path=f"r{route_id}_{i}.jpg",
                    action=f"forward:{0.5 + 0.1 * i:.2f}",
                    embedding=_vector(0.1 * (i + 1)),
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return route_id

    def test_add_route_and_steps_round_trip(self) -> None:
        route_id = self._record(self.kitchen_id, self.bedroom_id, n_steps=3)

        steps = db_routes.get_route_steps(self.conn, route_id)
        self.assertEqual([s.seq for s in steps], [0, 1, 2])
        self.assertEqual(
            [s.action for s in steps],
            ["forward:0.50", "forward:0.60", "forward:0.70"],
        )
        # Vector rows match step rows.
        for step in steps:
            vec_row = self.conn.execute(
                "SELECT rowid FROM route_step_vectors WHERE rowid = ?",
                (step.id,),
            ).fetchone()
            self.assertIsNotNone(vec_row)

    def test_list_routes_with_step_counts(self) -> None:
        self._record(self.kitchen_id, self.bedroom_id, n_steps=2)
        self._record(self.kitchen_id, self.hallway_id, n_steps=4)

        summaries = db_routes.list_routes_with_step_counts(self.conn)
        self.assertEqual(len(summaries), 2)
        by_pair = {
            (s.from_place_name, s.to_place_name): s for s in summaries
        }
        self.assertEqual(by_pair[("kitchen", "bedroom")].step_count, 2)
        self.assertEqual(by_pair[("kitchen", "hallway")].step_count, 4)

    def test_find_routes_between_returns_newest_first(self) -> None:
        first = self._record(self.kitchen_id, self.bedroom_id, n_steps=2)
        # Hop the clock forward by overwriting created_at on the second
        # row, since both rows would otherwise share a wallclock second.
        second = self._record(self.kitchen_id, self.bedroom_id, n_steps=5)
        self.conn.execute(
            "UPDATE routes SET created_at = created_at + 10 WHERE id = ?",
            (second,),
        )
        self.conn.commit()

        matches = db_routes.find_routes_between(self.conn, "kitchen", "bedroom")
        self.assertEqual([m.id for m in matches], [second, first])
        self.assertEqual(matches[0].step_count, 5)

    def test_find_routes_between_returns_empty_for_unknown_pair(self) -> None:
        self._record(self.kitchen_id, self.bedroom_id, n_steps=2)
        self.assertEqual(
            db_routes.find_routes_between(self.conn, "kitchen", "garage"),
            [],
        )

    def test_delete_route_removes_steps_and_vectors(self) -> None:
        route_id = self._record(self.kitchen_id, self.bedroom_id, n_steps=3)
        step_ids = [
            int(r["id"])
            for r in self.conn.execute(
                "SELECT id FROM route_steps WHERE route_id = ?", (route_id,)
            ).fetchall()
        ]

        self.assertTrue(db_routes.delete_route(self.conn, route_id))

        # Route, steps, and vectors are all gone.
        self.assertIsNone(
            self.conn.execute(
                "SELECT id FROM routes WHERE id = ?", (route_id,)
            ).fetchone()
        )
        for sid in step_ids:
            self.assertIsNone(
                self.conn.execute(
                    "SELECT id FROM route_steps WHERE id = ?", (sid,)
                ).fetchone()
            )
            self.assertIsNone(
                self.conn.execute(
                    "SELECT rowid FROM route_step_vectors WHERE rowid = ?",
                    (sid,),
                ).fetchone()
            )

    def test_delete_route_returns_false_for_unknown(self) -> None:
        self.assertFalse(db_routes.delete_route(self.conn, 99999))


if __name__ == "__main__":
    unittest.main()
