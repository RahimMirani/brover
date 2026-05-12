"""Tests for the lookup/summary/delete helpers in backend/db/places.py.

These run against a real (but per-test temp file) SQLite DB so we exercise
sqlite-vec for real. The teaching/localization tests inherit the same
setUp pattern.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.db import connection, places
from backend.db.connection import EMBEDDING_DIM


def _vector(seed: float) -> list[float]:
    """Make a deterministic 1024-d vector. Different `seed` -> different
    direction so cosine distance between two seeds is non-zero."""
    return [seed + 0.001 * i for i in range(EMBEDDING_DIM)]


class PlacesExtraTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = Path(self._tmp.name) / "test.db"
        self.conn = connection.connect(db_path=db_path)
        self.addCleanup(self.conn.close)

    def test_get_place_by_name_returns_none_for_missing(self) -> None:
        self.assertIsNone(places.get_place_by_name(self.conn, "nope"))

    def test_get_place_by_name_returns_existing(self) -> None:
        pid = places.add_place(self.conn, "kitchen")
        found = places.get_place_by_name(self.conn, "kitchen")
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.id, pid)
        self.assertEqual(found.name, "kitchen")

    def test_list_places_with_counts_includes_zero_view_places(self) -> None:
        places.add_place(self.conn, "kitchen")
        summaries = places.list_places_with_counts(self.conn)
        self.assertEqual(len(summaries), 1)
        s = summaries[0]
        self.assertEqual(s.name, "kitchen")
        self.assertEqual(s.view_count, 0)
        self.assertIsNone(s.last_taught_at)

    def test_list_places_with_counts_aggregates_views(self) -> None:
        pid = places.get_or_create_place(self.conn, "kitchen")
        places.add_place_view(
            self.conn, place_id=pid, image_path="a.jpg", embedding=_vector(0.1)
        )
        places.add_place_view(
            self.conn, place_id=pid, image_path="b.jpg", embedding=_vector(0.2)
        )
        places.get_or_create_place(self.conn, "bedroom")

        summaries = places.list_places_with_counts(self.conn)
        by_name = {s.name: s for s in summaries}
        self.assertEqual(by_name["kitchen"].view_count, 2)
        self.assertIsNotNone(by_name["kitchen"].last_taught_at)
        self.assertEqual(by_name["bedroom"].view_count, 0)
        self.assertIsNone(by_name["bedroom"].last_taught_at)

    def test_delete_place_removes_views_and_vectors(self) -> None:
        pid = places.get_or_create_place(self.conn, "kitchen")
        view_id = places.add_place_view(
            self.conn, place_id=pid, image_path="a.jpg", embedding=_vector(0.1)
        )

        self.assertTrue(places.delete_place_by_name(self.conn, "kitchen"))

        # Place row gone
        self.assertIsNone(places.get_place_by_name(self.conn, "kitchen"))

        # place_views row gone (cascade)
        row = self.conn.execute(
            "SELECT id FROM place_views WHERE id = ?", (view_id,)
        ).fetchone()
        self.assertIsNone(row)

        # vec0 row gone (manual delete in delete_place_by_name)
        vec_row = self.conn.execute(
            "SELECT rowid FROM place_view_vectors WHERE rowid = ?", (view_id,)
        ).fetchone()
        self.assertIsNone(vec_row)

    def test_delete_place_returns_false_for_unknown(self) -> None:
        self.assertFalse(places.delete_place_by_name(self.conn, "nope"))


if __name__ == "__main__":
    unittest.main()
