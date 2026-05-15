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

    # -- list_place_views / get_place_view / delete_place_view ---------------

    def test_list_place_views_orders_by_capture_time(self) -> None:
        pid = places.get_or_create_place(self.conn, "kitchen")
        v1 = places.add_place_view(
            self.conn, place_id=pid, image_path="a.jpg", embedding=_vector(0.1)
        )
        v2 = places.add_place_view(
            self.conn, place_id=pid, image_path="b.jpg", embedding=_vector(0.2)
        )
        v3 = places.add_place_view(
            self.conn, place_id=pid, image_path="c.jpg", embedding=_vector(0.3)
        )

        views = places.list_place_views(self.conn, pid)
        self.assertEqual([v.id for v in views], [v1, v2, v3])
        self.assertEqual([v.image_path for v in views], ["a.jpg", "b.jpg", "c.jpg"])

    def test_list_place_views_empty_for_unknown_place(self) -> None:
        self.assertEqual(places.list_place_views(self.conn, 9999), [])

    def test_get_place_view_returns_row_and_none(self) -> None:
        pid = places.get_or_create_place(self.conn, "kitchen")
        vid = places.add_place_view(
            self.conn, place_id=pid, image_path="a.jpg", embedding=_vector(0.1)
        )

        found = places.get_place_view(self.conn, vid)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.id, vid)
        self.assertEqual(found.place_id, pid)
        self.assertEqual(found.image_path, "a.jpg")

        self.assertIsNone(places.get_place_view(self.conn, 9999))

    def test_delete_place_view_drops_row_and_vector(self) -> None:
        pid = places.get_or_create_place(self.conn, "kitchen")
        vid = places.add_place_view(
            self.conn, place_id=pid, image_path="a.jpg", embedding=_vector(0.1)
        )

        deleted = places.delete_place_view(self.conn, vid)
        self.assertIsNotNone(deleted)
        assert deleted is not None
        self.assertEqual(deleted.image_path, "a.jpg")

        # place_views row gone
        self.assertIsNone(places.get_place_view(self.conn, vid))
        # vec0 row gone (orphan-embedding regression guard)
        vec_row = self.conn.execute(
            "SELECT rowid FROM place_view_vectors WHERE rowid = ?", (vid,)
        ).fetchone()
        self.assertIsNone(vec_row)

    def test_delete_place_view_unknown_returns_none(self) -> None:
        self.assertIsNone(places.delete_place_view(self.conn, 9999))

    def test_delete_place_view_leaves_other_views_alone(self) -> None:
        pid = places.get_or_create_place(self.conn, "kitchen")
        v1 = places.add_place_view(
            self.conn, place_id=pid, image_path="a.jpg", embedding=_vector(0.1)
        )
        v2 = places.add_place_view(
            self.conn, place_id=pid, image_path="b.jpg", embedding=_vector(0.2)
        )

        places.delete_place_view(self.conn, v1)
        remaining = places.list_place_views(self.conn, pid)
        self.assertEqual([v.id for v in remaining], [v2])

    def test_count_image_path_refs_counts_place_views_only(self) -> None:
        pid = places.get_or_create_place(self.conn, "kitchen")
        places.add_place_view(
            self.conn, place_id=pid, image_path="shared.jpg", embedding=_vector(0.1)
        )
        places.add_place_view(
            self.conn, place_id=pid, image_path="shared.jpg", embedding=_vector(0.2)
        )
        self.assertEqual(places.count_image_path_refs(self.conn, "shared.jpg"), 2)
        self.assertEqual(places.count_image_path_refs(self.conn, "missing.jpg"), 0)


if __name__ == "__main__":
    unittest.main()
