"""Tests for backend.localization.

The classification logic is the interesting bit. We seed the DB with
controlled embeddings, then craft query embeddings whose cosine distance
to each row is known, and verify the right status comes back.

Cosine distance between two vectors u, v is `1 - cos(u, v)` where cos is
dot(u, v) / (|u| * |v|). For two unit-length one-hot vectors that point
the same way distance is 0; for orthogonal ones it is 1. We build vectors
as 'mostly along axis i', so axis-i and axis-j queries are clearly far,
but two queries on the same axis are clearly close.
"""
from __future__ import annotations

import asyncio
import math
import tempfile
import unittest
from pathlib import Path

from backend import localization
from backend.db import connection, places
from backend.db.connection import EMBEDDING_DIM


def _axis_vector(axis: int, *, weight: float = 1.0) -> list[float]:
    """Vector that points along `axis` with a small amount of noise on other
    coordinates. Two `_axis_vector(i)` calls produce identical-direction
    vectors (distance ~0). Different axes produce nearly-orthogonal vectors
    (distance ~1).

    The noise (1e-3 on each non-axis dim) keeps cosine sim very close to 1.0
    for same-axis vectors but avoids ties at exactly 0 distance, which sometimes
    confuses vec0 ordering."""
    v = [1e-3] * EMBEDDING_DIM
    v[axis] = weight
    return v


def _blend(axis_a: int, axis_b: int, *, ratio: float) -> list[float]:
    """Vector that mixes two axes. ratio=0 -> all axis_a, ratio=1 -> all axis_b.
    Used to construct queries that are ambiguously between two known places."""
    a = _axis_vector(axis_a, weight=1.0 - ratio)
    b = _axis_vector(axis_b, weight=ratio)
    return [a[i] + b[i] for i in range(EMBEDDING_DIM)]


class _StaticEmbedder:
    """Returns whatever vector we tell it to, ignoring the input jpeg."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls = 0

    async def __call__(self, _jpeg: bytes) -> list[float]:
        self.calls += 1
        return list(self.vector)


class LocalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = Path(self._tmp.name) / "test.db"
        self.conn = connection.connect(db_path=db_path)
        self.addCleanup(self.conn.close)

    def _seed_place(self, name: str, axis: int, n_views: int = 3) -> int:
        pid = places.get_or_create_place(self.conn, name)
        for i in range(n_views):
            places.add_place_view(
                self.conn,
                place_id=pid,
                image_path=f"{name}_{i}.jpg",
                embedding=_axis_vector(axis),
            )
        return pid

    def test_empty_memory(self) -> None:
        embedder = _StaticEmbedder(_axis_vector(0))
        result = asyncio.run(
            localization.localize_jpeg(
                self.conn, b"jpeg", embed_image=embedder
            )
        )
        self.assertEqual(result.status, "empty_memory")
        self.assertIsNone(result.best)
        self.assertEqual(result.alternatives, [])
        # Don't waste an embed call if the DB is empty.
        self.assertEqual(embedder.calls, 0)

    def test_confident_match(self) -> None:
        self._seed_place("kitchen", axis=0)
        self._seed_place("bedroom", axis=100)

        embedder = _StaticEmbedder(_axis_vector(0))
        result = asyncio.run(
            localization.localize_jpeg(
                self.conn, b"jpeg", embed_image=embedder
            )
        )

        self.assertEqual(result.status, "confident")
        assert result.best is not None
        self.assertEqual(result.best.place_name, "kitchen")
        # Same-axis vectors should be very close to 0 distance.
        self.assertLess(result.best.distance, localization.HIGH_CONFIDENCE_DISTANCE)
        # 'kitchen' must not appear in alternatives.
        names = [c.place_name for c in result.alternatives]
        self.assertNotIn("kitchen", names)

    def test_ambiguous_when_blend_of_two_places(self) -> None:
        self._seed_place("kitchen", axis=0)
        self._seed_place("hallway", axis=100)

        # 50/50 blend between kitchen and hallway -> roughly equidistant
        embedder = _StaticEmbedder(_blend(0, 100, ratio=0.5))
        result = asyncio.run(
            localization.localize_jpeg(
                self.conn, b"jpeg", embed_image=embedder
            )
        )

        self.assertIn(result.status, ("ambiguous", "confident"))
        # If we got 'confident' from this query something's badly wrong with
        # the threshold tuning; force the test to fail loudly.
        self.assertEqual(result.status, "ambiguous")
        assert result.best is not None
        self.assertIn(result.best.place_name, {"kitchen", "hallway"})
        # The other place should show up as a runner-up.
        expected_alt = ({"kitchen", "hallway"} - {result.best.place_name}).pop()
        alt_names = {c.place_name for c in result.alternatives}
        self.assertIn(expected_alt, alt_names)

    def test_unrecognized_when_far_from_everything(self) -> None:
        self._seed_place("kitchen", axis=0)

        # Use an axis far from any taught place. Cosine between near-orthogonal
        # vectors is ~0, so distance is ~1 -- well past LOW_CONFIDENCE_DISTANCE.
        embedder = _StaticEmbedder(_axis_vector(500))
        result = asyncio.run(
            localization.localize_jpeg(
                self.conn, b"jpeg", embed_image=embedder
            )
        )

        self.assertEqual(result.status, "unrecognized")
        assert result.best is not None
        self.assertGreater(result.best.distance, localization.LOW_CONFIDENCE_DISTANCE)


class ClassifyHelperTests(unittest.TestCase):
    """Direct tests on the _classify branch logic so we cover the corners."""

    def _cand(self, name: str, distance: float) -> localization.LocalizeCandidate:
        return localization.LocalizeCandidate(
            place_name=name, view_id=1, image_path="x.jpg", distance=distance
        )

    def test_high_confidence_no_alternatives(self) -> None:
        status = localization._classify(self._cand("kitchen", 0.1), [])
        self.assertEqual(status, "confident")

    def test_high_confidence_with_distant_runner_up(self) -> None:
        status = localization._classify(
            self._cand("kitchen", 0.1), [self._cand("bedroom", 0.4)]
        )
        self.assertEqual(status, "confident")

    def test_high_confidence_with_close_runner_up_is_ambiguous(self) -> None:
        status = localization._classify(
            self._cand("kitchen", 0.1),
            [self._cand("bedroom", 0.11)],  # margin 0.01 < MIN_MARGIN
        )
        self.assertEqual(status, "ambiguous")

    def test_mid_distance_is_ambiguous(self) -> None:
        # Between HIGH (0.25) and LOW (0.45)
        status = localization._classify(self._cand("kitchen", 0.35), [])
        self.assertEqual(status, "ambiguous")

    def test_far_is_unrecognized(self) -> None:
        status = localization._classify(self._cand("kitchen", 0.6), [])
        self.assertEqual(status, "unrecognized")


if __name__ == "__main__":
    unittest.main()
