"""Tests for backend.teaching: stationary teach and tour buffer.

All hardware is faked: get_jpeg returns canned bytes, embed_image returns
deterministic vectors, save_jpeg returns synthetic paths without touching
disk. Real SQLite + sqlite-vec is used so the database side is exercised
end-to-end.
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Iterator

from backend import teaching
from backend.db import connection, places
from backend.db.connection import EMBEDDING_DIM


def _vector(seed: float) -> list[float]:
    return [seed + 0.001 * i for i in range(EMBEDDING_DIM)]


class _FakeCamera:
    """Returns a different deterministic jpeg every call. Lets us tell
    'frame 1' from 'frame 2' in tests."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bytes:
        self.calls += 1
        return f"jpeg_{self.calls}".encode()


class _FakeEmbedder:
    """Maps each unique jpeg payload to a unique deterministic vector."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    async def __call__(self, jpeg: bytes) -> list[float]:
        self.calls.append(jpeg)
        seed = 0.1 * len(self.calls)
        return _vector(seed)


def _fake_save(jpeg: bytes) -> str:
    return f"data/captures/{hash(jpeg) & 0xFFFFFFFF:08x}.jpg"


class _Clock:
    """Manually-advanced clock so tour tests can pretend N seconds have passed."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TeachPlaceStationaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = Path(self._tmp.name) / "test.db"
        self.conn = connection.connect(db_path=db_path)
        self.addCleanup(self.conn.close)

        self.camera = _FakeCamera()
        self.embedder = _FakeEmbedder()

    def test_teach_creates_place_and_stores_views(self) -> None:
        result = asyncio.run(
            teaching.teach_place_stationary(
                self.conn,
                "kitchen",
                get_jpeg=self.camera,
                embed_image=self.embedder,
                save_jpeg=_fake_save,
                frames=3,
                gap_s=0.0,
            )
        )

        self.assertEqual(result.views_added, 3)
        self.assertEqual(result.frames_requested, 3)
        self.assertTrue(result.fully_succeeded)
        self.assertEqual(self.embedder.calls.__len__(), 3)

        summaries = {s.name: s for s in places.list_places_with_counts(self.conn)}
        self.assertIn("kitchen", summaries)
        self.assertEqual(summaries["kitchen"].view_count, 3)

    def test_re_teaching_appends_views_not_replaces(self) -> None:
        asyncio.run(
            teaching.teach_place_stationary(
                self.conn,
                "kitchen",
                get_jpeg=self.camera,
                embed_image=self.embedder,
                save_jpeg=_fake_save,
                frames=2,
                gap_s=0.0,
            )
        )
        asyncio.run(
            teaching.teach_place_stationary(
                self.conn,
                "kitchen",
                get_jpeg=self.camera,
                embed_image=self.embedder,
                save_jpeg=_fake_save,
                frames=2,
                gap_s=0.0,
            )
        )

        summaries = {s.name: s for s in places.list_places_with_counts(self.conn)}
        self.assertEqual(summaries["kitchen"].view_count, 4)

    def test_empty_camera_skips_frame(self) -> None:
        def empty_camera() -> bytes:
            return b""

        result = asyncio.run(
            teaching.teach_place_stationary(
                self.conn,
                "ghost",
                get_jpeg=empty_camera,
                embed_image=self.embedder,
                save_jpeg=_fake_save,
                frames=2,
                gap_s=0.0,
            )
        )
        self.assertEqual(result.views_added, 0)
        self.assertEqual(self.embedder.calls, [])
        # Place was still created (so re-teaching later appends).
        self.assertIsNotNone(places.get_place_by_name(self.conn, "ghost"))


class TourBufferTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = Path(self._tmp.name) / "test.db"
        self.conn = connection.connect(db_path=db_path)
        self.addCleanup(self.conn.close)

        self.clock = _Clock()
        self.embedder = _FakeEmbedder()
        self.buffer = teaching.TourBuffer(
            max_seconds=10.0, sample_hz=2.0, now=self.clock
        )

    def _start(self) -> None:
        asyncio.run(self.buffer.start())

    def _record_burst(self, count: int, interval: float = 0.5) -> None:
        """Drop `count` frames in at `interval` apart, advancing the clock."""
        for i in range(count):
            self.buffer.record_frame(f"frame_{i}".encode())
            self.clock.advance(interval)

    def test_record_outside_tour_is_noop(self) -> None:
        self.buffer.record_frame(b"jpeg")
        self.assertEqual(self.buffer.buffered_frame_count, 0)

    def test_buffer_drops_old_frames(self) -> None:
        self._start()
        self._record_burst(count=30, interval=0.5)  # 15s of frames
        # max_seconds=10, sampled at 0.5s intervals -> at most 21 frames remain
        # (the newest one was just appended at t = 1000 + 15 - 0.5 = 1014.5)
        self.assertLessEqual(self.buffer.buffered_frame_count, 25)
        self.assertGreater(self.buffer.buffered_frame_count, 15)

    def test_tag_window_creates_place_and_views(self) -> None:
        self._start()
        self._record_burst(count=8, interval=0.5)  # 4s of frames

        result = asyncio.run(
            self.buffer.tag_window(
                self.conn,
                "kitchen",
                embed_image=self.embedder,
                save_jpeg=_fake_save,
                seconds=4.0,
                max_frames=5,
            )
        )

        self.assertGreater(result.views_added, 0)
        self.assertLessEqual(result.views_added, 5)
        self.assertEqual(result.place_name, "kitchen")
        summaries = {s.name: s for s in places.list_places_with_counts(self.conn)}
        self.assertEqual(summaries["kitchen"].view_count, result.views_added)

    def test_tag_consumes_window_so_followup_tag_is_fresh(self) -> None:
        self._start()
        self._record_burst(count=4, interval=0.5)  # 2s of frames

        first = asyncio.run(
            self.buffer.tag_window(
                self.conn,
                "kitchen",
                embed_image=self.embedder,
                save_jpeg=_fake_save,
                seconds=4.0,
                max_frames=5,
            )
        )
        self.assertGreater(first.views_added, 0)

        # Second tag with no new frames -> 0 views, but place still recorded
        second = asyncio.run(
            self.buffer.tag_window(
                self.conn,
                "kitchen",
                embed_image=self.embedder,
                save_jpeg=_fake_save,
                seconds=4.0,
                max_frames=5,
            )
        )
        self.assertEqual(second.views_added, 0)
        self.assertEqual(second.frames_in_window, 0)

    def test_two_distinct_places_recorded_in_one_tour(self) -> None:
        self._start()

        self._record_burst(count=4, interval=0.5)
        asyncio.run(
            self.buffer.tag_window(
                self.conn,
                "kitchen",
                embed_image=self.embedder,
                save_jpeg=_fake_save,
                seconds=4.0,
                max_frames=5,
            )
        )

        # Advance time so the previous window is fully in the past, then add
        # fresh frames for a different place.
        self.clock.advance(5.0)
        self._record_burst(count=4, interval=0.5)
        asyncio.run(
            self.buffer.tag_window(
                self.conn,
                "bedroom",
                embed_image=self.embedder,
                save_jpeg=_fake_save,
                seconds=4.0,
                max_frames=5,
            )
        )

        summary = asyncio.run(self.buffer.end())
        self.assertCountEqual(summary.places_taught, ["kitchen", "bedroom"])
        self.assertEqual(summary.tags_applied, 2)
        self.assertGreater(summary.total_views_added, 0)

    def test_tag_without_start_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            asyncio.run(
                self.buffer.tag_window(
                    self.conn,
                    "x",
                    embed_image=self.embedder,
                    save_jpeg=_fake_save,
                )
            )

    def test_double_start_raises(self) -> None:
        self._start()
        with self.assertRaises(RuntimeError):
            asyncio.run(self.buffer.start())


if __name__ == "__main__":
    unittest.main()
