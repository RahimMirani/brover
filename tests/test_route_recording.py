"""Tests for backend.route_recording.

Same setUp pattern as the Phase 3 tests: a temp-file SQLite per test so
sqlite-vec is exercised for real, with all hardware deps (camera + embed
+ JPEG storage) replaced by deterministic in-process fakes.
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from backend import route_recording
from backend.db import connection, places, routes as db_routes
from backend.db.connection import EMBEDDING_DIM


def _vector(seed: float) -> list[float]:
    return [seed + 0.001 * i for i in range(EMBEDDING_DIM)]


class _FakeCamera:
    """Each call returns a unique deterministic jpeg so we can tell frames apart."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bytes:
        self.calls += 1
        return f"jpeg_{self.calls}".encode()


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[bytes] = []

    async def __call__(self, jpeg: bytes) -> list[float]:
        self.calls.append(jpeg)
        return _vector(0.1 * len(self.calls))


def _fake_save(jpeg: bytes) -> str:
    return f"data/captures/{hash(jpeg) & 0xFFFFFFFF:08x}.jpg"


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class ActionCodecTests(unittest.TestCase):
    def test_encode_format_is_fixed_two_decimals(self) -> None:
        self.assertEqual(
            route_recording.encode_action("forward", 1.5), "forward:1.50"
        )
        self.assertEqual(
            route_recording.encode_action("turn_left", 0.3), "turn_left:0.30"
        )

    def test_decode_round_trip(self) -> None:
        for verb, duration in [
            ("forward", 1.5),
            ("backward", 0.8),
            ("turn_left", 0.3),
            ("turn_right", 0.45),
        ]:
            encoded = route_recording.encode_action(verb, duration)
            decoded_verb, decoded_duration = route_recording.decode_action(encoded)
            self.assertEqual(decoded_verb, verb)
            self.assertAlmostEqual(decoded_duration, duration, places=2)

    def test_encode_rejects_unknown_verb(self) -> None:
        with self.assertRaises(ValueError):
            route_recording.encode_action("strafe", 1.0)

    def test_encode_rejects_negative_duration(self) -> None:
        with self.assertRaises(ValueError):
            route_recording.encode_action("forward", -0.5)

    def test_decode_rejects_malformed_input(self) -> None:
        for bad in ["forward", "forward:abc", "spin:1.0", "forward:-0.5", "forward:1:0"]:
            with self.assertRaises(ValueError):
                route_recording.decode_action(bad)


class RouteRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = Path(self._tmp.name) / "test.db"
        self.conn = connection.connect(db_path=db_path)
        self.addCleanup(self.conn.close)

        self.clock = _Clock()
        self.camera = _FakeCamera()
        self.embedder = _FakeEmbedder()
        self.recorder = route_recording.RouteRecorder(
            max_steps=5, max_seconds=10.0, now=self.clock
        )

    def _start(self, name: str = "kitchen") -> None:
        self.recorder.start(name, self.camera)

    def _stop(self, name: str = "bedroom") -> route_recording.RouteRecordResult:
        return asyncio.run(
            self.recorder.stop(
                self.conn,
                name,
                embed_image=self.embedder,
                save_jpeg=_fake_save,
            )
        )

    # ----- basic lifecycle ----------------------------------------------------

    def test_start_then_stop_with_no_motion_raises_empty(self) -> None:
        self._start()
        with self.assertRaises(route_recording.RouteRecorderEmpty):
            self._stop()
        self.assertFalse(self.recorder.active)

    def test_double_start_raises(self) -> None:
        self._start()
        with self.assertRaises(RuntimeError):
            self._start("again")

    def test_stop_without_start_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self._stop()

    def test_cancel_when_inactive_is_noop(self) -> None:
        self.recorder.cancel()
        self.assertFalse(self.recorder.active)

    # ----- segment construction ----------------------------------------------

    def test_single_forward_segment_persists_one_step(self) -> None:
        self._start()
        self.recorder.on_manual_command("forward")
        self.clock.advance(1.5)
        self.recorder.on_manual_command("stop")
        result = self._stop()

        self.assertEqual(result.step_count, 1)
        steps = db_routes.get_route_steps(self.conn, result.route_id)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].action, "forward:1.50")
        # Voyage was called exactly once for the one step.
        self.assertEqual(len(self.embedder.calls), 1)

    def test_two_segments_with_stop_in_between(self) -> None:
        self._start()
        self.recorder.on_manual_command("forward")
        self.clock.advance(1.5)
        self.recorder.on_manual_command("stop")
        self.clock.advance(0.2)
        self.recorder.on_manual_command("left")
        self.clock.advance(0.3)
        self.recorder.on_manual_command("stop")
        result = self._stop()

        self.assertEqual(result.step_count, 2)
        steps = db_routes.get_route_steps(self.conn, result.route_id)
        self.assertEqual([s.action for s in steps], ["forward:1.50", "turn_left:0.30"])
        self.assertEqual([s.seq for s in steps], [0, 1])

    def test_direct_movement_to_movement_without_stop(self) -> None:
        """Holding forward, then directly pressing left (no key release)."""
        self._start()
        self.recorder.on_manual_command("forward")
        self.clock.advance(1.0)
        self.recorder.on_manual_command("left")
        self.clock.advance(0.4)
        self.recorder.on_manual_command("stop")
        result = self._stop()

        self.assertEqual(result.step_count, 2)
        steps = db_routes.get_route_steps(self.conn, result.route_id)
        self.assertEqual([s.action for s in steps], ["forward:1.00", "turn_left:0.40"])

    def test_in_flight_segment_closed_by_stop_call(self) -> None:
        """User says stop_route_recording while still holding the key down."""
        self._start()
        self.recorder.on_manual_command("forward")
        self.clock.advance(2.0)
        # No explicit "stop" before stop() -- the recorder should close
        # the in-flight segment with the elapsed duration.
        result = self._stop()

        self.assertEqual(result.step_count, 1)
        steps = db_routes.get_route_steps(self.conn, result.route_id)
        self.assertEqual(steps[0].action, "forward:2.00")

    # ----- edge cases --------------------------------------------------------

    def test_empty_camera_skips_step_but_state_stays_consistent(self) -> None:
        """If the camera produces no frame at command-start, that step is
        dropped but subsequent steps still record cleanly."""
        def flaky_camera() -> bytes:
            flaky_camera.calls += 1  # type: ignore[attr-defined]
            # First call returns empty, second returns a frame.
            return b"" if flaky_camera.calls == 1 else b"jpeg_ok"
        flaky_camera.calls = 0  # type: ignore[attr-defined]

        self.recorder.start("kitchen", flaky_camera)
        self.recorder.on_manual_command("forward")  # camera empty -> skipped
        self.clock.advance(1.0)
        self.recorder.on_manual_command("stop")
        self.clock.advance(0.1)
        self.recorder.on_manual_command("left")  # camera ok -> recorded
        self.clock.advance(0.3)
        self.recorder.on_manual_command("stop")
        result = self._stop()

        self.assertEqual(result.step_count, 1)
        steps = db_routes.get_route_steps(self.conn, result.route_id)
        self.assertEqual(steps[0].action, "turn_left:0.30")

    def test_max_steps_cap_auto_stops_further_recording(self) -> None:
        self._start()
        # Cap is 5 (set in setUp). Issue 6 movement commands, each closed.
        for i in range(6):
            self.recorder.on_manual_command("forward")
            self.clock.advance(0.1)
            self.recorder.on_manual_command("stop")
            self.clock.advance(0.05)
        self.assertTrue(self.recorder.overflowed)
        result = self._stop()
        # Only the first 5 were captured before the overflow kicked in.
        self.assertEqual(result.step_count, 5)

    def test_cancel_during_recording_writes_nothing(self) -> None:
        self._start()
        self.recorder.on_manual_command("forward")
        self.clock.advance(1.0)
        self.recorder.on_manual_command("stop")
        self.recorder.cancel()

        self.assertFalse(self.recorder.active)
        # Nothing made it to the DB.
        self.assertEqual(db_routes.list_routes_with_step_counts(self.conn), [])

    def test_record_with_no_active_recording_is_noop(self) -> None:
        self.recorder.on_manual_command("forward")
        self.assertEqual(self.recorder.step_count, 0)


class RouteRecorderEndpointTests(unittest.TestCase):
    """Tests focused on what the result + DB rows look like together."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = Path(self._tmp.name) / "test.db"
        self.conn = connection.connect(db_path=db_path)
        self.addCleanup(self.conn.close)

        self.clock = _Clock()
        self.camera = _FakeCamera()
        self.embedder = _FakeEmbedder()
        self.recorder = route_recording.RouteRecorder(now=self.clock)

    def test_places_get_or_created_via_stop(self) -> None:
        # Neither place exists before recording.
        self.assertIsNone(places.get_place_by_name(self.conn, "garage"))
        self.assertIsNone(places.get_place_by_name(self.conn, "kitchen"))

        self.recorder.start("garage", self.camera)
        self.recorder.on_manual_command("forward")
        self.clock.advance(0.5)
        self.recorder.on_manual_command("stop")
        asyncio.run(
            self.recorder.stop(
                self.conn,
                "kitchen",
                embed_image=self.embedder,
                save_jpeg=_fake_save,
            )
        )

        self.assertIsNotNone(places.get_place_by_name(self.conn, "garage"))
        self.assertIsNotNone(places.get_place_by_name(self.conn, "kitchen"))

    def test_two_routes_between_same_pair_both_persist(self) -> None:
        def record_once() -> None:
            self.recorder.start("kitchen", self.camera)
            self.recorder.on_manual_command("forward")
            self.clock.advance(0.5)
            self.recorder.on_manual_command("stop")
            asyncio.run(
                self.recorder.stop(
                    self.conn,
                    "bedroom",
                    embed_image=self.embedder,
                    save_jpeg=_fake_save,
                )
            )

        record_once()
        self.clock.advance(1.0)
        record_once()

        matches = db_routes.find_routes_between(self.conn, "kitchen", "bedroom")
        self.assertEqual(len(matches), 2)
        # Newest first.
        self.assertGreaterEqual(matches[0].created_at, matches[1].created_at)


if __name__ == "__main__":
    unittest.main()
