"""Tests for backend.training: the pending-captures TTL buffer.

The HTTP wiring around this lives in backend.main and is exercised at
the smoke-test level on the Pi rather than in unit tests -- it's three
lines of glue per endpoint. The interesting logic is the buffer itself
(TTL eviction, cap enforcement, pop-only-on-commit), and that's what we
cover here with a fake clock and no I/O.
"""
from __future__ import annotations

import unittest

from backend.training import PendingCaptures, PendingCapturesFull


class _Clock:
    """Manually-advanced wall clock so TTL tests can pretend N seconds have
    passed without sleeping."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class PendingCapturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        self.buf = PendingCaptures(ttl_seconds=60.0, max_pending=3, now=self.clock)

    def test_create_returns_unique_ids(self) -> None:
        a = self.buf.create(b"jpeg-a")
        b = self.buf.create(b"jpeg-b")
        self.assertNotEqual(a.id, b.id)
        self.assertEqual(self.buf.count, 2)

    def test_get_returns_same_bytes(self) -> None:
        item = self.buf.create(b"jpeg-data")
        fetched = self.buf.get(item.id)
        self.assertIsNotNone(fetched)
        assert fetched is not None  # narrow for mypy / pyright
        self.assertEqual(fetched.jpeg, b"jpeg-data")

    def test_get_returns_none_for_unknown_id(self) -> None:
        self.assertIsNone(self.buf.get("nope"))

    def test_pop_removes_entry(self) -> None:
        item = self.buf.create(b"x")
        popped = self.buf.pop(item.id)
        self.assertIsNotNone(popped)
        self.assertIsNone(self.buf.get(item.id))
        self.assertEqual(self.buf.count, 0)

    def test_pop_returns_none_for_unknown_id(self) -> None:
        self.assertIsNone(self.buf.pop("nope"))

    def test_discard_returns_true_when_present(self) -> None:
        item = self.buf.create(b"x")
        self.assertTrue(self.buf.discard(item.id))
        self.assertFalse(self.buf.discard(item.id))

    def test_empty_jpeg_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.buf.create(b"")

    def test_cap_enforced(self) -> None:
        self.buf.create(b"1")
        self.buf.create(b"2")
        self.buf.create(b"3")
        with self.assertRaises(PendingCapturesFull):
            self.buf.create(b"4")
        # Discarding one frees a slot.
        first_id = next(iter(self.buf._items))  # type: ignore[attr-defined]
        self.buf.discard(first_id)
        self.buf.create(b"4")  # no longer raises
        self.assertEqual(self.buf.count, 3)

    def test_expired_entries_are_evicted(self) -> None:
        a = self.buf.create(b"old")
        self.clock.advance(61.0)
        # Eviction is lazy: it happens inside the next public call.
        self.assertIsNone(self.buf.get(a.id))
        self.assertEqual(self.buf.count, 0)

    def test_expiry_makes_room_for_new_captures(self) -> None:
        self.buf.create(b"1")
        self.buf.create(b"2")
        self.buf.create(b"3")
        # All three age out -> next create succeeds despite the cap.
        self.clock.advance(61.0)
        self.buf.create(b"4")
        self.assertEqual(self.buf.count, 1)

    def test_pop_is_safe_after_expiry(self) -> None:
        item = self.buf.create(b"x")
        self.clock.advance(61.0)
        self.assertIsNone(self.buf.pop(item.id))

    def test_clear_drops_everything(self) -> None:
        self.buf.create(b"a")
        self.buf.create(b"b")
        self.buf.clear()
        self.assertEqual(self.buf.count, 0)

    def test_expires_at_reflects_ttl(self) -> None:
        item = self.buf.create(b"x")
        self.assertEqual(item.expires_at, item.created_at + 60.0)

    def test_invalid_ttl_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PendingCaptures(ttl_seconds=0)

    def test_invalid_max_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PendingCaptures(max_pending=0)


if __name__ == "__main__":
    unittest.main()
