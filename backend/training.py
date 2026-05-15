"""Manual training: UI-driven place captures and route start/stop.

The voice path (``backend.tools``) already exposes ``remember_here``,
``start_tour`` / ``tag_place`` / ``end_tour``, and the route-recording
tools. This module is the other half: the same memory writes, but
driven by explicit UI button presses instead of the LLM.

The piece this module owns is the **pending-captures buffer**. When the
user clicks ``CAPTURE`` in the UI we snapshot the live camera frame and
hold it in memory keyed by a uuid. They can then preview the thumbnail,
type a place name (with autocomplete from existing places), and click
``SAVE`` -- which is when the embed + DB write actually happens. Until
that click the JPEG never hits disk.

Why an in-memory pending dict rather than:

- *Re-encoding from a ``<canvas>`` on the client*:
  That would round-trip the JPEG through a base64 body and a re-encode,
  both of which soften the embedding. Keeping the original
  ``camera.latest_jpeg`` bytes means what we store at training time is
  byte-identical to what ``localize`` will see at query time.

- *Writing the JPEG straight to disk on capture*:
  The user might click ``DISCARD``; we don't want a stale frame
  cluttering ``data/captures``. Save on commit only.

Bounds:

- ``TTL_SECONDS``: a card the user never gets back to is evicted after
  10 minutes so an open browser tab cannot pin frames in RAM forever.
- ``MAX_PENDING``: cap on simultaneous pending captures so a stuck
  client cannot OOM the backend by clicking ``CAPTURE`` in a loop.

The route side of manual training does not need any state of its own:
the existing ``backend.route_recording.route_recorder`` singleton
already handles start/stop/cancel correctly, and ``mode.on_manual_input``
already forwards every D-pad command into it. The HTTP layer in
``main.py`` just wraps that singleton; there's no logic here for it.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


# Live-tunable but expected to stay constant in production. Imported by
# tests so they can monkey-patch shorter values without poking internals.
TTL_SECONDS: float = 600.0
MAX_PENDING: int = 20


class PendingCapturesFull(RuntimeError):
    """Raised when ``MAX_PENDING`` pending captures already exist.

    The UI should surface this as "save or discard older captures first"
    rather than a generic 500.
    """


@dataclass(frozen=True)
class PendingCapture:
    """One pending capture sitting in memory between ``CAPTURE`` and ``SAVE``.

    ``jpeg`` is the bytes we grabbed from ``camera.latest_jpeg`` at the
    instant of capture. ``expires_at`` is a wall-clock seconds value
    (matches ``time.time()``) so it can be returned directly to the UI
    without unit conversion.
    """

    id: str
    jpeg: bytes
    created_at: float
    expires_at: float


class PendingCaptures:
    """In-memory store of JPEGs awaiting a place-name tag.

    Single-process, single-thread by design -- the FastAPI server runs
    one worker and all access goes through the asyncio event loop. The
    only synchronisation we need is lazy TTL eviction, run inline on
    every public call so we don't have to manage a background task.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = TTL_SECONDS,
        max_pending: int = MAX_PENDING,
        now: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        if max_pending <= 0:
            raise ValueError(f"max_pending must be positive, got {max_pending}")

        self._ttl = float(ttl_seconds)
        self._max = int(max_pending)
        self._now = now
        self._items: dict[str, PendingCapture] = {}

    @property
    def count(self) -> int:
        """Number of pending captures right now. Evicts expired entries first
        so the value is meaningful (not stale)."""
        self._evict_expired()
        return len(self._items)

    def create(self, jpeg: bytes) -> PendingCapture:
        """Buffer one JPEG, return the entry (with id + expires_at).

        Raises ``ValueError`` on empty bytes (almost certainly an
        upstream camera-not-ready bug) and ``PendingCapturesFull`` when
        the cap is hit.
        """
        if not jpeg:
            raise ValueError("create: empty jpeg")

        self._evict_expired()
        if len(self._items) >= self._max:
            raise PendingCapturesFull(
                f"already {len(self._items)} pending captures (max {self._max}); "
                "save or discard older ones first"
            )

        now = self._now()
        # uuid4 hex is short, URL-safe, and collision-free at the scale
        # we run at. Don't use the int representation -- the value lands
        # in URLs directly via the preview endpoint.
        item = PendingCapture(
            id=uuid.uuid4().hex,
            jpeg=jpeg,
            created_at=now,
            expires_at=now + self._ttl,
        )
        self._items[item.id] = item
        return item

    def get(self, capture_id: str) -> PendingCapture | None:
        """Look up without consuming. Used by the preview endpoint."""
        self._evict_expired()
        return self._items.get(capture_id)

    def pop(self, capture_id: str) -> PendingCapture | None:
        """Look up *and remove*. Used by the save endpoint on commit.

        Pop is only called from the save path *after* the embed + DB
        write succeeds -- so a Voyage failure leaves the capture in the
        buffer for the user to retry without re-capturing.
        """
        self._evict_expired()
        return self._items.pop(capture_id, None)

    def discard(self, capture_id: str) -> bool:
        """Remove a capture explicitly. Returns whether anything was removed."""
        self._evict_expired()
        return self._items.pop(capture_id, None) is not None

    def clear(self) -> None:
        """Drop every pending capture. Used by the app's shutdown hook so
        a half-trained session does not leak across process restarts."""
        self._items.clear()

    def _evict_expired(self) -> None:
        now = self._now()
        # Iterate over a snapshot so we can mutate the dict during the loop.
        for capture_id, item in list(self._items.items()):
            if item.expires_at <= now:
                del self._items[capture_id]
                logger.debug(
                    "pending capture %s evicted (age %.0fs)",
                    capture_id,
                    now - item.created_at,
                )


# Live singleton -- the HTTP handlers in ``main.py`` poke this; tests
# construct their own ``PendingCaptures`` to keep state isolated.
pending_captures = PendingCaptures()


__all__ = [
    "MAX_PENDING",
    "PendingCapture",
    "PendingCaptures",
    "PendingCapturesFull",
    "TTL_SECONDS",
    "pending_captures",
]
