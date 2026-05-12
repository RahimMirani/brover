"""Route recording: capture (frame, motor-action) sequences during manual driving.

The user enters manual teleop mode, says "start recording a route from the
kitchen", drives the rover by hand to the bedroom, then says "stop, this is
the bedroom". This module turns those manual key-presses into a replayable
sequence of timed motor actions plus visual reference frames.

How the latched manual teleop is decoded into segments:

  Manual teleop sends a stream of `move` commands: `forward` on key-down,
  `stop` on key-up, with a 180 ms watchdog stopping the motors if commands
  stop arriving. So a 1.5 second forward press looks like:
    t=0.0  on_manual_command("forward")   -> open a new segment
    t=1.5  on_manual_command("stop")      -> close it: action="forward:1.50"

  Each non-stop command opens a segment (captures one camera frame for
  later re-localization). The next command -- stop or another movement --
  closes the previous segment by writing its action string. The codec
  (`encode_action` / `decode_action`) is the single source of truth for
  the action format so Phase 5's replay can parse what Phase 4 wrote.

Why everything stays in memory until stop:

  `route_steps.route_id` is `NOT NULL`, and the route's `to_place_id` is
  only known when the user names the destination. We could allocate the
  route row up front and update it later, but the cleaner shape is to
  buffer steps in memory, then on `stop` create the route and write all
  steps in one transaction. A short route is a few KB of JPEGs in RAM;
  the bounded caps (`MAX_STEPS`, `MAX_SECONDS`) prevent runaway memory
  use if the user forgets to stop. Embedding happens once at stop time so
  the manual-teleop callback stays synchronous and never blocks the
  websocket handler.

Cancellation:

  Mode transitions (entering AI, e-stop, server shutdown) call `cancel()`,
  which drops the in-memory buffer without persisting anything. This
  matches the existing "manual override wins" semantics in mode.py.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Awaitable, Callable

from backend.db import places, routes as db_routes

logger = logging.getLogger(__name__)

JpegSource = Callable[[], bytes]
EmbedFn = Callable[[bytes], Awaitable[list[float]]]
SaveJpegFn = Callable[[bytes], str]


DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_SECONDS = 120.0


# Manual-teleop command names (as the websocket sends them) mapped to the
# verb we store in `route_steps.action`. Anything not in this map (notably
# "stop") is treated as a segment terminator, not a new segment.
_CMD_TO_VERB: dict[str, str] = {
    "forward": "forward",
    "backward": "backward",
    "left": "turn_left",
    "right": "turn_right",
}


def encode_action(verb: str, duration_seconds: float) -> str:
    """Format the action string stored in `route_steps.action`.

    Phase 5's replay parses this back via `decode_action`. Keep both in
    sync; the round-trip test in tests/test_route_recording.py exercises
    this exact pair."""
    if verb not in _ACTION_VERBS:
        raise ValueError(f"unknown action verb: {verb!r}")
    if duration_seconds < 0:
        raise ValueError(f"duration must be non-negative, got {duration_seconds}")
    return f"{verb}:{duration_seconds:.2f}"


def decode_action(action: str) -> tuple[str, float]:
    """Inverse of `encode_action`. Raises ValueError on a malformed string."""
    parts = action.split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid action format: {action!r}")
    verb, duration_str = parts
    if verb not in _ACTION_VERBS:
        raise ValueError(f"unknown action verb in {action!r}")
    try:
        duration = float(duration_str)
    except ValueError as e:
        raise ValueError(f"invalid duration in {action!r}: {e}") from e
    if duration < 0:
        raise ValueError(f"duration must be non-negative in {action!r}")
    return verb, duration


_ACTION_VERBS = frozenset({"forward", "backward", "turn_left", "turn_right"})


class RouteRecorderError(RuntimeError):
    """Base class for recorder errors. Specific subclasses below."""


class RouteRecorderEmpty(RouteRecorderError):
    """Raised when `stop` is called but no motion was actually recorded."""


@dataclass
class _PendingStep:
    """One in-memory step. `action` is filled in when the segment closes."""

    jpeg: bytes
    captured_at: float
    verb: str
    action: str | None = None


@dataclass(frozen=True)
class RouteRecordResult:
    """Returned by `RouteRecorder.stop` on a successful save."""

    route_id: int
    from_place: str
    to_place: str
    step_count: int
    duration_seconds: float


class RouteRecorder:
    """Records one route at a time. Sync callbacks for the websocket
    handler, async `stop` for the embed + DB write phase.

    Designed for one recording at a time on the asyncio event loop. The
    sync surface (`start`, `on_manual_command`, `cancel`) is callable
    from anywhere -- in particular from `mode.on_manual_input`, which
    runs on the websocket task and must not await on Voyage."""

    def __init__(
        self,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {max_seconds}")

        self._max_steps = max_steps
        self._max_seconds = max_seconds
        self._now = now

        self._active = False
        self._from_place_name: str | None = None
        self._started_at: float | None = None
        self._current_cmd: str = "stop"
        self._current_cmd_started_at: float = 0.0
        self._steps: list[_PendingStep] = []
        self._in_flight_index: int | None = None
        self._jpeg_source: JpegSource | None = None
        self._overflowed = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    @property
    def from_place(self) -> str | None:
        return self._from_place_name

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self, from_place_name: str, get_jpeg: JpegSource) -> None:
        """Begin a recording from a named origin. Safe to call only when
        no recording is active."""
        if self._active:
            raise RuntimeError("route recording is already active")
        name = (from_place_name or "").strip()
        if not name:
            raise ValueError("from_place_name is required")

        self._active = True
        self._from_place_name = name
        self._started_at = self._now()
        self._current_cmd = "stop"
        self._current_cmd_started_at = self._started_at
        self._steps = []
        self._in_flight_index = None
        self._jpeg_source = get_jpeg
        self._overflowed = False
        logger.info("route recording started from %r", name)

    def on_manual_command(self, cmd: str) -> None:
        """Called by `mode.on_manual_input` for every manual teleop command.

        No-op when no recording is active or when the recorder has already
        overflowed its caps. Synchronous and fast -- a frame grab is the
        only I/O it does, and the camera property is already populated.
        """
        if not self._active or self._overflowed:
            return

        now = self._now()

        # Close the in-flight segment (if any) with its measured duration.
        if self._in_flight_index is not None:
            step = self._steps[self._in_flight_index]
            duration = max(0.0, now - self._current_cmd_started_at)
            step.action = encode_action(step.verb, duration)
            self._in_flight_index = None

        # If the new command is a movement, capture a frame and open a new
        # segment; otherwise it's a "stop" terminator and we just update state.
        if cmd in _CMD_TO_VERB:
            if self._exceeded_caps(now):
                self._overflowed = True
                logger.warning(
                    "route recording auto-stopped: exceeded cap "
                    "(steps=%d, elapsed=%.1fs)",
                    len(self._steps),
                    now - (self._started_at or now),
                )
                return

            jpeg = self._jpeg_source() if self._jpeg_source else b""
            if jpeg:
                self._steps.append(
                    _PendingStep(
                        jpeg=jpeg,
                        captured_at=now,
                        verb=_CMD_TO_VERB[cmd],
                    )
                )
                self._in_flight_index = len(self._steps) - 1
            else:
                logger.warning(
                    "route recording: no camera frame for cmd=%r; skipping step",
                    cmd,
                )

        self._current_cmd = cmd
        self._current_cmd_started_at = now

    def cancel(self) -> None:
        """Abandon the active recording without persisting anything.

        Called by `mode.enter_ai`, `mode.request_estop`, and the server
        shutdown path. No-op if not currently recording."""
        if not self._active:
            return
        logger.info(
            "route recording cancelled; %d step(s) discarded", len(self._steps)
        )
        self._reset()

    async def stop(
        self,
        conn: Connection,
        to_place_name: str,
        *,
        embed_image: EmbedFn,
        save_jpeg: SaveJpegFn,
    ) -> RouteRecordResult:
        """End the recording and persist it. Raises RouteRecorderEmpty if no
        motion was captured (start -> stop with nothing in between)."""
        if not self._active:
            raise RuntimeError("route recording is not active")

        name = (to_place_name or "").strip()
        if not name:
            raise ValueError("to_place_name is required")

        # Close any in-flight segment with its final duration.
        now = self._now()
        if self._in_flight_index is not None:
            step = self._steps[self._in_flight_index]
            duration = max(0.0, now - self._current_cmd_started_at)
            step.action = encode_action(step.verb, duration)
            self._in_flight_index = None

        from_place_name = self._from_place_name or ""
        started_at = self._started_at or now
        complete_steps = [s for s in self._steps if s.action is not None]

        if not complete_steps:
            self._reset()
            raise RouteRecorderEmpty(
                "no motion was recorded; route was not saved"
            )

        # Embed sequentially -- Voyage is rate-limited and a typical route
        # is 5-15 steps, so the wallclock cost is acceptable.
        embeddings_per_step: list[list[float]] = []
        for step in complete_steps:
            embedding = await embed_image(step.jpeg)
            embeddings_per_step.append(embedding)

        from_place_id = places.get_or_create_place(conn, from_place_name)
        to_place_id = places.get_or_create_place(conn, name)

        # One outer transaction so a half-written route never lands on disk.
        try:
            conn.execute("BEGIN")
            route_id = db_routes.add_route(conn, from_place_id, to_place_id)
            for seq, (step, embedding) in enumerate(
                zip(complete_steps, embeddings_per_step)
            ):
                image_path = save_jpeg(step.jpeg)
                assert step.action is not None  # already filtered above
                db_routes.add_route_step(
                    conn,
                    route_id=route_id,
                    seq=seq,
                    image_path=image_path,
                    action=step.action,
                    embedding=embedding,
                )
            conn.commit()
        except Exception:
            conn.rollback()
            self._reset()
            raise

        result = RouteRecordResult(
            route_id=route_id,
            from_place=from_place_name,
            to_place=name,
            step_count=len(complete_steps),
            duration_seconds=now - started_at,
        )
        logger.info(
            "route recorded: id=%d %s -> %s steps=%d duration=%.1fs",
            route_id,
            result.from_place,
            result.to_place,
            result.step_count,
            result.duration_seconds,
        )
        self._reset()
        return result

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _exceeded_caps(self, now: float) -> bool:
        if len(self._steps) >= self._max_steps:
            return True
        if self._started_at is not None and now - self._started_at > self._max_seconds:
            return True
        return False

    def _reset(self) -> None:
        self._active = False
        self._from_place_name = None
        self._started_at = None
        self._current_cmd = "stop"
        self._current_cmd_started_at = 0.0
        self._steps = []
        self._in_flight_index = None
        self._jpeg_source = None
        self._overflowed = False


# Live singleton, mirrors the `tour_buffer` pattern in backend/teaching.py.
route_recorder = RouteRecorder()


__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_SECONDS",
    "RouteRecorder",
    "RouteRecorderEmpty",
    "RouteRecorderError",
    "RouteRecordResult",
    "encode_action",
    "decode_action",
    "route_recorder",
]
