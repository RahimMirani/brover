"""Place teaching: the write path of Brover's spatial memory.

Composes camera + embeddings + places DB into two flows the LLM exposes
as tools:

- `teach_place_stationary`: stop-and-say. The rover sits still, the camera
  grabs a handful of frames, each is embedded and stored under one place
  name. Used by the `remember_here(name)` tool.

- `TourBuffer`: continuous-capture sweep. The user drives the rover manually
  while a background task pushes camera frames into a rolling time-bounded
  buffer. When the user says "this is the kitchen", `tag_window` pulls the
  most recent N seconds of buffered frames, embeds them, and stores them
  under that place name. Used by the `start_tour` / `tag_place(name)` /
  `end_tour` tools.

Why dependency-inject the camera, embedder, and storage:
    The live code wires in `camera.latest_jpeg`, `embeddings.embed_image`,
    and `captures.save_jpeg`. Tests pass deterministic fakes -- no Voyage
    network call, no rpicam-vid subprocess. Same module, two callers.

Errors:
    Embedding failures abort the rest of the teach call and propagate.
    Whatever views were stored before the failure stay in the DB; the
    `views_added` field of the returned `TeachResult` reflects only those
    that committed successfully. This matches the existing per-view
    transaction shape in `places.add_place_view`.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Awaitable, Callable, Iterable

from backend.db import places

logger = logging.getLogger(__name__)

# Callable shapes. Async embed so the live path can await voyageai without
# blocking the event loop; sync get_jpeg / save_jpeg because both the
# camera property and the file write are non-blocking in practice.
JpegSource = Callable[[], bytes]
EmbedFn = Callable[[bytes], Awaitable[list[float]]]
SaveJpegFn = Callable[[bytes], str]


@dataclass(frozen=True)
class TeachResult:
    """Outcome of one teaching call. `views_added` may be < `frames_requested`
    if the camera failed to produce a frame or an embed call errored out."""

    place_id: int
    place_name: str
    views_added: int
    frames_requested: int

    @property
    def fully_succeeded(self) -> bool:
        return self.views_added == self.frames_requested and self.views_added > 0


async def teach_place_stationary(
    conn: Connection,
    name: str,
    *,
    get_jpeg: JpegSource,
    embed_image: EmbedFn,
    save_jpeg: SaveJpegFn,
    frames: int = 5,
    gap_s: float = 0.4,
) -> TeachResult:
    """Capture `frames` frames spaced `gap_s` apart and store them under `name`.

    All captures attach to one place row; re-teaching an existing place
    appends new views rather than replacing (use `forget_place` first if
    you actually want a clean slate). Embeddings are awaited sequentially
    rather than in parallel -- Voyage is rate-limited and 5 sequential
    calls is fast enough that the UX cost is small.
    """
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")

    place_id = places.get_or_create_place(conn, name)
    views_added = 0

    for i in range(frames):
        if i > 0 and gap_s > 0:
            await asyncio.sleep(gap_s)

        try:
            jpeg = get_jpeg()
        except Exception:
            logger.exception("teach_place_stationary: camera frame fetch failed")
            break

        if not jpeg:
            logger.warning(
                "teach_place_stationary: no frame on iteration %d/%d", i + 1, frames
            )
            continue

        vector = await embed_image(jpeg)
        image_path = save_jpeg(jpeg)
        places.add_place_view(
            conn,
            place_id=place_id,
            image_path=image_path,
            embedding=vector,
        )
        views_added += 1

    logger.info(
        "teach_place_stationary: place=%r views_added=%d/%d",
        name,
        views_added,
        frames,
    )
    return TeachResult(
        place_id=place_id,
        place_name=name,
        views_added=views_added,
        frames_requested=frames,
    )


# -----------------------------------------------------------------------------
# Tour mode
# -----------------------------------------------------------------------------

DEFAULT_TOUR_BUFFER_SECONDS = 30.0
DEFAULT_TOUR_SAMPLE_HZ = 2.0
DEFAULT_TAG_WINDOW_SECONDS = 4.0
DEFAULT_TAG_MAX_FRAMES = 5


@dataclass(frozen=True)
class TourTagResult:
    """One `tag_window` outcome. Mirrors `TeachResult` but also reports how
    many frames were sitting in the recent window before this tag ran."""

    place_id: int
    place_name: str
    views_added: int
    frames_in_window: int


@dataclass(frozen=True)
class TourSummary:
    """Final summary returned by `TourBuffer.end()`."""

    duration_seconds: float
    tags_applied: int
    places_taught: list[str]
    total_views_added: int


class TourBuffer:
    """Rolling JPEG buffer for tour-mode teaching.

    Three responsibilities, kept deliberately small:

    - Keep the last N seconds of camera frames in memory, sampled at a fixed
      rate so the buffer size stays bounded regardless of camera FPS.
    - When asked, pull the most recent M seconds out of the buffer and turn
      them into stored place views. Tagged frames are *removed* from the
      buffer so a second tag a moment later does not re-store the same
      frames (and a "tag every few seconds" pattern keeps producing fresh
      views).
    - Track tour-level state (active / not active, when started, places
      taught) so `end()` can return a useful summary.

    Thread / task safety:
        Designed for one tour at a time on the asyncio event loop. `start`
        spawns a single polling task; `record_frame` is synchronous so
        tests can drive it without an event loop. `tag_window` is async
        because embedding is async.
    """

    def __init__(
        self,
        *,
        max_seconds: float = DEFAULT_TOUR_BUFFER_SECONDS,
        sample_hz: float = DEFAULT_TOUR_SAMPLE_HZ,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {max_seconds}")
        if sample_hz <= 0:
            raise ValueError(f"sample_hz must be positive, got {sample_hz}")

        self._max_seconds = float(max_seconds)
        self._sample_interval = 1.0 / float(sample_hz)
        self._now = now
        self._frames: deque[tuple[float, bytes]] = deque()
        self._active = False
        self._started_at: float | None = None
        self._tags_applied = 0
        self._total_views_added = 0
        self._places_taught: list[str] = []
        self._poll_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def buffered_frame_count(self) -> int:
        return len(self._frames)

    async def start(self, get_jpeg: JpegSource | None = None) -> None:
        """Begin a tour. If `get_jpeg` is provided, a background task starts
        polling it at `sample_hz` and recording frames. In tests we omit the
        callable and drive `record_frame` directly."""
        if self._active:
            raise RuntimeError("tour is already active")

        self._frames.clear()
        self._active = True
        self._started_at = self._now()
        self._tags_applied = 0
        self._total_views_added = 0
        self._places_taught = []

        if get_jpeg is not None:
            self._stop_event = asyncio.Event()
            self._poll_task = asyncio.create_task(
                self._poll_loop(get_jpeg), name="tour-buffer-poll"
            )

        logger.info("tour started: max_seconds=%.1f, sample_hz=%.2f",
                    self._max_seconds, 1.0 / self._sample_interval)

    def record_frame(self, jpeg: bytes, *, at: float | None = None) -> None:
        """Push one frame into the buffer, evicting anything older than
        `max_seconds`. Safe to call when the tour is not active (no-op)."""
        if not self._active or not jpeg:
            return
        t = self._now() if at is None else at
        self._frames.append((t, jpeg))
        cutoff = t - self._max_seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    async def tag_window(
        self,
        conn: Connection,
        name: str,
        *,
        embed_image: EmbedFn,
        save_jpeg: SaveJpegFn,
        seconds: float = DEFAULT_TAG_WINDOW_SECONDS,
        max_frames: int = DEFAULT_TAG_MAX_FRAMES,
    ) -> TourTagResult:
        """Store the most recent `seconds` of buffered frames under `name`.

        At most `max_frames` are kept, evenly spread across the window
        (Voyage is the slow expensive step, so we don't want to embed 60
        frames for one "this is the kitchen"). Tagged frames are consumed
        from the buffer; the next tag picks up from where this one left off.

        Errors raised by `embed_image` propagate; views committed so far
        stay in the DB (per-view transactions, same as stationary teach)."""
        if not self._active:
            raise RuntimeError("tour is not active; call start() first")

        cutoff = self._now() - seconds
        in_window: list[tuple[float, bytes]] = [
            entry for entry in self._frames if entry[0] >= cutoff
        ]
        if not in_window:
            place_id = places.get_or_create_place(conn, name)
            self._tags_applied += 1
            if name not in self._places_taught:
                self._places_taught.append(name)
            logger.info(
                "tour tag %r: no frames in the last %.1fs; no views added",
                name,
                seconds,
            )
            return TourTagResult(
                place_id=place_id,
                place_name=name,
                views_added=0,
                frames_in_window=0,
            )

        picked = _evenly_sampled(in_window, max_frames)

        place_id = places.get_or_create_place(conn, name)
        views_added = 0
        for _, jpeg in picked:
            vector = await embed_image(jpeg)
            image_path = save_jpeg(jpeg)
            places.add_place_view(
                conn,
                place_id=place_id,
                image_path=image_path,
                embedding=vector,
            )
            views_added += 1

        # Drop everything inside the tagged window so a quick follow-up tag
        # only picks up new material.
        self._frames = deque(entry for entry in self._frames if entry[0] < cutoff)

        self._tags_applied += 1
        self._total_views_added += views_added
        if name not in self._places_taught:
            self._places_taught.append(name)

        logger.info(
            "tour tag %r: added %d/%d views (window=%.1fs)",
            name,
            views_added,
            len(picked),
            seconds,
        )
        return TourTagResult(
            place_id=place_id,
            place_name=name,
            views_added=views_added,
            frames_in_window=len(in_window),
        )

    async def end(self) -> TourSummary:
        """Stop the tour and return a summary."""
        if not self._active:
            raise RuntimeError("tour is not active")

        if self._stop_event is not None:
            self._stop_event.set()
        if self._poll_task is not None:
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        self._stop_event = None

        duration = (
            0.0 if self._started_at is None else self._now() - self._started_at
        )
        summary = TourSummary(
            duration_seconds=duration,
            tags_applied=self._tags_applied,
            places_taught=list(self._places_taught),
            total_views_added=self._total_views_added,
        )

        self._active = False
        self._frames.clear()
        self._started_at = None
        logger.info(
            "tour ended: duration=%.1fs, tags=%d, places=%s",
            summary.duration_seconds,
            summary.tags_applied,
            summary.places_taught,
        )
        return summary

    async def _poll_loop(self, get_jpeg: JpegSource) -> None:
        assert self._stop_event is not None
        try:
            while not self._stop_event.is_set():
                try:
                    frame = get_jpeg()
                except Exception:
                    logger.exception("tour poll: camera read failed")
                    frame = b""
                if frame:
                    self.record_frame(frame)
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._sample_interval
                    )
                    return
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise


def _evenly_sampled(
    items: list[tuple[float, bytes]], max_count: int
) -> list[tuple[float, bytes]]:
    """Pick up to `max_count` items evenly spaced across `items`, preserving order.

    `items` is already sorted by time (it comes out of an append-only deque).
    For 5 items, max_count=5 -> all of them. For 60 items, max_count=5 ->
    indices roughly 0, 15, 30, 45, 59."""
    n = len(items)
    if n == 0 or max_count <= 0:
        return []
    if n <= max_count:
        return list(items)

    step = (n - 1) / (max_count - 1)
    picked_indices: list[int] = []
    for i in range(max_count):
        idx = int(round(i * step))
        if idx >= n:
            idx = n - 1
        if not picked_indices or idx != picked_indices[-1]:
            picked_indices.append(idx)
    return [items[i] for i in picked_indices]


# Live-singleton tour buffer. Tools poke this; tests instantiate their own.
tour_buffer = TourBuffer()


__all__ = [
    "TeachResult",
    "TourTagResult",
    "TourSummary",
    "TourBuffer",
    "teach_place_stationary",
    "tour_buffer",
]
