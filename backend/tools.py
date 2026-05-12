"""Tools exposed to Claude.

This module is the single source of truth for everything Claude can do:
what tools exist, how they are described to the model, and what actually
runs when Claude calls them.

Each handler returns a list of Anthropic content blocks suitable for use
as `tool_result.content`. Most return a single text block; `look` returns
an image block plus a short text caption.

Memory tools (`remember_here`, `start_tour`/`tag_place`/`end_tour`,
`find_place`, `localize`, `forget_place`) get the shared DB connection
lazily from `backend.db.connection.get_shared_connection` so they fail
with a clean error message in environments where the live server has
not initialised it (e.g. unit-test imports).
"""
from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Awaitable, Callable

from backend import camera as camera_mod
from backend import captures, embeddings, localization, motors, teaching
from backend.config import MAX_MOTOR_SECONDS
from backend.db import connection as db_connection
from backend.db import places
from backend.distance_sensor import distance_sensor
from backend.metrics import metrics

logger = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 5.0


ContentBlock = dict[str, Any]
ToolResult = list[ContentBlock]
ToolHandler = Callable[..., Awaitable[ToolResult]]


def _text(msg: str) -> ToolResult:
    return [{"type": "text", "text": msg}]


async def _forward(seconds: float) -> ToolResult:
    result = await motors.forward(float(seconds))
    if result.stopped_reason == "obstacle":
        distance = (
            "unknown distance"
            if result.distance_cm is None
            else f"{result.distance_cm:.1f} cm"
        )
        return _text(
            "Forward motion stopped after "
            f"{result.elapsed_seconds:.2f}s because an obstacle is ahead "
            f"at {distance}."
        )
    return _text(f"Drove forward for {result.elapsed_seconds:.2f}s.")


async def _backward(seconds: float) -> ToolResult:
    await motors.backward(float(seconds))
    clamped = min(float(seconds), MAX_MOTOR_SECONDS)
    return _text(f"Drove backward for {clamped:.2f}s.")


async def _turn(direction: str, seconds: float) -> ToolResult:
    if direction not in ("left", "right"):
        return _text(f"Error: direction must be 'left' or 'right', got {direction!r}.")
    await motors.turn(direction, float(seconds))
    clamped = min(float(seconds), MAX_MOTOR_SECONDS)
    return _text(f"Turned {direction} for {clamped:.2f}s.")


async def _stop() -> ToolResult:
    motors.stop()
    return _text("Motors stopped.")


async def _look() -> ToolResult:
    frame = camera_mod.camera.latest_jpeg
    if not frame:
        return _text("No camera frame is available yet.")
    b64 = base64.b64encode(frame).decode("ascii")
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        },
        {"type": "text", "text": "Current camera view."},
    ]


async def _distance() -> ToolResult:
    reading = distance_sensor.latest()
    if reading.distance_cm is None:
        return _text(f"No distance reading is available yet. Status: {reading.status}.")

    stale_note = " Reading is stale." if reading.stale else ""
    safety = "safe" if reading.safe_for_forward else "too close for forward motion"
    return _text(
        f"Forward distance is {reading.distance_cm:.1f} cm "
        f"({safety}; minimum safe forward distance is "
        f"{reading.min_safe_forward_cm:.1f} cm). Status: {reading.status}."
        f"{stale_note}"
    )


async def _wait(seconds: float) -> ToolResult:
    s = max(0.0, min(float(seconds), MAX_WAIT_SECONDS))
    await asyncio.sleep(s)
    return _text(f"Waited {s:.2f}s.")


# -----------------------------------------------------------------------------
# Memory tools
# -----------------------------------------------------------------------------

_TEACH_FRAMES = 5
_TEACH_GAP_S = 0.4
_TOUR_TAG_SECONDS = 4.0
_TOUR_TAG_MAX_FRAMES = 5


def _get_jpeg() -> bytes:
    return camera_mod.camera.latest_jpeg


async def _remember_here(name: str) -> ToolResult:
    """Stop-and-say teach. Captures a handful of frames where the rover is
    currently sitting and files them all under one place name."""
    name = (name or "").strip()
    if not name:
        return _text("Error: place name is required.")

    try:
        conn = db_connection.get_shared_connection()
    except RuntimeError as e:
        return _text(f"Error: memory not available ({e}).")

    if not _get_jpeg():
        return _text("No camera frame is available yet, cannot remember.")

    try:
        result = await teaching.teach_place_stationary(
            conn,
            name,
            get_jpeg=_get_jpeg,
            embed_image=embeddings.embed_image,
            save_jpeg=captures.save_jpeg,
            frames=_TEACH_FRAMES,
            gap_s=_TEACH_GAP_S,
        )
    except embeddings.EmbeddingError as e:
        return _text(f"Error: embedding service failed ({e}).")

    if result.views_added == 0:
        return _text(
            f"Tried to remember {name!r} but no frames were captured. "
            "Check the camera and try again."
        )
    return _text(
        f"Remembered {name!r}: stored {result.views_added} views."
    )


async def _start_tour() -> ToolResult:
    """Begin a continuous-capture tour. The user drives manually and tags
    places along the way; `end_tour` finishes the session."""
    if teaching.tour_buffer.active:
        return _text(
            "A tour is already running. Tag any pending places, then call end_tour."
        )
    await teaching.tour_buffer.start(get_jpeg=_get_jpeg)
    return _text(
        "Tour started. Drive the rover around and call tag_place(name) when "
        "you reach a place worth remembering. Call end_tour when finished."
    )


async def _tag_place(name: str) -> ToolResult:
    """Tag the most recent few seconds of buffered frames as a named place."""
    name = (name or "").strip()
    if not name:
        return _text("Error: place name is required.")
    if not teaching.tour_buffer.active:
        return _text(
            "No tour is currently running. Call start_tour first, then tag places."
        )

    try:
        conn = db_connection.get_shared_connection()
    except RuntimeError as e:
        return _text(f"Error: memory not available ({e}).")

    try:
        result = await teaching.tour_buffer.tag_window(
            conn,
            name,
            embed_image=embeddings.embed_image,
            save_jpeg=captures.save_jpeg,
            seconds=_TOUR_TAG_SECONDS,
            max_frames=_TOUR_TAG_MAX_FRAMES,
        )
    except embeddings.EmbeddingError as e:
        return _text(f"Error: embedding service failed ({e}).")

    if result.views_added == 0:
        return _text(
            f"Tagged {name!r}, but no fresh frames were in the recent window "
            "(maybe you tagged the same spot twice in a row). Drive a bit and "
            "try again."
        )
    return _text(
        f"Tagged {name!r}: stored {result.views_added} views from the last "
        f"{_TOUR_TAG_SECONDS:.0f}s of the tour."
    )


async def _end_tour() -> ToolResult:
    """End the active tour and report a summary."""
    if not teaching.tour_buffer.active:
        return _text("No tour is currently running.")
    summary = await teaching.tour_buffer.end()
    if not summary.places_taught:
        return _text(
            f"Tour ended after {summary.duration_seconds:.0f}s. No places were "
            "tagged, so nothing was saved."
        )
    names = ", ".join(summary.places_taught)
    return _text(
        f"Tour ended after {summary.duration_seconds:.0f}s. "
        f"Tagged {summary.tags_applied} places ({names}) for a total of "
        f"{summary.total_views_added} stored views."
    )


async def _find_place(name: str) -> ToolResult:
    """Look up a place by name without touching the camera. Useful for
    'do you know the kitchen?' style questions."""
    name = (name or "").strip()
    if not name:
        return _text("Error: place name is required.")

    try:
        conn = db_connection.get_shared_connection()
    except RuntimeError as e:
        return _text(f"Error: memory not available ({e}).")

    place = places.get_place_by_name(conn, name)
    if place is None:
        known = [s.name for s in places.list_places_with_counts(conn)]
        if not known:
            return _text(
                f"I don't have a place called {name!r}, and I haven't been "
                "taught anywhere yet."
            )
        return _text(
            f"I don't have a place called {name!r}. I know: {', '.join(known)}."
        )

    summaries = {s.id: s for s in places.list_places_with_counts(conn)}
    summary = summaries.get(place.id)
    view_count = 0 if summary is None else summary.view_count
    return _text(
        f"Yes, I know {name!r}: {view_count} stored view"
        f"{'s' if view_count != 1 else ''}."
    )


async def _localize() -> ToolResult:
    """Embed the current camera view and report the most likely place."""
    try:
        conn = db_connection.get_shared_connection()
    except RuntimeError as e:
        return _text(f"Error: memory not available ({e}).")

    jpeg = _get_jpeg()
    if not jpeg:
        return _text("No camera frame is available yet, cannot localize.")

    try:
        result = await localization.localize_jpeg(
            conn, jpeg, embed_image=embeddings.embed_image
        )
    except embeddings.EmbeddingError as e:
        return _text(f"Error: embedding service failed ({e}).")

    if result.status == "empty_memory":
        return _text(
            "I don't recognize this place — I haven't been taught anywhere yet. "
            "Drive me somewhere and say 'remember this as <name>'."
        )

    assert result.best is not None
    best = result.best

    if result.status == "confident":
        return _text(
            f"I'm pretty sure this is {best.place_name!r} "
            f"(similarity {best.similarity:.2f})."
        )
    if result.status == "ambiguous":
        if result.alternatives:
            runner = result.alternatives[0]
            return _text(
                f"I think this might be {best.place_name!r} "
                f"(similarity {best.similarity:.2f}), but it could also be "
                f"{runner.place_name!r} (similarity {runner.similarity:.2f})."
            )
        return _text(
            f"I think this might be {best.place_name!r} "
            f"(similarity {best.similarity:.2f}), but I'm not sure."
        )
    return _text(
        "I don't recognize this place. The closest stored place is "
        f"{best.place_name!r} but only at similarity {best.similarity:.2f}."
    )


async def _forget_place(name: str) -> ToolResult:
    """Remove a place and every view stored for it. Used to fix mis-teaches."""
    name = (name or "").strip()
    if not name:
        return _text("Error: place name is required.")

    try:
        conn = db_connection.get_shared_connection()
    except RuntimeError as e:
        return _text(f"Error: memory not available ({e}).")

    deleted = places.delete_place_by_name(conn, name)
    if not deleted:
        return _text(f"No place called {name!r} to forget.")
    return _text(f"Forgot {name!r} and all of its stored views.")


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "forward",
        "description": (
            "Drive the rover forward for a given duration. The rover cannot steer "
            "while driving; to change heading, stop and call `turn` first. Max single "
            f"call is {MAX_MOTOR_SECONDS} seconds -- longer requests are clamped."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "How long to drive forward, in seconds.",
                    "minimum": 0,
                    "maximum": MAX_MOTOR_SECONDS,
                }
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "backward",
        "description": (
            "Drive the rover backward for a given duration. Max single call is "
            f"{MAX_MOTOR_SECONDS} seconds -- longer requests are clamped."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "How long to drive backward, in seconds.",
                    "minimum": 0,
                    "maximum": MAX_MOTOR_SECONDS,
                }
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "turn",
        "description": (
            "Spin the rover in place. The rover cannot drive forward while turning. "
            "Rough calibration on a hard floor: 0.5s -> ~45 degrees, 0.3s -> ~25 "
            "degrees. Actual angle varies with surface friction and battery level; "
            "call `look` afterwards to verify."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Which way to spin.",
                },
                "seconds": {
                    "type": "number",
                    "description": "How long to spin, in seconds.",
                    "minimum": 0,
                    "maximum": MAX_MOTOR_SECONDS,
                },
            },
            "required": ["direction", "seconds"],
        },
    },
    {
        "name": "stop",
        "description": "Immediately stop all motor motion. Always safe to call.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "look",
        "description": (
            "Capture the current camera view and return it as an image. Use this "
            "to check surroundings, identify obstacles, or find objects or people "
            "the user asked about. You receive no visual input except what comes "
            "back from this tool, so call it whenever you need to see."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "distance",
        "description": (
            "Return the latest live ultrasonic distance reading straight ahead. "
            "Use this when you need to know how far the nearest forward obstacle "
            "is. The backend also uses this sensor automatically for forward "
            "motion safety, so you do not need to call this before every move."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "wait",
        "description": f"Pause for a given number of seconds (max {MAX_WAIT_SECONDS}).",
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "How long to wait.",
                    "minimum": 0,
                    "maximum": MAX_WAIT_SECONDS,
                }
            },
            "required": ["seconds"],
        },
    },
    # ----- Memory tools -------------------------------------------------------
    {
        "name": "remember_here",
        "description": (
            "Stationary teach: stand still and capture a handful of camera "
            "frames as a named place. Use when the user says things like "
            "'remember this as the kitchen' or 'this spot is the bedroom'. "
            f"Captures {_TEACH_FRAMES} frames over about "
            f"{_TEACH_FRAMES * _TEACH_GAP_S:.0f} seconds and takes one "
            "Voyage embedding call per frame, so don't call speculatively. "
            "Re-using a name appends views to the existing place; use "
            "forget_place first if you want a clean replacement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the place to remember (e.g. 'kitchen').",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "start_tour",
        "description": (
            "Begin a tour-mode teaching session. The user will drive the "
            "rover manually around the space; you should then call "
            "tag_place(name) each time the user identifies a place ('this "
            "is the kitchen', 'we're now in the hallway'). Call end_tour "
            "when the user says they are done. Only one tour can be active "
            "at a time."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "tag_place",
        "description": (
            "While a tour is active, mark the most recent few seconds of "
            "buffered camera frames as a named place. Use whenever the user "
            "identifies a location during the tour. Requires start_tour to "
            "have been called first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the place the rover is currently in.",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "end_tour",
        "description": (
            "End the active tour and get a summary of places tagged. Use "
            "when the user signals the tour is over."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "find_place",
        "description": (
            "Look up a place by name without using the camera. Use for "
            "'do you know the kitchen?' or 'how many places have you been "
            "taught?'-style questions. Returns whether the place exists and "
            "how many views are stored for it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the place to look up.",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "localize",
        "description": (
            "Embed the current camera view and find the closest matching "
            "place in memory. Use for 'where am I?'-style questions. May "
            "return that no places are taught yet, that the current view "
            "is ambiguous between two places, or that nothing recognisable "
            "is in view — report the result honestly to the user."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "forget_place",
        "description": (
            "Permanently delete a place and every stored view for it. Only "
            "use when the user explicitly asks to forget or to start over "
            "for that place. Returns whether anything was actually deleted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the place to forget.",
                }
            },
            "required": ["name"],
        },
    },
]


TOOL_REGISTRY: dict[str, ToolHandler] = {
    "forward": _forward,
    "backward": _backward,
    "turn": _turn,
    "stop": _stop,
    "look": _look,
    "distance": _distance,
    "wait": _wait,
    "remember_here": _remember_here,
    "start_tour": _start_tour,
    "tag_place": _tag_place,
    "end_tour": _end_tour,
    "find_place": _find_place,
    "localize": _localize,
    "forget_place": _forget_place,
}


async def dispatch(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Run a tool by name and return Anthropic content blocks for tool_result."""
    handler = TOOL_REGISTRY.get(name)
    if handler is None:
        return _text(f"Error: unknown tool {name!r}.")
    metrics.record_tool(name)
    try:
        return await handler(**arguments)
    except TypeError as e:
        logger.warning("tool %s called with bad args %r: %s", name, arguments, e)
        return _text(f"Error: bad arguments for {name}: {e}")
    except Exception as e:
        logger.exception("tool %s crashed", name)
        return _text(f"Error running {name}: {e}")
