"""Tools exposed to Claude.

This module is the single source of truth for everything Claude can do:
what tools exist, how they are described to the model, and what actually
runs when Claude calls them.

Each handler returns a list of Anthropic content blocks suitable for use
as `tool_result.content`. Most return a single text block; `look` returns
an image block plus a short text caption.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Awaitable, Callable

from backend import camera as camera_mod
from backend import motors
from backend.config import MAX_MOTOR_SECONDS
from backend.metrics import metrics

logger = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 5.0


ContentBlock = dict[str, Any]
ToolResult = list[ContentBlock]
ToolHandler = Callable[..., Awaitable[ToolResult]]


def _text(msg: str) -> ToolResult:
    return [{"type": "text", "text": msg}]


async def _forward(seconds: float) -> ToolResult:
    await motors.forward(float(seconds))
    clamped = min(float(seconds), MAX_MOTOR_SECONDS)
    return _text(f"Drove forward for {clamped:.2f}s.")


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


async def _wait(seconds: float) -> ToolResult:
    s = max(0.0, min(float(seconds), MAX_WAIT_SECONDS))
    await asyncio.sleep(s)
    return _text(f"Waited {s:.2f}s.")


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
]


TOOL_REGISTRY: dict[str, ToolHandler] = {
    "forward": _forward,
    "backward": _backward,
    "turn": _turn,
    "stop": _stop,
    "look": _look,
    "wait": _wait,
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
