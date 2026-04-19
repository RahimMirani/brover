"""Claude agent loop.

Single entrypoint: run_agent(user_text, ws_send, cancel_event).

Flow per user command:
    1. Snapshot the current camera frame and prepend it to the user message
       so the model starts with visual context.
    2. Call Claude with the tool schemas.
    3. If Claude returns tool_use blocks, dispatch each via
       backend.tools.dispatch, then append the tool_results and loop.
    4. If Claude returns end_turn (or stops calling tools), extract the
       final text and return it. The caller TTS's that text for the user.

The loop is bounded by MAX_AGENT_ITERATIONS and honours cancel_event between
iterations and between tool calls, so the user's e-stop or a manual-drive
override can interrupt cleanly.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Awaitable, Callable

from anthropic import AsyncAnthropic

from backend import camera as camera_mod
from backend.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    MAX_AGENT_ITERATIONS,
    MAX_MOTOR_SECONDS,
)
from backend.tools import TOOL_SCHEMAS, dispatch

logger = logging.getLogger(__name__)

_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


SYSTEM_PROMPT = f"""You are Brover, an AI agent embedded in a small two-wheel-drive RC rover.

You control the rover by calling tools. You can drive forward, drive backward,
spin in place (tank-style turning: the two wheels spin opposite directions),
stop, pause, and capture a still image from the forward-facing camera.

Important constraints:
- You cannot steer while driving. To change heading, stop and call `turn`.
- Every motor call is hard-capped at {MAX_MOTOR_SECONDS} seconds server-side.
  If you need more motion, chain multiple calls.
- You have no visual input unless you call `look`. Call it whenever you
  need to check surroundings. After any significant motion, a fresh `look`
  before the next move is usually wise.
- The user can override you with manual controls at any time. If your turn
  is cancelled mid-sequence, just wrap up and report what you did.

Rough motion calibration on a hard floor (varies with surface and battery):
- turn(0.5s) -> ~45 degrees
- turn(0.3s) -> ~25 degrees
- forward(1.0s) -> ~0.5 meters

Reply style: your final response is spoken aloud to the user. Keep it to one
or two short sentences. Say what you did and, if relevant, what you saw.
"""


WsSend = Callable[[dict[str, Any]], Awaitable[None]]


def _initial_user_content(user_text: str, frame: bytes) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if frame:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(frame).decode("ascii"),
                },
            }
        )
    content.append({"type": "text", "text": user_text})
    return content


def _extract_text(assistant_blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in assistant_blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "") or ""
            if text:
                parts.append(text)
    return " ".join(parts).strip()


def _strip_images(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop image blocks for WS relay to the phone (the phone has the live stream)."""
    return [c for c in content if c.get("type") != "image"]


async def run_agent(
    user_text: str,
    ws_send: WsSend,
    cancel_event: asyncio.Event,
) -> str:
    """Run the Claude tool-use loop. Returns the final text for TTS."""
    frame = camera_mod.camera.latest_jpeg
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _initial_user_content(user_text, frame)},
    ]

    for iteration in range(MAX_AGENT_ITERATIONS):
        if cancel_event.is_set():
            return "Cancelled."

        logger.info("agent iteration %d", iteration + 1)
        response = await _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        assistant_blocks = list(response.content)
        messages.append({"role": "assistant", "content": assistant_blocks})

        if response.stop_reason == "end_turn":
            return _extract_text(assistant_blocks) or "Done."

        tool_use_blocks = [
            b for b in assistant_blocks if getattr(b, "type", None) == "tool_use"
        ]
        if not tool_use_blocks:
            return _extract_text(assistant_blocks) or "Done."

        tool_result_content: list[dict[str, Any]] = []
        for block in tool_use_blocks:
            if cancel_event.is_set():
                break
            name = block.name
            arguments = dict(block.input) if block.input else {}

            await ws_send(
                {"type": "tool_call", "name": name, "arguments": arguments}
            )

            result_content = await dispatch(name, arguments)

            await ws_send(
                {
                    "type": "tool_result",
                    "name": name,
                    "content": _strip_images(result_content),
                }
            )

            tool_result_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                }
            )

        if cancel_event.is_set():
            return "Cancelled."

        messages.append({"role": "user", "content": tool_result_content})

    return "I hit my iteration limit."
