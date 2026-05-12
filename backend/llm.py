"""Provider-agnostic agent loop.

Single entrypoint: run_agent(user_text, ws_send, cancel_event, model).

Flow per user command:
    1. Snapshot the current camera frame and prepend it to the user message
       so the model starts with visual context.
    2. Call the selected LLM provider with the tool schemas.
    3. If the model returns tool_use blocks, dispatch each via
       backend.tools.dispatch, then append the tool_results and loop.
    4. If the model returns end_turn (or stops calling tools), extract the
       final text and return it. The caller TTS's that text for the user.

The loop is bounded by MAX_AGENT_ITERATIONS and honours cancel_event between
iterations and between tool calls, so the user's e-stop or a manual-drive
override can interrupt cleanly.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from backend import camera as camera_mod
from backend import registry
from backend.config import MAX_AGENT_ITERATIONS, MAX_MOTOR_SECONDS
from backend.metrics import metrics
from backend.providers import LLMProvider
from backend.providers.anthropic_provider import AnthropicProvider
from backend.providers.grok_provider import GrokProvider
from backend.tools import TOOL_SCHEMAS, dispatch

logger = logging.getLogger(__name__)

_providers: dict[str, LLMProvider] = {
    "anthropic": AnthropicProvider(),
    "grok": GrokProvider(),
}


SYSTEM_PROMPT = f"""You are Brover, an AI agent embedded in a small two-wheel-drive RC rover.

You control the rover by calling tools. You can drive forward, drive backward,
spin in place (tank-style turning: the two wheels spin opposite directions),
stop, pause, capture a still image from the forward-facing camera, and read the
latest forward ultrasonic distance.

Important constraints:
- You cannot steer while driving. To change heading, stop and call `turn`.
- Every motor call is hard-capped at {MAX_MOTOR_SECONDS} seconds server-side.
  If you need more motion, chain multiple calls.
- You have no visual input unless you call `look`. Call it whenever you
  need to check surroundings. After any significant motion, a fresh `look`
  before the next move is usually wise.
- The backend continuously monitors the forward ultrasonic sensor and can stop
  unsafe forward motion automatically. Call `distance` when you need the current
  distance for reasoning or to report it to the user.
- The user can override you with manual controls at any time. If your turn
  is cancelled mid-sequence, just wrap up and report what you did.

Rough motion calibration on a hard floor (varies with surface and battery):
- turn(0.5s) -> ~45 degrees
- turn(0.3s) -> ~25 degrees
- forward(1.0s) -> ~0.5 meters

Memory:
You have a local memory of places the user has taught you. Use these tools:
- `remember_here(name)` for "remember this as the kitchen"-style commands.
  Captures a few frames where the rover is currently sitting.
- `start_tour()` / `tag_place(name)` / `end_tour()` for a guided tour:
  the user drives manually around the space and calls out place names as
  they go ("this is the kitchen", "we're now in the hallway"). Tag each
  one when the user identifies it; end the tour when the user is done.
- `find_place(name)` to answer "do you know the X?" without touching the
  camera, and to enumerate what you know.
- `localize()` to answer "where am I?". Be honest with the result: it can
  report empty memory, ambiguity, or that the place is unrecognised.
- `forget_place(name)` only when the user explicitly asks to forget or
  re-teach a place.
Each memory call that takes a picture also makes a Voyage embedding API
call, so don't call them speculatively.

Reply style: your final response is spoken aloud to the user. Keep it to one
or two short sentences. Say what you did and, if relevant, what you saw.
"""


WsSend = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class AgentResult:
    text: str
    model: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    iterations: int = 0


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


def _extract_text(assistant_blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in assistant_blocks:
        if block.get("type") == "text":
            text = block.get("text", "") or ""
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
    model: str = registry.DEFAULT_MODEL_ID,
) -> AgentResult:
    """Run the model tool-use loop. Returns final text and telemetry."""
    spec = registry.MODELS[model]
    provider = _providers[spec.provider]
    frame = camera_mod.camera.latest_jpeg
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _initial_user_content(user_text, frame)},
    ]
    session_latency_ms = 0.0
    session_input_tokens = 0
    session_output_tokens = 0
    session_cost_usd = 0.0

    def result(text: str, iterations: int) -> AgentResult:
        return AgentResult(
            text=text,
            model=model,
            latency_ms=session_latency_ms,
            input_tokens=session_input_tokens,
            output_tokens=session_output_tokens,
            cost_usd=session_cost_usd,
            iterations=iterations,
        )

    for iteration in range(MAX_AGENT_ITERATIONS):
        if cancel_event.is_set():
            return result("Cancelled.", iteration)

        logger.info("agent iteration %d using %s", iteration + 1, model)
        response = await provider.call(
            model=spec.api_id,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        call_cost = (
            response.cost_usd
            if response.cost_usd is not None
            else registry.estimated_cost_usd(
                model, response.input_tokens, response.output_tokens
            )
        )
        metrics.record_llm_call(
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            cost_usd=call_cost,
        )
        session_latency_ms += response.latency_ms
        session_input_tokens += response.input_tokens
        session_output_tokens += response.output_tokens
        session_cost_usd += call_cost

        assistant_blocks = response.assistant_blocks
        messages.append({"role": "assistant", "content": assistant_blocks})

        if response.stop_reason == "end_turn":
            return result(_extract_text(assistant_blocks) or "Done.", iteration + 1)

        tool_use_blocks = [
            b for b in assistant_blocks if b.get("type") == "tool_use"
        ]
        if not tool_use_blocks:
            return result(_extract_text(assistant_blocks) or "Done.", iteration + 1)

        tool_result_content: list[dict[str, Any]] = []
        for block in tool_use_blocks:
            if cancel_event.is_set():
                break
            name = block.get("name", "")
            arguments = dict(block.get("input") or {})

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
                    "tool_use_id": block.get("id", ""),
                    "content": result_content,
                }
            )

        if cancel_event.is_set():
            return result("Cancelled.", iteration + 1)

        messages.append({"role": "user", "content": tool_result_content})

    return result("I hit my iteration limit.", MAX_AGENT_ITERATIONS)
