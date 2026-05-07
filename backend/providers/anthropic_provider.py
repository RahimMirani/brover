"""Anthropic provider adapter."""
from __future__ import annotations

import time
from typing import Any

from anthropic import AsyncAnthropic

from backend.config import ANTHROPIC_API_KEY
from backend.providers import ContentBlock, ProviderResponse


def _block_to_dict(block: Any) -> ContentBlock:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}) or {},
        }
    return {"type": btype or "unknown"}


class AnthropicProvider:
    def __init__(self) -> None:
        self._client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def call(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ProviderResponse:
        started = time.perf_counter()
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return ProviderResponse(
            assistant_blocks=[_block_to_dict(block) for block in response.content],
            stop_reason=response.stop_reason or "",
            input_tokens=int(getattr(response.usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(response.usage, "output_tokens", 0) or 0),
            latency_ms=latency_ms,
        )
