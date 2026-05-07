"""Provider contract for Brover LLM adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


ContentBlock = dict[str, Any]
Message = dict[str, Any]


@dataclass
class ProviderResponse:
    assistant_blocks: list[ContentBlock]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None = None


class LLMProvider(Protocol):
    async def call(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ProviderResponse:
        ...
