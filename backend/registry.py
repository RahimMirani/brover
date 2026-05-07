"""Model catalog for Brover's LLM providers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    api_id: str
    display_name: str
    input_per_mtok: float
    output_per_mtok: float
    supports_vision: bool


DEFAULT_MODEL_ID = "claude-sonnet-4-5"


# Pricing last verified 2026-05-07. Provider-reported exact cost should win
# when available because cached, image, and reasoning tokens may bill separately.
MODELS: dict[str, ModelSpec] = {
    "claude-sonnet-4-5": ModelSpec(
        provider="anthropic",
        api_id="claude-sonnet-4-5",
        display_name="Claude Sonnet 4.5",
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        supports_vision=True,
    ),
    "claude-haiku-4-5": ModelSpec(
        provider="anthropic",
        api_id="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        input_per_mtok=1.0,
        output_per_mtok=5.0,
        supports_vision=True,
    ),
    "grok-4.3": ModelSpec(
        provider="grok",
        api_id="grok-4.3",
        display_name="Grok 4.3",
        input_per_mtok=1.25,
        output_per_mtok=2.5,
        supports_vision=True,
    ),
}


def estimated_cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    spec = MODELS[model_id]
    return (
        (input_tokens / 1_000_000) * spec.input_per_mtok
        + (output_tokens / 1_000_000) * spec.output_per_mtok
    )
