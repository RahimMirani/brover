"""xAI Grok provider adapter using OpenAI-compatible chat completions."""
from __future__ import annotations

import json
import time
from typing import Any

from backend.config import XAI_API_KEY
from backend.providers import ContentBlock, Message, ProviderResponse

TICKS_PER_USD = 10_000_000_000
NANO_USD_PER_USD = 1_000_000_000


def anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        }
        for tool in tools
    ]


def messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", [])
        if role == "user" and _is_tool_result_turn(content):
            translated.extend(_tool_result_messages(content))
        elif role == "user":
            translated.append({"role": "user", "content": _content_to_openai(content)})
        elif role == "assistant":
            translated.append(_assistant_to_openai(content))
        elif role == "system":
            translated.append({"role": "system", "content": _text_from_content(content)})
    return translated


def response_message_to_blocks(message: Any) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    content = getattr(message, "content", None)
    if content:
        blocks.append({"type": "text", "text": content})

    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        raw_args = getattr(function, "arguments", "") if function else ""
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            arguments = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": getattr(tool_call, "id", ""),
                "name": getattr(function, "name", "") if function else "",
                "input": arguments,
            }
        )
    return blocks


def finish_reason_to_stop_reason(finish_reason: str | None) -> str:
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "stop":
        return "end_turn"
    if finish_reason == "length":
        return "max_tokens"
    return finish_reason or ""


def usage_cost_usd(usage: Any) -> float | None:
    ticks = _get_usage_value(usage, "cost_in_usd_ticks")
    if ticks is not None:
        return float(ticks) / TICKS_PER_USD
    nano_usd = _get_usage_value(usage, "cost_in_nano_usd")
    if nano_usd is not None:
        return float(nano_usd) / NANO_USD_PER_USD
    return None


class GrokProvider:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )

    async def call(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ProviderResponse:
        started = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, *messages_to_openai(messages)],
            tools=anthropic_tools_to_openai(tools),
            max_completion_tokens=max_tokens,
            extra_body={"search_parameters": {"mode": "off"}},
        )
        latency_ms = (time.perf_counter() - started) * 1000
        choice = response.choices[0]
        usage = response.usage
        return ProviderResponse(
            assistant_blocks=response_message_to_blocks(choice.message),
            stop_reason=finish_reason_to_stop_reason(choice.finish_reason),
            input_tokens=int(_get_usage_value(usage, "prompt_tokens") or 0),
            output_tokens=int(_get_usage_value(usage, "completion_tokens") or 0),
            latency_ms=latency_ms,
            cost_usd=usage_cost_usd(usage),
        )


def _is_tool_result_turn(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _tool_result_messages(content: list[ContentBlock]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") != "tool_result":
            continue
        tool_call_id = block.get("tool_use_id", "")
        result_content = block.get("content", [])
        if not isinstance(result_content, list):
            result_content = [{"type": "text", "text": str(result_content)}]

        text = _text_from_content(result_content)
        images = [c for c in result_content if c.get("type") == "image"]
        if not images:
            messages.append(
                {"role": "tool", "tool_call_id": tool_call_id, "content": text or "ok"}
            )
            continue

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": text or "Camera frame captured; image attached next.",
            }
        )
        user_content = [_image_to_openai(image) for image in images]
        user_content.append(
            {"type": "text", "text": "(camera frame for the previous tool call)"}
        )
        messages.append({"role": "user", "content": user_content})
    return messages


def _assistant_to_openai(content: list[ContentBlock]) -> dict[str, Any]:
    text = _text_from_content(content)
    tool_calls = [
        {
            "id": block.get("id", ""),
            "type": "function",
            "function": {
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {}) or {}),
            },
        }
        for block in content
        if block.get("type") == "tool_use"
    ]
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _content_to_openai(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    out: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            out.append(_image_to_openai(block))
    return out


def _image_to_openai(block: ContentBlock) -> dict[str, Any]:
    source = block.get("source", {})
    media_type = source.get("media_type", "image/jpeg")
    data = source.get("data", "")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{data}"},
    }


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return " ".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ).strip()


def _get_usage_value(usage: Any, name: str) -> Any:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(name)
    return getattr(usage, name, None)
