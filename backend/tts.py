"""Text-to-speech wrapper around OpenAI's speech synthesis API.

After the agent loop finishes, the server synthesizes the final reply
to MP3 and sends the bytes to the phone over the WebSocket (base64).
The phone plays it via an HTML Audio element.
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

from backend.config import OPENAI_API_KEY, OPENAI_TTS_MODEL, OPENAI_TTS_VOICE

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def synthesize(text: str) -> bytes:
    """Synthesize text to MP3 bytes. Returns empty bytes for empty input."""
    if not text or not text.strip():
        return b""

    async with _client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text,
        response_format="mp3",
    ) as response:
        chunks: list[bytes] = []
        async for chunk in response.iter_bytes():
            chunks.append(chunk)
        return b"".join(chunks)
