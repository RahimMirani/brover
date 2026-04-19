"""Speech-to-text wrapper around OpenAI's transcription API.

The phone records a short audio blob (webm/opus from MediaRecorder) and
ships it to the server over the WebSocket as a single base64-encoded
message. We forward the decoded bytes to OpenAI and return the text.
"""
from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI

from backend.config import OPENAI_API_KEY, OPENAI_STT_MODEL

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def transcribe(audio: bytes, filename: str = "audio.webm") -> str:
    """Transcribe an audio blob. Returns the plain transcript text.

    filename is passed through to OpenAI so the API can infer the format;
    the extension must match the actual audio container (webm, mp3, wav...).
    """
    if not audio:
        return ""

    response = await _client.audio.transcriptions.create(
        model=OPENAI_STT_MODEL,
        file=(filename, audio),
    )
    text: Optional[str] = getattr(response, "text", None)
    return (text or "").strip()
