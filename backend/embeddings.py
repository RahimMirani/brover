"""Image embedding via Voyage AI's multimodal model.

The whole spatial-memory system rests on this module. Every captured frame
is sent here, returns a 1024-dim vector, and that vector is what
`backend/db/places.py` stores or queries against.

Two `kind` values matter:

  "document" — used when storing a frame for later retrieval (teaching).
  "query"    — used when looking up a frame against stored ones (localizing).

Voyage normalizes the two slightly differently and using the right one on
each side measurably improves retrieval quality. Callers from the
teaching path use "document"; callers from the localization path use
"query".

Errors from Voyage (network failures, auth, rate limits, dimension
mismatch) are wrapped in `EmbeddingError` so callers don't have to import
`voyageai` to handle them. Higher layers decide whether to retry, buffer,
or hard-fail based on context.
"""
from __future__ import annotations

import io
import logging
from typing import Literal

import voyageai
from PIL import Image

from backend.config import VOYAGE_API_KEY, VOYAGE_EMBED_MODEL
from backend.db.connection import EMBEDDING_DIM

logger = logging.getLogger(__name__)

EmbedKind = Literal["document", "query"]


class EmbeddingError(RuntimeError):
    """Raised when the Voyage call fails or returns an unexpected result."""


# Lazy singleton: defer client creation until the first call so importing
# this module never fails just because the key isn't loaded yet (e.g. in
# tooling that imports the package without running the app).
_client: voyageai.AsyncClient | None = None


def _get_client() -> voyageai.AsyncClient:
    global _client
    if _client is None:
        if not VOYAGE_API_KEY:
            raise EmbeddingError(
                "VOYAGE_API_KEY is not set; add it to .env before calling embed_image"
            )
        _client = voyageai.AsyncClient(api_key=VOYAGE_API_KEY)
    return _client


async def embed_image(jpeg_bytes: bytes, *, kind: EmbedKind = "document") -> list[float]:
    """Return a 1024-dim embedding for the given JPEG.

    `kind="document"` for frames being stored (teaching path).
    `kind="query"` for frames being looked up (localization path).
    """
    if not jpeg_bytes:
        raise EmbeddingError("embed_image: empty jpeg_bytes")

    try:
        image = Image.open(io.BytesIO(jpeg_bytes))
        image.load()
    except Exception as e:
        raise EmbeddingError(f"failed to decode jpeg: {e}") from e

    client = _get_client()

    try:
        result = await client.multimodal_embed(
            inputs=[[image]],
            model=VOYAGE_EMBED_MODEL,
            input_type=kind,
        )
    except Exception as e:
        raise EmbeddingError(f"voyage call failed: {e}") from e

    if not result.embeddings:
        raise EmbeddingError("voyage returned no embeddings")

    vector = list(result.embeddings[0])
    if len(vector) != EMBEDDING_DIM:
        raise EmbeddingError(
            f"voyage returned {len(vector)}-dim vector, expected {EMBEDDING_DIM}"
        )

    logger.debug("embedded jpeg (%d bytes) as %d-dim vector", len(jpeg_bytes), len(vector))
    return vector
