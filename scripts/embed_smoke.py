"""Voyage embedding smoke test.

Captures two frames a few seconds apart from the live camera, embeds both
via the Voyage multimodal API, and prints the cosine similarity between
the two vectors. This proves three things at once:

  1. The Voyage SDK is installed and the API key works.
  2. Camera output flows into the embedding path cleanly.
  3. The embeddings are well-shaped (1024-dim, normalized enough that a
     same-scene similarity comes out high).

Two views of the same scene should land at ~0.9+. If you point the
camera at very different scenes between the two captures, the value
drops noticeably -- that's the signal we'll lean on for localization.

Run on the Pi only (needs rpicam-vid and a configured VOYAGE_API_KEY):

    python -m scripts.embed_smoke
"""
from __future__ import annotations

import asyncio
import math
import sys

from backend import camera as camera_mod
from backend import embeddings


WARMUP_SECONDS = 2.0
GAP_SECONDS = 3.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _wait_for_frame(timeout_seconds: float) -> bytes:
    """Block until the camera has produced its first frame, or time out."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not camera_mod.camera.latest_jpeg:
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("camera produced no frames before timeout")
        await asyncio.sleep(0.1)
    return camera_mod.camera.latest_jpeg


async def main() -> int:
    print("starting camera...")
    await camera_mod.camera.start()

    try:
        print(f"warming up for {WARMUP_SECONDS}s...")
        frame_a = await _wait_for_frame(timeout_seconds=WARMUP_SECONDS + 5.0)
        await asyncio.sleep(WARMUP_SECONDS)
        frame_a = camera_mod.camera.latest_jpeg
        print(f"captured frame 1 ({len(frame_a)} bytes); embedding as 'document'...")
        embed_a = await embeddings.embed_image(frame_a, kind="document")
        print(f"  got {len(embed_a)}-dim vector")

        print(f"\nwait {GAP_SECONDS}s, then capture frame 2...")
        await asyncio.sleep(GAP_SECONDS)
        frame_b = camera_mod.camera.latest_jpeg
        print(f"captured frame 2 ({len(frame_b)} bytes); embedding as 'query'...")
        embed_b = await embeddings.embed_image(frame_b, kind="query")
        print(f"  got {len(embed_b)}-dim vector")

        sim = cosine_similarity(embed_a, embed_b)
        print(f"\ncosine similarity (frame_a, frame_b): {sim:.4f}")

        if sim >= 0.85:
            print("OK: same-scene similarity is high as expected")
            return 0
        if sim >= 0.6:
            print(
                "note: similarity is lower than typical for a static scene "
                "-- did the camera move significantly? this isn't a hard fail."
            )
            return 0
        print(
            "WARN: similarity is unexpectedly low. Confirm the model returned "
            "valid vectors and that the camera was on a stable scene."
        )
        return 1
    except embeddings.EmbeddingError as e:
        print(f"FAIL: embedding error: {e}")
        return 2
    finally:
        await camera_mod.camera.stop()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
