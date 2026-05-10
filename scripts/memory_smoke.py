"""End-to-end memory smoke test: db + embeddings + camera.

This is the Phase 2 acceptance criterion. Teaches two scenes, queries
one of them, and verifies the right scene comes back as the top match.
Proves the entire memory stack works on the Pi with real visual data.

Run on the Pi (needs camera + VOYAGE_API_KEY in .env):

    python -m scripts.memory_smoke

Flow:
    1. Start camera, wait for warm-up.
    2. Prompt: point at scene A, press Enter.
       Capture 5 frames, embed each, store under place 'test_scene_a'.
    3. Prompt: point at scene B (clearly different), press Enter.
       Capture 5 frames, store under 'test_scene_b'.
    4. Prompt: point camera back at scene A, press Enter.
       Capture 1 query frame, embed, find nearest 5 views, print results.
    5. Pass if the top match's place is 'test_scene_a'.

Re-running just adds more frames per scene (get_or_create_place is
idempotent). To start fresh, delete data/brover.db.
"""
from __future__ import annotations

import asyncio
import sys

from backend import camera as camera_mod
from backend import captures, embeddings
from backend.db import connection as db_connection
from backend.db import places

FRAMES_PER_SCENE = 5
GAP_BETWEEN_FRAMES = 0.6  # seconds between captures within one scene
WARMUP_SECONDS = 2.0


async def _wait_for_first_frame(timeout: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not camera_mod.camera.latest_jpeg:
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("camera produced no frames before timeout")
        await asyncio.sleep(0.1)


def _grab_frame() -> bytes:
    frame = camera_mod.camera.latest_jpeg
    if not frame:
        raise RuntimeError("camera has no frame yet")
    return frame


async def _teach_scene(conn, place_name: str) -> int:
    place_id = places.get_or_create_place(conn, place_name)
    print(f"  place_id for {place_name!r}: {place_id}")
    for i in range(FRAMES_PER_SCENE):
        await asyncio.sleep(GAP_BETWEEN_FRAMES)
        frame = _grab_frame()
        image_path = captures.save_jpeg(frame)
        vector = await embeddings.embed_image(frame, kind="document")
        view_id = places.add_place_view(
            conn,
            place_id=place_id,
            image_path=image_path,
            embedding=vector,
        )
        print(f"  [{i + 1}/{FRAMES_PER_SCENE}] view_id={view_id} {image_path}")
    return place_id


async def main() -> int:
    print("starting camera...")
    await camera_mod.camera.start()
    conn = db_connection.connect()

    try:
        await _wait_for_first_frame(timeout=WARMUP_SECONDS + 5.0)
        await asyncio.sleep(WARMUP_SECONDS)

        input("\n>>> Point camera at SCENE A and press Enter")
        print("teaching test_scene_a:")
        await _teach_scene(conn, "test_scene_a")

        input("\n>>> Point camera at SCENE B (clearly different) and press Enter")
        print("teaching test_scene_b:")
        await _teach_scene(conn, "test_scene_b")

        input("\n>>> Point camera BACK at SCENE A and press Enter")
        await asyncio.sleep(0.5)
        query_frame = _grab_frame()
        query_path = captures.save_jpeg(query_frame)
        print(f"\nquery frame stored at {query_path}")

        query_vector = await embeddings.embed_image(query_frame, kind="query")
        results = places.find_nearest_place_views(conn, query_vector, k=5)

        print("\nnearest views (lower distance = more similar):")
        for r in results:
            print(
                f"  place={r.place_name!r} view_id={r.view_id} "
                f"distance={r.distance:.4f}"
            )

        if not results:
            print("\nFAIL: no results returned")
            return 1

        top = results[0]
        if top.place_name == "test_scene_a":
            print(
                f"\nOK: top match is 'test_scene_a' "
                f"(distance={top.distance:.4f}). End-to-end stack works."
            )
            return 0

        print(
            f"\nFAIL: top match was {top.place_name!r}, expected 'test_scene_a'. "
            "Two likely causes: the two scenes were too visually similar, or "
            "the embedding stack is misbehaving."
        )
        return 1
    except embeddings.EmbeddingError as e:
        print(f"FAIL: embedding error: {e}")
        return 2
    finally:
        conn.close()
        await camera_mod.camera.stop()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
