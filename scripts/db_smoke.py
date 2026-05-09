"""Database smoke test using hand-rolled fake vectors.

Goal: prove that the schema applies cleanly, the place CRUD is wired up
correctly, and sqlite-vec returns the right rowids back -- all without
needing Voyage configured. Runs end-to-end in a few seconds.

Run from the repo root:
    python -m scripts.db_smoke

What it does:
1. Open data/brover.db (creating data/ and the file on first run).
2. Register a 'smoketest' place.
3. Insert three views with deterministic random vectors (seeds 100/101/102).
4. Query with the same vector as view 0; expect view 0 back as top match.
5. Query with the same vector as view 1; expect view 1 back as top match.
6. List all known places and view counts.

Exit code is 0 on success and non-zero on any unexpected result, so this
script doubles as a CI-friendly health check later.
"""
from __future__ import annotations

import random
import sys

from backend.db import connection, places


def fake_vector(seed: int) -> list[float]:
    """Deterministic pseudo-random vector at the configured embedding dim."""
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(connection.EMBEDDING_DIM)]


def main() -> int:
    conn = connection.connect()
    print(f"opened db at {connection.DB_PATH}")

    place_id = places.get_or_create_place(conn, "smoketest")
    print(f"place id for 'smoketest': {place_id}")

    seeds = [100, 101, 102]
    view_ids: list[int] = []
    for i, seed in enumerate(seeds):
        vid = places.add_place_view(
            conn,
            place_id=place_id,
            image_path=f"data/captures/smoketest_{seed}.jpg",
            embedding=fake_vector(seed),
            heading_deg=float(i * 90),
            distance_cm=42.0,
        )
        view_ids.append(vid)
    print(f"inserted view ids: {view_ids}")

    failures: list[str] = []

    for i, seed in enumerate(seeds):
        matches = places.find_nearest_place_views(conn, fake_vector(seed), k=3)
        print(f"\nquery with seed={seed} (expected top match view_id={view_ids[i]}):")
        for m in matches:
            print(
                f"  view_id={m.view_id} place={m.place_name!r} "
                f"cosine_distance={m.distance:.6f}"
            )
        if not matches or matches[0].view_id != view_ids[i]:
            failures.append(
                f"seed {seed}: expected view_id {view_ids[i]} as top match, "
                f"got {matches[0].view_id if matches else 'nothing'}"
            )

    print("\nplaces in db:")
    for p in places.list_places(conn):
        print(f"  id={p.id} name={p.name!r} created_at={p.created_at:.3f}")

    conn.close()

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nOK: schema applies, inserts work, nearest-neighbor returns expected rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
