"""Place localization: the read path of Brover's spatial memory.

Given a current camera frame, embed it and ask the places DB which
stored views are closest. Turn that raw distance list into a four-state
answer the agent can act on:

  "empty_memory"  the DB has no place views yet. Agent should ask the
                  user to teach somewhere first.
  "confident"     top match is comfortably below HIGH_CONFIDENCE_DISTANCE
                  and beats its runner-up of a different place by at
                  least MIN_MARGIN. Agent should report the name plainly.
  "ambiguous"     top match is plausible but the runner-up is too close,
                  or it sits between HIGH and LOW thresholds. Agent
                  should hedge ("might be the kitchen, but I'm not sure").
  "unrecognized"  top match is further than LOW_CONFIDENCE_DISTANCE.
                  Agent should say it does not recognise this place.

Thresholds are set conservatively and will need recalibration once a
real apartment has been taught. They live as module constants so the
test suite and a future calibration script can read them.

Embedding is dependency-injected (same pattern as teaching.py): the live
path passes embeddings.embed_image; tests pass a deterministic fake.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Awaitable, Callable, Literal

from backend.db import places

logger = logging.getLogger(__name__)


# Cosine distance thresholds (sqlite-vec: 0 = identical, 2 = opposite).
# These are starting points; expect to recalibrate after the first real
# apartment is taught.
HIGH_CONFIDENCE_DISTANCE = 0.25
LOW_CONFIDENCE_DISTANCE = 0.45
MIN_MARGIN = 0.05


LocalizeStatus = Literal["empty_memory", "confident", "ambiguous", "unrecognized"]

EmbedFn = Callable[[bytes], Awaitable[list[float]]]


@dataclass(frozen=True)
class LocalizeCandidate:
    """One nearest-neighbor hit, normalised for the agent's consumption."""

    place_name: str
    view_id: int
    image_path: str
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity in [-1, 1]. Friendlier than 'distance' for prompts."""
        return 1.0 - self.distance


@dataclass(frozen=True)
class LocalizeResult:
    """Outcome of one `localize_jpeg` call.

    `best` is None only when `status == "empty_memory"`. `alternatives` is
    the runner-up candidates (already filtered: each is a *different place*
    from `best`, so the agent can mention plausible second guesses without
    five rows of the same kitchen)."""

    status: LocalizeStatus
    best: LocalizeCandidate | None
    alternatives: list[LocalizeCandidate]


async def localize_jpeg(
    conn: Connection,
    jpeg: bytes,
    *,
    embed_image: EmbedFn,
    k: int = 5,
) -> LocalizeResult:
    """Embed the frame, query the place index, classify confidence.

    `k` controls the breadth of the nearest-neighbor search; the agent
    only ever sees the top match and a couple alternatives, but a larger
    `k` makes the margin check more meaningful when the top several hits
    are all the same place."""
    if not jpeg:
        raise ValueError("localize_jpeg: empty jpeg")

    if _is_place_views_empty(conn):
        logger.info("localize: empty memory")
        return LocalizeResult(status="empty_memory", best=None, alternatives=[])

    vector = await embed_image(jpeg)
    rows = places.find_nearest_place_views(conn, vector, k=k)
    if not rows:
        # Belt-and-braces: empty result despite non-empty table (shouldn't
        # happen with sqlite-vec, but treat it like empty_memory).
        logger.warning("localize: vector search returned no rows")
        return LocalizeResult(status="empty_memory", best=None, alternatives=[])

    candidates = [
        LocalizeCandidate(
            place_name=row.place_name,
            view_id=row.view_id,
            image_path=row.image_path,
            distance=row.distance,
        )
        for row in rows
    ]
    best = candidates[0]
    alternatives = _alternatives_of_other_places(candidates, best.place_name)

    status = _classify(best, alternatives)
    logger.info(
        "localize: status=%s top=%r distance=%.4f alternatives=%d",
        status,
        best.place_name,
        best.distance,
        len(alternatives),
    )
    return LocalizeResult(status=status, best=best, alternatives=alternatives)


def _is_place_views_empty(conn: Connection) -> bool:
    """Return True if no place views are stored. Cheaper than running a
    vector query just to discover the index is empty."""
    row = conn.execute("SELECT COUNT(*) AS n FROM place_views").fetchone()
    return int(row["n"]) == 0


def _alternatives_of_other_places(
    candidates: list[LocalizeCandidate], top_name: str
) -> list[LocalizeCandidate]:
    """Filter out hits for the same place as `top_name`. Keeps the agent's
    'second guess' list meaningfully different from the first guess."""
    seen: set[str] = {top_name}
    result: list[LocalizeCandidate] = []
    for c in candidates:
        if c.place_name in seen:
            continue
        seen.add(c.place_name)
        result.append(c)
    return result


def _classify(
    best: LocalizeCandidate, alternatives: list[LocalizeCandidate]
) -> LocalizeStatus:
    if best.distance > LOW_CONFIDENCE_DISTANCE:
        return "unrecognized"

    if best.distance <= HIGH_CONFIDENCE_DISTANCE:
        # Top hit is close. Only call it ambiguous if a *different* place is
        # also close enough that the margin between them is small.
        if alternatives:
            margin = alternatives[0].distance - best.distance
            if margin < MIN_MARGIN:
                return "ambiguous"
        return "confident"

    # best.distance is between HIGH and LOW: we have something, but not
    # sharply. Default to ambiguous so the agent hedges.
    return "ambiguous"


__all__ = [
    "HIGH_CONFIDENCE_DISTANCE",
    "LOW_CONFIDENCE_DISTANCE",
    "MIN_MARGIN",
    "LocalizeStatus",
    "LocalizeCandidate",
    "LocalizeResult",
    "localize_jpeg",
]
