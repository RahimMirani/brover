"""Frame storage on disk.

JPEGs from the camera land in `data/captures/` with content-hashed
filenames. Two identical frames produce the same filename and dedup
naturally. The database stores the *relative* path so the value is
portable across devices (different absolute paths still resolve from
the project root).
"""
from __future__ import annotations

import hashlib

from backend.db.connection import CAPTURES_DIR

# CAPTURES_DIR is .../data/captures, so .parent.parent is the project root.
_PROJECT_ROOT = CAPTURES_DIR.parent.parent


def save_jpeg(jpeg_bytes: bytes) -> str:
    """Write the JPEG to `data/captures/<sha256>.jpg` and return its path.

    Returns a path relative to the project root, with forward slashes for
    cross-platform consistency (the value gets stored in SQLite as a string,
    and we want the same path string to work on the Pi and on a laptop).
    """
    if not jpeg_bytes:
        raise ValueError("save_jpeg: empty jpeg_bytes")

    sha = hashlib.sha256(jpeg_bytes).hexdigest()
    abs_path = CAPTURES_DIR / f"{sha}.jpg"
    if not abs_path.exists():
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(jpeg_bytes)

    return abs_path.relative_to(_PROJECT_ROOT).as_posix()
