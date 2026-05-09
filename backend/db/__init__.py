"""Database layer for Brover's spatial memory.

Phase 2 of the training pipeline. See `training_plan.md` for the bigger
picture. The shape:

  schema.sql      Source of truth for tables. Re-applied on every startup;
                  every CREATE statement is `IF NOT EXISTS` so that's safe.
  connection.py   Opens `data/brover.db`, loads sqlite-vec, applies the schema.
  places.py       CRUD for the place tables (Phase 2 — exercised now).
  routes.py       Stubs for route tables (Phase 4 — schema exists, code later).
  people.py       Stubs for face tables (Phase 7 — schema exists, code later).

The actual `.db` file lives at `data/brover.db` on the device's SD card and
is never committed to git — `data/` is gitignored. Each Pi grows its own
memory; `git pull` propagates code, not data.
"""
