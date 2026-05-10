# Brover — Training & Spatial Memory Plan

High-level strategy for how Brover learns and remembers its environment. Covers spatial memory, navigation, and people recognition. Companion to the project overview in [`README.md`](README.md) and the architecture / hardware notes in [`plan.md`](plan.md).

---

## Core idea

Brover doesn't get a trained model of the house — it builds a **retrieval-based memory** that grows as it's taught. The embedding model is pretrained (accessed via API), and "training" Brover means populating a local database of embedded camera frames, place labels, routes between places, and known faces. Knowledge lives in the data, not in model weights.

This is the modern pattern for spatially-grounded agents: pretrained foundation model + retrieval + tool use. No GPU, no training pipeline, no per-deployment ML work. To deploy to a new house, you teach that house from scratch — same code, fresh database.

---

## What gets stored

Three kinds of memory, kept separate because they fail differently and have different privacy implications:

1. **Places** — named locations (kitchen, bedroom, etc.), each represented by multiple frames captured from different angles and headings. Multiple views per place make recognition robust to approach direction.
2. **Routes** — recorded traversals between adjacent places, capturing both the visual sequence (frames along the path) and the motor actions taken. This is what enables actual navigation, not just recognition.
3. **People** — face embeddings tagged with names, stored separately from place data. Treated as biometric data: stays local, encrypted, only taught with consent.

---

## Teaching workflow

Two modes the user drives explicitly:

- **Teach a place:** drive Brover to the location, hit record, label it, slowly rotate so it captures multiple headings. ~30 seconds per place.
- **Teach a route:** put Brover at place A, hit record, drive it to place B, stop. Frames + motor actions are captured along the way as a sequence.

Teaching is explicit and user-driven for v1. Passive/automatic memory ("Brover remembers as it goes") is a later addition — predictable, debuggable memory first.

For a typical home: teach 6-10 places and 8-15 short route segments. Total time ~30 minutes. You don't need to teach every (start, destination) pair — see route composition below.

---

## How navigation actually works

The key insight: **you don't need to teach every possible route**. You teach short segments between adjacent places, and the system composes longer routes via graph search.

Places are nodes. Taught routes are edges. To go from bedroom to kitchen, run shortest-path search over the graph (e.g. bedroom → hallway → kitchen) and execute each edge in sequence.

Each edge is executed via **teach-and-repeat**: replay the stored motor actions while continuously verifying that the current camera view still matches the expected frame in the recorded sequence. If it drifts, stop and re-localize. This is a well-established robotics technique and works with camera alone.

For destinations with no taught route, Brover falls back to careful exploration + localization, and saves what it learns so the next attempt is direct. The map grows organically.

Claude can also reason over the graph symbolically — given the topology, it's good at inferring sensible intermediate steps even for unmapped pairs.

---

## Spatial awareness ("what's to my left?")

Recognizing a place isn't enough — Brover should know directions within it. Two mechanisms:

- **Headings per place:** each place stores frames at multiple headings (0°, 90°, 180°, 270°). Brover tracks its current heading from cumulative motor actions, so "look left" maps to a stored view.
- **Scene captions:** during teaching, each frame is captioned (by Claude or a cheaper vision model) with a short description. This gives the agent rich textual context per direction, not just vector similarity scores.

Combined, Brover can answer "what's on my right?" without moving — by looking up the stored view at heading + 90° and reading its caption.

---

## People recognition

Separate system from place memory. Use a face embedding model (hosted API or local library), teach by capturing several frames of a person's face from different angles tagged with their name, recognize at runtime via nearest-neighbor against stored face embeddings.

Kept fully local on the Pi for privacy. Only taught explicitly with consent. Not synced to any cloud DB.

---

## Embedding model

Hosted API, not local. Voyage AI's multimodal embedding model is the default choice — joint image-text embedding space means Brover can also be queried by text ("find frames that look like a kitchen") without extra infrastructure.

Local embedding (CLIP on the Pi) is possible but burns RAM and CPU that Brover needs for other work. The API call adds 200-500ms latency, which is acceptable and is the dominant latency in the system either way.

Voyage is the fourth external dependency alongside Anthropic, OpenAI, and xAI. When it's unreachable, teaching should buffer-and-retry rather than hard-fail; localization against already-stored vectors keeps working without it.

---

## Storage strategy

All memory lives on the Pi in a local SQLite database with the `sqlite-vec` extension for vector similarity search. Images stored as files on disk; embeddings and metadata in SQLite.

The DB file lives at `data/brover.db` on the Pi's SD card. Captured frames go to `data/captures/` with content-hash filenames (`<sha256>.jpg`) for automatic dedup, written straight from the camera's existing JPEG output without re-encoding. A soft cap on stored frames plus a `prune` command keeps the SD card from filling silently over time. The schema covers `places`, `place_views`, `routes`, `route_steps`, `people`, and `face_views`, with `sqlite-vec` virtual tables for the embeddings — created up front so later phases don't need migrations.

Rationale:
- At one-house scale (a few thousand frames, <500 MB total), local search is faster than network round-trips to a hosted DB.
- Brover already depends on the Claude API and the embedding API being reachable. Putting the DB in the cloud adds a third point of failure with no upside.
- Pi 5 with 4GB RAM and a 32GB+ SD card has plenty of headroom — storage and memory are not the constraints.

A hosted DB (Supabase / Chroma) only makes sense later if multiple Brovers need to share a map, or if a remote admin UI for inspecting data becomes valuable. Reasonable evolution: local primary, periodic cloud backup.

---

## Sensing limits and a small upgrade

Camera-only navigation works — every step described here is pure vision plus motor odometry. Modern home robots largely run camera-based, no LIDAR.

The one weak spot is fast obstacle detection in the forward path. A $2 ultrasonic sensor (HC-SR04) wired to the front of the car closes that gap with a hardware-level distance check that's faster and more reliable than any vision pipeline. Brover should read this sensor continuously in the backend, keep the latest `distance_cm` in memory, expose it through a `distance` tool when Claude asks, and use it automatically to stop unsafe forward motion.

For training, distance is metadata rather than memory by itself. During explicit place/route recording, save the latest `distance_cm` beside each captured frame/action sample. Do not rely on old stored distance readings for safety, because furniture, doors, people, boxes, and pets move; safety decisions always use the live reading.

Wheel encoders or an IMU would also help. Time-based motor control drifts fast — the current calibration is roughly `turn(0.5s) ≈ 45°` and varies with battery level and floor surface — so teach-and-repeat in Phase 5 will lean on continuous visual re-localization between steps rather than dead reckoning. Encoders or an IMU are pencilled in for after Phase 5 if drift turns out to be the dominant failure mode.

---

## Phases

Work one phase at a time. Each phase has a single goal and a clear "done" signal.

**Phase 1 — Hardware and Safety [DONE].** Camera streaming, L298N motor control with safety-clamped durations, HC-SR04 ultrasonic with continuous polling and forward-motion auto-stop, WebSocket UI with manual teleop and e-stop, LLM agent loop with tool dispatch. The live `distance_cm` reading carries forward into Phase 3 as per-sample training metadata.

**Phase 2 — Memory Foundation [DONE].** Voyage embedding client + local SQLite with `sqlite-vec`. Schema covers all of `places`, `place_views`, `routes`, `route_steps`, `people`, and `face_views` so later phases don't need migrations. Shipped as PR #4 (DB layer), PR #5 (Voyage client), and a memory smoke test on `main`. End-to-end Pi verification happens the next time Brover is powered on.

**Phase 3 — Place Teaching [NEXT — training pipeline v1].** Domain layer (`backend/teaching.py`, `backend/localization.py`) that composes camera + Voyage + SQLite. New tools for Claude: `remember_here`, `find_place`, `localize`. After this phase, the user can speak "remember this as the kitchen" or "where am I?" and Brover does the right thing. A phone-UI Teach button is a follow-up; voice-driven teaching ships first because it's the project's main interaction model anyway.

**Phase 4 — Route Recording.** Extend teaching to capture (frame, motor-action) sequences between adjacent places, stored as `routes` + `route_steps`.

**Phase 5 — Graph Navigation.** Build the place/route graph in memory, run shortest-path search for unseen pairs, replay edges via teach-and-repeat with continuous visual re-localization between steps. Tools: `find_route`, `execute_route`. Critical: `execute_route` must honor the existing `cancel_event` between every step so the manual override and e-stop keep working — long-running navigation must not become a silent override.

**Phase 6 — Spatial Awareness.** Multiple headings per place view + per-frame scene captions written during teaching. New tool `scan_room` answers "what's to my left?" without driving.

**Phase 7 — People Recognition.** Separate face-embedding pipeline using `people` + `face_views`. Opt-in only, local only, never synced. Tools: `remember_person`, `recognize_faces`.

**Phase 8 — Resilience and polish.** "I'm lost" recovery, "refresh this place" command, optional passive memory updates, optional cloud backup of `brover.db`, and a small admin endpoint for inspecting stored memory.

---

## Known failure modes to design around from the start

- **Visually similar rooms** (two bedrooms, two bathrooms) can confuse embedding-based recognition. Capture distinctive frames including unique objects.
- **Featureless corridors and empty walls** produce near-identical embeddings. Teach corridor frames that look toward distinctive ends.
- **Moved furniture / changed lighting** causes drift between stored frames and current view. Need a "refresh this place" command and graceful low-confidence handling ("I'm lost, where am I?").
- **Tool-use loops** can run away — cap iterations and always have a stop fallback.
- **Cold start** with empty memory — agent should handle gracefully, not crash.