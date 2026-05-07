# Brover

AI agent inside an RC car. A Raspberry Pi serves a phone-facing web UI, streams a live camera feed, and runs a **FastAPI** backend that forwards commands to **Claude** with motor and sensing tools so the car can act on natural language.

For hardware, wiring, and the high-level system design, see [`plan.md`](plan.md). For spatial memory, teaching places and routes, and the longer-term roadmap, see [`training_plan.md`](training_plan.md).

---

## Features

- **Web UI** — static frontend with voice input (WebSocket sends audio), live MJPEG preview, manual drive, AI mode, and emergency stop.
- **AI loop** — server transcribes speech, grabs a camera frame when needed, calls the LLM API, executes returned tool calls on GPIO motors, optionally replies with synthesized speech.
- **Metrics** — basic sampling and optional analytics frontend (`analytics.html`).
- **Mode state** — idle, manual teleop, and AI-controlled operation coordinated over the socket.

---

## Requirements

- **Runtime:** Raspberry Pi (project targets Pi 5, 4 GB) with CSI camera (`IMX708` in the current hardware notes), GPIO motor driver (**L298N** or similar — pin map is in [`plan.md`](plan.md)).
- **Python:** 3.10+ recommended (matches typical Pi OS images).
- **API keys:** `.env.example` lists `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` (used for Claude, transcription, etc., per backend modules).

Separate power rails for Pi and motors with a common ground are required — see [`plan.md`](plan.md).

---

## Quick start

From the repo root (on the Pi, after cloning):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # edit with real keys
```

Run the app (documentation in `backend/main.py` emphasizes **single worker** — shared camera and GPIO cannot be forked safely):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://<pi-ip>:8000/` from a phone on the same LAN. The MJPEG stream is at `/stream.mjpg`.

---

## Repo layout

| Path | Role |
|------|------|
| `backend/` | FastAPI app (`main.py`), camera, motors, STT/TTS, LLM agent, metrics, tooling |
| `frontend/` | `index.html`, `app.js`, `style.css`; `analytics.*` for optional dashboards |
| `plan.md` | Product plan, architecture diagram, GPIO mapping |
| `training_plan.md` | Spatial memory, teaching workflow, embeddings, SQLite strategy |
| `requirements.txt` | Python dependencies |
| `.env.example` | Required environment variables |

---

## Contributing / next steps

The **spatial memory stack** described in [`training_plan.md`](training_plan.md) (SQLite + embeddings, teaching UI, localization tools) is planned work; current code focuses on voice + vision + motor tool use over the websocket.
