"""FastAPI app: phone web UI, MJPEG camera stream, and a WebSocket that
handles voice commands, manual teleop, and e-stop.

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Single worker required: there are shared singletons (camera subprocess,
gpiozero motor pins, mode state machine) that cannot be safely forked.

WebSocket protocol
------------------
Inbound from phone:
    {"type": "ping"}                                   keepalive, no-op
    {"type": "move", "cmd": "<direction|stop>"}        manual teleop
    {"type": "estop"}                                  emergency stop
    {"type": "set_model", "model": "..."}              select LLM model
    {"type": "audio", "data": "<base64>", "mime": "audio/webm"}

Outbound to phone:
    {"type": "mode", "state": "idle|manual|ai"}
    {"type": "model", "model": "..."}
    {"type": "transcript", "text": "..."}
    {"type": "tool_call", "name": "...", "arguments": {...}}
    {"type": "tool_result", "name": "...", "content": [...]}
    {"type": "final", "text": "..."}
    {"type": "audio_reply", "data": "<base64 mp3>"}
    {"type": "error", "text": "..."}
"""
from __future__ import annotations

import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import Body, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend import captures, embeddings, motors, registry, route_recording, teaching, training
from backend.camera import camera, mjpeg_generator
from backend.db import connection as db_connection
from backend.db import places as places_db
from backend.db import routes as routes_db
from backend.distance_sensor import distance_sensor
from backend.llm import run_agent
from backend.metrics import install_error_counter, metrics
from backend.mode import mode
from backend.stt import transcribe
from backend.tts import synthesize

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
# Resolve the captures directory once. db_connection.CAPTURES_DIR is the same
# Path the storage layer writes to via captures.save_jpeg, so the URL the UI
# hits through the /captures mount lines up with the bytes on disk exactly.
CAPTURES_DIR = db_connection.CAPTURES_DIR

Send = Callable[[dict[str, Any]], Awaitable[None]]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logger.info("brover starting up")
    install_error_counter()
    await camera.start()
    await distance_sensor.start()
    db_connection.init_shared_connection()
    metrics.start_sampler()
    try:
        yield
    finally:
        logger.info("brover shutting down")
        await metrics.stop_sampler()
        # End any orphaned tour cleanly so its background poll task doesn't
        # outlive the camera it polls.
        if teaching.tour_buffer.active:
            try:
                await teaching.tour_buffer.end()
            except Exception:
                logger.exception("failed to end tour cleanly during shutdown")
        # Drop any in-flight route recording. We don't try to persist it on
        # shutdown -- a half-saved route from an unexpected restart would
        # confuse future replays more than the missing data does.
        route_recording.route_recorder.cancel()
        # Drop any pending manual-training captures. They live in RAM only
        # so this just frees memory; no on-disk cleanup required.
        training.pending_captures.clear()
        motors.stop()
        await distance_sensor.stop()
        await camera.stop()
        db_connection.close_shared_connection()


app = FastAPI(lifespan=lifespan)


@app.get("/stream.mjpg")
async def stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/metrics")
async def api_metrics() -> JSONResponse:
    """Latest metrics snapshot + 5-minute rolling history. Polled by /analytics."""
    return JSONResponse(metrics.snapshot())


@app.get("/api/sensors")
async def api_sensors() -> JSONResponse:
    """Latest live hardware sensor readings."""
    reading = distance_sensor.latest()
    age_seconds = (
        None
        if reading.updated_at is None
        else max(0.0, time.monotonic() - reading.updated_at)
    )
    return JSONResponse(
        {
            "distance": {
                "distance_cm": reading.distance_cm,
                "status": reading.status,
                "stale": reading.stale,
                "age_seconds": age_seconds,
                "safe_for_forward": reading.safe_for_forward,
                "min_safe_forward_cm": reading.min_safe_forward_cm,
            }
        }
    )


@app.get("/api/routes")
async def api_routes() -> JSONResponse:
    """Every recorded route as `from -> to (N steps)`.

    Mirrors `/api/places`: returns an empty list and `ready=false` rather
    than 500-ing when the shared DB connection has not been opened yet.
    """
    try:
        conn = db_connection.get_shared_connection()
    except RuntimeError:
        return JSONResponse({"routes": [], "ready": False})

    summaries = routes_db.list_routes_with_step_counts(conn)
    return JSONResponse(
        {
            "ready": True,
            "routes": [
                {
                    "id": s.id,
                    "from_place": s.from_place_name,
                    "to_place": s.to_place_name,
                    "step_count": s.step_count,
                    "created_at": s.created_at,
                }
                for s in summaries
            ],
        }
    )


@app.get("/api/places")
async def api_places() -> JSONResponse:
    """Names and view counts of every place Brover has been taught.

    Backs both manual debugging ("what does the rover actually know?") and
    the LLM's `find_place` answers. Returns an empty list, not an error,
    when the DB has no places yet -- cold-start should not look like a
    server failure.
    """
    try:
        conn = db_connection.get_shared_connection()
    except RuntimeError:
        return JSONResponse({"places": [], "ready": False})

    summaries = places_db.list_places_with_counts(conn)
    return JSONResponse(
        {
            "ready": True,
            "places": [
                {
                    "id": s.id,
                    "name": s.name,
                    "view_count": s.view_count,
                    "created_at": s.created_at,
                    "last_taught_at": s.last_taught_at,
                }
                for s in summaries
            ],
        }
    )


@app.get("/api/models")
async def api_models() -> JSONResponse:
    """Available LLM models and the UI default."""
    return JSONResponse(
        {
            "default_model": registry.DEFAULT_MODEL_ID,
            "models": [
                {
                    "id": model_id,
                    "display_name": spec.display_name,
                    "provider": spec.provider,
                    "supports_vision": spec.supports_vision,
                    "input_per_mtok": spec.input_per_mtok,
                    "output_per_mtok": spec.output_per_mtok,
                }
                for model_id, spec in registry.MODELS.items()
            ],
        }
    )


# -----------------------------------------------------------------------------
# Manual training: UI-driven place captures and route start/stop.
#
# The voice/agent path keeps its own tools (remember_here, start_tour, etc.);
# these endpoints are the parallel UI surface. They reuse the same memory
# primitives (embed_image, save_jpeg, places.add_place_view, route_recorder),
# so any view stored here is indistinguishable from one stored by Claude.
# -----------------------------------------------------------------------------


def _get_db_or_503():
    """Return the shared DB connection or raise 503 if not initialised yet.

    Mirrors the readiness check used by /api/places and /api/routes so a
    cold-start request gets a clear message instead of an opaque 500.
    """
    try:
        return db_connection.get_shared_connection()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"memory not ready: {e}")


def _image_url(image_path: str) -> str:
    """Translate a stored image_path (e.g. ``data/captures/abc.jpg``) into
    the public URL served by the /captures static mount.

    `captures.save_jpeg` always writes into ``data/captures/`` with a
    bare ``<sha256>.jpg`` filename, so the URL is always
    ``/captures/<basename>``. Building it from ``Path(...).name`` keeps
    us safe if an image_path ever has subdirectories in it.
    """
    return f"/captures/{Path(image_path).name}"


@app.get("/api/places/{place_id}/views")
async def api_place_views(place_id: int) -> JSONResponse:
    """List every stored frame for one place.

    Returns each row's ``id`` (needed for the per-view delete button)
    plus an ``image_url`` already shaped for the /captures static
    mount, so the UI can drop it straight into an ``<img src>`` tag.
    """
    conn = _get_db_or_503()
    place = places_db.list_places_with_counts(conn)
    summaries = {s.id: s for s in place}
    summary = summaries.get(place_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"place {place_id} not found")

    views = places_db.list_place_views(conn, place_id)
    return JSONResponse(
        {
            "place_id": place_id,
            "name": summary.name,
            "view_count": len(views),
            "views": [
                {
                    "id": v.id,
                    "image_path": v.image_path,
                    "image_url": _image_url(v.image_path),
                    "captured_at": v.captured_at,
                    "heading_deg": v.heading_deg,
                    "distance_cm": v.distance_cm,
                }
                for v in views
            ],
        }
    )


@app.delete("/api/training/place_views/{view_id}")
async def api_delete_place_view(view_id: int) -> JSONResponse:
    """Delete one stored frame: DB rows, embedding, and the JPEG on disk.

    JPEG cleanup is conditional on no other row referencing the file --
    `captures.save_jpeg` dedups by content hash, so the same bytes can
    legitimately back multiple views. We only unlink when the reference
    count drops to zero after the DB delete.
    """
    conn = _get_db_or_503()
    deleted = places_db.delete_place_view(conn, view_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail=f"view {view_id} not found")

    refs = places_db.count_image_path_refs(conn, deleted.image_path)
    file_removed = False
    if refs == 0:
        # The stored path is relative to the project root; resolve from
        # the captures dir's parent so the join stays portable.
        abs_path = CAPTURES_DIR.parent.parent / deleted.image_path
        try:
            if abs_path.is_file():
                abs_path.unlink()
                file_removed = True
        except OSError:
            logger.exception(
                "failed to unlink JPEG for deleted view %d (%s)",
                view_id,
                abs_path,
            )

    return JSONResponse(
        {
            "view_id": view_id,
            "place_id": deleted.place_id,
            "image_path": deleted.image_path,
            "file_removed": file_removed,
            "remaining_refs": refs,
        }
    )


@app.post("/api/training/captures")
async def api_training_capture() -> JSONResponse:
    """Snapshot the live camera frame and hold it in the pending buffer.

    Returns the capture id the UI uses for the preview, save, and discard
    endpoints. The JPEG is never written to disk until the save call.
    """
    frame = camera.latest_jpeg
    if not frame:
        raise HTTPException(
            status_code=503,
            detail="camera has no frame yet; wait a moment and try again",
        )

    try:
        item = training.pending_captures.create(frame)
    except training.PendingCapturesFull as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValueError as e:
        # Defensive: pending_captures.create rejects empty bytes, which we
        # already filtered above. If we somehow get here, surface a 500.
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(
        {
            "capture_id": item.id,
            "created_at": item.created_at,
            "expires_at": item.expires_at,
            "byte_size": len(item.jpeg),
        }
    )


@app.get("/api/training/captures/{capture_id}")
async def api_training_capture_preview(capture_id: str) -> Response:
    """Stream the buffered JPEG back to the UI for the preview thumbnail.

    no-store keeps a previously-shown capture from being served after it's
    been saved or discarded -- the capture_id stays in the URL, but the
    bytes behind it should not be cached by an intermediate browser.
    """
    item = training.pending_captures.get(capture_id)
    if item is None:
        raise HTTPException(status_code=404, detail="capture not found or expired")
    return Response(
        content=item.jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/training/captures/{capture_id}/save_place")
async def api_training_save_place(
    capture_id: str, payload: dict = Body(...)
) -> JSONResponse:
    """Embed the buffered frame and store it under a place name.

    Order matters: embed FIRST, then persist + pop. A Voyage failure
    leaves the pending capture in place so the user can retry without
    re-capturing the frame.
    """
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    item = training.pending_captures.get(capture_id)
    if item is None:
        raise HTTPException(status_code=404, detail="capture not found or expired")

    conn = _get_db_or_503()

    try:
        vector = await embeddings.embed_image(item.jpeg)
    except embeddings.EmbeddingError as e:
        # Capture stays in the buffer; the UI can retry the save.
        raise HTTPException(status_code=502, detail=f"embedding failed: {e}")

    image_path = captures.save_jpeg(item.jpeg)
    place_id = places_db.get_or_create_place(conn, name)
    places_db.add_place_view(
        conn,
        place_id=place_id,
        image_path=image_path,
        embedding=vector,
    )

    # Only consume the pending capture after the DB write committed.
    training.pending_captures.pop(capture_id)

    summaries = {s.id: s for s in places_db.list_places_with_counts(conn)}
    summary = summaries.get(place_id)
    view_count = 0 if summary is None else summary.view_count
    return JSONResponse(
        {
            "place_id": place_id,
            "name": name,
            "view_count": view_count,
            "image_path": image_path,
        }
    )


@app.delete("/api/training/captures/{capture_id}")
async def api_training_discard(capture_id: str) -> JSONResponse:
    """Drop a pending capture without saving. Idempotent."""
    discarded = training.pending_captures.discard(capture_id)
    return JSONResponse({"discarded": discarded})


@app.post("/api/training/routes/start")
async def api_training_route_start(payload: dict = Body(...)) -> JSONResponse:
    """Begin a route recording driven by manual teleop.

    The existing route_recorder + mode.on_manual_input wiring picks up
    every D-pad command from here on. Nothing else changes -- this is
    just the HTTP face of the same singleton the voice path uses.
    """
    from_place = (payload.get("from_place") or "").strip()
    if not from_place:
        raise HTTPException(status_code=400, detail="from_place is required")

    if route_recording.route_recorder.active:
        raise HTTPException(
            status_code=409,
            detail=(
                f"a route recording from "
                f"{route_recording.route_recorder.from_place!r} is already active"
            ),
        )

    if not camera.latest_jpeg:
        raise HTTPException(
            status_code=503,
            detail="camera has no frame yet; wait a moment and try again",
        )

    try:
        route_recording.route_recorder.start(
            from_place, lambda: camera.latest_jpeg
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))

    return JSONResponse(
        {
            "active": True,
            "from_place": from_place,
            "step_count": route_recording.route_recorder.step_count,
        }
    )


@app.post("/api/training/routes/stop")
async def api_training_route_stop(payload: dict = Body(...)) -> JSONResponse:
    """End the active recording, embed every step, and persist the route."""
    to_place = (payload.get("to_place") or "").strip()
    if not to_place:
        raise HTTPException(status_code=400, detail="to_place is required")

    if not route_recording.route_recorder.active:
        raise HTTPException(
            status_code=409, detail="no route recording is currently active"
        )

    conn = _get_db_or_503()

    try:
        result = await route_recording.route_recorder.stop(
            conn,
            to_place,
            embed_image=embeddings.embed_image,
            save_jpeg=captures.save_jpeg,
        )
    except route_recording.RouteRecorderEmpty as e:
        raise HTTPException(status_code=409, detail=str(e))
    except embeddings.EmbeddingError as e:
        # Recorder has already reset its state on this exception path -- the
        # recording is lost. Surface that clearly to the UI.
        raise HTTPException(
            status_code=502, detail=f"embedding failed; recording lost: {e}"
        )

    return JSONResponse(
        {
            "route_id": result.route_id,
            "from_place": result.from_place,
            "to_place": result.to_place,
            "step_count": result.step_count,
            "duration_seconds": result.duration_seconds,
        }
    )


@app.post("/api/training/routes/cancel")
async def api_training_route_cancel() -> JSONResponse:
    """Drop the in-flight recording without saving. Idempotent."""
    was_active = route_recording.route_recorder.active
    route_recording.route_recorder.cancel()
    return JSONResponse({"cancelled": was_active})


@app.get("/api/training/routes/state")
async def api_training_route_state() -> JSONResponse:
    """Snapshot of the route recorder for the UI status badge."""
    rec = route_recording.route_recorder
    return JSONResponse(
        {
            "active": rec.active,
            "from_place": rec.from_place,
            "step_count": rec.step_count,
            "overflowed": rec.overflowed,
        }
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    metrics.ws_client_connected()
    logger.info("ws client connected from %s", ws.client)
    current_model = registry.DEFAULT_MODEL_ID

    async def send(msg: dict[str, Any]) -> None:
        try:
            await ws.send_text(json.dumps(msg))
        except (WebSocketDisconnect, RuntimeError):
            pass

    await send({"type": "mode", "state": mode.state})
    await send({"type": "model", "model": current_model})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send({"type": "error", "text": "invalid json"})
                continue

            t = msg.get("type")
            if t == "ping":
                continue
            elif t == "set_model":
                requested_model = msg.get("model", "")
                if requested_model not in registry.MODELS:
                    await send(
                        {"type": "error", "text": f"unknown model: {requested_model!r}"}
                    )
                    continue
                current_model = requested_model
                await send({"type": "model", "model": current_model})
            elif t == "move":
                cmd = msg.get("cmd", "stop")
                mode.on_manual_input(cmd)
                await send({"type": "mode", "state": mode.state})
            elif t == "estop":
                mode.request_estop()
                await send({"type": "mode", "state": mode.state})
            elif t == "audio":
                await _handle_audio(msg, send, current_model)
            else:
                await send({"type": "error", "text": f"unknown message type: {t!r}"})

    except WebSocketDisconnect:
        logger.info("ws client disconnected")
    except Exception:
        logger.exception("ws handler crashed")
    finally:
        metrics.ws_client_disconnected()
        mode.request_estop()


async def _handle_audio(msg: dict[str, Any], send: Send, current_model: str) -> None:
    b64 = msg.get("data", "")
    mime = msg.get("mime", "audio/webm")
    if not b64:
        await send({"type": "error", "text": "audio: missing data"})
        return

    try:
        audio_bytes = base64.b64decode(b64)
    except Exception as e:
        await send({"type": "error", "text": f"audio: bad base64: {e}"})
        return

    filename = "audio.webm" if "webm" in mime else "audio.mp4"

    try:
        text = await transcribe(audio_bytes, filename=filename)
    except Exception as e:
        logger.exception("stt failed")
        await send({"type": "error", "text": f"stt failed: {e}"})
        return

    await send({"type": "transcript", "text": text})

    if not text.strip():
        await send({"type": "final", "text": ""})
        return

    mode.enter_ai()
    await send({"type": "mode", "state": mode.state})

    try:
        agent_result = await run_agent(text, send, mode.cancel_event, current_model)
        final_text = agent_result.text
    except Exception as e:
        logger.exception("agent loop crashed")
        agent_result = None
        final_text = f"Sorry, something went wrong: {e}"
    finally:
        mode.enter_idle()

    final_msg: dict[str, Any] = {"type": "final", "text": final_text}
    if agent_result is not None:
        final_msg.update(
            {
                "model": agent_result.model,
                "latency_ms": agent_result.latency_ms,
                "input_tokens": agent_result.input_tokens,
                "output_tokens": agent_result.output_tokens,
                "cost_usd": agent_result.cost_usd,
                "iterations": agent_result.iterations,
            }
        )
    await send(final_msg)
    await send({"type": "mode", "state": mode.state})

    if final_text:
        try:
            mp3 = await synthesize(final_text)
            if mp3:
                await send(
                    {
                        "type": "audio_reply",
                        "data": base64.b64encode(mp3).decode("ascii"),
                    }
                )
        except Exception:
            logger.exception("tts failed")


# Serve stored capture JPEGs read-only at /captures/<filename>. Backs the
# Memory gallery in the training panel: each <img src="/captures/abc.jpg">
# in the UI resolves to the same file the storage layer wrote via
# captures.save_jpeg. Registered before the catch-all frontend mount so
# the prefix wins, and the directory exists by the time we mount because
# db_connection.connect() creates it on first DB open. Mounting before
# DB init means the directory may not exist yet on a brand-new install;
# create it defensively here so StaticFiles doesn't raise on startup.
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")

# Mount the frontend static files at the root. Registered LAST so that
# /stream.mjpg, /ws, /api/*, and /captures/* win against this catch-all.
# html=True means a GET / returns frontend/index.html.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
