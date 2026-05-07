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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend import motors, registry
from backend.camera import camera, mjpeg_generator
from backend.llm import run_agent
from backend.metrics import install_error_counter, metrics
from backend.mode import mode
from backend.stt import transcribe
from backend.tts import synthesize

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

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
    metrics.start_sampler()
    try:
        yield
    finally:
        logger.info("brover shutting down")
        await metrics.stop_sampler()
        motors.stop()
        await camera.stop()


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


# Mount the frontend static files at the root. Registered LAST so that
# /stream.mjpg and /ws win against this catch-all. html=True means a GET /
# returns frontend/index.html.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
