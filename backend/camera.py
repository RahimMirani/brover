"""Camera: rpicam-vid subprocess + JPEG frame parser + async fan-out.

One rpicam-vid process streams MJPEG to stdout. A background reader thread
splits the stream on JPEG SOI (FFD8) / EOI (FFD9) markers and publishes
each complete frame to:

  - self.latest_jpeg         a bytes attribute, always holds the most
                             recent full frame. Used by the look() tool
                             to give Claude a single still image.

  - subscriber asyncio.Queues one per connected MJPEG HTTP client. Each
                             queue has maxsize=1; a slow consumer just
                             sees the latest frame, not a backlog.

Subscribe/unsubscribe is handled inside mjpeg_generator(), so route
handlers just return StreamingResponse(mjpeg_generator(), ...).
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
from typing import Optional

from backend.config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH
from backend.metrics import metrics

logger = logging.getLogger(__name__)

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"


class CameraStream:
    def __init__(
        self,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: set[asyncio.Queue[bytes]] = set()
        self._running = False
        self.latest_jpeg: bytes = b""

    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        cmd = [
            "rpicam-vid",
            "-t", "0",
            "--width", str(self._width),
            "--height", str(self._height),
            "--framerate", str(self._fps),
            "--codec", "mjpeg",
            "-o", "-",
        ]
        # stderr is piped (not DEVNULL) so that rpicam-vid's diagnostics --
        # "no cameras available", "pipeline handler in use", missing modules,
        # etc. -- show up in our logs instead of disappearing. Without this
        # an early subprocess failure looks like a silent "pipe closed".
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="camera-reader", daemon=True
        )
        self._thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop, name="camera-stderr", daemon=True
        )
        self._stderr_thread.start()
        logger.info(
            "camera started: %dx%d @ %dfps", self._width, self._height, self._fps
        )

    async def stop(self) -> None:
        self._running = False
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
            except Exception:
                logger.exception("error terminating rpicam-vid")
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        logger.info("camera stopped")

    def subscribe(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        self._subscribers.discard(q)

    def _reader_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        buffer = b""
        try:
            while self._running:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    if self._running:
                        # Give the subprocess a moment to fully exit so we
                        # can report a real return code alongside the warning;
                        # the actual error text already arrived via _stderr_loop.
                        try:
                            rc = proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            rc = None
                        logger.warning(
                            "rpicam-vid pipe closed unexpectedly (exit code=%s); "
                            "see preceding 'rpicam-vid:' lines for the cause",
                            rc,
                        )
                    break
                buffer += chunk
                while True:
                    start = buffer.find(_SOI)
                    if start == -1:
                        break
                    end = buffer.find(_EOI, start + 2)
                    if end == -1:
                        if start > 0:
                            buffer = buffer[start:]
                        break
                    jpg = buffer[start : end + 2]
                    buffer = buffer[end + 2 :]
                    self._publish(jpg)
        except Exception:
            logger.exception("camera reader thread crashed")

    def _stderr_loop(self) -> None:
        """Forward rpicam-vid stderr into our logger.

        rpicam-vid prints both routine startup info ("Camera 0 imx219 ...")
        and fatal errors ("ERROR: *** no cameras available ***") to stderr.
        We log them all at INFO so they're visible during normal operation
        and especially when the subprocess dies early.
        """
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for raw_line in iter(proc.stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    logger.info("rpicam-vid: %s", line)
        except Exception:
            logger.exception("camera stderr reader crashed")

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def _publish(self, frame: bytes) -> None:
        self.latest_jpeg = frame
        metrics.record_frame(len(frame))
        loop = self._loop
        if loop is None or not self._running:
            return
        for q in list(self._subscribers):
            try:
                loop.call_soon_threadsafe(_push_latest, q, frame)
            except RuntimeError:
                return


def _push_latest(q: "asyncio.Queue[bytes]", frame: bytes) -> None:
    """Drop any pending frame and put the newest one. Keeps slow consumers current."""
    try:
        while True:
            q.get_nowait()
    except asyncio.QueueEmpty:
        pass
    try:
        q.put_nowait(frame)
    except asyncio.QueueFull:
        pass


camera = CameraStream()


async def mjpeg_generator():
    """Multipart MJPEG stream for one HTTP client. Subscribes and cleans up itself."""
    queue = camera.subscribe()
    try:
        while True:
            frame = await queue.get()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )
    finally:
        camera.unsubscribe(queue)
