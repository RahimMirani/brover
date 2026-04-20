"""In-memory metrics store + background sampler.

The store is a process-wide singleton (`metrics`) that holds:

- A rolling 5-minute window of host samples (1 Hz), used for sparklines.
- Event counters and windows updated synchronously by the rest of the app
  (camera frame sizes, motor commands, mode transitions, WebSocket client
  count, tool-call counts, error count).

A single async task `sampler_loop()` ticks at 1 Hz, reads host stats via
backend.system_stats, and appends a HostSample to the history.

Nothing here is persisted to disk. A restart wipes everything.

All entry points are safe to call from any thread / any task. The counters
are simple ints, the deques are bounded, and we never block the sampler.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from backend import system_stats

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_SECONDS = 1.0
HISTORY_SECONDS = 300  # 5 min at 1 Hz
HISTORY_SIZE = int(HISTORY_SECONDS / SAMPLE_INTERVAL_SECONDS)

FPS_WINDOW_SECONDS = 5.0
FPS_RING_SIZE = 300  # plenty of headroom at 30 fps over FPS_WINDOW_SECONDS

MOTOR_DUTY_WINDOW_SECONDS = 60.0
MODE_TIMELINE_WINDOW_SECONDS = 300.0


@dataclass
class HostSample:
    """One tick of host state. Rate fields are computed vs the previous sample."""

    t: float
    cpu_percent: Optional[float]
    cpu_per_core: list[float]
    cpu_freq_mhz: Optional[float]
    cpu_freq_max_mhz: Optional[float]
    loadavg: Optional[tuple[float, float, float]]
    mem_used_bytes: Optional[int]
    mem_total_bytes: Optional[int]
    mem_percent: Optional[float]
    mem_cached_bytes: Optional[int]
    swap_used_bytes: Optional[int]
    swap_total_bytes: Optional[int]
    disk_used_bytes: Optional[int]
    disk_total_bytes: Optional[int]
    disk_percent: Optional[float]
    disk_read_bytes_per_s: Optional[float]
    disk_write_bytes_per_s: Optional[float]
    net_rx_bytes_per_s: Optional[float]
    net_tx_bytes_per_s: Optional[float]
    net_total_rx_bytes: Optional[int]
    net_total_tx_bytes: Optional[int]
    cpu_temp_c: Optional[float]
    throttled: Optional[dict]
    core_voltage_v: Optional[float]
    arm_clock_hz: Optional[int]
    wifi: Optional[dict]
    uptime_seconds: Optional[float]
    self_proc: Optional[dict]
    camera_proc: Optional[dict]


@dataclass
class _AppState:
    ws_clients: int = 0
    estop_count: int = 0
    error_count: int = 0
    last_frame_size_bytes: int = 0
    camera_frames_dropped: int = 0
    tool_calls: Counter = field(default_factory=Counter)
    mode_current: str = "idle"
    frame_times: Deque[float] = field(
        default_factory=lambda: deque(maxlen=FPS_RING_SIZE)
    )
    # Tuples of (start_ts, stop_ts or None). Open-ended when motors are
    # currently active -- closed when a "stop" arrives.
    motor_intervals: Deque[list] = field(default_factory=deque)
    motor_last_cmd: str = "stop"
    motor_last_start: Optional[float] = None
    # Mode transitions: (timestamp, mode_name).
    mode_timeline: Deque[tuple[float, str]] = field(default_factory=deque)


class MetricsStore:
    def __init__(self) -> None:
        self.started_at: float = time.time()
        self._host_history: Deque[HostSample] = deque(maxlen=HISTORY_SIZE)
        self._app = _AppState()
        self._sampler_task: Optional[asyncio.Task[None]] = None
        self._prev_disk: Optional[dict] = None
        self._prev_net: Optional[dict] = None
        self._prev_sample_t: Optional[float] = None
        # seed mode timeline so duration maths are defined from process start
        self._app.mode_timeline.append((self.started_at, "idle"))

    # ---------------- camera ----------------
    def record_frame(self, size_bytes: int) -> None:
        now = time.monotonic()
        self._app.frame_times.append(now)
        self._app.last_frame_size_bytes = int(size_bytes)

    def record_frame_drop(self) -> None:
        self._app.camera_frames_dropped += 1

    # ---------------- motors ----------------
    def record_motor(self, cmd: str) -> None:
        now = time.monotonic()
        active = cmd in ("forward", "backward", "left", "right")
        was_active = self._app.motor_last_cmd in ("forward", "backward", "left", "right")

        if active and not was_active:
            self._app.motor_last_start = now
            self._app.motor_intervals.append([now, None])
        elif (not active) and was_active:
            start = self._app.motor_last_start
            if self._app.motor_intervals and self._app.motor_intervals[-1][1] is None:
                self._app.motor_intervals[-1][1] = now
            self._app.motor_last_start = None
            _ = start  # kept for readability

        self._app.motor_last_cmd = cmd if active else "stop"
        self._trim_motor_intervals(now)

    def _trim_motor_intervals(self, now: float) -> None:
        cutoff = now - MOTOR_DUTY_WINDOW_SECONDS * 2
        while self._app.motor_intervals:
            start, stop = self._app.motor_intervals[0]
            end = stop if stop is not None else now
            if end < cutoff:
                self._app.motor_intervals.popleft()
            else:
                break

    # ---------------- mode ------------------
    def record_mode(self, state: str) -> None:
        if state == self._app.mode_current:
            return
        self._app.mode_current = state
        self._app.mode_timeline.append((time.time(), state))
        cutoff = time.time() - MODE_TIMELINE_WINDOW_SECONDS * 2
        while (
            len(self._app.mode_timeline) > 2
            and self._app.mode_timeline[0][0] < cutoff
        ):
            self._app.mode_timeline.popleft()

    def record_estop(self) -> None:
        self._app.estop_count += 1

    # ---------------- websocket -------------
    def ws_client_connected(self) -> None:
        self._app.ws_clients += 1

    def ws_client_disconnected(self) -> None:
        self._app.ws_clients = max(0, self._app.ws_clients - 1)

    # ---------------- tools -----------------
    def record_tool(self, name: str) -> None:
        self._app.tool_calls[name] += 1

    # ---------------- errors ----------------
    def record_error(self) -> None:
        self._app.error_count += 1

    # ---------------- sampler ---------------
    def start_sampler(self) -> None:
        if self._sampler_task is not None and not self._sampler_task.done():
            return
        system_stats.prime_cpu_percent()
        self._sampler_task = asyncio.create_task(
            self._sampler_loop(), name="metrics-sampler"
        )
        logger.info("metrics sampler started (interval=%.1fs)", SAMPLE_INTERVAL_SECONDS)

    async def stop_sampler(self) -> None:
        task = self._sampler_task
        self._sampler_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _sampler_loop(self) -> None:
        try:
            while True:
                try:
                    self._tick()
                except Exception:
                    logger.exception("metrics sampler tick failed")
                await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            raise

    def _tick(self) -> None:
        now_mono = time.monotonic()
        now_wall = time.time()

        cpu = system_stats.read_cpu()
        mem = system_stats.read_memory()
        disk_usage = system_stats.read_disk_usage("/")
        disk_io = system_stats.read_disk_io_raw()
        net_io = system_stats.read_net_io_raw()

        dt = (
            now_mono - self._prev_sample_t
            if self._prev_sample_t is not None
            else None
        )

        disk_read_rate: Optional[float] = None
        disk_write_rate: Optional[float] = None
        if disk_io and self._prev_disk and dt and dt > 0:
            disk_read_rate = max(
                0.0, (disk_io["read_bytes"] - self._prev_disk["read_bytes"]) / dt
            )
            disk_write_rate = max(
                0.0, (disk_io["write_bytes"] - self._prev_disk["write_bytes"]) / dt
            )

        rx_rate: Optional[float] = None
        tx_rate: Optional[float] = None
        total_rx = 0
        total_tx = 0
        if net_io:
            for nic, counters in net_io.items():
                if nic == "lo":
                    continue
                total_rx += counters["bytes_recv"]
                total_tx += counters["bytes_sent"]
            if self._prev_net and dt and dt > 0:
                prev_rx = sum(
                    c["bytes_recv"]
                    for nic, c in self._prev_net.items()
                    if nic != "lo"
                )
                prev_tx = sum(
                    c["bytes_sent"]
                    for nic, c in self._prev_net.items()
                    if nic != "lo"
                )
                rx_rate = max(0.0, (total_rx - prev_rx) / dt)
                tx_rate = max(0.0, (total_tx - prev_tx) / dt)

        sample = HostSample(
            t=now_wall,
            cpu_percent=cpu.get("total_percent"),
            cpu_per_core=cpu.get("per_core_percent", []),
            cpu_freq_mhz=cpu.get("freq_current_mhz"),
            cpu_freq_max_mhz=cpu.get("freq_max_mhz"),
            loadavg=system_stats.read_loadavg(),
            mem_used_bytes=mem.get("used_bytes"),
            mem_total_bytes=mem.get("total_bytes"),
            mem_percent=mem.get("percent"),
            mem_cached_bytes=mem.get("cached_bytes"),
            swap_used_bytes=mem.get("swap_used_bytes"),
            swap_total_bytes=mem.get("swap_total_bytes"),
            disk_used_bytes=disk_usage.get("used_bytes"),
            disk_total_bytes=disk_usage.get("total_bytes"),
            disk_percent=disk_usage.get("percent"),
            disk_read_bytes_per_s=disk_read_rate,
            disk_write_bytes_per_s=disk_write_rate,
            net_rx_bytes_per_s=rx_rate,
            net_tx_bytes_per_s=tx_rate,
            net_total_rx_bytes=total_rx if net_io else None,
            net_total_tx_bytes=total_tx if net_io else None,
            cpu_temp_c=system_stats.read_cpu_temp_c(),
            throttled=system_stats.read_throttled(),
            core_voltage_v=system_stats.read_core_voltage(),
            arm_clock_hz=system_stats.read_arm_clock_hz(),
            wifi=system_stats.read_wifi(),
            uptime_seconds=system_stats.read_uptime_seconds(),
            self_proc=system_stats.read_self_process(),
            camera_proc=system_stats.read_camera_process(),
        )

        self._host_history.append(sample)
        self._prev_disk = disk_io
        self._prev_net = net_io
        self._prev_sample_t = now_mono

    # ---------------- snapshot / queries ----
    def _camera_fps(self) -> float:
        now = time.monotonic()
        cutoff = now - FPS_WINDOW_SECONDS
        times = self._app.frame_times
        count = sum(1 for t in times if t >= cutoff)
        if count == 0:
            return 0.0
        return count / FPS_WINDOW_SECONDS

    def _motor_duty_percent(self) -> float:
        now = time.monotonic()
        cutoff = now - MOTOR_DUTY_WINDOW_SECONDS
        active_time = 0.0
        for start, stop in self._app.motor_intervals:
            end = stop if stop is not None else now
            clipped_start = max(start, cutoff)
            clipped_end = min(end, now)
            if clipped_end > clipped_start:
                active_time += clipped_end - clipped_start
        return 100.0 * active_time / MOTOR_DUTY_WINDOW_SECONDS

    def _mode_time_share(self) -> dict:
        now = time.time()
        cutoff = now - MODE_TIMELINE_WINDOW_SECONDS
        timeline = list(self._app.mode_timeline)
        if not timeline:
            return {"idle": 0.0, "manual": 0.0, "ai": 0.0}
        # Build segments clipped to [cutoff, now].
        segments: list[tuple[float, float, str]] = []
        for i, (ts, state) in enumerate(timeline):
            start = max(ts, cutoff)
            end = timeline[i + 1][0] if i + 1 < len(timeline) else now
            end = min(end, now)
            if end > start:
                segments.append((start, end, state))
        total = max(1e-9, now - max(cutoff, timeline[0][0]))
        shares = {"idle": 0.0, "manual": 0.0, "ai": 0.0}
        for start, end, state in segments:
            if state in shares:
                shares[state] += end - start
        return {k: 100.0 * v / total for k, v in shares.items()}

    def _import_camera_subscribers(self) -> int:
        try:
            from backend.camera import camera

            return len(getattr(camera, "_subscribers", set()))
        except Exception:
            return 0

    def snapshot(self) -> dict:
        latest = self._host_history[-1] if self._host_history else None
        history = list(self._host_history)

        def series(attr: str) -> list:
            return [getattr(s, attr) for s in history]

        def history_ts() -> list[float]:
            return [s.t for s in history]

        host_dict: Optional[dict] = None
        if latest is not None:
            host_dict = {
                "t": latest.t,
                "cpu_percent": latest.cpu_percent,
                "cpu_per_core": latest.cpu_per_core,
                "cpu_freq_mhz": latest.cpu_freq_mhz,
                "cpu_freq_max_mhz": latest.cpu_freq_max_mhz,
                "loadavg": latest.loadavg,
                "mem_used_bytes": latest.mem_used_bytes,
                "mem_total_bytes": latest.mem_total_bytes,
                "mem_percent": latest.mem_percent,
                "mem_cached_bytes": latest.mem_cached_bytes,
                "swap_used_bytes": latest.swap_used_bytes,
                "swap_total_bytes": latest.swap_total_bytes,
                "disk_used_bytes": latest.disk_used_bytes,
                "disk_total_bytes": latest.disk_total_bytes,
                "disk_percent": latest.disk_percent,
                "disk_read_bytes_per_s": latest.disk_read_bytes_per_s,
                "disk_write_bytes_per_s": latest.disk_write_bytes_per_s,
                "net_rx_bytes_per_s": latest.net_rx_bytes_per_s,
                "net_tx_bytes_per_s": latest.net_tx_bytes_per_s,
                "net_total_rx_bytes": latest.net_total_rx_bytes,
                "net_total_tx_bytes": latest.net_total_tx_bytes,
                "cpu_temp_c": latest.cpu_temp_c,
                "throttled": latest.throttled,
                "core_voltage_v": latest.core_voltage_v,
                "arm_clock_hz": latest.arm_clock_hz,
                "wifi": latest.wifi,
                "uptime_seconds": latest.uptime_seconds,
                "self_proc": latest.self_proc,
                "camera_proc": latest.camera_proc,
            }

        return {
            "server_time": time.time(),
            "app_uptime_seconds": time.time() - self.started_at,
            "host": host_dict,
            "history": {
                "t": history_ts(),
                "cpu_percent": series("cpu_percent"),
                "mem_percent": series("mem_percent"),
                "cpu_temp_c": series("cpu_temp_c"),
                "net_rx_bytes_per_s": series("net_rx_bytes_per_s"),
                "net_tx_bytes_per_s": series("net_tx_bytes_per_s"),
                "disk_read_bytes_per_s": series("disk_read_bytes_per_s"),
                "disk_write_bytes_per_s": series("disk_write_bytes_per_s"),
            },
            "app": {
                "ws_clients": self._app.ws_clients,
                "camera_subscribers": self._import_camera_subscribers(),
                "camera_fps": round(self._camera_fps(), 2),
                "camera_last_frame_bytes": self._app.last_frame_size_bytes,
                "camera_frames_dropped": self._app.camera_frames_dropped,
                "motor_duty_percent_60s": round(self._motor_duty_percent(), 2),
                "motor_last_cmd": self._app.motor_last_cmd,
                "mode_current": self._app.mode_current,
                "mode_time_share_5m_percent": self._mode_time_share(),
                "estop_count": self._app.estop_count,
                "error_count": self._app.error_count,
                "tool_calls": dict(self._app.tool_calls),
            },
        }


metrics = MetricsStore()


class _ErrorCountingHandler(logging.Handler):
    """Logging handler that bumps the error counter on WARNING+."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            metrics.record_error()


def install_error_counter() -> None:
    """Attach the error-counting handler to the root logger (idempotent)."""
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, _ErrorCountingHandler):
            return
    handler = _ErrorCountingHandler(level=logging.WARNING)
    root.addHandler(handler)
