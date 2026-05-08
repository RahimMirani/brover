"""Motor control for the L298N driver via gpiozero.

Two APIs live in this module:

  AI teleop (async, timed, duration-clamped)
    - await forward(seconds)
    - await backward(seconds)
    - await turn(direction, seconds)
    - stop()                # sync, always safe

  Manual teleop (sync, instantaneous, relies on a client-driven watchdog)
    - set_motion(cmd)       # cmd in {"forward","backward","left","right","stop"}
    - stop()

The AI functions drive, sleep, then stop inside a single await, so the
caller does not have to remember to stop the motors afterwards. The
manual function just latches the motor direction; the ModeManager
watchdog (see mode.py) calls stop() if the client stops refreshing the
command within MANUAL_WATCHDOG_SECONDS.

Motors always run at full speed in MVP (the L298N ENA/ENB jumpers are on;
no PWM speed control is wired). Direction semantics match test.py exactly.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

from gpiozero import Motor

from backend.config import (
    MAX_MOTOR_SECONDS,
    PIN_LEFT_IN1,
    PIN_LEFT_IN2,
    PIN_RIGHT_IN1,
    PIN_RIGHT_IN2,
    ULTRASONIC_POLL_SECONDS,
)
from backend.distance_sensor import DistanceReading, distance_sensor
from backend.metrics import metrics

Direction = Literal["left", "right"]
ManualCmd = Literal["forward", "backward", "left", "right", "stop"]


@dataclass(frozen=True)
class MotionResult:
    command: str
    requested_seconds: float
    elapsed_seconds: float
    stopped_reason: str | None = None
    distance_cm: float | None = None


_left = Motor(forward=PIN_LEFT_IN1, backward=PIN_LEFT_IN2)
_right = Motor(forward=PIN_RIGHT_IN1, backward=PIN_RIGHT_IN2)
_current_motion: ManualCmd = "stop"


def _clamp_seconds(seconds: float) -> float:
    return max(0.0, min(float(seconds), MAX_MOTOR_SECONDS))


def stop() -> None:
    """Stop both motors. Safe to call at any time from any context."""
    global _current_motion
    _left.stop()
    _right.stop()
    _current_motion = "stop"
    metrics.record_motor("stop")


def _unsafe_forward_reading() -> DistanceReading | None:
    reading = distance_sensor.latest()
    return None if reading.safe_for_forward else reading


def _set_forward() -> DistanceReading | None:
    blocked = _unsafe_forward_reading()
    if blocked is not None:
        stop()
        return blocked

    global _current_motion
    _left.forward()
    _right.forward()
    _current_motion = "forward"
    metrics.record_motor("forward")
    return None


async def forward(seconds: float) -> MotionResult:
    seconds = _clamp_seconds(seconds)
    if seconds <= 0:
        return MotionResult("forward", seconds, 0.0)

    blocked = _set_forward()
    if blocked is not None:
        return MotionResult(
            "forward",
            seconds,
            0.0,
            stopped_reason="obstacle",
            distance_cm=blocked.distance_cm,
        )

    started = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - started
            remaining = seconds - elapsed
            if remaining <= 0:
                return MotionResult("forward", seconds, seconds)

            await asyncio.sleep(min(ULTRASONIC_POLL_SECONDS, remaining))
            blocked = _unsafe_forward_reading()
            if blocked is not None:
                return MotionResult(
                    "forward",
                    seconds,
                    time.monotonic() - started,
                    stopped_reason="obstacle",
                    distance_cm=blocked.distance_cm,
                )
    finally:
        stop()


async def backward(seconds: float) -> None:
    seconds = _clamp_seconds(seconds)
    global _current_motion
    _left.backward()
    _right.backward()
    _current_motion = "backward"
    metrics.record_motor("backward")
    try:
        await asyncio.sleep(seconds)
    finally:
        stop()


async def turn(direction: Direction, seconds: float) -> None:
    """Spin in place. Direction is the rover's rotation direction.

    Matches left()/right() semantics from test.py:
        left  -> left wheel forward, right wheel backward (rover spins left)
        right -> left wheel backward, right wheel forward (rover spins right)
    """
    seconds = _clamp_seconds(seconds)

    global _current_motion
    if direction == "left":
        _left.forward()
        _right.backward()
        _current_motion = "left"
    elif direction == "right":
        _left.backward()
        _right.forward()
        _current_motion = "right"
    else:
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

    metrics.record_motor(direction)
    try:
        await asyncio.sleep(seconds)
    finally:
        stop()


def set_motion(cmd: ManualCmd) -> None:
    """Latch the motors into a direction for manual teleop.

    This does NOT auto-stop. The caller (the WebSocket handler) is responsible
    for sending "stop" on key release, and the ModeManager watchdog will stop
    the motors if commands stop arriving within MANUAL_WATCHDOG_SECONDS.
    """
    global _current_motion
    if cmd == "forward":
        blocked = _set_forward()
        if blocked is not None:
            return
        return
    elif cmd == "backward":
        _current_motion = "backward"
        _left.backward()
        _right.backward()
    elif cmd == "left":
        _current_motion = "left"
        _left.forward()
        _right.backward()
    elif cmd == "right":
        _current_motion = "right"
        _left.backward()
        _right.forward()
    elif cmd == "stop":
        stop()
        return
    else:
        raise ValueError(f"unknown manual cmd: {cmd!r}")

    metrics.record_motor(cmd)


def enforce_forward_safety() -> DistanceReading | None:
    """Stop latched manual forward motion if the live distance becomes unsafe."""
    if _current_motion != "forward":
        return None

    blocked = _unsafe_forward_reading()
    if blocked is not None:
        stop()
        return blocked
    return None
