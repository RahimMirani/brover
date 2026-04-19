"""Motor control for the L298N driver via gpiozero.

Two APIs live in this module:

  AI teleop (async, timed, duration-clamped)
    - await forward(seconds, speed)
    - await backward(seconds, speed)
    - await turn(direction, seconds, speed)
    - stop()                # sync, always safe

  Manual teleop (sync, instantaneous, relies on a short client-driven watchdog)
    - set_motion(cmd)       # cmd in {"forward","backward","left","right","stop"}
    - stop()

The AI functions do the whole motion-plus-sleep-plus-stop sequence inside
one await, so the caller does not have to remember to stop the motors
afterwards. The manual function just latches the motor direction; the
ModeManager watchdog (see mode.py) calls stop() if the client stops
refreshing the command within MANUAL_WATCHDOG_SECONDS.

Wiring matches test.py with the addition of PWM enable pins. See the
block comment in config.py for L298N jumper requirements.
"""
from __future__ import annotations

import asyncio
from typing import Literal

from gpiozero import Motor

from backend.config import (
    ENABLE_PWM,
    MAX_MOTOR_SECONDS,
    PIN_LEFT_EN,
    PIN_LEFT_IN1,
    PIN_LEFT_IN2,
    PIN_RIGHT_EN,
    PIN_RIGHT_IN1,
    PIN_RIGHT_IN2,
)

Direction = Literal["left", "right"]
ManualCmd = Literal["forward", "backward", "left", "right", "stop"]


def _make_motor(forward_pin: int, backward_pin: int, enable_pin: int) -> Motor:
    if ENABLE_PWM:
        return Motor(forward=forward_pin, backward=backward_pin, enable=enable_pin, pwm=True)
    return Motor(forward=forward_pin, backward=backward_pin)


_left = _make_motor(PIN_LEFT_IN1, PIN_LEFT_IN2, PIN_LEFT_EN)
_right = _make_motor(PIN_RIGHT_IN1, PIN_RIGHT_IN2, PIN_RIGHT_EN)


def _clamp_seconds(seconds: float) -> float:
    return max(0.0, min(float(seconds), MAX_MOTOR_SECONDS))


def _clamp_speed(speed: float) -> float:
    return max(0.0, min(float(speed), 1.0))


def stop() -> None:
    """Stop both motors. Safe to call at any time from any context."""
    _left.stop()
    _right.stop()


async def forward(seconds: float, speed: float = 0.7) -> None:
    seconds = _clamp_seconds(seconds)
    speed = _clamp_speed(speed)
    _left.forward(speed)
    _right.forward(speed)
    try:
        await asyncio.sleep(seconds)
    finally:
        stop()


async def backward(seconds: float, speed: float = 0.7) -> None:
    seconds = _clamp_seconds(seconds)
    speed = _clamp_speed(speed)
    _left.backward(speed)
    _right.backward(speed)
    try:
        await asyncio.sleep(seconds)
    finally:
        stop()


async def turn(direction: Direction, seconds: float, speed: float = 0.6) -> None:
    """Spin in place. Direction is the rover's rotation direction.

    Matches the left()/right() semantics from test.py:
        left  -> left wheel forward, right wheel backward (rover spins left)
        right -> left wheel backward, right wheel forward (rover spins right)
    """
    seconds = _clamp_seconds(seconds)
    speed = _clamp_speed(speed)

    if direction == "left":
        _left.forward(speed)
        _right.backward(speed)
    elif direction == "right":
        _left.backward(speed)
        _right.forward(speed)
    else:
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

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
    if cmd == "forward":
        _left.forward(1.0)
        _right.forward(1.0)
    elif cmd == "backward":
        _left.backward(1.0)
        _right.backward(1.0)
    elif cmd == "left":
        _left.forward(1.0)
        _right.backward(1.0)
    elif cmd == "right":
        _left.backward(1.0)
        _right.forward(1.0)
    elif cmd == "stop":
        stop()
    else:
        raise ValueError(f"unknown manual cmd: {cmd!r}")
