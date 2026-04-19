"""Environment variables + GPIO pin constants for the backend.

Loads the .env file (if present) on import so every module can read
the same values. Keep this module side-effect-light: no I/O, no hardware.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# -----------------------------------------------------------------------------
# GPIO pin assignments (BCM numbering)
# -----------------------------------------------------------------------------
# L298N mapping. Matches the working wiring in test.py, with the addition of
# PWM enable pins for speed control.
#
#   Left motor:  IN1=17 (forward), IN2=27 (backward), ENA=12 (PWM enable)
#   Right motor: IN3=22 (forward), IN4=23 (backward), ENB=13 (PWM enable)
#
# IMPORTANT: for PWM to actually vary motor speed, the ENA and ENB jumpers
# on the L298N board must be physically removed, and GPIO 12 and GPIO 13
# wired to those header pins. If the jumpers are left in place, GPIO will
# fight the 5V jumper and you can damage the board. If you prefer to keep
# the jumpers on (always-full-speed, no PWM), set ENABLE_PWM=false in .env
# and the Motor objects will be constructed without enable pins.
PIN_LEFT_IN1 = 17
PIN_LEFT_IN2 = 27
PIN_LEFT_EN = 12

PIN_RIGHT_IN1 = 22
PIN_RIGHT_IN2 = 23
PIN_RIGHT_EN = 13

ENABLE_PWM = (os.getenv("ENABLE_PWM", "true").strip().lower() != "false")

# -----------------------------------------------------------------------------
# API keys + models
# -----------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

# -----------------------------------------------------------------------------
# Safety tunables
# -----------------------------------------------------------------------------
MAX_MOTOR_SECONDS = _float("MAX_MOTOR_SECONDS", 3.0)
MAX_AGENT_ITERATIONS = _int("MAX_AGENT_ITERATIONS", 10)
MANUAL_WATCHDOG_SECONDS = _float("MANUAL_WATCHDOG_SECONDS", 0.18)

# -----------------------------------------------------------------------------
# Camera
# -----------------------------------------------------------------------------
CAMERA_WIDTH = _int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT = _int("CAMERA_HEIGHT", 480)
CAMERA_FPS = _int("CAMERA_FPS", 30)
