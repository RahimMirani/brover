"""Environment variables + GPIO pin constants for the backend.

Secrets and model choices live in .env (loaded below). Everything else
(pin numbers, safety caps, camera settings) lives here as plain Python
constants so you can tweak them in one place without restarting shells
or touching deploy pipelines.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# -----------------------------------------------------------------------------
# Secrets (from .env)
# -----------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")


# -----------------------------------------------------------------------------
# Embedding model
# -----------------------------------------------------------------------------
# voyage-multimodal-3 returns 1024-dim vectors in a joint image-text space.
# If we change models here we must also drop the *_vectors virtual tables in
# data/brover.db so they get recreated with the right dimension.
VOYAGE_EMBED_MODEL = "voyage-multimodal-3"


# -----------------------------------------------------------------------------
# Voice choices
# -----------------------------------------------------------------------------
OPENAI_STT_MODEL = "whisper-1"
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
# Voice options for OpenAI TTS: alloy, echo, fable, onyx, nova, shimmer
OPENAI_TTS_VOICE = "alloy"


# -----------------------------------------------------------------------------
# GPIO pin assignments (BCM numbering)
# -----------------------------------------------------------------------------
# L298N mapping. Matches the working wiring in test.py exactly.
#
#   Left motor:  IN1=17 (forward), IN2=27 (backward)
#   Right motor: IN3=22 (forward), IN4=23 (backward)
#
# No PWM enable pins for MVP; the L298N ENA/ENB jumpers stay in place,
# motors run at full speed whenever driven. PWM speed control can be
# added post-MVP by removing those jumpers and wiring ENA/ENB to GPIO.
PIN_LEFT_IN1 = 17
PIN_LEFT_IN2 = 27
PIN_RIGHT_IN1 = 22
PIN_RIGHT_IN2 = 23


# -----------------------------------------------------------------------------
# Ultrasonic distance sensor (BCM numbering)
# -----------------------------------------------------------------------------
# HC-SR04 front distance sensor. Trigger is safe from the Pi; echo is commonly
# 5V and must be level-shifted or divided down before reaching the GPIO pin.
PIN_ULTRASONIC_TRIGGER = 24
PIN_ULTRASONIC_ECHO = 25

# Distance readings are live safety data. They are kept in memory continuously
# and later can be stored alongside explicit training samples.
ULTRASONIC_MAX_DISTANCE_CM = 300.0
ULTRASONIC_MIN_SAFE_FORWARD_CM = 25.0
ULTRASONIC_POLL_SECONDS = 0.05
ULTRASONIC_STALE_SECONDS = 0.5
ULTRASONIC_SMOOTHING_SAMPLES = 5


# -----------------------------------------------------------------------------
# Safety caps
# -----------------------------------------------------------------------------
# Hard upper bound on any AI-driven timed motion call. Claude can ask for
# forward(seconds=10) but it will be clamped to this value.
MAX_MOTOR_SECONDS = 15.0

# Max number of tool_use iterations per user command before the agent loop
# aborts. Prevents runaway sequences.
MAX_AGENT_ITERATIONS = 30

# If the phone stops sending manual-drive commands for this long while in
# manual mode, the mode watchdog stops the motors. Matches the 180ms used
# in test.py.
MANUAL_WATCHDOG_SECONDS = 0.18


# -----------------------------------------------------------------------------
# Camera
# -----------------------------------------------------------------------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
