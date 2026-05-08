"""Front ultrasonic distance sensor.

The HC-SR04 is treated like the camera: a background task keeps the latest
reading in memory, and the rest of the app reads that cached value. This keeps
motor safety checks fast and lets future training code attach `distance_cm`
metadata without making Claude call a tool first.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from gpiozero import DistanceSensor as GpioDistanceSensor

from backend.config import (
    PIN_ULTRASONIC_ECHO,
    PIN_ULTRASONIC_TRIGGER,
    ULTRASONIC_MAX_DISTANCE_CM,
    ULTRASONIC_MIN_SAFE_FORWARD_CM,
    ULTRASONIC_POLL_SECONDS,
    ULTRASONIC_SMOOTHING_SAMPLES,
    ULTRASONIC_STALE_SECONDS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistanceReading:
    """Snapshot of the latest forward distance measurement."""

    distance_cm: Optional[float]
    status: str
    updated_at: Optional[float]
    stale: bool
    min_safe_forward_cm: float

    @property
    def safe_for_forward(self) -> bool:
        if self.distance_cm is None or self.stale:
            return True
        return self.distance_cm >= self.min_safe_forward_cm


class DistanceSensor:
    def __init__(self) -> None:
        self._sensor: Optional[GpioDistanceSensor] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._samples: Deque[float] = deque(maxlen=ULTRASONIC_SMOOTHING_SAMPLES)
        self._latest_cm: Optional[float] = None
        self._updated_at: Optional[float] = None
        self._status = "not_started"

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._sensor = GpioDistanceSensor(
            echo=PIN_ULTRASONIC_ECHO,
            trigger=PIN_ULTRASONIC_TRIGGER,
            max_distance=ULTRASONIC_MAX_DISTANCE_CM / 100.0,
            queue_len=1,
        )
        self._status = "starting"
        self._task = asyncio.create_task(self._poll_loop(), name="distance-sensor")
        logger.info(
            "distance sensor started: trigger=%d echo=%d",
            PIN_ULTRASONIC_TRIGGER,
            PIN_ULTRASONIC_ECHO,
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._sensor is not None:
            self._sensor.close()
            self._sensor = None
        self._status = "stopped"
        logger.info("distance sensor stopped")

    def latest(self) -> DistanceReading:
        now = time.monotonic()
        stale = (
            self._updated_at is None
            or now - self._updated_at > ULTRASONIC_STALE_SECONDS
        )
        status = "stale" if stale and self._latest_cm is not None else self._status
        return DistanceReading(
            distance_cm=self._latest_cm,
            status=status,
            updated_at=self._updated_at,
            stale=stale,
            min_safe_forward_cm=ULTRASONIC_MIN_SAFE_FORWARD_CM,
        )

    def is_forward_safe(self) -> bool:
        return self.latest().safe_for_forward

    async def _poll_loop(self) -> None:
        sensor = self._sensor
        if sensor is None:
            self._status = "unavailable"
            return

        while True:
            try:
                raw_cm = float(sensor.distance) * 100.0
                if 0.0 < raw_cm <= ULTRASONIC_MAX_DISTANCE_CM:
                    self._samples.append(raw_cm)
                    self._latest_cm = statistics.median(self._samples)
                    self._updated_at = time.monotonic()
                    self._status = "ok"
                else:
                    self._status = "out_of_range"
                await asyncio.sleep(ULTRASONIC_POLL_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._status = "error"
                logger.exception("distance sensor read failed")
                await asyncio.sleep(ULTRASONIC_POLL_SECONDS)


distance_sensor = DistanceSensor()
