"""HC-SR04 distance sensor smoke test.

Streams live readings from the front ultrasonic sensor to stdout so you
can confirm the wiring is correct before trusting it from the main
backend. Bypasses FastAPI, asyncio, the camera, and the DB - just
gpiozero on the pins configured in backend/config.py.

Run on the Pi (stop the FastAPI server first - gpiozero claims the pins
exclusively):

    python -m scripts.distance_smoke

What to expect:
    - Wave your hand 5-50 cm in front of the sensor and the printed
      cm value should track it smoothly.
    - With nothing in front you should see ~max range (close to 300 cm),
      not 0 and not a stream of "no echo" warnings.
    - "WARN DistanceSensorNoEcho: echo pin set high" on every line means
      the echo line is stuck HIGH - usually missing voltage divider on
      ECHO, swapped TRIG/ECHO, no 5V on VCC, or a dead sensor.

Press Ctrl+C to stop. A min/avg/max summary is printed on exit.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from typing import Optional

from gpiozero import DistanceSensor

from backend.config import (
    PIN_ULTRASONIC_ECHO,
    PIN_ULTRASONIC_TRIGGER,
    ULTRASONIC_MAX_DISTANCE_CM,
    ULTRASONIC_MIN_SAFE_FORWARD_CM,
)

POLL_SECONDS = 0.1
BAR_WIDTH = 30


def _on_warning(message, category, filename, lineno, file=None, line=None) -> None:
    """Surface gpiozero warnings inline with the readings.

    Python's default filter shows each warning only once, which hides
    persistent wiring problems. We want every occurrence here.
    """
    print(f"  WARN {category.__name__}: {message}", file=sys.stderr, flush=True)


def _bar(distance_cm: float, max_cm: float) -> str:
    frac = max(0.0, min(1.0, distance_cm / max_cm))
    filled = int(frac * BAR_WIDTH)
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"


def _classify(distance_cm: float, max_cm: float) -> str:
    if distance_cm >= max_cm * 0.999:
        return "out of range / no echo"
    if distance_cm <= 0.0:
        return "invalid (0)"
    if distance_cm < ULTRASONIC_MIN_SAFE_FORWARD_CM:
        return f"TOO CLOSE (< {ULTRASONIC_MIN_SAFE_FORWARD_CM:.0f} cm)"
    return "ok"


def main() -> int:
    warnings.simplefilter("always")
    warnings.showwarning = _on_warning

    print("HC-SR04 distance sensor smoke test")
    print(f"  trigger pin (BCM) = {PIN_ULTRASONIC_TRIGGER}")
    print(f"  echo pin    (BCM) = {PIN_ULTRASONIC_ECHO}")
    print(f"  max range         = {ULTRASONIC_MAX_DISTANCE_CM:.0f} cm")
    print(f"  min safe forward  = {ULTRASONIC_MIN_SAFE_FORWARD_CM:.0f} cm")
    print(f"  poll interval     = {POLL_SECONDS:.2f} s")
    print(f"  pin factory       = {os.environ.get('GPIOZERO_PIN_FACTORY', 'default')}")
    print("\nReading distance... (Ctrl+C to stop)\n", flush=True)

    sensor = DistanceSensor(
        echo=PIN_ULTRASONIC_ECHO,
        trigger=PIN_ULTRASONIC_TRIGGER,
        max_distance=ULTRASONIC_MAX_DISTANCE_CM / 100.0,
        queue_len=1,
    )

    n = 0
    min_d: Optional[float] = None
    max_d: Optional[float] = None
    sum_d = 0.0

    try:
        while True:
            distance_cm = float(sensor.distance) * 100.0
            label = _classify(distance_cm, ULTRASONIC_MAX_DISTANCE_CM)
            bar = _bar(distance_cm, ULTRASONIC_MAX_DISTANCE_CM)
            ts = time.strftime("%H:%M:%S")

            print(
                f"[{ts}] {distance_cm:6.1f} cm {bar} {label}",
                flush=True,
            )

            n += 1
            sum_d += distance_cm
            min_d = distance_cm if min_d is None else min(min_d, distance_cm)
            max_d = distance_cm if max_d is None else max(max_d, distance_cm)

            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print()  # newline after the ^C
        if n > 0 and min_d is not None and max_d is not None:
            avg = sum_d / n
            print(
                f"Stats over {n} samples: "
                f"min={min_d:.1f} cm  avg={avg:.1f} cm  max={max_d:.1f} cm"
            )
        return 0
    finally:
        sensor.close()


if __name__ == "__main__":
    sys.exit(main())
