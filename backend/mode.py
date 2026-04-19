"""Mode state machine coordinating manual teleop and AI agent control.

Three modes:
    idle    no motor activity, no watchdog. Initial state.
    manual  user is driving with D-pad/WASD. A 180ms watchdog stops
            the motors if the phone stops refreshing the command, so a
            dropped connection or tab-switch can never leave the rover
            driving away.
    ai      the Claude agent loop owns the motors. The agent's own
            timed asyncio.sleep(seconds) handles motor cutoff, so the
            manual watchdog is disabled while in this mode.

Conflict rule: any manual input that arrives while in ai mode
    1. sets cancel_event so the agent loop exits cleanly,
    2. stops the motors immediately,
    3. transitions into manual mode and executes the new command.
Manual always wins.

The cancel_event is never replaced -- it is cleared at the start of each
AI turn and set when something asks the agent to stop. That way, a reference
to mode.cancel_event captured by a running agent is always the right one.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal, Optional

from backend import motors
from backend.config import MANUAL_WATCHDOG_SECONDS

logger = logging.getLogger(__name__)

Mode = Literal["idle", "manual", "ai"]


class ModeManager:
    def __init__(self) -> None:
        self._state: Mode = "idle"
        self.cancel_event: asyncio.Event = asyncio.Event()
        self._last_move_at: float = 0.0
        self._watchdog_task: Optional[asyncio.Task[None]] = None

    @property
    def state(self) -> Mode:
        return self._state

    def on_manual_input(self, cmd: str) -> None:
        """Handle an incoming manual drive command.

        Safe to call from any state. Seizes the motors from the AI if
        an agent loop is currently running.
        """
        if self._state == "ai":
            logger.info("manual override: cancelling agent")
            self.cancel_event.set()
            motors.stop()
            self._state = "idle"

        if self._state != "manual":
            self._enter_manual()

        self._last_move_at = time.monotonic()
        try:
            motors.set_motion(cmd)  # type: ignore[arg-type]
        except ValueError as e:
            logger.warning("bad manual cmd %r: %s", cmd, e)

    def enter_ai(self) -> None:
        """Transition to AI mode at the start of a voice command."""
        self._cancel_watchdog()
        motors.stop()
        self.cancel_event.clear()
        self._state = "ai"
        logger.info("mode -> ai")

    def enter_idle(self) -> None:
        """Return to idle. Called after AI finishes normally."""
        self._cancel_watchdog()
        motors.stop()
        self._state = "idle"
        logger.info("mode -> idle")

    def request_estop(self) -> None:
        """E-stop: kill motors, cancel agent, cancel watchdog, go idle.

        Called on the user's red-button press and on WebSocket disconnect.
        """
        motors.stop()
        self.cancel_event.set()
        self._cancel_watchdog()
        self._state = "idle"
        logger.info("estop -> idle")

    def _enter_manual(self) -> None:
        self._state = "manual"
        self._last_move_at = time.monotonic()
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(), name="manual-watchdog"
        )
        logger.info("mode -> manual")

    def _cancel_watchdog(self) -> None:
        task = self._watchdog_task
        if task is not None and not task.done():
            task.cancel()
        self._watchdog_task = None

    async def _watchdog_loop(self) -> None:
        """Stop motors if manual commands stop arriving within the timeout."""
        try:
            while self._state == "manual":
                await asyncio.sleep(0.03)
                if self._state != "manual":
                    return
                if time.monotonic() - self._last_move_at > MANUAL_WATCHDOG_SECONDS:
                    logger.debug("manual watchdog timeout -> idle")
                    motors.stop()
                    self._state = "idle"
                    self._watchdog_task = None
                    return
        except asyncio.CancelledError:
            raise


mode = ModeManager()
