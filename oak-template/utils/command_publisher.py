# utils/command_publisher.py
"""
Rate-limited command publisher.

Sends when the command changes or on a heartbeat interval.
No automatic FREE recovery from STOP — the decision layer owns commands.
"""

import time
from typing import Optional

from utils.config import WalkerConfig
from utils.arduino_serial import ArduinoSerial


class CommandPublisher:
    def __init__(self, cfg: WalkerConfig, serial: ArduinoSerial):
        self.cfg = cfg
        self._serial = serial
        self._last_sent: Optional[str] = None
        self._last_send_time: float = 0.0
        self._heartbeat_s: float = cfg.COMMAND_REFRESH_S
        self._min_interval_s: float = cfg.MIN_COMMAND_INTERVAL_S

    def _is_allowed(self, command: str, state: dict) -> tuple[bool, str]:
        if command.startswith("GO:"):
            if not state.get("ready", False):
                return False, "arduino_not_ready"
            if not state.get("sensor_ok", True):
                return False, "sensor_error"
            if not state.get("calibrated", True):
                return False, "not_calibrated"
            if command == "GO:LEFT" and state.get("locked_left", False):
                return False, "locked_left"
            if command == "GO:RIGHT" and state.get("locked_right", False):
                return False, "locked_right"
        return True, "allowed"

    def publish(
        self,
        command: str,
        state: dict,
        reason: str = "",
        **_,
    ) -> bool:
        """Publish a command. Returns True if sent to Arduino."""
        if command in ("NONE", ""):
            return False

        now = time.time()
        elapsed = now - self._last_send_time
        changed = command != self._last_sent

        allowed, block_reason = self._is_allowed(command, state)
        if not allowed:
            if changed:
                print(
                    f"[Publisher] BLOCKED {self._last_sent or '(none)'} -> {command} "
                    f"({block_reason})"
                )
            return False

        if elapsed < self._min_interval_s and not changed:
            return False

        if changed or elapsed >= self._heartbeat_s:
            self._send(command, now, reason or ("changed" if changed else "heartbeat"))
            return True

        return False

    def _send(self, command: str, now: float, reason: str) -> None:
        prev = self._last_sent
        self._serial.send_command(command)
        self._last_sent = command
        self._last_send_time = now
        print(f"[Publisher] {prev or '(none)'} -> {command} | {reason}")

    @property
    def last_command(self) -> Optional[str]:
        return self._last_sent

    def reset(self) -> None:
        self._last_sent = None
        self._last_send_time = 0.0
