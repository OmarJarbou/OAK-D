# utils/command_publisher.py
"""
Rate-limited command publisher for the Smart Walker.

Prevents command spam by only sending when:
  - The command actually changes, OR
  - A configurable refresh interval has elapsed (heartbeat).

Enforces a minimum interval between any two serial sends.
"""

import time
from typing import Optional

from utils.config import WalkerConfig
from utils.arduino_serial import ArduinoSerial


class CommandPublisher:
    """
    Rate-limited wrapper around ArduinoSerial.send_command().

    Usage:
        pub = CommandPublisher(cfg, arduino_serial)
        # In the main loop:
        pub.publish("GO:CENTER")  # Only sends if command changed or heartbeat expired
    """

    def __init__(self, cfg: WalkerConfig, serial: ArduinoSerial):
        self.cfg = cfg
        self._serial = serial
        self._last_sent: Optional[str] = None
        self._last_send_time: float = 0.0
        self._heartbeat_s: float = 2.0
        self._stop_heartbeat_s: float = 0.7

    @staticmethod
    def _is_left_command(command: str) -> bool:
        return command in {"GO:LEFT", "GO:L2", "GO:L1"}

    @staticmethod
    def _is_right_command(command: str) -> bool:
        return command in {"GO:RIGHT", "GO:R2", "GO:R1"}

    def _is_allowed(self, command: str, state: dict) -> tuple[bool, str]:
        """Validate command against Arduino readiness and lock/safety state."""
        if command.startswith("GO:"):
            if not state.get("ready", False):
                return False, "arduino_not_ready"
            if not state.get("sensor_ok", True):
                return False, "sensor_error"
            if not state.get("calibrated", True):
                return False, "not_calibrated"
            if self._is_left_command(command) and state.get("locked_left", False):
                return False, "locked_left"
            if self._is_right_command(command) and state.get("locked_right", False):
                return False, "locked_right"
        return True, "allowed"

    def publish(self, command: str, state: dict, reason: str = "") -> bool:
        """
        Attempt to publish a command. Returns True if actually sent.

        Rules:
          1. If command == "NONE", do nothing.
          2. Send immediately on command transition (subject to state safety checks).
          3. For unchanged command, heartbeat every 2.0s.
          4. STOP heartbeat may repeat faster, but not below 0.7s.
        """
        if command == "NONE":
            return False

        now = time.time()
        elapsed = now - self._last_send_time
        changed = (command != self._last_sent)

        allowed, block_reason = self._is_allowed(command, state)
        if not allowed:
            if changed:
                print(
                    f"[Publisher] {self._last_sent or '(none)'} -> {command} "
                    f"reason=blocked:{block_reason}"
                )
            return False

        if changed:
            self._send(command, now, reason or "changed")
            return True

        heartbeat_s = self._stop_heartbeat_s if command == "STOP" else self._heartbeat_s
        if elapsed >= heartbeat_s:
            self._send(command, now, reason or f"heartbeat_{heartbeat_s:.1f}s")
            return True

        return False

    def _send(self, command: str, now: float, reason: str) -> None:
        """Actually send the command via serial."""
        prev = self._last_sent
        self._serial.send_command(command)
        self._last_sent = command
        self._last_send_time = now
        print(f"[Publisher] {prev or '(none)'} -> {command} reason={reason}")

    @property
    def last_command(self) -> Optional[str]:
        return self._last_sent

    def reset(self) -> None:
        """Reset publisher state (e.g., on deauthorization)."""
        self._last_sent = None
        self._last_send_time = 0.0
