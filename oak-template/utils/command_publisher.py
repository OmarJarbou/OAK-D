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

    def publish(self, command: str) -> bool:
        """
        Attempt to publish a command. Returns True if actually sent.

        Rules:
          1. If command == "NONE", do nothing (no nav commands before auth).
          2. If command differs from last sent → send immediately
             (respecting MIN_COMMAND_INTERVAL_S).
          3. If same command → send only after COMMAND_REFRESH_S
             (STOP uses STOP_REPEAT_INTERVAL_S for faster heartbeat).
          4. Never send faster than MIN_COMMAND_INTERVAL_S.
        """
        if command == "NONE":
            return False

        now = time.time()
        elapsed = now - self._last_send_time

        # Respect minimum interval
        if elapsed < self.cfg.MIN_COMMAND_INTERVAL_S:
            return False

        changed = (command != self._last_sent)

        if changed:
            # New command → send immediately
            self._send(command, now)
            return True

        # Same command → heartbeat refresh (same interval for all commands)
        if elapsed >= self.cfg.COMMAND_REFRESH_S:
            self._send(command, now)
            return True

        return False

    def _send(self, command: str, now: float) -> None:
        """Actually send the command via serial."""
        self._serial.send_command(command)
        prev = self._last_sent
        self._last_sent = command
        self._last_send_time = now
        if prev != command:
            print(f"[Publisher] {prev or '(none)'} → {command}")

    @property
    def last_command(self) -> Optional[str]:
        return self._last_sent

    def reset(self) -> None:
        """Reset publisher state (e.g., on deauthorization)."""
        self._last_sent = None
        self._last_send_time = 0.0
