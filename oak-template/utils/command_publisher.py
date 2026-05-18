# utils/command_publisher.py
"""
Simplified rate-limited command publisher for the Smart Walker.

Rules:
  - NONE → skip.
  - STOP → send immediately, always.
  - Other commands → send on change; no send if < MIN_SWITCH_S since last change.
  - Heartbeat every 2 s for unchanged non-STOP commands.
  - No STOP_HOLD, no RECOVERY_FREE, no POST_RECOVERY_GRACE.
"""

import time
from typing import Optional

from utils.config import WalkerConfig
from utils.arduino_serial import ArduinoSerial


class CommandPublisher:
    """
    Rate-limited wrapper around ArduinoSerial.send_command().
    """

    _MIN_SWITCH_S: float = 0.3    # min seconds before a non-STOP command change
    _HEARTBEAT_S: float = 2.0     # re-send unchanged command interval

    def __init__(self, cfg: WalkerConfig, serial: ArduinoSerial):
        self.cfg = cfg
        self._serial = serial
        self._last_sent: Optional[str] = None
        self._last_send_time: float = 0.0
        self._last_change_time: float = 0.0
        self._last_stable_count: int = 0
        self._last_depth_mm: float = 0.0
        self._flip_lr: bool = bool(cfg.FLIP_LR)

    # ── Lock helpers (kept for arduino state check) ───────

    def _is_left_command(self, command: str) -> bool:
        if not self._flip_lr:
            return command in {"GO:LEFT", "GO:L2", "GO:L1"}
        return command in {"GO:RIGHT", "GO:R2", "GO:R1"}

    def _is_right_command(self, command: str) -> bool:
        if not self._flip_lr:
            return command in {"GO:RIGHT", "GO:R2", "GO:R1"}
        return command in {"GO:LEFT", "GO:L2", "GO:L1"}

    def _is_allowed(self, command: str, state: dict) -> tuple[bool, str]:
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

    # ── Main publish ──────────────────────────────────────

    def publish(
        self,
        command: str,
        state: dict,
        reason: str = "",
        min_p20_depth: float = 0.0,
        stable_count: int = 0,
        critical_stop: bool = False,
        allow_recenter: bool = False,
    ) -> bool:
        """
        Attempt to publish command. Returns True if actually sent.
        """
        if command == "NONE":
            return False

        now = time.time()
        changed = command != self._last_sent
        elapsed = now - self._last_send_time

        self._last_depth_mm = float(min_p20_depth or 0.0)
        self._last_stable_count = int(stable_count or 0)

        allowed, block_reason = self._is_allowed(command, state)
        if not allowed:
            if changed:
                print(
                    f"[Publisher] BLOCKED {self._last_sent or '(none)'} -> {command} "
                    f"reason={block_reason}"
                )
            return False

        # ── STOP: always send immediately ────────────────
        if command == "STOP":
            if changed or elapsed >= self._HEARTBEAT_S:
                self._send(command, now, reason or "STOP")
                return True
            return False

        # ── Non-STOP: respect minimum switch interval ────
        if changed:
            if now - self._last_change_time < self._MIN_SWITCH_S:
                return False
            self._send(command, now, reason or "changed")
            self._last_change_time = now
            return True

        # ── Heartbeat for unchanged commands ─────────────
        if elapsed >= self._HEARTBEAT_S:
            self._send(command, now, reason or f"heartbeat_{self._HEARTBEAT_S:.1f}s")
            return True

        return False

    # ── Internal send ─────────────────────────────────────

    def _send(self, command: str, now: float, reason: str) -> None:
        prev = self._last_sent
        self._serial.send_command(command)
        self._last_sent = command
        self._last_send_time = now
        print(
            f"[Publisher] {prev or '(none)'} -> {command} | {reason} | "
            f"stable_count={self._last_stable_count} | depth_mm={self._last_depth_mm:.0f}"
        )

    # ── Properties ───────────────────────────────────────

    @property
    def last_command(self) -> Optional[str]:
        return self._last_sent

    def reset(self) -> None:
        self._last_sent = None
        self._last_send_time = 0.0
        self._last_change_time = 0.0
