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
        self._free_sticky_seconds: float = cfg.FREE_STICKY_SECONDS
        self._post_recovery_grace_s: float = cfg.POST_RECOVERY_GRACE_S
        self._post_recovery_until: float = 0.0
        self._stop_entered_time: float = 0.0
        self._stop_hold_seconds: float = cfg.STOP_HOLD_SECONDS
        self._critical_stop_distance_mm: float = cfg.CRITICAL_STOP_DISTANCE_MM
        self._min_command_hold_s: float = cfg.MIN_COMMAND_HOLD_MS / 1000.0
        self._current_cmd_entered_time: float = 0.0

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

    def publish(self, command: str, state: dict, reason: str = "",
            min_p20_depth: float = 0.0,
            stable_count: int = 0,
            critical_stop: bool = False) -> bool:
        """
        Attempt to publish a command. Returns True if actually sent.

        Rules:
        1. NONE → skip.
        2. Critical STOP overrides all cooldowns immediately.
        3. Non-critical transitions must hold MIN_COMMAND_HOLD_MS.
        4. STOP is edge-triggered (latch), with recovery FREE after hold.
        5. Non-STOP unchanged commands heartbeat every 2.0s.
        """
        if command == "NONE":
            return False

        now = time.time()
        elapsed = now - self._last_send_time
        held_current = now - self._current_cmd_entered_time
        changed = (command != self._last_sent)

        allowed, block_reason = self._is_allowed(command, state)
        if not allowed:
            if changed:
                print(
                    f"[Publisher] BLOCKED {self._last_sent or '(none)'} -> {command} "
                    f"reason=blocked:{block_reason} stable={stable_count}"
                )
            return False

        # POST_RECOVERY grace: after STOP -> FREE recovery, allow GO:* immediately
        # so the user can steer out, but block STOP unless it's truly critical.
        if now < self._post_recovery_until:
            if command == "STOP":
                if not (min_p20_depth > 0.0 and min_p20_depth < self._critical_stop_distance_mm):
                    print("[Publisher] POST_RECOVERY grace: blocking STOP, allowing GO")
                    return False
            elif command.startswith("GO:") and changed:
                print("[Publisher] POST_RECOVERY grace: blocking STOP, allowing GO")
                self._send(command, now, reason or "POST_RECOVERY_GO")
                return True

        # Task B: FREE should be sticky. If we're in FREE and the engine tries to
        # switch to GO:CENTER shortly after, suppress it. Only STOP or a genuine
        # side-steer GO:* should break out of FREE.
        if (
            self._last_sent == "FREE"
            and command == "GO:CENTER"
            and held_current < self._free_sticky_seconds
        ):
            print(
                f"[Publisher] STICKY_FREE FREE -> GO:CENTER blocked "
                f"held={held_current:.2f}s < {self._free_sticky_seconds:.2f}s stable={stable_count}"
            )
            return False

        # ── Critical STOP: override all cooldowns immediately ────────
        if command == "STOP" and critical_stop and changed:
            print(
                f"[Publisher] {self._last_sent or '(none)'} -> STOP "
                f"reason=CRITICAL_STOP stable={stable_count} "
                f"(cooldown override)"
            )
            self._stop_entered_time = now
            self._send(command, now, "CRITICAL_STOP")
            return True

        # ── Entering STOP (non-critical) ─────────────────────────────
        if command == "STOP" and changed:
            if held_current < self._min_command_hold_s:
                print(
                    f"[Publisher] COOLDOWN {self._last_sent or '(none)'} -> STOP "
                    f"held={held_current*1000:.0f}ms < {self._min_command_hold_s*1000:.0f}ms "
                    f"stable={stable_count}"
                )
                return False
            self._stop_entered_time = now
            self._send(command, now, reason or "STOP_transition")
            return True

        # ── Already in STOP ──────────────────────────────────────────
        if command == "STOP" and self._last_sent == "STOP":
            held = now - self._stop_entered_time

            if min_p20_depth > 0.0 and min_p20_depth < self._critical_stop_distance_mm:
                print(
                    f"[Publisher] CRITICAL_STOP held={held:.2f}s "
                    f"depth={min_p20_depth:.0f}mm stable={stable_count}"
                )
                return False

            if held < self._stop_hold_seconds:
                print(
                    f"[Publisher] STOP_HOLD {held:.2f}s / {self._stop_hold_seconds}s "
                    f"stable={stable_count}"
                )
                return False

            # Recovery FREE
            print(
                f"[Publisher] STOP -> FREE reason=RECOVERY_FREE "
                f"held={held:.2f}s stable={stable_count}"
            )
            self._send("FREE", now, "RECOVERY_FREE")
            self._post_recovery_until = now + self._post_recovery_grace_s
            return True

        # ── Non-STOP transition with cooldown guard ───────────────────
        if changed:
            if held_current < self._min_command_hold_s:
                print(
                    f"[Publisher] COOLDOWN {self._last_sent or '(none)'} -> {command} "
                    f"held={held_current*1000:.0f}ms < {self._min_command_hold_s*1000:.0f}ms "
                    f"stable={stable_count}"
                )
                return False
            self._send(command, now, reason or "changed")
            return True

        # ── Heartbeat for unchanged non-STOP ─────────────────────────
        if elapsed >= self._heartbeat_s:
            self._send(command, now, reason or f"heartbeat_{self._heartbeat_s:.1f}s")
            return True

        return False

    def _send(self, command: str, now: float, reason: str) -> None:
        """Actually send the command via serial and log transition."""
        prev = self._last_sent
        self._serial.send_command(command)
        self._last_sent = command
        self._last_send_time = now
        self._current_cmd_entered_time = now
        print(f"[Publisher] {prev or '(none)'} -> {command} reason={reason}")

    @property
    def last_command(self) -> Optional[str]:
        return self._last_sent

    def reset(self) -> None:
        """Reset publisher state (e.g., on deauthorization)."""
        self._last_sent = None
        self._last_send_time = 0.0
