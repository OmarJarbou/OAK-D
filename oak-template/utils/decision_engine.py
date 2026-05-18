# utils/decision_engine.py
"""
Simplified Smart Walker decision engine — v4.0

Logic:
  - Score every zone by p20_depth / 1500 (clip 0-1), zero if valid_ratio < 0.1
  - Pick best_zone = highest score
  - Map best_zone → GO:LEFT / GO:CENTER / GO:RIGHT / FREE
  - LiDAR: correct direction only when camera AND lidar agree on blockage
  - STOP only for genuine front obstruction confirmed by both sensors (or camera alone)
  - No STOP_HOLD, no RECOVERY_FREE, no POST_RECOVERY_GRACE
"""

import time
from dataclasses import dataclass
from typing import Optional

from utils.config import WalkerConfig
from utils.corridor_analyzer import AnalysisResult, FreeSpaceGroup


@dataclass
class DecisionResult:
    """Complete output of one decision cycle."""
    raw_command: str
    stable_command: str
    confidence: float
    chosen_corridor: str
    chosen_group: Optional[FreeSpaceGroup]
    reason: str
    valid_groups: list
    center_blocked_reason: str
    critical_stop: bool = False
    stable_count: int = 0
    allow_recenter: bool = False


class DecisionEngine:
    """
    Reactive navigation decision engine.

    Priority:
      1. Safety gates (authorized, ready, sensor_ok, calibrated)
      2. Emergency: obstacle < EMERGENCY_STOP_MM confirmed by camera (± LiDAR)
      3. Best zone selection (highest depth score)
      4. LiDAR side correction
      5. FREE when center is deeply clear
    """

    # Minimum seconds before switching to a different non-STOP command
    _MIN_SWITCH_INTERVAL_S: float = 0.3

    def __init__(self, cfg: WalkerConfig):
        self.cfg = cfg
        self._last_stable: str = "STOP"
        self._last_change_time: float = 0.0

    # ── Public API ────────────────────────────────────────

    def decide(
        self,
        analysis: AnalysisResult,
        arduino_state: dict,
        fusion_boost: float = 0.0,
        lidar_left_mm: float = 9999.0,
        lidar_right_mm: float = 9999.0,
        side_escape_left: bool = False,
        side_escape_right: bool = False,
        fusion_reason: str = "",
        lidar_front_mm: float = 9999.0,
    ) -> DecisionResult:
        cfg = self.cfg

        # ── 1. Safety Gates ──────────────────────────────
        if not arduino_state.get("authorized", False):
            return self._result("NONE", "NONE", 0.0, "", None,
                                "Not authorized", [], "")

        if not arduino_state.get("ready", False):
            return self._result("NONE", "NONE", 0.0, "", None,
                                "Arduino not ready", [], "")

        if not arduino_state.get("sensor_ok", True):
            return self._result("STOP", "STOP", 1.0, "", None,
                                "STOP: sensor error", [], "", critical_stop=True)

        if not arduino_state.get("calibrated", True):
            return self._result("STOP", "STOP", 1.0, "", None,
                                "STOP: not calibrated", [], "", critical_stop=True)

        # ── 2. No depth data ─────────────────────────────
        if not analysis.corridors:
            return self._result("STOP", "STOP", 0.5, "", None,
                                "STOP: no depth data", [], "", critical_stop=True)

        corridors = analysis.corridors

        # ── 3. Score every zone ──────────────────────────
        # score = p20 / 1500, zero when valid_ratio < 0.1
        zone_scores: dict[str, float] = {}
        for name, m in corridors.items():
            if m.valid_ratio < 0.1:
                zone_scores[name] = 0.0
            else:
                zone_scores[name] = float(min(1.0, max(0.0, m.p20_depth / 1500.0)))

        # ── 4. Emergency check ───────────────────────────
        center_m = corridors.get("CENTER")
        p20_center = center_m.p20_depth if center_m else 0.0

        # Camera-side emergency: has_emergency flag or center p20 < threshold
        camera_emergency = (
            analysis.has_emergency
            or (center_m is not None
                and center_m.valid_ratio > 0.1
                and 0 < p20_center < cfg.EMERGENCY_STOP_MM)
        )

        # LiDAR front close — trust only when camera p20 also small
        # (ignore LiDAR when camera sees far, likely low obstacle)
        lidar_front_close = (
            lidar_front_mm < cfg.EMERGENCY_STOP_MM
            and p20_center < 800
        )

        emergency = camera_emergency or lidar_front_close

        # ── 5. Pick best zone ────────────────────────────
        best_name = max(zone_scores, key=zone_scores.__getitem__)
        best_score = zone_scores[best_name]
        best_m = corridors.get(best_name)

        # ── 6. Map zone → command ────────────────────────
        if best_name == "CENTER":
            cmd = "GO:CENTER"
            reason = f"CENTER deepest (p20={int(p20_center)}mm)"
        elif best_name in ("L1", "L2", "LEFT"):
            cmd = "GO:LEFT"
            reason = f"{best_name} clearer (p20={int(best_m.p20_depth if best_m else 0)}mm)"
        elif best_name in ("R1", "R2", "RIGHT"):
            cmd = "GO:RIGHT"
            reason = f"{best_name} clearer (p20={int(best_m.p20_depth if best_m else 0)}mm)"
        else:
            cmd = "GO:CENTER"
            reason = "Fallback center"

        # ── 7. LiDAR side correction ─────────────────────
        # Correct direction only when BOTH camera and LiDAR indicate a problem.
        if cmd == "GO:LEFT" and lidar_left_mm < 400:
            left_m = corridors.get("L1") or corridors.get("LEFT")
            left_p20 = left_m.p20_depth if left_m else 0.0
            if left_p20 < 700:
                # Camera AND LiDAR agree left is blocked — switch right
                right_m = corridors.get("R1") or corridors.get("RIGHT")
                right_p20 = right_m.p20_depth if right_m else 0.0
                if right_p20 > 600 and lidar_right_mm > 400:
                    cmd = "GO:RIGHT"
                    reason = f"LiDAR+cam: left blocked, switching right"
                # else trust camera, keep GO:LEFT
            # else camera sees left as clear → trust camera, ignore LiDAR
        elif cmd == "GO:RIGHT" and lidar_right_mm < 400:
            right_m = corridors.get("R1") or corridors.get("RIGHT")
            right_p20 = right_m.p20_depth if right_m else 0.0
            if right_p20 < 700:
                # Camera AND LiDAR agree right is blocked — switch left
                left_m = corridors.get("L1") or corridors.get("LEFT")
                left_p20 = left_m.p20_depth if left_m else 0.0
                if left_p20 > 600 and lidar_left_mm > 400:
                    cmd = "GO:LEFT"
                    reason = f"LiDAR+cam: right blocked, switching left"

        # ── 8. Handle emergency ──────────────────────────
        if emergency:
            # Try to steer to an open side before stopping
            l1_m = corridors.get("L1") or corridors.get("LEFT")
            r1_m = corridors.get("R1") or corridors.get("RIGHT")
            l_p20 = l1_m.p20_depth if l1_m else 0.0
            r_p20 = r1_m.p20_depth if r1_m else 0.0
            l_ok = l_p20 > 600 and (lidar_left_mm > 500 or lidar_left_mm >= 9000)
            r_ok = r_p20 > 600 and (lidar_right_mm > 500 or lidar_right_mm >= 9000)

            if l_ok and not r_ok:
                cmd = "GO:LEFT"
                reason = "Emergency: left open"
            elif r_ok and not l_ok:
                cmd = "GO:RIGHT"
                reason = "Emergency: right open"
            elif l_ok and r_ok:
                cmd = "GO:LEFT" if l_p20 >= r_p20 else "GO:RIGHT"
                reason = f"Emergency: both open, picking {'left' if l_p20 >= r_p20 else 'right'}"
            else:
                cmd = "STOP"
                reason = "Emergency: all blocked"

        # ── 9. FREE mode when center is very clear ───────
        if cmd == "GO:CENTER" and not emergency:
            if (center_m is not None
                    and center_m.valid_ratio > 0.1
                    and p20_center >= cfg.FREE_CLEAR_DISTANCE_MM):
                cmd = "FREE"
                reason = f"FREE: center clear ({int(p20_center)}mm)"

        # ── 10. Apply minimum switch interval (except STOP) ──
        now = time.time()
        if cmd != "STOP" and cmd != self._last_stable:
            if now - self._last_change_time < self._MIN_SWITCH_INTERVAL_S:
                cmd = self._last_stable
                reason = "Hold: min interval"

        # Update state
        if cmd != self._last_stable:
            self._last_stable = cmd
            self._last_change_time = now

        confidence = float(best_score)
        critical = (cmd == "STOP" and emergency)

        return self._result(
            raw_command=cmd,
            stable_command=cmd,
            confidence=confidence,
            chosen_corridor=best_name,
            chosen_group=None,
            reason=reason,
            valid_groups=analysis.valid_groups,
            center_blocked_reason="" if best_name == "CENTER" else best_name,
            critical_stop=critical,
            stable_count=1,
            allow_recenter=False,
        )

    def reset(self) -> None:
        """Reset state (e.g., on deauthorization)."""
        self._last_stable = "STOP"
        self._last_change_time = 0.0

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _result(
        raw_command: str,
        stable_command: str,
        confidence: float,
        chosen_corridor: str,
        chosen_group,
        reason: str,
        valid_groups: list,
        center_blocked_reason: str,
        critical_stop: bool = False,
        stable_count: int = 0,
        allow_recenter: bool = False,
    ) -> DecisionResult:
        return DecisionResult(
            raw_command=raw_command,
            stable_command=stable_command,
            confidence=confidence,
            chosen_corridor=chosen_corridor,
            chosen_group=chosen_group,
            reason=reason,
            valid_groups=valid_groups,
            center_blocked_reason=center_blocked_reason,
            critical_stop=critical_stop,
            stable_count=stable_count,
            allow_recenter=allow_recenter,
        )
