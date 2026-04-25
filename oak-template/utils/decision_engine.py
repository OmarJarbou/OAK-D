# utils/decision_engine.py
"""
Smart decision engine for the walker navigation system — v3.0

Takes AnalysisResult (corridors + merged groups) + Arduino state →
produces a single command string using SAFETY-SCORE-BASED target selection.

Key changes from v2.1:
  - Target zone chosen by safety_score, not distance-from-center
  - CENTER gets a soft bias (CENTER_SAFETY_BIAS) but does NOT hard-win
  - Emergency stop only when center danger + no valid side escape
  - Improved hysteresis: no more STOP-dominates-all failsafe
  - center_blocked_reason field for debugging
"""

from collections import Counter, deque
import time
from dataclasses import dataclass
from typing import Optional, List

from utils.config import WalkerConfig
from utils.corridor_analyzer import AnalysisResult, FreeSpaceGroup


@dataclass
class DecisionResult:
    """Complete output of one decision cycle."""
    raw_command: str             # Before smoothing
    stable_command: str          # After temporal smoothing
    confidence: float            # 0.0 – 1.0
    chosen_corridor: str         # Which zone was selected as target
    chosen_group: Optional[FreeSpaceGroup]  # The merged group it belongs to
    reason: str                  # Human-readable explanation
    valid_groups: List[FreeSpaceGroup]
    center_blocked_reason: str   # Why CENTER was not chosen (empty if chosen)
    critical_stop: bool = False
    stable_count: int = 0


class DecisionEngine:
    """
    Produces stable navigation commands from merged free-space groups.

    Decision priority:
      1. Safety gates (auth, sensor, calibration)
      2. Emergency stop (center danger + no side escape)
      3. Best valid merged group → safest zone within it
      4. Temporal smoothing (deque voting with hysteresis)
    """

    def __init__(self, cfg: WalkerConfig):
        self.cfg = cfg
        self._history: deque = deque(maxlen=cfg.HISTORY_SIZE)
        self._last_stable: str = "STOP"
        self._last_change_time: float = 0.0
        self._unsafe_streak: int = 0
        self._free_streak: int = 0
        self._go_streak: int = 0
        self._go_candidate: str = ""

        # Task-4 hysteresis/cooldown (from config/.env)
        self._stop_frames_required: int = cfg.STOP_FRAMES_REQUIRED
        self._free_frames_required: int = cfg.FREE_FRAMES_REQUIRED
        self._go_frames_required: int = cfg.GO_FRAMES_REQUIRED
        self._command_change_cooldown_s: float = cfg.COMMAND_CHANGE_COOLDOWN_S
        self._unsafe_conf_threshold: float = cfg.UNSAFE_CONF_THRESHOLD
        self._free_conf_threshold: float = cfg.FREE_CONF_THRESHOLD
        self._free_center_close_obs_max: float = cfg.FREE_CENTER_CLOSE_OBS_MAX
        self._free_center_min_valid_ratio: float = cfg.FREE_CENTER_MIN_VALID_RATIO
        self._free_center_min_p20_mm: float = cfg.FREE_CENTER_MIN_P20_MM
        self._free_stable_frames: int = cfg.FREE_STABLE_FRAMES
        self._free_clear_distance_mm: float = cfg.FREE_CLEAR_DISTANCE_MM
        self._free_stable_streak: int = 0   # consecutive frames where FREE conditions met

    def decide(
        self,
        analysis: AnalysisResult,
        arduino_state: dict,
    ) -> DecisionResult:
        """Main entry point. Returns a DecisionResult."""
        cfg = self.cfg

        # ── 1. Safety Gates ─────────────────────────────────
        if not arduino_state.get("authorized", False):
            return DecisionResult(
                raw_command="NONE", stable_command="NONE",
                confidence=0.0, chosen_corridor="", chosen_group=None,
                reason="Not authorized - waiting for RFID",
                valid_groups=[], center_blocked_reason="",
                critical_stop=False, stable_count=0,
            )

        if not arduino_state.get("ready", False):
            return DecisionResult(
                raw_command="NONE", stable_command="NONE",
                confidence=0.0, chosen_corridor="", chosen_group=None,
                reason="Arduino not ready (auth sequence in progress)",
                valid_groups=[], center_blocked_reason="",
                critical_stop=False, stable_count=0,
            )

        if not arduino_state.get("sensor_ok", True):
            stable = self._apply_mode_hysteresis("STOP", critical_stop=True)
            return DecisionResult(
                raw_command="STOP", stable_command=stable,
                confidence=1.0, chosen_corridor="", chosen_group=None,
                reason="STOP: sensor error",
                valid_groups=[], center_blocked_reason="",
                critical_stop=False, stable_count=0,
            )

        if not arduino_state.get("calibrated", True):
            stable = self._apply_mode_hysteresis("STOP", critical_stop=True)
            return DecisionResult(
                raw_command="STOP", stable_command=stable,
                confidence=1.0, chosen_corridor="", chosen_group=None,
                reason="STOP: not calibrated",
                valid_groups=[], center_blocked_reason="",
                critical_stop=False, stable_count=0,
            )

        # ── 2. No Depth Data ────────────────────────────────
        if not analysis.corridors:
            raw = "STOP"
            stable = self._apply_mode_hysteresis(raw, critical_stop=True)
            return DecisionResult(
                raw_command=raw, stable_command=stable,
                confidence=0.5, chosen_corridor="", chosen_group=None,
                reason="No depth data",
                valid_groups=[], center_blocked_reason="",
                critical_stop=False, stable_count=0,
            )

        # ── 3. Choose from Valid Merged Groups ──────────────
        valid = analysis.valid_groups
        locked_left = arduino_state.get("locked_left", False)
        locked_right = arduino_state.get("locked_right", False)

        # For each valid group, pick the safest target zone
        eligible = []
        for g in valid:
            best = self._pick_safest_target(g, analysis, locked_left, locked_right)
            if best is not None:
                eligible.append((g, best))

        confidence = self._compute_confidence(analysis)

        # ── 4. Determine raw mode candidate by priority ──────
        if not eligible:
            raw_cmd = "STOP"
            reason = "STOP: no valid group width"
            best_group = None
            best_target = ""
            center_blocked_reason = "No eligible group"
            critical_stop = False
        else:
            # ── 5. Pick Best Group (by avg safety_score, not has_center) ──
            def group_priority(item):
                g, _target = item
                zone_safety_scores = [
                    analysis.corridors[name].safety_score
                    for name in g.zone_names
                    if name in analysis.corridors
                ]
                avg_safety = sum(zone_safety_scores) / max(len(zone_safety_scores), 1)
                return (avg_safety, g.total_width_m)

            eligible.sort(key=group_priority, reverse=True)
            best_group, best_target = eligible[0]

            # ── 6. Determine why CENTER was not chosen ──────────
            center_blocked_reason = self._why_not_center(
                best_target, best_group, analysis, locked_left, locked_right
            )

            # ── 7. Stickiness: prefer current GO if still valid ──
            if self._last_stable.startswith("GO:"):
                current_zone = self._last_stable.replace("GO:", "")
                for g, _t in eligible:
                    if current_zone in g.zone_names:
                        current_m = analysis.corridors.get(current_zone)
                        best_m = analysis.corridors.get(best_target)
                        if current_m and best_m:
                            if current_m.safety_score >= best_m.safety_score * 0.80:
                                best_group = g
                                best_target = current_zone
                                center_blocked_reason = self._why_not_center(
                                    best_target, best_group, analysis,
                                    locked_left, locked_right
                                )
                        break

            go_cmd = cfg.ZONE_TO_CMD.get(best_target, "GO:CENTER")
            free_candidate = self._is_free_candidate(analysis, best_target, confidence)
            critical_stop = bool(analysis.has_emergency)
            unsafe_condition = (
                analysis.has_emergency
                or len(valid) == 0
                or confidence < self._unsafe_conf_threshold
            )

            if unsafe_condition:
                raw_cmd = "STOP"
                if analysis.has_emergency:
                    reason = "STOP: danger close ratio high"
                elif len(valid) == 0:
                    reason = "STOP: no valid group width"
                else:
                    reason = f"STOP: depth confidence unsafe ({confidence:.2f})"
            elif free_candidate:
                raw_cmd = "FREE"
                reason = (
                    f"FREE: clear path "
                    f"(p20={int(analysis.corridors['CENTER'].p20_depth)}mm "
                    f"streak={self._free_stable_streak}/{self._free_stable_frames} "
                    f"conf={confidence:.2f})"
                )
            else:
                raw_cmd = go_cmd
                reason = self._build_go_reason(best_target, analysis, confidence)
        # ── 8. Hysteresis + cooldown ─────────────────────────
        stable_cmd, stable_count = self._apply_mode_hysteresis(raw_cmd, critical_stop=critical_stop)

        return DecisionResult(
            raw_command=raw_cmd, stable_command=stable_cmd,
            confidence=confidence,
            chosen_corridor=best_target,
            chosen_group=best_group,
            reason=reason,
            valid_groups=valid,
            center_blocked_reason=center_blocked_reason,
            critical_stop=critical_stop,
            stable_count=stable_count,
        )

    def _is_free_candidate(
        self, analysis: AnalysisResult, best_target: str, confidence: float
    ) -> bool:
        """
        Return True when the center path is safely clear and no steering
        correction is needed — prefer FREE over GO:CENTER in this case.

        Conditions (all must pass):
        - best target is CENTER (no side-steering needed)
        - confidence is acceptable
        - center zone is clear with no close obstacles
        - p20 depth exceeds FREE_CLEAR_DISTANCE_MM
        - stable for FREE_STABLE_FRAMES consecutive frames
        """
        center = analysis.corridors.get("CENTER")
        if center is None:
            self._free_stable_streak = 0
            return False

        # Must not need side steering
        if best_target != "CENTER":
            self._free_stable_streak = 0
            return False

        # Confidence gate
        if confidence < self._free_conf_threshold:
            self._free_stable_streak = 0
            return False

        # Center must be genuinely clear
        center_ok = (
            center.is_clear
            and center.valid_ratio >= self._free_center_min_valid_ratio
            and center.close_obstacle_ratio <= self._free_center_close_obs_max
            and center.p20_depth >= self._free_clear_distance_mm
        )

        if not center_ok:
            self._free_stable_streak = 0
            return False

        # Increment stability counter
        self._free_stable_streak += 1
        return self._free_stable_streak >= self._free_stable_frames

    def _build_go_reason(
        self, best_target: str, analysis: AnalysisResult, confidence: float
    ) -> str:
        if best_target == "CENTER":
            center = analysis.corridors.get("CENTER")
            if center is not None:
                return (
                    f"GO:CENTER: active recenter needed "
                    f"(p20={int(center.p20_depth)}mm "
                    f"streak={self._free_stable_streak}/{self._free_stable_frames} "
                    f"conf={confidence:.2f})"
                )
            return f"GO:CENTER: active recenter needed (conf={confidence:.2f})"
        side = "right" if best_target.startswith("R") or best_target == "RIGHT" else "left"
        return f"GO:{best_target}: center weak, {side} safer (conf={confidence:.2f})"

    def _apply_mode_hysteresis(self, raw: str, critical_stop: bool = False) -> tuple[str, int]:
        """Apply STOP/FREE/GO frame hysteresis. Returns (stable_command, stable_count)."""
        self._history.append(raw)
        now = time.time()

        if raw == "STOP":
            self._unsafe_streak += 1
            self._free_streak = 0
            self._go_streak = 0
            self._go_candidate = ""
            if critical_stop or self._unsafe_streak >= self._stop_frames_required:
                if self._last_stable != "STOP":
                    self._last_stable = "STOP"
                    self._last_change_time = now
                return self._last_stable, self._unsafe_streak

            return self._last_stable, self._unsafe_streak

        self._unsafe_streak = 0

        if raw == "FREE":
            self._free_streak += 1
            self._go_streak = 0
            self._go_candidate = ""
            if self._free_streak < self._free_frames_required:
                return self._last_stable, self._free_streak
            if (
                self._last_stable != "FREE"
                and now - self._last_change_time < self._command_change_cooldown_s
            ):
                return self._last_stable, self._free_streak
            if self._last_stable != "FREE":
                self._last_stable = "FREE"
                self._last_change_time = now
            return self._last_stable, self._free_streak

        # GO:* candidate
        self._free_streak = 0
        if self._go_candidate == raw:
            self._go_streak += 1
        else:
            self._go_candidate = raw
            self._go_streak = 1
        if self._go_streak < self._go_frames_required:
            return self._last_stable, self._go_streak
        if (
            self._last_stable != raw
            and now - self._last_change_time < self._command_change_cooldown_s
        ):
            return self._last_stable, self._go_streak
        if self._last_stable != raw:
            self._last_stable = raw
            self._last_change_time = now
        return self._last_stable, self._go_streak

    # ── Safety-Score Target Selection ────────────────────────

    def _pick_safest_target(
        self, group: FreeSpaceGroup, analysis: AnalysisResult,
        locked_left: bool, locked_right: bool,
    ) -> Optional[str]:
        """Pick the safest zone in a group using safety_score with soft CENTER bias."""
        cfg = self.cfg
        left_blocked = {"LEFT", "L2", "L1"}
        right_blocked = {"RIGHT", "R2", "R1"}
        center_idx = cfg.NUM_ZONES // 2

        candidates = []
        for name in group.zone_names:
            if locked_left and name in left_blocked:
                continue
            if locked_right and name in right_blocked:
                continue
            m = analysis.corridors.get(name)
            if m is not None:
                candidates.append((name, m))

        if not candidates:
            return None

        # Find the zone with the best safety_score
        best_name = None
        best_effective_score = -1.0
        for name, m in candidates:
            effective = m.safety_score
            if m.zone_index == center_idx:
                effective += cfg.CENTER_SAFETY_BIAS
            if effective > best_effective_score:
                best_effective_score = effective
                best_name = name

        # Check if CENTER is in the group and close enough to the best
        center_name = cfg.ZONE_NAMES[center_idx]
        center_m = None
        for name, m in candidates:
            if name == center_name:
                center_m = m
                break

        if center_m is not None and best_name != center_name:
            center_effective = center_m.safety_score + cfg.CENTER_SAFETY_BIAS
            if center_effective >= best_effective_score * cfg.CENTER_ACCEPT_RATIO:
                best_name = center_name

        return best_name

    # ── CENTER Rejection Reason ──────────────────────────────

    def _why_not_center(
        self, chosen: str, group: FreeSpaceGroup, analysis: AnalysisResult,
        locked_left: bool, locked_right: bool,
    ) -> str:
        cfg = self.cfg
        center_idx = cfg.NUM_ZONES // 2
        center_name = cfg.ZONE_NAMES[center_idx]

        if chosen == center_name:
            return ""

        center_m = analysis.corridors.get(center_name)
        chosen_m = analysis.corridors.get(chosen)

        if center_m is None:
            return "CENTER: no data"

        if center_name not in group.zone_names:
            return f"CENTER not in best group [{','.join(group.zone_names)}]"

        if not center_m.is_clear:
            parts = []
            if center_m.valid_ratio < cfg.MIN_VALID_RATIO:
                parts.append(f"low valid {center_m.valid_ratio:.0%}")
            if center_m.p20_depth <= cfg.EMERGENCY_STOP_MM:
                parts.append(f"p20={int(center_m.p20_depth)}mm too close")
            if center_m.close_obstacle_ratio >= 0.60:
                parts.append(f"close_obs={center_m.close_obstacle_ratio:.0%}")
            return f"CENTER blocked: {', '.join(parts) if parts else 'unsafe'}"

        if chosen_m:
            return (
                f"CENTER safety={center_m.safety_score:.2f} < "
                f"{chosen} safety={chosen_m.safety_score:.2f} "
                f"(need {cfg.CENTER_ACCEPT_RATIO:.0%})"
            )

        return f"CENTER weaker than {chosen}"

    # ── Confidence ──────────────────────────────────────────

    def _compute_confidence(self, analysis: AnalysisResult) -> float:
        if not analysis.corridors:
            return 0.0

        avg_valid = sum(c.valid_ratio for c in analysis.corridors.values()) / len(analysis.corridors)
        avg_emergency = sum(c.emergency_ratio for c in analysis.corridors.values()) / len(analysis.corridors)
        valid_group_count = len(analysis.valid_groups)
        group_score = min(valid_group_count / 3.0, 1.0)

        if self._history:
            counts = Counter(self._history)
            unanimity = counts.most_common(1)[0][1] / len(self._history)
        else:
            unanimity = 0.5

        confidence = (
            0.25 * avg_valid
            + 0.25 * group_score
            + 0.25 * (1.0 - avg_emergency)
            + 0.25 * unanimity
        )
        return max(0.0, min(1.0, confidence))

    def reset(self) -> None:
        self._history.clear()
        self._last_stable = "STOP"
        self._last_change_time = 0.0
        self._unsafe_streak = 0
        self._free_streak = 0
        self._go_streak = 0
        self._go_candidate = ""
        self._free_stable_streak = 0
