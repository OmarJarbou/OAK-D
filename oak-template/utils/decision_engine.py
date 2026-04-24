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
                reason="Not authorized — waiting for RFID",
                valid_groups=[], center_blocked_reason="",
            )

        if not arduino_state.get("ready", False):
            return DecisionResult(
                raw_command="NONE", stable_command="NONE",
                confidence=0.0, chosen_corridor="", chosen_group=None,
                reason="Arduino not ready (auth sequence in progress)",
                valid_groups=[], center_blocked_reason="",
            )

        if not arduino_state.get("sensor_ok", True):
            self._push_and_stabilize("STOP")
            return DecisionResult(
                raw_command="STOP", stable_command="STOP",
                confidence=1.0, chosen_corridor="", chosen_group=None,
                reason="Sensor error — STOP for safety",
                valid_groups=[], center_blocked_reason="",
            )

        if not arduino_state.get("calibrated", True):
            self._push_and_stabilize("STOP")
            return DecisionResult(
                raw_command="STOP", stable_command="STOP",
                confidence=1.0, chosen_corridor="", chosen_group=None,
                reason="Not calibrated — cannot steer",
                valid_groups=[], center_blocked_reason="",
            )

        # ── 2. Emergency Stop ───────────────────────────────
        if analysis.has_emergency:
            raw = "STOP"
            stable = self._push_and_stabilize(raw)
            return DecisionResult(
                raw_command=raw, stable_command=stable,
                confidence=0.95, chosen_corridor="", chosen_group=None,
                reason="Emergency: center danger + no side escape",
                valid_groups=analysis.valid_groups,
                center_blocked_reason="Emergency: too close in center",
            )

        # ── 3. No Depth Data ────────────────────────────────
        if not analysis.corridors:
            self._push_and_stabilize("STOP")
            return DecisionResult(
                raw_command="STOP", stable_command="STOP",
                confidence=0.5, chosen_corridor="", chosen_group=None,
                reason="No depth data",
                valid_groups=[], center_blocked_reason="",
            )

        # ── 4. Choose from Valid Merged Groups ──────────────
        valid = analysis.valid_groups
        locked_left = arduino_state.get("locked_left", False)
        locked_right = arduino_state.get("locked_right", False)

        # For each valid group, pick the safest target zone
        eligible = []
        for g in valid:
            best = self._pick_safest_target(g, analysis, locked_left, locked_right)
            if best is not None:
                eligible.append((g, best))

        if not eligible:
            raw = "STOP"
            stable = self._push_and_stabilize(raw)
            confidence = self._compute_confidence(analysis)
            return DecisionResult(
                raw_command=raw, stable_command=stable,
                confidence=confidence, chosen_corridor="", chosen_group=None,
                reason=f"No valid path (groups={len(analysis.groups)}, valid={len(valid)}, all locked/blocked)",
                valid_groups=valid,
                center_blocked_reason="No eligible group",
            )

        # ── 5. Pick Best Group (by avg safety_score, not has_center) ──
        def group_priority(item):
            g, target = item
            # Compute average safety_score for the group
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

        # ── 7. Stickiness: prefer current command if still valid ──
        if self._last_stable.startswith("GO:"):
            current_zone = self._last_stable.replace("GO:", "")
            for g, t in eligible:
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

        # ── 8. Map to Command ───────────────────────────────
        raw_cmd = cfg.ZONE_TO_CMD.get(best_target, "GO:CENTER")
        stable_cmd = self._push_and_stabilize(raw_cmd)
        confidence = self._compute_confidence(analysis)

        return DecisionResult(
            raw_command=raw_cmd, stable_command=stable_cmd,
            confidence=confidence,
            chosen_corridor=best_target,
            chosen_group=best_group,
            reason=f"Group [{','.join(best_group.zone_names)}] w={best_group.total_width_m:.2f}m → {best_target}",
            valid_groups=valid,
            center_blocked_reason=center_blocked_reason,
        )

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

    # ── Temporal Smoothing ──────────────────────────────────

    def _push_and_stabilize(self, raw: str) -> str:
        """Add raw command to history and return stabilized command."""
        self._history.append(raw)
        counts = Counter(self._history)
        best_cmd, best_count = counts.most_common(1)[0]
        cfg = self.cfg

        # STOP hysteresis: once stopped, require strong evidence to move
        if self._last_stable == "STOP" and best_cmd != "STOP":
            if best_count >= cfg.STOP_CLEAR_COUNT:
                self._last_stable = best_cmd
        # Turn stickiness: don't flip direction easily
        elif (
            self._last_stable.startswith("GO:")
            and self._last_stable != "GO:CENTER"
            and best_cmd != self._last_stable
        ):
            if best_cmd == "STOP":
                if best_count >= cfg.MIN_STABLE_COUNT:
                    self._last_stable = best_cmd
            else:
                if best_count >= cfg.TURN_STICK_COUNT:
                    self._last_stable = best_cmd
        else:
            if best_count >= cfg.MIN_STABLE_COUNT:
                self._last_stable = best_cmd

        return self._last_stable

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
