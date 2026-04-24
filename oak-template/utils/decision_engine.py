# utils/decision_engine.py
"""
Smart decision engine for the walker navigation system.

Takes AnalysisResult (corridors + merged groups) + Arduino state →
produces a single command string using merged free-space groups.
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


class DecisionEngine:
    """
    Produces stable navigation commands from merged free-space groups.

    Decision priority:
      1. Safety gates (auth, sensor, calibration)
      2. Emergency stop (obstacles dangerously close in center)
      3. Best valid merged group → target zone within it
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
                valid_groups=[],
            )

        if not arduino_state.get("ready", False):
            return DecisionResult(
                raw_command="NONE", stable_command="NONE",
                confidence=0.0, chosen_corridor="", chosen_group=None,
                reason="Arduino not ready (auth sequence in progress)",
                valid_groups=[],
            )

        if not arduino_state.get("sensor_ok", True):
            self._push_and_stabilize("STOP")
            return DecisionResult(
                raw_command="STOP", stable_command="STOP",
                confidence=1.0, chosen_corridor="", chosen_group=None,
                reason="Sensor error — STOP for safety",
                valid_groups=[],
            )

        if not arduino_state.get("calibrated", True):
            self._push_and_stabilize("STOP")
            return DecisionResult(
                raw_command="STOP", stable_command="STOP",
                confidence=1.0, chosen_corridor="", chosen_group=None,
                reason="Not calibrated — cannot steer",
                valid_groups=[],
            )

        # ── 2. Emergency Stop ───────────────────────────────
        if analysis.has_emergency:
            raw = "STOP"
            stable = self._push_and_stabilize(raw)
            return DecisionResult(
                raw_command=raw, stable_command=stable,
                confidence=0.95, chosen_corridor="", chosen_group=None,
                reason="Emergency: obstacle very close in center",
                valid_groups=analysis.valid_groups,
            )

        # ── 3. No Depth Data ────────────────────────────────
        if not analysis.corridors:
            self._push_and_stabilize("STOP")
            return DecisionResult(
                raw_command="STOP", stable_command="STOP",
                confidence=0.5, chosen_corridor="", chosen_group=None,
                reason="No depth data",
                valid_groups=[],
            )

        # ── 4. Choose from Valid Merged Groups ──────────────
        valid = analysis.valid_groups
        locked_left = arduino_state.get("locked_left", False)
        locked_right = arduino_state.get("locked_right", False)

        # Filter out groups whose best zone is blocked by Arduino lock
        eligible = []
        for g in valid:
            # Recalculate best zone considering locks
            best = self._pick_target_in_group(g, locked_left, locked_right)
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
            )

        # ── 5. Pick Best Group ──────────────────────────────
        # Prefer group containing CENTER, then highest avg_score, then widest
        center_idx = cfg.NUM_ZONES // 2

        def group_priority(item):
            g, target = item
            has_center = center_idx in g.zone_indices
            return (has_center, g.avg_score, g.total_width_m)

        eligible.sort(key=group_priority, reverse=True)
        best_group, best_target = eligible[0]

        # ── 6. Stickiness: prefer current command if still valid ──
        if self._last_stable.startswith("GO:"):
            current_zone = self._last_stable.replace("GO:", "")
            # Check if current zone is in any valid group
            for g, t in eligible:
                if current_zone in g.zone_names:
                    # Current zone is still valid → stick with it if score is decent
                    current_m = analysis.corridors.get(current_zone)
                    best_m = analysis.corridors.get(best_target)
                    if current_m and best_m:
                        if current_m.score >= best_m.score * 0.75:
                            best_group = g
                            best_target = current_zone
                    break

        # ── 7. Map to Command ───────────────────────────────
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
        )

    # ── Target Selection Within Group ────────────────────────

    def _pick_target_in_group(
        self, group: FreeSpaceGroup,
        locked_left: bool, locked_right: bool,
    ) -> Optional[str]:
        """Pick the best target zone in a group, respecting Arduino locks."""
        left_blocked = {"LEFT", "L2", "L1"}
        right_blocked = {"RIGHT", "R2", "R1"}

        center_idx = self.cfg.NUM_ZONES // 2
        candidates = []
        for name, idx in zip(group.zone_names, group.zone_indices):
            if locked_left and name in left_blocked:
                continue
            if locked_right and name in right_blocked:
                continue
            candidates.append((name, idx))

        if not candidates:
            return None

        # Prefer closest to center
        candidates.sort(key=lambda x: abs(x[1] - center_idx))
        return candidates[0][0]

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

        # Failsafe: STOP dominates
        if "STOP" in counts and self._last_stable != "STOP":
            non_stop = sum(v for k, v in counts.items() if k != "STOP")
            if counts["STOP"] >= non_stop:
                self._last_stable = "STOP"

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
