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
    allow_recenter: bool = False  # bypass FREE→GO:CENTER sticky when recentring


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
        self._free_exit_bad_streak: int = 0
        self._free_exit_bad_frames: int = int(cfg.FREE_EXIT_BAD_FRAMES)
        self._unsafe_conf_streak_frames: int = int(cfg.UNSAFE_CONF_STREAK_FRAMES)
        self._low_conf_streak: int = 0
        self._effective_go_frames: int = self._go_frames_required
        self._last_non_center_go: str = ""

        # Latched True once Arduino reports CENTER reached/at-target.
        # While True and center path remains clear, we suppress GO:CENTER and emit FREE only.
        # Cleared on STOP, side-steer need, or obstacle/center degradation.
        self._confirmed_centered: bool = False

        # LiDAR side-escape direction latch.
        # Once a side is chosen ("left" / "right" / ""), stick with it until the
        # front arc is no longer in lidar_veto_side_escape mode.
        # This prevents the walker from flip-flopping between left and right
        # mid-escape when sensor noise shifts the apparent clearance.
        self._escape_latch_side: str = ""
        self._escape_latch_clear_streak: int = 0  # consecutive non-escape frames
        self._escape_latch_clear_required: int = 3  # frames needed to clear latch

        # Lightweight temporal smoothing (EMA) for stability without vision changes.
        self._ema_alpha: float = float(cfg.TEMP_EMA_ALPHA)
        self._ema_warmup_frames: int = int(cfg.TEMP_EMA_WARMUP_FRAMES)
        self._ema_frames_seen: int = 0
        self._ema_p20: dict[str, float] = {}
        self._ema_close: dict[str, float] = {}

    def _update_ema(self, analysis: AnalysisResult) -> None:
        """Update EMA dicts for per-zone p20_depth and close_obstacle_ratio."""
        if not analysis.corridors:
            return
        a = self._ema_alpha
        for name, m in analysis.corridors.items():
            p20 = float(m.p20_depth or 0.0)
            close = float(m.close_obstacle_ratio or 0.0)
            if name not in self._ema_p20:
                self._ema_p20[name] = p20
            else:
                self._ema_p20[name] = a * p20 + (1.0 - a) * self._ema_p20[name]
            if name not in self._ema_close:
                self._ema_close[name] = close
            else:
                self._ema_close[name] = a * close + (1.0 - a) * self._ema_close[name]
        self._ema_frames_seen += 1

    def _ema_ready(self) -> bool:
        return self._ema_frames_seen >= self._ema_warmup_frames

    def _center_path_clear(self, analysis: AnalysisResult) -> bool:
        """Return True when CENTER remains clearly passable (used for confirmed-center latch)."""
        center = analysis.corridors.get("CENTER")
        if center is None:
            return False
        p20 = self._ema_p20.get("CENTER", center.p20_depth) if self._ema_ready() else center.p20_depth
        close = (
            self._ema_close.get("CENTER", center.close_obstacle_ratio)
            if self._ema_ready()
            else center.close_obstacle_ratio
        )
        return (
            center.is_clear
            and center.valid_ratio >= self._free_center_min_valid_ratio
            and close <= self._free_center_close_obs_max
            and p20 >= self._free_clear_distance_mm
        )

    def _recenter_side_ref(self, prev_stable: str) -> str:
        """Lateral GO:* in effect for recenter logic (persists across FREE)."""
        if prev_stable.startswith("GO:") and prev_stable != "GO:CENTER":
            return prev_stable
        if prev_stable == "FREE" and self._last_non_center_go:
            return self._last_non_center_go
        return ""

    def _should_recenter(self, analysis: AnalysisResult, side_full_cmd: str) -> bool:
        """True when center is open again after a side avoidance command."""
        if (
            not side_full_cmd
            or not side_full_cmd.startswith("GO:")
            or side_full_cmd == "GO:CENTER"
        ):
            return False
        side_zone = side_full_cmd.replace("GO:", "")
        center = analysis.corridors.get("CENTER")
        side_m = analysis.corridors.get(side_zone)
        if center is None or not center.is_clear:
            return False
        if center.p20_depth < self.cfg.RECENTER_MIN_P20_MM:
            return False
        if side_m is None:
            return True
        if center.safety_score + self.cfg.RECENTER_SAFETY_GAP < side_m.safety_score:
            return False
        return True

    def decide(
        self,
        analysis: AnalysisResult,
        arduino_state: dict,
        fusion_boost: float = 0.0,
        lidar_left_mm: float = 0.0,
        lidar_right_mm: float = 0.0,
        side_escape_left: bool = False,
        side_escape_right: bool = False,
        fusion_reason: str = "",
        lidar_front_mm: float = 0.0,
    ) -> DecisionResult:
        """Main entry point. Returns a DecisionResult."""
        cfg = self.cfg
        prev_stable = self._last_stable

        # Fix 2: Ignore center_confirmed from Arduino entirely.
        # Previously this latched _confirmed_centered=True and suppressed GO:CENTER,
        # causing the system to miss steering corrections after the user moved.
        if arduino_state.get("center_confirmed", False):
            pass  # intentionally ignored — do not latch confirmed_centered

        # ── 1. Safety Gates ─────────────────────────────────
        if not arduino_state.get("authorized", False):
            return DecisionResult(
                raw_command="NONE", stable_command="NONE",
                confidence=0.0, chosen_corridor="", chosen_group=None,
                reason="Not authorized - waiting for RFID",
                valid_groups=[], center_blocked_reason="",
                critical_stop=False, stable_count=0, allow_recenter=False,
            )

        if not arduino_state.get("ready", False):
            return DecisionResult(
                raw_command="NONE", stable_command="NONE",
                confidence=0.0, chosen_corridor="", chosen_group=None,
                reason="Arduino not ready (auth sequence in progress)",
                valid_groups=[], center_blocked_reason="",
                critical_stop=False, stable_count=0, allow_recenter=False,
            )

        if not arduino_state.get("sensor_ok", True):
            stable = self._apply_mode_hysteresis("STOP", critical_stop=True)
            return DecisionResult(
                raw_command="STOP", stable_command=stable,
                confidence=1.0, chosen_corridor="", chosen_group=None,
                reason="STOP: sensor error",
                valid_groups=[], center_blocked_reason="",
                critical_stop=False, stable_count=0, allow_recenter=False,
            )

        if not arduino_state.get("calibrated", True):
            stable = self._apply_mode_hysteresis("STOP", critical_stop=True)
            return DecisionResult(
                raw_command="STOP", stable_command=stable,
                confidence=1.0, chosen_corridor="", chosen_group=None,
                reason="STOP: not calibrated",
                valid_groups=[], center_blocked_reason="",
                critical_stop=False, stable_count=0, allow_recenter=False,
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
                critical_stop=False, stable_count=0, allow_recenter=False,
            )

        # Update lightweight temporal smoothing (EMA). Used only by
        # _center_path_clear() and _is_free_candidate().
        self._update_ema(analysis)

        confidence = self._compute_confidence(analysis, fusion_boost)
        if confidence < self._unsafe_conf_threshold:
            self._low_conf_streak += 1
        else:
            self._low_conf_streak = 0
        unsafe_low_conf = self._low_conf_streak >= self._unsafe_conf_streak_frames

        recenter_side = self._recenter_side_ref(prev_stable)
        recenter_hold = (
            bool(recenter_side)
            and self._should_recenter(analysis, recenter_side)
        )

        # ── 3. Choose from Valid Merged Groups ──────────────
        valid = analysis.valid_groups
        locked_left = arduino_state.get("locked_left", False)
        locked_right = arduino_state.get("locked_right", False)

        # -- LiDAR Side-Escape Fast Path --------------------------------------------------
        # When fusion reports a LiDAR front obstacle with a side escape
        # available, bypass normal group scoring and steer directly toward
        # the open side. This avoids the STOP->FREE->STOP infinite loop.
        # ── Proactive escape latch release ───────────────────────────────────────
        # Release when LiDAR front is genuinely clear again.
        # Uses lidar_front_mm (passed from fused.front_clear_mm) — not
        # lidar_left/right_mm which are side arcs (30-90°), not front.
        if cfg.LIDAR_STEERING_ENABLED and self._escape_latch_side and lidar_front_mm > cfg.LIDAR_SAFETY_MM:
            print(
                f"[Decision] Escape latch RELEASED (front clear): "
                f"lidar_front={lidar_front_mm:.0f}mm > {cfg.LIDAR_SAFETY_MM:.0f}mm"
            )
            self._escape_latch_side = ""

        if cfg.LIDAR_STEERING_ENABLED and fusion_reason == "lidar_veto_side_escape":
            # Use latched direction if already committed; pick fresh only if none.
            if self._escape_latch_side == "left" and locked_left:
                self._escape_latch_side = ""   # locked out — reset latch
            if self._escape_latch_side == "right" and locked_right:
                self._escape_latch_side = ""   # locked out — reset latch

            if self._escape_latch_side:
                # Already committed — keep same direction regardless of noise.
                force_side = self._escape_latch_side
            else:
                # First frame in this escape: pick the side with greater clearance.
                force_side = None
                if side_escape_left and side_escape_right:
                    force_side = "left" if lidar_left_mm >= lidar_right_mm else "right"
                elif side_escape_left and not locked_left:
                    force_side = "left"
                elif side_escape_right and not locked_right:
                    force_side = "right"
                if force_side:
                    self._escape_latch_side = force_side  # lock in direction
                    self._escape_latch_clear_streak = 0  # reset clear counter

            if force_side is not None:
                # Determine zone key (L1/L2/R1/R2) then translate via
                # ZONE_TO_CMD so FLIP_LR is automatically applied.
                # < 900mm = tighter space → sharper turn (L2/R2)
                if force_side == "left":
                    zone_key = "L2" if lidar_left_mm < 900 else "L1"
                else:
                    zone_key = "R2" if lidar_right_mm < 900 else "R1"
                zone_cmd = self.cfg.ZONE_TO_CMD.get(zone_key, f"GO:{zone_key}")
                zone_name = zone_key
                print(
                    f"[Decision] LiDAR side-escape -> {zone_cmd} "
                    f"(L={lidar_left_mm:.0f}mm R={lidar_right_mm:.0f}mm latch={force_side})"
                )
                stable_cmd, stable_count = self._apply_mode_hysteresis(
                    zone_cmd, critical_stop=False
                )
                return DecisionResult(
                    raw_command=zone_cmd, stable_command=stable_cmd,
                    confidence=confidence,
                    chosen_corridor=zone_name,
                    chosen_group=None,
                    reason=f"{zone_cmd}: LiDAR side-escape (front blocked, {force_side} open)",
                    valid_groups=valid,
                    center_blocked_reason="LiDAR front obstacle - side escape active",
                    critical_stop=False,
                    stable_count=stable_count,
                    allow_recenter=False,
                )
        elif cfg.LIDAR_STEERING_ENABLED:
            # Not in side-escape mode: require N consecutive non-escape frames
            # before clearing latch (prevents single-frame noise from resetting).
            self._escape_latch_clear_streak += 1
            if self._escape_latch_clear_streak >= self._escape_latch_clear_required:
                self._escape_latch_side = ""
                self._escape_latch_clear_streak = 0

        # Determine LiDAR directional preference for low-confidence frames.
        lidar_pref = (
            self._lidar_side_preference(confidence, lidar_left_mm, lidar_right_mm)
            if cfg.LIDAR_STEERING_ENABLED
            else None
        )

        # For each valid group, pick the safest target zone
        eligible = []
        for g in valid:
            best = self._pick_safest_target(
                g, analysis, locked_left, locked_right, lidar_pref,
                lidar_left_mm=lidar_left_mm,
                lidar_right_mm=lidar_right_mm,
            )
            if best is not None:
                eligible.append((g, best))

        # ── 4. Determine raw mode candidate by priority ──────
        narrow_zone = self._try_camera_narrow_escape(
            analysis, locked_left, locked_right
        )
        if not eligible:
            if narrow_zone is not None:
                go_cmd = cfg.ZONE_TO_CMD.get(narrow_zone, "GO:CENTER")
                raw_cmd = go_cmd
                best_target = narrow_zone
                best_group = None
                reason = (
                    f"{go_cmd}: camera narrow escape "
                    f"(group too narrow, {narrow_zone} open)"
                )
                center_blocked_reason = self._why_not_center(
                    narrow_zone,
                    FreeSpaceGroup(
                        zone_names=[narrow_zone],
                        zone_indices=[
                            analysis.corridors[narrow_zone].zone_index
                        ],
                        total_width_m=0.0,
                        is_valid=False,
                        avg_p20_depth=analysis.corridors[narrow_zone].p20_depth,
                        avg_score=analysis.corridors[narrow_zone].score,
                        best_zone=narrow_zone,
                    ),
                    analysis,
                    locked_left,
                    locked_right,
                )
                critical_stop = False
                self._confirmed_centered = False
            else:
                raw_cmd = "STOP"
                reason = "STOP: no valid group width"
                best_group = None
                best_target = ""
                center_blocked_reason = "No eligible group"
                critical_stop = False
                self._confirmed_centered = False
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
            # Skip while recentring: side command was for avoidance; once center is
            # safe again, release stickiness so GO:CENTER can win.
            if self._last_stable.startswith("GO:") and not recenter_hold:
                current_zone = self._last_stable.replace("GO:", "")
                for g, _t in eligible:
                    if current_zone in g.zone_names:
                        current_m = analysis.corridors.get(current_zone)
                        best_m = analysis.corridors.get(best_target)
                        if current_m and best_m:
                            if current_m.safety_score >= best_m.safety_score * 0.70:  # Fix 3: was 0.80
                                best_group = g
                                best_target = current_zone
                                center_blocked_reason = self._why_not_center(
                                    best_target, best_group, analysis,
                                    locked_left, locked_right
                                )
                        break

            go_cmd = cfg.ZONE_TO_CMD.get(best_target, "GO:CENTER")
            critical_stop = bool(analysis.has_emergency)
            
            # Allow turning left/right even if front is blocked (has_emergency),
            # so the walker can pivot and escape the obstacle.
            if narrow_zone is not None and best_target in ("", "CENTER"):
                best_target = narrow_zone
                go_cmd = cfg.ZONE_TO_CMD.get(best_target, "GO:CENTER")
                center_blocked_reason = self._why_not_center(
                    best_target, best_group, analysis, locked_left, locked_right
                )

            can_side_escape = (
                analysis.has_emergency
                and not unsafe_low_conf
                and best_target not in ("", "CENTER")
                and (len(valid) > 0 or narrow_zone is not None)
            )

            unsafe_condition = (
                (analysis.has_emergency and not can_side_escape)
                or (len(valid) == 0 and narrow_zone is None)
                or unsafe_low_conf
            )

            # Clear confirmed-centered latch if steering away from center is needed
            # or if center is no longer clearly passable.
            if best_target != "CENTER" or not self._center_path_clear(analysis):
                self._confirmed_centered = False

            if unsafe_condition:
                raw_cmd = "STOP"
                # STOP clears confirmed centering.
                self._confirmed_centered = False
                if analysis.has_emergency and not can_side_escape:
                    reason = "STOP: danger close ratio high"
                elif len(valid) == 0:
                    reason = "STOP: no valid group width"
                else:
                    reason = f"STOP: depth confidence unsafe ({confidence:.2f})"
            else:
                # If Arduino already confirmed we're centered and CENTER remains clear,
                # force FREE only (avoid GO:CENTER↔FREE oscillation).
                if (
                    self._confirmed_centered
                    and best_target == "CENTER"
                    and self._center_path_clear(analysis)
                ):
                    raw_cmd = "FREE"
                    reason = "FREE: confirmed centered (suppress GO:CENTER)"
                else:
                    free_candidate = self._is_free_candidate(analysis, best_target, confidence)
                    if free_candidate:
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

                        # If we are trying to re-center (GO:CENTER), that implies we are
                        # not yet confirmed centered.
                        if best_target == "CENTER" and raw_cmd == "GO:CENTER":
                            self._confirmed_centered = False

            if raw_cmd == "FREE":
                # FREE does not clear confirmed-centered latch by itself.
                pass

        # ── 7.5 FREE exit hysteresis (Option C) ─────────────────────
        # When currently stable FREE, require N consecutive "not free" frames
        # before leaving FREE (unless critical STOP or a clear side-steer win).
        free_exit_bad_frames = self._free_exit_bad_frames

        def _is_side_steer(cmd: str) -> bool:
            return cmd in {
                "GO:LEFT", "GO:L2", "GO:L1",
                "GO:R1", "GO:R2", "GO:RIGHT",
            }

        bypass_bad_streak = False
        if raw_cmd == "STOP" and critical_stop:
            bypass_bad_streak = True
        elif _is_side_steer(raw_cmd):
            # Bypass only if side safety beats CENTER by SIDE_PREFER_MARGIN.
            side_name = raw_cmd.replace("GO:", "")
            side_m = analysis.corridors.get(side_name)
            center_m = analysis.corridors.get("CENTER")
            if side_m is not None and center_m is not None:
                if (side_m.safety_score - center_m.safety_score) > cfg.SIDE_PREFER_MARGIN:
                    bypass_bad_streak = True
        elif raw_cmd == "GO:CENTER" and recenter_hold:
            bypass_bad_streak = True

        if raw_cmd == "FREE":
            self._free_exit_bad_streak = 0
        elif self._last_stable == "FREE" and not bypass_bad_streak:
            self._free_exit_bad_streak += 1
            if self._free_exit_bad_streak < free_exit_bad_frames:
                raw_cmd = "FREE"
                reason = f"FREE: exit-hold bad_streak={self._free_exit_bad_streak}/{free_exit_bad_frames}"
            else:
                self._free_exit_bad_streak = 0
        # ── 8. Hysteresis + cooldown ─────────────────────────
        self._effective_go_frames = self._go_frames_required
        if raw_cmd.startswith("GO:") and raw_cmd != "GO:CENTER":
            sn = raw_cmd.replace("GO:", "")
            sm = analysis.corridors.get(sn)
            cm = analysis.corridors.get("CENTER")
            if (
                sm is not None
                and cm is not None
                and (sm.safety_score - cm.safety_score) >= cfg.GO_FAST_MARGIN
            ):
                self._effective_go_frames = min(
                    self._go_frames_required, cfg.GO_FAST_FRAMES_REQUIRED
                )
        elif raw_cmd == "GO:CENTER" and recenter_hold:
            self._effective_go_frames = min(
                self._go_frames_required, cfg.GO_FAST_FRAMES_REQUIRED
            )

        stable_cmd, stable_count = self._apply_mode_hysteresis(raw_cmd, critical_stop=critical_stop)

        # If we exit STOP into a side-steer, treat that as losing confirmed centered state.
        # This makes a later return-to-center eligible to fire GO:CENTER again.
        if (
            prev_stable == "STOP"
            and stable_cmd.startswith("GO:")
            and stable_cmd != "GO:CENTER"
        ):
            self._confirmed_centered = False

        # Any STOP clears confirmed centered state.
        if stable_cmd == "STOP":
            self._confirmed_centered = False

        if stable_cmd.startswith("GO:") and stable_cmd != "GO:CENTER":
            self._last_non_center_go = stable_cmd
        elif stable_cmd in ("GO:CENTER", "STOP", "NONE"):
            self._last_non_center_go = ""

        eligible_nonempty = bool(eligible)
        allow_recenter = (
            eligible_nonempty
            and best_target == "CENTER"
            and recenter_hold
            and stable_cmd == "GO:CENTER"
        )

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
            allow_recenter=allow_recenter,
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
        # Side steering needed → FREE must not engage.
        if best_target != "CENTER":
            self._free_stable_streak = 0
            return False

        center = analysis.corridors.get("CENTER")

        # Structural failures — hard reset streak
        if center is None:
            self._free_stable_streak = 0
            return False

        if confidence < self._free_conf_threshold:
            self._free_stable_streak = 0
            return False

        if not center.is_clear:
            self._free_stable_streak = 0
            return False

        if center.valid_ratio < self._free_center_min_valid_ratio:
            self._free_stable_streak = 0
            return False

        if center.close_obstacle_ratio > self._free_center_close_obs_max:
            self._free_stable_streak = 0
            return False

        p20 = self._ema_p20.get("CENTER", center.p20_depth) if self._ema_ready() else center.p20_depth
        if p20 < self._free_clear_distance_mm:
            # Soft failure — borderline depth, don't reset streak fully,
            # just don't increment. This prevents one bad frame killing progress.
            return False

        # All conditions passed — increment streak
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
            # FREE streak is already guaranteed by _is_free_candidate's
            # FREE_STABLE_FRAMES check — don't double-gate here.
            # Only enforce cooldown.
            # Fix 7: FREE cooldown is halved — FREE is safer than GO so
            # it should engage faster once conditions are stable.
            free_cooldown = self._command_change_cooldown_s * 0.5
            if (
                self._last_stable != "FREE"
                and now - self._last_change_time < free_cooldown
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
        if self._go_streak < self._effective_go_frames:
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

    def _try_camera_narrow_escape(
        self,
        analysis: AnalysisResult,
        locked_left: bool,
        locked_right: bool,
    ) -> Optional[str]:
        """Pick a side zone when no merged group meets walker width.

        Uses camera safety_score only (no LiDAR). Requires the side to be
        is_clear, p20 >= CAMERA_NARROW_ESCAPE_MIN_P20_MM, and to beat CENTER
        by SIDE_PREFER_MARGIN (or center is not clear).
        """
        cfg = self.cfg
        center = analysis.corridors.get("CENTER")
        if center is None:
            return None

        left_zones = frozenset({"LEFT", "L2", "L1"})
        right_zones = frozenset({"R1", "R2", "RIGHT"})

        best_name: Optional[str] = None
        best_safety = -1.0
        for name in ("LEFT", "L2", "L1", "R1", "R2", "RIGHT"):
            if locked_left and name in left_zones:
                continue
            if locked_right and name in right_zones:
                continue
            m = analysis.corridors.get(name)
            if m is None or not m.is_clear:
                continue
            if m.p20_depth < cfg.CAMERA_NARROW_ESCAPE_MIN_P20_MM:
                continue
            if m.safety_score > best_safety:
                best_safety = m.safety_score
                best_name = name

        if best_name is None:
            return None

        margin = best_safety - center.safety_score
        if center.is_clear and margin < cfg.SIDE_PREFER_MARGIN:
            return None
        if not center.is_clear and margin < cfg.SIDE_PREFER_MARGIN * 0.5:
            return None

        print(
            f"[Decision] Camera narrow escape -> {best_name} "
            f"(safety={best_safety:.2f} vs CENTER={center.safety_score:.2f})"
        )
        return best_name

    def _lidar_side_preference(
        self,
        confidence: float,
        lidar_left_mm: float,
        lidar_right_mm: float,
    ) -> Optional[str]:
        """Return 'left', 'right', or None — which physical side LiDAR favours.

        Only applied when OAK-D confidence is below threshold AND the side
        distance gap is large enough to be actionable (not just noise).
        """
        if not self.cfg.LIDAR_STEERING_ENABLED:
            return None
        if confidence >= self.cfg.LIDAR_SIDE_BIAS_CONF_THRESHOLD:
            return None  # OAK-D trusted — no LiDAR nudge needed
        if lidar_left_mm <= 0.0 and lidar_right_mm <= 0.0:
            return None  # No LiDAR data available
        gap = lidar_left_mm - lidar_right_mm
        if abs(gap) < self.cfg.LIDAR_SIDE_BIAS_MM_MIN_GAP:
            return None  # Both sides too similar — avoid noise-driven flip
        pref = "left" if gap > 0 else "right"
        print(
            f"[LiDAR-Bias] conf={confidence:.2f} left={lidar_left_mm:.0f}mm "
            f"right={lidar_right_mm:.0f}mm gap={gap:+.0f}mm -> prefer {pref}"
        )
        return pref

    def _pick_safest_target(
        self, group: FreeSpaceGroup, analysis: AnalysisResult,
        locked_left: bool, locked_right: bool,
        lidar_pref: Optional[str] = None,
        lidar_left_mm: float = 0.0,
        lidar_right_mm: float = 0.0,
    ) -> Optional[str]:
        """Pick the safest zone in a group using safety_score with soft CENTER bias.

        LiDAR distance multipliers (Fix 3) are applied unconditionally:
          - Side with > LIDAR_BOOST_MM clearance  → +15% on that side's zones
          - Side with 0 < dist < LIDAR_SAFETY_MM  → -15% on that side's zones
          (These act on top of the existing LiDAR preference bonus.)
        """
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

        # ── LiDAR distance-based score multiplier (Fix 3) ────────
        # Applied regardless of camera confidence.
        # Thresholds: clear side > 900mm → boost; blocked side < safety_mm → penalty.
        _LIDAR_BOOST_MM: float = 900.0
        _LIDAR_BOOST: float = 1.15   # Clear side: boost score significantly
        _LIDAR_PENALTY: float = 0.65 # Blocked side: heavy penalty so chairs/obstacles lose

        _left_zones = {"LEFT", "L2", "L1"}
        _right_zones = {"R1", "R2", "RIGHT"}

        def _lidar_distance_mult(name: str) -> float:
            if not cfg.LIDAR_STEERING_ENABLED:
                return 1.0
            mult = 1.0
            if name in _left_zones and lidar_left_mm > 0.0:
                if lidar_left_mm > _LIDAR_BOOST_MM:
                    mult *= _LIDAR_BOOST
                elif lidar_left_mm < cfg.LIDAR_SAFETY_MM:
                    mult *= _LIDAR_PENALTY
            elif name in _right_zones and lidar_right_mm > 0.0:
                if lidar_right_mm > _LIDAR_BOOST_MM:
                    mult *= _LIDAR_BOOST
                elif lidar_right_mm < cfg.LIDAR_SAFETY_MM:
                    mult *= _LIDAR_PENALTY
            return mult

        # If a side zone clearly beats CENTER by SIDE_PREFER_MARGIN, prefer it
        # unconditionally (ignore CENTER bias and accept-ratio).
        center_name = cfg.ZONE_NAMES[center_idx]
        center_m = analysis.corridors.get(center_name)
        if center_m is not None:
            center_safety = center_m.safety_score
            best_side_name = None
            best_side_safety = -1.0
            for name, m in candidates:
                if name == center_name:
                    continue
                adjusted = m.safety_score * _lidar_distance_mult(name)
                if adjusted > best_side_safety:
                    best_side_safety = adjusted
                    best_side_name = name
            if (
                best_side_name is not None
                and (best_side_safety - center_safety) > cfg.SIDE_PREFER_MARGIN
            ):
                print(
                    f"[Decision] Side zone {best_side_name} preferred: "
                    f"safety={best_side_safety:.2f} vs CENTER={center_safety:.2f}"
                )
                return best_side_name

        # LiDAR side-bias: compute per-zone preference bonus when OAK-D confidence is low.
        # LEFT/L2/L1 are on the left; R1/R2/RIGHT are on the right.

        def _lidar_bonus(name: str) -> float:
            if lidar_pref is None:
                return 0.0
            if lidar_pref == "left" and name in _left_zones:
                return cfg.LIDAR_SIDE_BIAS_BONUS
            if lidar_pref == "right" and name in _right_zones:
                return cfg.LIDAR_SIDE_BIAS_BONUS
            return 0.0

        # Find the zone with the best safety_score (+ LiDAR distance mult + side bonus)
        best_name = None
        best_effective_score = -1.0
        for name, m in candidates:
            effective = m.safety_score * _lidar_distance_mult(name)
            if m.zone_index == center_idx:
                effective += cfg.CENTER_SAFETY_BIAS
            effective += _lidar_bonus(name)
            if effective > best_effective_score:
                best_effective_score = effective
                best_name = name

        # Check if CENTER is in the group and close enough to the best
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
                parts.append(f"low nav {center_m.valid_ratio:.0%}")
            if center_m.p20_depth <= cfg.EMERGENCY_STOP_MM:
                parts.append(f"p20={int(center_m.p20_depth)}mm too close")
            if center_m.close_obstacle_ratio >= 0.60:
                parts.append(f"close_obs={center_m.close_obstacle_ratio:.0%}")
            if center_m.largest_close_blob_px >= cfg.MIN_OBSTACLE_CLUSTER_PX:
                parts.append(f"blob={center_m.largest_close_blob_px}px")
            if center_m.vertical_close_run_frac >= cfg.VERTICAL_CLOSE_RUN_FRAC:
                parts.append(f"vrun={center_m.vertical_close_run_frac:.2f}")
            if center_m.emergency_ratio >= cfg.SPREAD_EMERGENCY_RATIO:
                parts.append(f"emerg={center_m.emergency_ratio:.0%}")
            return f"CENTER blocked: {', '.join(parts) if parts else 'unsafe'}"

        if chosen_m:
            return (
                f"CENTER safety={center_m.safety_score:.2f} < "
                f"{chosen} safety={chosen_m.safety_score:.2f} "
                f"(need {cfg.CENTER_ACCEPT_RATIO:.0%})"
            )

        return f"CENTER weaker than {chosen}"

    # ── Confidence ──────────────────────────────────────────

    def _compute_confidence(self, analysis: AnalysisResult, fusion_boost: float = 0.0) -> float:
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
        confidence = max(0.0, min(1.0, confidence + float(fusion_boost or 0.0)))
        return confidence

    def reset(self) -> None:
        self._history.clear()
        self._last_stable = "STOP"
        self._last_change_time = 0.0
        self._unsafe_streak = 0
        self._free_streak = 0
        self._go_streak = 0
        self._go_candidate = ""
        self._free_stable_streak = 0
        self._free_exit_bad_streak = 0
        self._confirmed_centered = False
        self._ema_frames_seen = 0
        self._ema_p20.clear()
        self._ema_close.clear()
        self._low_conf_streak = 0
        self._last_non_center_go = ""
        self._effective_go_frames = self._go_frames_required
        self._escape_latch_side = ""
        self._escape_latch_clear_streak = 0
