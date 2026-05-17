from __future__ import annotations

from dataclasses import dataclass, replace

from utils.corridor_analyzer import AnalysisResult
from utils.lidar_analyzer import LidarAnalyzer, LidarScan


@dataclass
class FusedAnalysis:
    oak_analysis: AnalysisResult
    has_emergency: bool
    front_clear_mm: float
    side_escape_left: bool
    side_escape_right: bool
    lidar_active: bool
    confidence_boost: float  # range: -0.10 to +0.10
    fusion_reason: str
    # Raw LiDAR side distances — used by DecisionEngine for directional bias.
    # 0.0 when no scan is available.
    lidar_left_mm: float = 0.0
    lidar_right_mm: float = 0.0


class FusionLayer:
    DISAGREEMENT_TRUST_THRESHOLD_MM = 800.0

    def __init__(self, lidar: LidarAnalyzer, cfg=None):
        self._lidar = lidar
        self._cfg = cfg

    # ── OAK-FIRST POLICY ─────────────────────────────────────────────────
    # OAK camera is the SOLE authority on obstacle avoidance.
    # LiDAR contributes side-distance data for directional bias ONLY.
    # LiDAR never overrides OAK's emergency/clear decision.
    # ─────────────────────────────────────────────────────────────────────

    def fuse(self, oak_analysis: AnalysisResult) -> FusedAnalysis:
        """
        OAK-FIRST fusion:
          - OAK camera ALWAYS decides has_emergency and front_clear_mm.
          - LiDAR contributes side distances (left/right) for directional bias only.
          - LiDAR never triggers or suppresses an emergency stop.
        """
        oak_front_mm = self._oak_front_mm(oak_analysis)
        oak_emergency = bool(oak_analysis.has_emergency)

        scan: LidarScan | None = self._lidar.latest_scan
        if scan is None:
            # No LiDAR data — OAK decides everything, small confidence penalty.
            return FusedAnalysis(
                oak_analysis=oak_analysis,
                has_emergency=oak_emergency,
                front_clear_mm=float(oak_front_mm),
                side_escape_left=False,
                side_escape_right=False,
                lidar_active=False,
                confidence_boost=-0.05,
                fusion_reason="lidar_stale",
                lidar_left_mm=0.0,
                lidar_right_mm=0.0,
            )

        # LiDAR is active — OAK still owns has_emergency / front_clear_mm.
        # LiDAR only provides: side escape flags and side distances.
        # Confidence boost when LiDAR agrees with OAK's front assessment.
        lidar_emergency = not bool(scan.front_clear)
        if oak_emergency == lidar_emergency:
            confidence_boost = +0.10
            fusion_reason = "oak_lidar_agree"
        elif lidar_emergency and not oak_emergency:
            # LiDAR sees front obstacle, OAK says clear.
            # OAK wins — LiDAR may be detecting a low/floor-level obstacle.
            confidence_boost = +0.03
            fusion_reason = "oak_clear_lidar_ignored"
        else:
            # OAK sees obstacle, LiDAR says clear.
            # OAK wins — trust the camera that's at head/chest height.
            confidence_boost = 0.0
            fusion_reason = "oak_emergency_lidar_clear"

        return FusedAnalysis(
            oak_analysis=oak_analysis,          # unmodified — OAK owns this
            has_emergency=oak_emergency,         # OAK decides
            front_clear_mm=float(oak_front_mm), # OAK decides
            side_escape_left=bool(scan.side_escape_left),
            side_escape_right=bool(scan.side_escape_right),
            lidar_active=True,
            confidence_boost=confidence_boost,
            fusion_reason=fusion_reason,
            lidar_left_mm=float(scan.left_min_mm),
            lidar_right_mm=float(scan.right_min_mm),
        )

    @staticmethod
    def _oak_front_mm(analysis: AnalysisResult) -> float:
        # Fix 5: Return 9999 when no valid data so we don't falsely trigger
        # emergency stops (0 was interpreted as "obstacle at 0mm").
        # Also require valid_ratio >= 0.14 to filter zones with near-zero depth pixels.
        zones = analysis.corridors
        vals = []
        for k in ("L1", "CENTER", "R1"):
            m = zones.get(k)
            if m is not None:
                try:
                    v = float(m.p20_depth or 0.0)
                except Exception:
                    v = 0.0
                if v > 0.0 and float(m.valid_ratio or 0.0) >= 0.14:
                    vals.append(v)
        if not vals:
            return 9999.0  # No data → assume path clear, don't stop
        return float(min(vals))

    @staticmethod
    def _override_emergency(analysis: AnalysisResult, value: bool) -> AnalysisResult:
        return replace(analysis, has_emergency=bool(value))
