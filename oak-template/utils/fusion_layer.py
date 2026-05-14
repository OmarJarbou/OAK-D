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

    def __init__(self, lidar: LidarAnalyzer):
        self._lidar = lidar

    def fuse(self, oak_analysis: AnalysisResult) -> FusedAnalysis:
        oak_front_mm = self._oak_front_mm(oak_analysis)

        scan: LidarScan | None = self._lidar.latest_scan
        if scan is None:
            return FusedAnalysis(
                oak_analysis=oak_analysis,
                has_emergency=bool(oak_analysis.has_emergency),
                front_clear_mm=float(oak_front_mm),
                side_escape_left=False,
                side_escape_right=False,
                lidar_active=False,
                confidence_boost=-0.05,
                fusion_reason="lidar_stale",
                lidar_left_mm=0.0,
                lidar_right_mm=0.0,
            )

        lidar_emergency = not bool(scan.front_clear)
        oak_emergency = bool(oak_analysis.has_emergency)
        lidar_front_mm = float(scan.front_min_mm)

        # 2) Agreement: boost confidence, tighten clearance estimate.
        if oak_emergency == lidar_emergency:
            fused_emergency = oak_emergency
            fused_front_mm = min(float(oak_front_mm), lidar_front_mm)
            fused_oak = self._override_emergency(oak_analysis, fused_emergency)
            return FusedAnalysis(
                oak_analysis=fused_oak,
                has_emergency=fused_emergency,
                front_clear_mm=float(fused_front_mm),
                side_escape_left=bool(scan.side_escape_left),
                side_escape_right=bool(scan.side_escape_right),
                lidar_active=True,
                confidence_boost=+0.10,
                fusion_reason="oak_lidar_agree",
                lidar_left_mm=float(scan.left_min_mm),
                lidar_right_mm=float(scan.right_min_mm),
            )

        # 3) LiDAR veto: LiDAR says emergency, OAK says clear.
        if lidar_emergency and (not oak_emergency):
            # ── Side-escape override ─────────────────────────────────────
            # If either side has a clear escape path, do NOT declare emergency.
            # Let DecisionEngine steer the walker left or right directly
            # without forcing a STOP first. This prevents the STOP→FREE→STOP
            # infinite loop when an obstacle is ahead but a side path exists.
            has_side_escape = bool(scan.side_escape_left) or bool(scan.side_escape_right)
            if has_side_escape:
                # Keep OAK analysis unchanged (no emergency) but carry LiDAR
                # side distances so DecisionEngine can pick the open side.
                return FusedAnalysis(
                    oak_analysis=oak_analysis,
                    has_emergency=False,
                    front_clear_mm=float(lidar_front_mm),
                    side_escape_left=bool(scan.side_escape_left),
                    side_escape_right=bool(scan.side_escape_right),
                    lidar_active=True,
                    confidence_boost=+0.05,
                    fusion_reason="lidar_veto_side_escape",
                    lidar_left_mm=float(scan.left_min_mm),
                    lidar_right_mm=float(scan.right_min_mm),
                )

            # No side escape available → true emergency, stop the walker.
            fused_oak = self._override_emergency(oak_analysis, True)
            return FusedAnalysis(
                oak_analysis=fused_oak,
                has_emergency=True,
                front_clear_mm=float(lidar_front_mm),
                side_escape_left=False,
                side_escape_right=False,
                lidar_active=True,
                confidence_boost=+0.05,
                fusion_reason="lidar_veto_emergency",
                lidar_left_mm=float(scan.left_min_mm),
                lidar_right_mm=float(scan.right_min_mm),
            )

        # 4) OAK says emergency, LiDAR says clear.
        if (not lidar_emergency) and oak_emergency:
            if float(oak_front_mm) < self.DISAGREEMENT_TRUST_THRESHOLD_MM:
                fused_oak = self._override_emergency(oak_analysis, True)
                return FusedAnalysis(
                    oak_analysis=fused_oak,
                    has_emergency=True,
                    front_clear_mm=min(float(oak_front_mm), lidar_front_mm),
                    side_escape_left=bool(scan.side_escape_left),
                    side_escape_right=bool(scan.side_escape_right),
                    lidar_active=True,
                    confidence_boost=0.0,
                    fusion_reason="oak_trusted_close_obstacle",
                    lidar_left_mm=float(scan.left_min_mm),
                    lidar_right_mm=float(scan.right_min_mm),
                )

            fused_oak = self._override_emergency(oak_analysis, False)
            return FusedAnalysis(
                oak_analysis=fused_oak,
                has_emergency=False,
                front_clear_mm=min(float(oak_front_mm), lidar_front_mm),
                side_escape_left=bool(scan.side_escape_left),
                side_escape_right=bool(scan.side_escape_right),
                lidar_active=True,
                confidence_boost=+0.08,
                fusion_reason="oak_suppressed_false_stop",
                lidar_left_mm=float(scan.left_min_mm),
                lidar_right_mm=float(scan.right_min_mm),
            )

        # Defensive fallback: should be unreachable.
        return FusedAnalysis(
            oak_analysis=oak_analysis,
            has_emergency=bool(oak_analysis.has_emergency),
            front_clear_mm=float(oak_front_mm),
            side_escape_left=bool(scan.side_escape_left),
            side_escape_right=bool(scan.side_escape_right),
            lidar_active=True,
            confidence_boost=0.0,
            fusion_reason="fallback",
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
