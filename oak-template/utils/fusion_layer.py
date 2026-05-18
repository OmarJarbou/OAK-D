from __future__ import annotations

from dataclasses import dataclass

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
    confidence_boost: float       # always 0.0 in this simplified version
    fusion_reason: str
    lidar_left_mm: float = 0.0
    lidar_right_mm: float = 0.0


class FusionLayer:
    """
    Simplified fusion: pass LiDAR distances through without overriding
    the camera's has_emergency flag.

    The DecisionEngine receives raw LiDAR distances and decides when
    to trust them — no complex agreement logic here.
    """

    def __init__(self, lidar: LidarAnalyzer, cfg=None):
        self._lidar = lidar
        self._cfg = cfg

    def fuse(self, oak_analysis: AnalysisResult) -> FusedAnalysis:
        scan: LidarScan | None = self._lidar.latest_scan

        if scan is None:
            return FusedAnalysis(
                oak_analysis=oak_analysis,
                has_emergency=oak_analysis.has_emergency,
                front_clear_mm=9999.0,
                side_escape_left=False,
                side_escape_right=False,
                lidar_active=False,
                confidence_boost=0.0,
                fusion_reason="no_lidar",
                lidar_left_mm=9999.0,
                lidar_right_mm=9999.0,
            )

        # Pass LiDAR distances through; camera's emergency is authoritative.
        return FusedAnalysis(
            oak_analysis=oak_analysis,
            has_emergency=oak_analysis.has_emergency,   # camera is authoritative
            front_clear_mm=float(scan.front_min_mm),
            side_escape_left=bool(scan.side_escape_left),
            side_escape_right=bool(scan.side_escape_right),
            lidar_active=True,
            confidence_boost=0.0,
            fusion_reason="lidar_data",
            lidar_left_mm=float(scan.left_min_mm),
            lidar_right_mm=float(scan.right_min_mm),
        )
