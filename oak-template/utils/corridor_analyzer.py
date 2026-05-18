# utils/corridor_analyzer.py
"""
Simplified 7-zone corridor analysis.

Scores each zone by p20_depth / 2000 (clipped 0-1).
is_clear = True if valid_ratio > 0.1 and p20_depth > 500.
No blob detection, no vertical run fractions, no spread emergency.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from utils.config import WalkerConfig


# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class CorridorMetrics:
    """Analysis result for a single corridor zone."""
    name: str
    zone_index: int
    valid_ratio: float      # Fraction of navigable pixels (valid depth, not floor)
    p20_depth: float        # 20th percentile depth on navigable pixels (mm)
    mean_depth: float       # Mean depth on navigable pixels (mm)
    score: float            # Navigation score [0, 1]  = p20_depth / 2000, clipped
    safety_score: float     # Same as score (kept for compatibility)
    is_clear: bool          # valid_ratio > 0.1 and p20_depth > 500

    # Legacy / compatibility fields (kept so main.py debug overlay doesn't crash)
    thin_depth: float = 0.0
    p25_depth: float = 0.0
    p50_depth: float = 0.0
    close_obstacle_ratio: float = 0.0
    danger_obstacle_ratio: float = 0.0
    emergency_ratio: float = 0.0
    largest_close_blob_px: int = 0
    vertical_close_run_frac: float = 0.0
    zone_width_m: float = 0.0


@dataclass
class FreeSpaceGroup:
    """A merged group of adjacent clear zones (kept for compatibility)."""
    zone_names: List[str]
    zone_indices: List[int]
    total_width_m: float
    is_valid: bool
    avg_p20_depth: float
    avg_score: float
    best_zone: str
    center_index: int = 3


@dataclass(slots=False)
class AnalysisResult:
    """Complete analysis output."""
    corridors: Dict[str, CorridorMetrics]
    groups: List[FreeSpaceGroup]
    valid_groups: List[FreeSpaceGroup]
    floor_mask: np.ndarray
    roi_box: Tuple[int, int, int, int]
    has_emergency: bool


# ──────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────

class CorridorAnalyzer:
    """
    Analyzes a raw depth frame → per-zone metrics.
    Simplified: score = p20_depth / 2000 (clipped 0-1).
    """

    def __init__(self, cfg: WalkerConfig):
        self.cfg = cfg
        self._half_fov_tan = math.tan(cfg.HFOV_RAD / 2.0)

    def analyze(self, depth_frame: np.ndarray) -> AnalysisResult:
        if depth_frame is None or depth_frame.size == 0:
            return self._empty_result()

        cfg = self.cfg
        h, w = depth_frame.shape

        # ── Extract ROI ──────────────────────────────────────
        y1 = int(h * cfg.TOP_CROP_RATIO)
        y2 = int(h * cfg.BOTTOM_CROP_RATIO)
        x1 = int(w * cfg.SIDE_MARGIN_RATIO)
        x2 = int(w * (1.0 - cfg.SIDE_MARGIN_RATIO))
        roi_box = (x1, y1, x2, y2)

        roi_depth = depth_frame[y1:y2, x1:x2]
        roi_valid = (roi_depth >= cfg.MIN_DEPTH_MM) & (roi_depth <= cfg.MAX_DEPTH_MM)

        # ── Simple Floor Removal ────────────────────────────
        floor_mask = self._simple_floor_removal(roi_depth, roi_valid)
        roi_valid = roi_valid & (~floor_mask)

        # ── Per-Zone Metrics ────────────────────────────────
        roi_w = x2 - x1
        zone_width_px = roi_w // cfg.NUM_ZONES
        corridors: Dict[str, CorridorMetrics] = {}
        has_emergency = False

        for i, name in enumerate(cfg.ZONE_NAMES):
            zx1 = i * zone_width_px
            zx2 = (i + 1) * zone_width_px if i < cfg.NUM_ZONES - 1 else roi_w

            z_depth = roi_depth[:, zx1:zx2]
            z_valid = roi_valid[:, zx1:zx2]

            m = self._compute_zone(name, i, z_depth, z_valid)
            corridors[name] = m

            # Emergency: obstacle very close in central zones
            if name in ("L1", "CENTER", "R1"):
                if m.valid_ratio > 0.1 and 0 < m.p20_depth < cfg.EMERGENCY_STOP_MM:
                    has_emergency = True

        # Build minimal group list (just for debug overlay compatibility)
        groups = self._build_groups(corridors)
        valid_groups = [g for g in groups if g.is_valid]

        return AnalysisResult(
            corridors=corridors,
            groups=groups,
            valid_groups=valid_groups,
            floor_mask=floor_mask,
            roi_box=roi_box,
            has_emergency=has_emergency,
        )

    # ── Simple Floor Removal ──────────────────────────────

    def _simple_floor_removal(
        self, depth: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        """
        Column-wise floor propagation from the bottom row upward.
        A pixel is floor if the row below it is floor and the depth
        increases slightly (camera looks down at ground).
        """
        cfg = self.cfg
        h, w = depth.shape
        floor = np.zeros((h, w), dtype=bool)
        if h < 2:
            return floor

        # Bottom row: floor if depth looks like ground (not a wall right there)
        floor[h - 1, :] = (
            valid[h - 1, :]
            & (depth[h - 1, :] > cfg.FLOOR_MIN_DEPTH)
            & (depth[h - 1, :] < cfg.FLOOR_MAX_DEPTH)
        )

        for y in range(h - 2, -1, -1):
            diff = depth[y, :].astype(np.int32) - depth[y + 1, :].astype(np.int32)
            floor[y, :] = (
                floor[y + 1, :]
                & (diff > cfg.FLOOR_GRADIENT_MIN)
                & (diff < cfg.FLOOR_GRADIENT_MAX)
            )

        return floor

    # ── Per-Zone Computation ──────────────────────────────

    def _compute_zone(
        self,
        name: str,
        idx: int,
        z_depth: np.ndarray,
        z_valid: np.ndarray,
    ) -> CorridorMetrics:
        cfg = self.cfg
        total_px = max(z_depth.size, 1)
        nav_count = int(np.count_nonzero(z_valid))

        if nav_count == 0:
            return CorridorMetrics(
                name=name, zone_index=idx,
                valid_ratio=0.0, p20_depth=0.0, mean_depth=0.0,
                score=0.0, safety_score=0.0, is_clear=False,
            )

        valid_ratio = nav_count / total_px
        nav_vals = z_depth[z_valid]
        p20 = float(np.percentile(nav_vals, 20))
        mean = float(np.mean(nav_vals))

        # Simple score: deeper = safer
        score = float(np.clip(p20 / 2000.0, 0.0, 1.0))

        # Clear: enough valid pixels and not too close
        is_clear = valid_ratio > 0.1 and p20 > cfg.EMERGENCY_STOP_MM

        return CorridorMetrics(
            name=name, zone_index=idx,
            valid_ratio=valid_ratio,
            p20_depth=p20,
            mean_depth=mean,
            score=score,
            safety_score=score,   # same as score
            is_clear=is_clear,
            # legacy fields
            thin_depth=p20,
            p25_depth=p20,
            p50_depth=mean,
            close_obstacle_ratio=0.0,
            danger_obstacle_ratio=0.0,
            emergency_ratio=0.0,
            largest_close_blob_px=0,
            vertical_close_run_frac=0.0,
            zone_width_m=0.0,
        )

    # ── Group Builder (for debug overlay compatibility) ───

    def _build_groups(
        self, corridors: Dict[str, CorridorMetrics]
    ) -> List[FreeSpaceGroup]:
        """Build simple contiguous clear-zone groups."""
        cfg = self.cfg
        zone_list = [corridors[n] for n in cfg.ZONE_NAMES]
        groups: List[FreeSpaceGroup] = []
        current: List[CorridorMetrics] = []

        for m in zone_list:
            if m.is_clear:
                current.append(m)
            else:
                if current:
                    groups.append(self._make_group(current))
                    current = []
        if current:
            groups.append(self._make_group(current))

        return groups

    def _make_group(self, zones: List[CorridorMetrics]) -> FreeSpaceGroup:
        names = [z.name for z in zones]
        indices = [z.zone_index for z in zones]
        avg_p20 = sum(z.p20_depth for z in zones) / len(zones)
        avg_score = sum(z.score for z in zones) / len(zones)
        best = max(zones, key=lambda z: z.score)
        # is_valid: at least one clear zone (width check relaxed)
        is_valid = len(zones) >= 1
        return FreeSpaceGroup(
            zone_names=names,
            zone_indices=indices,
            total_width_m=0.0,
            is_valid=is_valid,
            avg_p20_depth=avg_p20,
            avg_score=avg_score,
            best_zone=best.name,
        )

    # ── Empty Fallback ────────────────────────────────────

    def _empty_result(self) -> AnalysisResult:
        empty_corridors = {
            name: CorridorMetrics(
                name=name, zone_index=i,
                valid_ratio=0.0, p20_depth=0.0, mean_depth=0.0,
                score=0.0, safety_score=0.0, is_clear=False,
            )
            for i, name in enumerate(self.cfg.ZONE_NAMES)
        }
        return AnalysisResult(
            corridors=empty_corridors,
            groups=[],
            valid_groups=[],
            floor_mask=np.zeros((1, 1), dtype=bool),
            roi_box=(0, 0, 0, 0),
            has_emergency=False,
        )
