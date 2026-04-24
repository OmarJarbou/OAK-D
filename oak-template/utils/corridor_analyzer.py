# utils/corridor_analyzer.py
"""
7-zone corridor analysis with MERGED free-space group width estimation.

Splits the depth ROI into 7 equal horizontal zones matching Arduino positions.
After computing per-zone depth metrics, merges adjacent "clear" zones into
continuous free-space groups and estimates the total physical width of each
group. A path is valid only if the merged group width >= REQUIRED_CLEAR_WIDTH_M.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from utils.config import WalkerConfig


# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class CorridorMetrics:
    """Analysis result for a single corridor zone."""
    name: str
    zone_index: int             # 0=LEFT ... 6=RIGHT
    valid_ratio: float          # Fraction of valid depth pixels
    p20_depth: float            # 20th percentile depth (mm)
    p25_depth: float            # 25th percentile depth (mm)
    p50_depth: float            # Median depth (mm)
    mean_depth: float           # Mean depth (mm)
    close_obstacle_ratio: float # Fraction of pixels < CLOSE_OBSTACLE_MM
    emergency_ratio: float      # Fraction of pixels < EMERGENCY_STOP_MM
    zone_width_m: float         # Physical width of this single zone (meters)
    is_clear: bool              # True if zone has safe depth (not width-gated)
    score: float                # Depth-based navigation score [0, 1] (includes center bias)
    safety_score: float         # Pure safety score [0, 1] (NO center bias)


@dataclass
class FreeSpaceGroup:
    """A merged group of adjacent clear zones."""
    zone_names: List[str]       # e.g. ["L1", "CENTER", "R1"]
    zone_indices: List[int]     # e.g. [2, 3, 4]
    total_width_m: float        # Sum of individual zone widths
    is_valid: bool              # total_width_m >= REQUIRED_CLEAR_WIDTH_M
    avg_p20_depth: float        # Average p20 depth across group
    avg_score: float            # Average score across group
    best_zone: str              # Recommended target zone (closest to center)
    center_index: int = 3       # Index of CENTER zone


@dataclass
class AnalysisResult:
    """Complete analysis output."""
    corridors: Dict[str, CorridorMetrics]
    groups: List[FreeSpaceGroup]
    valid_groups: List[FreeSpaceGroup]  # Only groups with is_valid=True
    floor_mask: np.ndarray
    roi_box: Tuple[int, int, int, int]
    has_emergency: bool         # Any zone has dangerously close obstacle


# ──────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────

class CorridorAnalyzer:
    """
    Analyzes a raw depth frame → per-zone metrics → merged free-space groups.
    """

    def __init__(self, cfg: WalkerConfig):
        self.cfg = cfg
        self._half_fov_tan = math.tan(cfg.HFOV_RAD / 2.0)

    def analyze(self, depth_frame: np.ndarray) -> AnalysisResult:
        """
        Analyze a raw depth frame (uint16, millimeters).
        Returns AnalysisResult with corridors, merged groups, and emergency flag.
        """
        cfg = self.cfg

        if depth_frame is None or depth_frame.size == 0:
            return self._empty_result()

        h, w = depth_frame.shape
        depth = depth_frame.copy()

        # ── Extract ROI ─────────────────────────────────────
        y1 = int(h * cfg.TOP_CROP_RATIO)
        y2 = int(h * cfg.BOTTOM_CROP_RATIO)
        x1 = int(w * cfg.SIDE_MARGIN_RATIO)
        x2 = int(w * (1.0 - cfg.SIDE_MARGIN_RATIO))
        roi_box = (x1, y1, x2, y2)

        roi_depth = depth[y1:y2, x1:x2]
        roi_valid = (roi_depth >= cfg.MIN_DEPTH_MM) & (roi_depth <= cfg.MAX_DEPTH_MM)

        # ── Floor Removal ───────────────────────────────────
        floor_mask = self._remove_floor(roi_depth, roi_valid)
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

            m = self._compute_zone(name, i, z_depth, z_valid, zone_width_px, roi_w)
            corridors[name] = m

        # ── Merge Adjacent Clear Zones ──────────────────────
        groups = self._merge_clear_zones(corridors)
        valid_groups = [g for g in groups if g.is_valid]

        # ── Emergency Detection (smarter) ────────────────
        center_zone_names = ("L1", "CENTER", "R1")
        emergency_count = 0
        for cname in center_zone_names:
            m = corridors[cname]
            if m.emergency_ratio > 0.35 and m.valid_ratio > 0.20:
                emergency_count += 1

        has_emergency = False
        if emergency_count >= len(center_zone_names) * cfg.EMERGENCY_CENTER_COVERAGE:
            # Center is dangerous — but check if sides offer escape
            side_escape = False
            for g in groups:
                if g.is_valid and g.avg_p20_depth > cfg.MIN_SIDE_DEPTH_MM:
                    # Check group doesn’t overlap with blocked center
                    center_idx = cfg.NUM_ZONES // 2
                    if center_idx not in g.zone_indices:
                        side_escape = True
                        break
            if not side_escape:
                has_emergency = True

        return AnalysisResult(
            corridors=corridors,
            groups=groups,
            valid_groups=valid_groups,
            floor_mask=floor_mask,
            roi_box=roi_box,
            has_emergency=has_emergency,
        )

    # ── Per-Zone Computation ─────────────────────────────────

    def _compute_zone(
        self,
        name: str,
        zone_index: int,
        z_depth: np.ndarray,
        z_valid: np.ndarray,
        zone_pixel_width: int,
        total_roi_width: int,
    ) -> CorridorMetrics:
        cfg = self.cfg
        total_pixels = max(z_depth.size, 1)
        valid_values = z_depth[z_valid]

        if valid_values.size == 0:
            return CorridorMetrics(
                name=name, zone_index=zone_index,
                valid_ratio=0.0, p20_depth=0.0, p25_depth=0.0,
                p50_depth=0.0, mean_depth=0.0,
                close_obstacle_ratio=1.0, emergency_ratio=1.0,
                zone_width_m=0.0, is_clear=False, score=0.0,
                safety_score=0.0,
            )

        valid_ratio = valid_values.size / total_pixels
        p20_depth = float(np.percentile(valid_values, 20))
        p25_depth = float(np.percentile(valid_values, 25))
        p50_depth = float(np.percentile(valid_values, 50))
        mean_depth = float(np.mean(valid_values))
        close_obstacle_ratio = float(np.mean(valid_values < cfg.CLOSE_OBSTACLE_MM))
        emergency_ratio = float(np.mean(valid_values < cfg.EMERGENCY_STOP_MM))

        # Physical width of this single zone at its median depth
        depth_m = p50_depth / 1000.0
        total_visible_width = 2.0 * depth_m * self._half_fov_tan
        zone_fraction = zone_pixel_width / max(total_roi_width, 1)
        zone_width_m = total_visible_width * zone_fraction

        # A zone is "clear" based on depth, NOT on width
        # Clear = enough valid data AND obstacles aren't critically close
        is_clear = (
            valid_ratio >= cfg.MIN_VALID_RATIO
            and p20_depth > cfg.EMERGENCY_STOP_MM
            and close_obstacle_ratio < 0.60
        )

        # Depth-based score (used for ranking within merged groups)
        depth_score = float(np.clip(
            (p20_depth - cfg.MIN_DEPTH_MM) / (cfg.SAFE_CORRIDOR_MM - cfg.MIN_DEPTH_MM),
            0.0, 1.0,
        ))
        close_penalty = 1.0 - close_obstacle_ratio

        # Center preference (baked into corridor score, NOT into safety_score)
        center_idx = cfg.NUM_ZONES // 2
        dist_from_center = abs(zone_index - center_idx)
        center_bonus = cfg.CENTER_BIAS if dist_from_center == 0 else (
            cfg.ADJACENT_BIAS if dist_from_center == 1 else 0.0
        )

        score = (
            cfg.WEIGHT_DEPTH_P20 * depth_score
            + cfg.WEIGHT_CLOSE_OBS * close_penalty
            + cfg.WEIGHT_VALID_RATIO * valid_ratio
            + center_bonus
        )

        # ── Safety Score (pure safety, NO center bias) ───────
        depth_norm_25 = float(np.clip(
            (p25_depth - cfg.MIN_DEPTH_MM) / (cfg.SAFE_CORRIDOR_MM - cfg.MIN_DEPTH_MM),
            0.0, 1.0,
        ))
        floor_invalid_ratio = 1.0 - valid_ratio
        safety_score = (
            cfg.SAFETY_W_DEPTH * depth_norm_25
            + cfg.SAFETY_W_CLOSE_OBS * close_penalty
            + cfg.SAFETY_W_VALID * valid_ratio
            + cfg.SAFETY_W_FLOOR_INV * (1.0 - floor_invalid_ratio)
        )

        return CorridorMetrics(
            name=name, zone_index=zone_index,
            valid_ratio=valid_ratio, p20_depth=p20_depth,
            p25_depth=p25_depth,
            p50_depth=p50_depth, mean_depth=mean_depth,
            close_obstacle_ratio=close_obstacle_ratio,
            emergency_ratio=emergency_ratio,
            zone_width_m=zone_width_m, is_clear=is_clear,
            score=score, safety_score=safety_score,
        )

    # ── Merge Adjacent Clear Zones ───────────────────────────

    def _merge_clear_zones(
        self, corridors: Dict[str, CorridorMetrics]
    ) -> List[FreeSpaceGroup]:
        """
        Scan zones left-to-right. Merge consecutive clear zones into groups.
        Compute total physical width of each group.
        Choose the best target zone within each valid group.
        """
        cfg = self.cfg
        zone_list = [corridors[name] for name in cfg.ZONE_NAMES]

        groups: List[FreeSpaceGroup] = []
        current_group: List[CorridorMetrics] = []

        for m in zone_list:
            if m.is_clear:
                current_group.append(m)
            else:
                if current_group:
                    groups.append(self._build_group(current_group))
                    current_group = []

        # Don't forget the last group
        if current_group:
            groups.append(self._build_group(current_group))

        return groups

    def _build_group(self, zones: List[CorridorMetrics]) -> FreeSpaceGroup:
        """Build a FreeSpaceGroup from a list of adjacent clear zones."""
        cfg = self.cfg
        names = [z.name for z in zones]
        indices = [z.zone_index for z in zones]
        total_width = sum(z.zone_width_m for z in zones)
        is_valid = total_width >= cfg.REQUIRED_CLEAR_WIDTH_M
        avg_p20 = sum(z.p20_depth for z in zones) / len(zones)
        avg_score = sum(z.score for z in zones) / len(zones)

        # Choose best target: zone with highest safety_score (not closest to center)
        best_zone_m = max(zones, key=lambda z: z.safety_score)

        return FreeSpaceGroup(
            zone_names=names,
            zone_indices=indices,
            total_width_m=total_width,
            is_valid=is_valid,
            avg_p20_depth=avg_p20,
            avg_score=avg_score,
            best_zone=best_zone_m.name,
        )

    # ── Floor Removal ─────────────────────────────────────

    def _remove_floor(
        self, roi_depth: np.ndarray, roi_valid: np.ndarray
    ) -> np.ndarray:
        cfg = self.cfg
        h, w = roi_depth.shape
        is_floor = np.zeros((h, w), dtype=bool)
        if h < 2:
            return is_floor

        is_floor[h - 1, :] = (
            roi_valid[h - 1, :]
            & (roi_depth[h - 1, :] > cfg.FLOOR_MIN_DEPTH)
            & (roi_depth[h - 1, :] < cfg.FLOOR_MAX_DEPTH)
        )

        for y in range(h - 2, -1, -1):
            diff = roi_depth[y, :].astype(np.int32) - roi_depth[y + 1, :].astype(np.int32)
            continues = (
                is_floor[y + 1, :]
                & (diff > cfg.FLOOR_GRADIENT_MIN)
                & (diff < cfg.FLOOR_GRADIENT_MAX)
            )
            is_floor[y, :] = continues

        return is_floor

    # ── Empty Fallback ────────────────────────────────────

    def _empty_result(self) -> AnalysisResult:
        empty_corridors = {
            name: CorridorMetrics(
                name=name, zone_index=i,
                valid_ratio=0.0, p20_depth=0.0, p25_depth=0.0,
                p50_depth=0.0, mean_depth=0.0,
                close_obstacle_ratio=1.0, emergency_ratio=1.0,
                zone_width_m=0.0, is_clear=False, score=0.0,
                safety_score=0.0,
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
