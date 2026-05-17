# utils/corridor_analyzer.py
"""
7-zone corridor analysis with MERGED free-space group width estimation.

All navigation metrics use raw stereo depth in millimeters (uint16 frame from
stereo.depth). RGB depth colormap previews are never used for classification.

Splits the depth ROI into 7 equal horizontal zones matching Arduino positions.
Ground / floor is treated as traversable: obstacles are evaluated on navigable
pixels (valid range, not classified as floor), using percentile depth, close-pixel
ratio, connected close-blob size, and vertical close runs — not mean depth.
Merged group width must be >= REQUIRED_CLEAR_WIDTH_M (walker width + side margins).
"""

import math
import numpy as np
import cv2
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
    valid_ratio: float          # Fraction of navigable pixels (valid depth, not floor)
    p20_depth: float            # 20th percentile depth on navigable (mm)
    thin_depth: float           # Low percentile depth (sparse / thin obstacles)
    p25_depth: float            # 25th percentile depth on navigable (mm)
    p50_depth: float            # Median depth on navigable (mm)
    mean_depth: float           # Median navigable depth (mm); legacy field name
    close_obstacle_ratio: float # On navigable: fraction with depth < CLOSE_OBSTACLE_MM
    danger_obstacle_ratio: float # On navigable: fraction < EMERGENCY_STOP_MM
    emergency_ratio: float      # Same as danger_obstacle_ratio (navigable)
    largest_close_blob_px: int  # Largest 8-connected close-obstacle blob (pixels)
    vertical_close_run_frac: float  # Max vertical run of emergency-close / zone height
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


@dataclass(slots=False)
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
            z_floor = floor_mask[:, zx1:zx2]

            m = self._compute_zone(
                name, i, z_depth, z_valid, z_floor, zone_width_px, roi_w,
            )
            corridors[name] = m

        # ── Merge Adjacent Clear Zones ──────────────────────
        groups = self._merge_clear_zones(corridors)
        valid_groups = [g for g in groups if g.is_valid]

        # ── Emergency Detection (navigable / non-floor only) ────────────────
        center_zone_names = ("L1", "CENTER", "R1")
        emergency_count = 0
        for cname in center_zone_names:
            m = corridors[cname]
            tail_too_close = (
                m.thin_depth > 0
                and m.thin_depth <= cfg.EMERGENCY_STOP_MM
                and m.valid_ratio > cfg.MIN_NAVIGABLE_RATIO
            )
            if tail_too_close:
                emergency_count += 1
            elif (
                m.emergency_ratio > cfg.SPREAD_EMERGENCY_RATIO
                and m.valid_ratio > cfg.MIN_NAVIGABLE_RATIO
            ):
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

    @staticmethod
    def _largest_cc_pixel_count(mask: np.ndarray) -> int:
        if mask is None or mask.size == 0 or not np.any(mask):
            return 0
        img = (mask.astype(np.uint8) * 255)
        n, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        if n <= 1:
            return 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        return int(np.max(areas))

    @staticmethod
    def _max_vertical_run_rows(mask: np.ndarray) -> int:
        """Longest contiguous run of True in any single column."""
        h, w = mask.shape
        if h == 0 or w == 0 or not np.any(mask):
            return 0
        best = 0
        for x in range(w):
            col = mask[:, x]
            padded = np.concatenate(([False], col, [False]))
            d = np.diff(padded.astype(np.int8))
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            if starts.size:
                best = max(best, int((ends - starts).max()))
        return best

    def _compute_zone(
        self,
        name: str,
        zone_index: int,
        z_depth: np.ndarray,
        z_valid: np.ndarray,
        z_floor: np.ndarray,
        zone_pixel_width: int,
        total_roi_width: int,
    ) -> CorridorMetrics:
        cfg = self.cfg
        total_pixels = max(z_depth.size, 1)
        navigable = z_valid & (~z_floor)
        nav_count = int(np.count_nonzero(navigable))
        nav_values = z_depth[navigable]

        if nav_count == 0:
            return CorridorMetrics(
                name=name, zone_index=zone_index,
                valid_ratio=0.0, p20_depth=0.0, thin_depth=0.0, p25_depth=0.0,
                p50_depth=0.0, mean_depth=0.0,
                close_obstacle_ratio=1.0, danger_obstacle_ratio=1.0, emergency_ratio=1.0,
                largest_close_blob_px=0, vertical_close_run_frac=0.0,
                zone_width_m=0.0, is_clear=False, score=0.0,
                safety_score=0.0,
            )

        # Coverage of walker-relevant pixels (excludes large flat floor)
        valid_ratio = nav_count / total_pixels

        p20_depth = float(np.percentile(nav_values, 20))
        thin_depth = float(np.percentile(nav_values, cfg.THIN_OBSTACLE_PERCENTILE))
        p25_depth = float(np.percentile(nav_values, 25))
        p50_depth = float(np.percentile(nav_values, 50))
        # Intentionally not using mean depth for decisions; keep robust center depth for debug
        mean_depth = float(np.median(nav_values))

        close_mask = navigable & (z_depth < cfg.CLOSE_OBSTACLE_MM)
        emerg_mask = navigable & (z_depth < cfg.EMERGENCY_STOP_MM)
        close_obstacle_ratio = float(np.count_nonzero(close_mask) / nav_count)
        danger_obstacle_ratio = float(np.count_nonzero(emerg_mask) / nav_count)
        emergency_ratio = danger_obstacle_ratio

        largest_close_blob_px = self._largest_cc_pixel_count(close_mask)
        zh = max(z_depth.shape[0], 1)
        max_v_run = self._max_vertical_run_rows(emerg_mask)
        vertical_close_run_frac = max_v_run / float(zh)

        compact_obstacle = largest_close_blob_px >= cfg.MIN_OBSTACLE_CLUSTER_PX
        vertical_obstacle = vertical_close_run_frac >= cfg.VERTICAL_CLOSE_RUN_FRAC
        spread_emergency = emergency_ratio >= cfg.SPREAD_EMERGENCY_RATIO
        blocked_obstacle = compact_obstacle or vertical_obstacle or spread_emergency

        # Physical width of this single zone at its median navigable depth
        depth_m = p50_depth / 1000.0
        total_visible_width = 2.0 * depth_m * self._half_fov_tan
        zone_fraction = zone_pixel_width / max(total_roi_width, 1)
        zone_width_m = total_visible_width * zone_fraction

        # Clear = enough navigable data, percentiles safe, no compact/vertical/spread threat
        is_clear = (
            valid_ratio >= max(cfg.MIN_VALID_RATIO, cfg.MIN_NAVIGABLE_RATIO)
            and not blocked_obstacle
            and p20_depth > cfg.EMERGENCY_STOP_MM
            and thin_depth > cfg.EMERGENCY_STOP_MM
            and close_obstacle_ratio < 0.58
            and danger_obstacle_ratio < max(0.26, cfg.SPREAD_EMERGENCY_RATIO - 0.02)
        )

        depth_score = float(np.clip(
            (p20_depth - cfg.MIN_DEPTH_MM) / (cfg.SAFE_CORRIDOR_MM - cfg.MIN_DEPTH_MM),
            0.0, 1.0,
        ))
        close_penalty = 1.0 - close_obstacle_ratio
        danger_penalty = 1.0 - danger_obstacle_ratio
        blob_penalty = float(
            np.clip(
                1.0 - largest_close_blob_px / float(max(cfg.MIN_OBSTACLE_CLUSTER_PX * 4, 1)),
                0.0, 1.0,
            )
        )

        center_idx = cfg.NUM_ZONES // 2
        dist_from_center = abs(zone_index - center_idx)
        center_bonus = cfg.CENTER_BIAS if dist_from_center == 0 else (
            cfg.ADJACENT_BIAS if dist_from_center == 1 else 0.0
        )

        score = (
            cfg.WEIGHT_DEPTH_P20 * depth_score
            + cfg.WEIGHT_CLOSE_OBS * close_penalty
            + 0.18 * danger_penalty
            + 0.07 * blob_penalty
            + cfg.WEIGHT_VALID_RATIO * valid_ratio
            + center_bonus
        )

        # Use p20 (not p25) for safety_score: more sensitive to thin/close obstacles
        # such as chair legs or narrow poles that p25 may partially miss.
        depth_norm_25 = float(np.clip(
            (p20_depth - cfg.MIN_DEPTH_MM) / (cfg.SAFE_CORRIDOR_MM - cfg.MIN_DEPTH_MM),
            0.0, 1.0,
        ))
        floor_invalid_ratio = 1.0 - valid_ratio
        vert_penalty = float(np.clip(1.0 - vertical_close_run_frac * 2.5, 0.0, 1.0))
        safety_score = (
            cfg.SAFETY_W_DEPTH * depth_norm_25
            + cfg.SAFETY_W_CLOSE_OBS * close_penalty
            + 0.10 * danger_penalty
            + 0.08 * blob_penalty
            + 0.05 * vert_penalty
            + cfg.SAFETY_W_VALID * valid_ratio
            + cfg.SAFETY_W_FLOOR_INV * (1.0 - floor_invalid_ratio)
        )

        return CorridorMetrics(
            name=name, zone_index=zone_index,
            valid_ratio=valid_ratio, p20_depth=p20_depth, thin_depth=thin_depth,
            p25_depth=p25_depth,
            p50_depth=p50_depth, mean_depth=mean_depth,
            close_obstacle_ratio=close_obstacle_ratio,
            danger_obstacle_ratio=danger_obstacle_ratio,
            emergency_ratio=emergency_ratio,
            largest_close_blob_px=largest_close_blob_px,
            vertical_close_run_frac=vertical_close_run_frac,
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
        
        # FIX: Account for zone quantization error. The physical gap is usually wider
        # than the sum of purely clear zones because obstacles might only partially 
        # occupy the boundary zones. We add 1.2x the width of a single zone as tolerance.
        zone_w = zones[0].zone_width_m if zones else 0.0
        effective_width = total_width + (1.2 * zone_w)
        
        is_valid = effective_width >= cfg.REQUIRED_CLEAR_WIDTH_M
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
        """
        Mark floor / ground as traversable: column-wise continuity from the bottom
        plus morphological fill in the lower band of the ROI (flat floor dominates).
        """
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

        # Widen floor mask in the lower portion of the ROI (large flat ground plane).
        y0 = int(h * (1.0 - cfg.GROUND_LOWER_BAND_RATIO))
        y0 = max(0, min(y0, h - 1))
        lower = np.zeros((h, w), dtype=np.uint8)
        lower[y0:, :] = 255
        sub_h, sub_w = h - y0, w
        band = (is_floor.astype(np.uint8) * 255)
        if sub_h >= 3 and sub_w >= 3 and int(cfg.FLOOR_DILATE_ITERATIONS) > 0:
            kv = max(3, int(cfg.FLOOR_DILATE_KERNEL_V) | 1)
            kh = max(3, int(cfg.FLOOR_DILATE_KERNEL_H) | 1)
            if kv > sub_h:
                kv = sub_h - 1 if sub_h % 2 == 0 else sub_h
            if kh > sub_w:
                kh = sub_w - 1 if sub_w % 2 == 0 else sub_w
            kv = max(3, kv | 1)
            kh = max(3, kh | 1)
            k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))
            k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
            band[y0:, :] = cv2.dilate(
                band[y0:, :], k_v, iterations=int(cfg.FLOOR_DILATE_ITERATIONS),
            )
            band[y0:, :] = cv2.dilate(
                band[y0:, :], k_h, iterations=int(cfg.FLOOR_DILATE_ITERATIONS),
            )
        grown = band.astype(bool) & lower.astype(bool)
        # Only treat grown pixels as floor if depth still looks like ground (not a pole).
        plane_like = (
            roi_valid
            & (roi_depth > cfg.FLOOR_MIN_DEPTH)
            & (roi_depth < cfg.FLOOR_MAX_DEPTH)
        )
        is_floor = is_floor | (grown & plane_like)

        return is_floor

    # ── Empty Fallback ────────────────────────────────────

    def _empty_result(self) -> AnalysisResult:
        empty_corridors = {
            name: CorridorMetrics(
                name=name, zone_index=i,
                valid_ratio=0.0, p20_depth=0.0, thin_depth=0.0, p25_depth=0.0,
                p50_depth=0.0, mean_depth=0.0,
                close_obstacle_ratio=1.0, danger_obstacle_ratio=1.0, emergency_ratio=1.0,
                largest_close_blob_px=0, vertical_close_run_frac=0.0,
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
