"""
Simplified 3-zone navigation.

Camera → LEFT / CENTER / RIGHT steering by p20 depth.
LiDAR  → front emergency STOP only (no side steering).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from utils.config import WalkerConfig

ZONE_NAMES = ("LEFT", "CENTER", "RIGHT")
MIN_VALID_PIXELS = 50
INVALID_ZONE_DEPTH = 9999.0


@dataclass
class SimpleAnalysis:
    metrics: Dict[str, float]
    roi_box: Tuple[int, int, int, int]


@dataclass
class SimpleDecision:
    raw_command: str
    stable_command: str
    reason: str


def analyze_simple(depth_frame: np.ndarray, cfg: WalkerConfig) -> Optional[SimpleAnalysis]:
    """Split ROI into 3 vertical zones; return p20 depth (mm) per zone."""
    h, w = depth_frame.shape[:2]
    y1 = int(h * cfg.TOP_CROP_RATIO)
    y2 = int(h * cfg.BOTTOM_CROP_RATIO)
    x1 = int(w * cfg.SIDE_MARGIN_RATIO)
    x2 = int(w * (1.0 - cfg.SIDE_MARGIN_RATIO))
    roi = depth_frame[y1:y2, x1:x2]
    valid = (roi >= cfg.MIN_DEPTH_MM) & (roi <= cfg.MAX_DEPTH_MM)
    if not np.any(valid):
        return None

    zone_w = roi.shape[1] // 3
    metrics: Dict[str, float] = {}
    for i, name in enumerate(ZONE_NAMES):
        zx1 = i * zone_w
        zx2 = roi.shape[1] if i == 2 else (i + 1) * zone_w
        z_valid = valid[:, zx1:zx2]
        z_depth = roi[:, zx1:zx2][z_valid]
        if z_depth.size < MIN_VALID_PIXELS or np.mean(z_valid) < 0.15:
            metrics[name] = 0.0
        else:
            metrics[name] = float(np.percentile(z_depth, 20))

    return SimpleAnalysis(metrics=metrics, roi_box=(x1, y1, x2, y2))


def _norm_depth(v: float) -> float:
    """Treat missing/blocked zones as far so we do not STOP by mistake."""
    return INVALID_ZONE_DEPTH if v <= 0 else v


def decide_simple(
    metrics: Dict[str, float],
    lidar_front_mm: Optional[float],
    cfg: WalkerConfig,
) -> Tuple[str, str]:
    """
    Pick GO:LEFT / GO:CENTER / GO:RIGHT or STOP.
    STOP only when front blocked AND no side escape.
    """
    stop_mm = cfg.STOP_DISTANCE_MM
    min_side = cfg.MIN_SIDE_DEPTH_MM

    if lidar_front_mm is not None and lidar_front_mm > 0 and lidar_front_mm < stop_mm:
        return "STOP", f"LiDAR front {lidar_front_mm:.0f}mm < {stop_mm:.0f}mm"

    center = _norm_depth(metrics.get("CENTER", 0.0))
    left = _norm_depth(metrics.get("LEFT", 0.0))
    right = _norm_depth(metrics.get("RIGHT", 0.0))
    center_biased = center + cfg.CENTER_BIAS_MM

    if (
        center < stop_mm
        and left < min_side
        and right < min_side
    ):
        return "STOP", (
            f"Front blocked ({metrics.get('CENTER', 0):.0f}mm) "
            f"no side path (L={metrics.get('LEFT', 0):.0f} R={metrics.get('RIGHT', 0):.0f})"
        )

    if center_biased >= left and center_biased >= right:
        return "GO:CENTER", f"Center clear ({center:.0f}mm biased {center_biased:.0f})"
    if left > right:
        return "GO:LEFT", f"Left clearer ({left:.0f} vs {right:.0f}mm)"
    return "GO:RIGHT", f"Right clearer ({right:.0f} vs {left:.0f}mm)"


def apply_flip_lr(command: str, flip: bool) -> str:
    if not flip:
        return command
    swap = {"GO:LEFT": "GO:RIGHT", "GO:RIGHT": "GO:LEFT"}
    return swap.get(command, command)


class SimpleSmoother:
    """Frame hysteresis + cooldown before accepting a new stable command."""

    def __init__(self, hysteresis: int = 3, cooldown_ms: float = 500.0):
        self.hysteresis = max(1, hysteresis)
        self.cooldown_s = cooldown_ms / 1000.0
        self._history: list[str] = []
        self.stable_command: str = "NONE"
        self._last_change_time: float = 0.0

    def reset(self) -> None:
        self._history.clear()
        self.stable_command = "NONE"
        self._last_change_time = 0.0

    def update(self, raw_cmd: str) -> SimpleDecision:
        self._history.append(raw_cmd)
        if len(self._history) > self.hysteresis:
            self._history.pop(0)

        candidate = max(set(self._history), key=self._history.count)
        now = time.time()

        if candidate != self.stable_command:
            if self.stable_command == "NONE":
                self.stable_command = candidate
                self._last_change_time = now
            elif (now - self._last_change_time) >= self.cooldown_s:
                self.stable_command = candidate
                self._last_change_time = now

        return SimpleDecision(
            raw_command=raw_cmd,
            stable_command=self.stable_command,
            reason=f"stable={self.stable_command} raw={raw_cmd}",
        )
