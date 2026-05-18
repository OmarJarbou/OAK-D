#!/usr/bin/env python3
"""
Follow the Gap algorithm using LiDAR.
Outputs: "LEFT", "CENTER", "RIGHT" based on the largest safe gap.
"""
import time
import numpy as np
from utils.config import WalkerConfig
from utils.lidar_analyzer import LidarAnalyzer


class LidarGapFollower:
    def __init__(self, cfg: WalkerConfig, lidar: LidarAnalyzer):
        self.cfg = cfg
        self.lidar = lidar
        self.last_command = "CENTER"
        self.last_time = 0.0
        self.cooldown = 0.3  # seconds

    def get_command(self):
        """Return one of: 'LEFT', 'CENTER', 'RIGHT', or None if no LiDAR"""
        scan = self.lidar.latest_scan
        if scan is None:
            return None

        angles = scan.raw_angles
        dists = scan.raw_distances
        if len(angles) == 0:
            return None

        # 1. Focus on front arc (±75 degrees)
        front_mask = np.abs(angles) <= 75.0
        angles_front = angles[front_mask]
        dists_front = dists[front_mask]
        if len(angles_front) == 0:
            return None

        # 2. Inflate obstacles: set all points closer than SAFETY_DIST to 0
        safety_mm = self.cfg.LIDAR_SAFETY_MM
        dists_inflated = dists_front.copy()
        # Find the minimum distance in front
        min_idx = np.argmin(dists_front)
        min_angle = angles_front[min_idx]
        min_dist = dists_front[min_idx]

        # Inflate radius = safety margin + robot half-width (~0.3m)
        inflation_radius_deg = np.degrees(np.arctan2(0.35, min_dist / 1000.0))
        for i, ang in enumerate(angles_front):
            if abs(ang - min_angle) < inflation_radius_deg:
                dists_inflated[i] = 0.0

        # 3. Identify gaps: sequences where dists_inflated > safety_mm/2 (free)
        free = dists_inflated > (safety_mm / 2.0)
        gaps = []
        start = None
        for i, is_free in enumerate(free):
            if is_free and start is None:
                start = i
            elif not is_free and start is not None:
                gaps.append((start, i - 1))
                start = None
        if start is not None:
            gaps.append((start, len(free)-1))

        if not gaps:
            return "STOP"

        # 4. Pick the widest gap (in degrees)
        best_gap = max(gaps, key=lambda g: angles_front[g[1]] - angles_front[g[0]])
        start_deg = angles_front[best_gap[0]]
        end_deg = angles_front[best_gap[1]]
        center_deg = (start_deg + end_deg) / 2.0

        # 5. Convert angle to command
        if abs(center_deg) < 20.0:
            cmd = "CENTER"
        elif center_deg < 0:
            cmd = "LEFT"
        else:
            cmd = "RIGHT"

        # Cooldown to avoid rapid changes
        now = time.time()
        if cmd != self.last_command and now - self.last_time < self.cooldown:
            cmd = self.last_command
        else:
            self.last_command = cmd
            self.last_time = now

        return cmd
