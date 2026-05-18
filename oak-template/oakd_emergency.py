#!/usr/bin/env python3
"""
Emergency stop using OAK-D depth.
Checks only the central region. Returns True if obstacle closer than threshold.
"""
import numpy as np
from utils.config import WalkerConfig


class OakdEmergency:
    def __init__(self, cfg: WalkerConfig):
        self.cfg = cfg
        self.emergency_mm = cfg.EMERGENCY_STOP_MM   # 600 mm default

    def check(self, depth_frame):
        """Return True if an obstacle is too close in front."""
        if depth_frame is None or depth_frame.size == 0:
            return False

        h, w = depth_frame.shape
        # Central crop (40% of image width and height)
        cx, cy = w // 2, h // 2
        crop_w, crop_h = w // 3, h // 3
        x1 = max(0, cx - crop_w//2)
        x2 = min(w, cx + crop_w//2)
        y1 = max(0, cy - crop_h//2)
        y2 = min(h, cy + crop_h//2)

        center_region = depth_frame[y1:y2, x1:x2]
        valid = center_region[(center_region > self.cfg.MIN_DEPTH_MM) &
                              (center_region < self.cfg.MAX_DEPTH_MM)]
        if len(valid) == 0:
            return False
        min_depth = np.min(valid)
        return min_depth < self.emergency_mm
