# safe_direction.py
"""
Unified Depth & AI Navigation Engine.

Fuses YOLO obstacle detections with depth analysis.
Splits the ROI into LEFT / CENTER / RIGHT zones.
Depth provides the baseline clearance.
YOLO detections (if any in that zone) penalize the zone's score drastically
based on their proximity.

Outputs one stable command: FORWARD / LEFT / RIGHT / STOP.
"""

import cv2
import numpy as np
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


# ============================================================
# Configuration  (distances in millimeters)
# ============================================================

MIN_DEPTH_MM = 300
MAX_DEPTH_MM = 5000

# Navigation thresholds
STOP_DISTANCE_MM = 700
TURN_DISTANCE_MM = 1200
FORWARD_CLEAR_MM = 1600

# ROI settings
TOP_CROP_RATIO = 0.45
BOTTOM_CROP_RATIO = 0.95
SIDE_MARGIN_RATIO = 0.03

# Strong Smoothing & Hysteresis
HISTORY_SIZE = 5
MIN_STABLE_COUNT = 3
HYSTERESIS_STOP_CLEAR_COUNT = 4  # How many FORWARDs needed to exit STOP


# ============================================================
# Data structures
# ============================================================

@dataclass
class ZoneMetrics:
    name: str
    valid_ratio: float
    p25_depth: float
    mean_depth: float
    close_turn_ratio: float
    close_stop_ratio: float
    yolo_penalty: float  # Added by YOLO detections
    score: float

@dataclass
class YoloObstacle:
    label: str
    distance_mm: float
    cx_ratio: float  # Center X ratio [0.0, 1.0] of depth frame width


# ============================================================
# Engine
# ============================================================

class SafeDirectionEngine:
    def __init__(self) -> None:
        self.history: deque = deque(maxlen=HISTORY_SIZE)
        self.last_stable_command: str = "STOP"

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def analyze_scene(
        self, depth_frame: np.ndarray, yolo_obstacles: List[YoloObstacle]
    ) -> Tuple[str, Dict[str, ZoneMetrics], np.ndarray]:
        """
        Analyze depth frame alongside YOLO obstacles.
        Return (raw_command, zone_metrics_dict, visualisation_bgr).
        """
        if depth_frame is None or depth_frame.size == 0:
            return "STOP", {}, self._make_fallback_vis()

        depth = depth_frame.copy()
        valid_mask = (depth >= MIN_DEPTH_MM) & (depth <= MAX_DEPTH_MM)

        h, w = depth.shape
        y1 = int(h * TOP_CROP_RATIO)
        y2 = int(h * BOTTOM_CROP_RATIO)
        x1 = int(w * SIDE_MARGIN_RATIO)
        x2 = int(w * (1.0 - SIDE_MARGIN_RATIO))

        roi_depth = depth[y1:y2, x1:x2]
        roi_valid = valid_mask[y1:y2, x1:x2]

        zone_metrics = self._compute_zone_metrics(
            w, roi_depth, roi_valid, yolo_obstacles, x1, x2
        )
        raw_decision = self._decide(zone_metrics)
        vis = self._build_visualization(
            depth, zone_metrics, yolo_obstacles, raw_decision, (x1, y1, x2, y2)
        )

        return raw_decision, zone_metrics, vis

    def smooth_decision(self, raw_decision: str) -> str:
        """Apply temporal smoothing with hysteresis for STOP."""
        self.history.append(raw_decision)
        counts = Counter(self.history)
        
        # Determine the most frequent command
        best_cmd, best_count = counts.most_common(1)[0]

        # Hysteresis: If we are stopped, make it harder to start again
        if self.last_stable_command == "STOP" and best_cmd == "FORWARD":
            if best_count >= HYSTERESIS_STOP_CLEAR_COUNT:
                self.last_stable_command = best_cmd
        else:
            if best_count >= MIN_STABLE_COUNT:
                self.last_stable_command = best_cmd

        # Failsafe: if STOP is anywhere in recent history, and we're not heavily biased to FORWARD, stay STOP
        if "STOP" in self.history and self.last_stable_command == "FORWARD":
             if counts["FORWARD"] < HYSTERESIS_STOP_CLEAR_COUNT:
                 self.last_stable_command = "STOP"

        return self.last_stable_command

    # ----------------------------------------------------------
    # Zone analysis
    # ----------------------------------------------------------

    def _compute_zone_metrics(
        self,
        full_width: int,
        roi_depth: np.ndarray,
        roi_valid: np.ndarray,
        obstacles: List[YoloObstacle],
        roi_x1: int,
        roi_x2: int
    ) -> Dict[str, ZoneMetrics]:
        
        roi_w = roi_x2 - roi_x1
        third = roi_w // 3

        zones = {
            "LEFT": (0, third),
            "CENTER": (third, 2 * third),
            "RIGHT": (2 * third, roi_w),
        }

        # Map obstacles to zones based on their absolute X pixel
        zone_penalties = {"LEFT": 0.0, "CENTER": 0.0, "RIGHT": 0.0}
        for obs in obstacles:
            abs_x = int(obs.cx_ratio * full_width)
            rel_x = abs_x - roi_x1
            
            # Identify which zone it belongs to
            target_zone = "CENTER"
            if rel_x < third:
                target_zone = "LEFT"
            elif rel_x > 2 * third:
                target_zone = "RIGHT"
            
            # Closer obstacles = exponentially worse penalty
            if obs.distance_mm < STOP_DISTANCE_MM:
                pen = 1.0  # Instant block
            elif obs.distance_mm < TURN_DISTANCE_MM:
                pen = 0.5
            else:
                pen = 0.2
            
            zone_penalties[target_zone] = max(zone_penalties[target_zone], pen)

        metrics: Dict[str, ZoneMetrics] = {}

        for name, (xs, xe) in zones.items():
            z_depth = roi_depth[:, xs:xe]
            z_valid = roi_valid[:, xs:xe]

            valid_values = z_depth[z_valid]
            total_pixels = z_depth.size
            penalty = zone_penalties[name]

            if valid_values.size == 0:
                metrics[name] = ZoneMetrics(name, 0.0, 0.0, 0.0, 1.0, 1.0, penalty, 0.0)
                continue

            valid_ratio = valid_values.size / total_pixels
            p25_depth = float(np.percentile(valid_values, 25))
            mean_depth = float(np.mean(valid_values))
            close_turn_ratio = float(np.mean(valid_values < TURN_DISTANCE_MM))
            close_stop_ratio = float(np.mean(valid_values < STOP_DISTANCE_MM))

            depth_score = np.clip(
                (p25_depth - MIN_DEPTH_MM) / (MAX_DEPTH_MM - MIN_DEPTH_MM),
                0.0,
                1.0,
            )

            # Base score from depth analysis
            base_score = (
                0.55 * depth_score
                + 0.30 * (1.0 - close_turn_ratio)
                + 0.15 * valid_ratio
            )
            
            # Apply YOLO penalty subtractively
            final_score = max(0.0, base_score - penalty)

            metrics[name] = ZoneMetrics(
                name=name,
                valid_ratio=valid_ratio,
                p25_depth=p25_depth,
                mean_depth=mean_depth,
                close_turn_ratio=close_turn_ratio,
                close_stop_ratio=close_stop_ratio,
                yolo_penalty=penalty,
                score=final_score,
            )

        return metrics

    # ----------------------------------------------------------
    # Decision logic
    # ----------------------------------------------------------

    def _decide(self, m: Dict[str, ZoneMetrics]) -> str:
        left = m["LEFT"]
        center = m["CENTER"]
        right = m["RIGHT"]

        # A zone is completely blocked if depth says so OR YOLO sees something super close
        left_blocked = (left.p25_depth < TURN_DISTANCE_MM) or (left.close_turn_ratio > 0.40) or (left.yolo_penalty >= 1.0)
        center_blocked = (center.p25_depth < TURN_DISTANCE_MM) or (center.close_turn_ratio > 0.35) or (center.yolo_penalty >= 1.0)
        right_blocked = (right.p25_depth < TURN_DISTANCE_MM) or (right.close_turn_ratio > 0.40) or (right.yolo_penalty >= 1.0)

        # Emergency stop
        if center.yolo_penalty >= 1.0 or (
            center.close_stop_ratio > 0.28
            and left.close_stop_ratio > 0.18
            and right.close_stop_ratio > 0.18
        ):
            return "STOP"

        # Center is clearly open and not severely penalized
        if (
            not center_blocked
            and center.p25_depth > FORWARD_CLEAR_MM
            and center.close_turn_ratio < 0.18
            and center.valid_ratio > 0.20
            and center.yolo_penalty < 0.3
        ):
            return "FORWARD"

        # Evaluate best alternative
        if center_blocked:
            if left_blocked and right_blocked:
                return "STOP"
            if left.score > right.score + 0.05:
                return "LEFT"
            if right.score > left.score + 0.05:
                return "RIGHT"
            return "LEFT" if left.p25_depth >= right.p25_depth else "RIGHT"

        best_side_gap = abs(left.score - right.score)
        if best_side_gap > 0.22:
            return "LEFT" if left.score > right.score else "RIGHT"

        return "FORWARD"

    # ----------------------------------------------------------
    # Visualization helpers
    # ----------------------------------------------------------

    def _build_visualization(
        self,
        depth_frame: np.ndarray,
        metrics: Dict[str, ZoneMetrics],
        obstacles: List[YoloObstacle],
        raw_decision: str,
        roi_box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        depth_vis = self._colorize_depth(depth_frame)
        h, w, _ = depth_vis.shape

        x1, y1, x2, y2 = roi_box
        cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

        roi_w = x2 - x1
        third = roi_w // 3

        zone_colors = {
            "LEFT": (255, 200, 0),
            "CENTER": (0, 255, 255),
            "RIGHT": (255, 0, 255),
        }

        # Draw zone separators
        for i in range(1, 3):
            x = x1 + i * third
            cv2.line(depth_vis, (x, y1), (x, y2), (255, 255, 255), 2)

        # Draw metrics
        if metrics:
            zone_positions = {
                "LEFT": (x1 + 10, y1 + 30),
                "CENTER": (x1 + third + 10, y1 + 30),
                "RIGHT": (x1 + 2 * third + 10, y1 + 30),
            }

            for zone_name, pos in zone_positions.items():
                m = metrics[zone_name]
                color = zone_colors[zone_name]
                lines = [
                    zone_name,
                    f"p25: {int(m.p25_depth)} mm",
                    f"close: {m.close_turn_ratio:.2f}",
                    f"pen: {m.yolo_penalty:.2f}",
                    f"score: {m.score:.2f}",
                ]
                for i, txt in enumerate(lines):
                    cv2.putText(
                        depth_vis, txt, (pos[0], pos[1] + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
                    )

        # Draw YOLO obstacles
        for obs in obstacles:
            cx = int(obs.cx_ratio * w)
            cy = h // 2  # Approximate vertical center for visualization
            
            color = (0, 0, 255) if obs.distance_mm < STOP_DISTANCE_MM else (0, 165, 255)
            cv2.circle(depth_vis, (cx, cy), 15, color, -1)
            cv2.circle(depth_vis, (cx, cy), 15, (255, 255, 255), 2)
            cv2.putText(
                depth_vis, f"{obs.label}", (cx - 20, cy - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA
            )

        # Draw raw decision
        decision_color = {
            "FORWARD": (0, 255, 0),
            "LEFT": (0, 165, 255),
            "RIGHT": (255, 100, 0),
            "STOP": (0, 0, 255),
        }.get(raw_decision, (255, 255, 255))

        cv2.putText(
            depth_vis, f"RAW: {raw_decision}", (20, h - 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, decision_color, 3, cv2.LINE_AA
        )

        return depth_vis

    def _colorize_depth(self, depth_frame: np.ndarray) -> np.ndarray:
        clipped = np.clip(depth_frame, MIN_DEPTH_MM, MAX_DEPTH_MM)
        norm = (
            (clipped - MIN_DEPTH_MM) / (MAX_DEPTH_MM - MIN_DEPTH_MM) * 255.0
        ).astype(np.uint8)
        norm = 255 - norm
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def _make_fallback_vis(self) -> np.ndarray:
        img = np.zeros((400, 640, 3), dtype=np.uint8)
        cv2.putText(
            img, "No depth frame", (180, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
        )
        return img
