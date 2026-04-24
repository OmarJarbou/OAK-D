# safe_direction.py - DEPRECATED
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
STOP_DISTANCE_MM = 800
TURN_DISTANCE_MM = 1500
FORWARD_CLEAR_MM = 1800

# ROI settings
TOP_CROP_RATIO = 0.45
BOTTOM_CROP_RATIO = 0.95
SIDE_MARGIN_RATIO = 0.03

# Strong Smoothing & Hysteresis
HISTORY_SIZE = 5
MIN_STABLE_COUNT = 3
HYSTERESIS_STOP_CLEAR_COUNT = 4  # How many FORWARDs needed to exit STOP
TURN_BIAS_BONUS = 0.40  # Bonus score given to maintain a turn


# ============================================================
# Data structures
# ============================================================

@dataclass
class ZoneMetrics:
    name: str
    valid_ratio: float
    p5_depth: float    # Absolute closest structures
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

        floor_mask = self._remove_floor(roi_depth, roi_valid)
        roi_valid = roi_valid & (~floor_mask)

        zone_metrics = self._compute_zone_metrics(
            w, roi_depth, roi_valid, yolo_obstacles, x1, x2
        )
        raw_decision = self._decide(zone_metrics)
        vis = self._build_visualization(
            depth, zone_metrics, yolo_obstacles, raw_decision, (x1, y1, x2, y2), floor_mask
        )

        return raw_decision, zone_metrics, vis

    def smooth_decision(self, raw_decision: str) -> str:
        """Apply temporal smoothing with hysteresis for STOP."""
        self.history.append(raw_decision)
        counts = Counter(self.history)
        
        # Determine the most frequent command
        best_cmd, best_count = counts.most_common(1)[0]
        
        # Directional stickiness: if we were turning, require overwhelming evidence to switch back
        if self.last_stable_command in ["LEFT", "RIGHT"] and best_cmd != self.last_stable_command:
            # Only switch away from a turn if the new command is STOP or FORWARD(clearly open)
            if best_cmd not in ["STOP", "FORWARD"] or counts[best_cmd] < HYSTERESIS_STOP_CLEAR_COUNT:
                best_cmd = self.last_stable_command

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

    def _remove_floor(self, roi_depth: np.ndarray, roi_valid: np.ndarray) -> np.ndarray:
        """
        Calculates a dynamic mask of the floor by relying on the vertical depth gradient.
        Returns a boolean mask where True corresponds to the floor.
        """
        h, w = roi_depth.shape
        is_floor = np.zeros((h, w), dtype=bool)
        if h < 2: return is_floor
        
        # The very bottom row is assumed to be floor if its depth is typical for a cart's ground base
        is_floor[h-1, :] = roi_valid[h-1, :] & (roi_depth[h-1, :] > 300) & (roi_depth[h-1, :] < 3000)
        
        # Raycast vertically from bottom to top
        for y in range(h-2, -1, -1):
            # diff > 0 means the row above is FURTHER than the row below (Property of Ground)
            diff = roi_depth[y, :] - roi_depth[y+1, :]
            
            # Floor continues if the row above is smoothly further away
            # We accept a tight gradient threshold to reject noisy obstacles
            continues_floor = is_floor[y+1, :] & (diff > 1) & (diff < 150)
            is_floor[y, :] = continues_floor
            
        return is_floor

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
                metrics[name] = ZoneMetrics(name, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, penalty, 0.0)
                continue

            valid_ratio = valid_values.size / total_pixels
            p5_depth = float(np.percentile(valid_values, 5))
            p25_depth = float(np.percentile(valid_values, 25))
            mean_depth = float(np.mean(valid_values))
            close_turn_ratio = float(np.mean(valid_values < TURN_DISTANCE_MM))
            close_stop_ratio = float(np.mean(valid_values < STOP_DISTANCE_MM))

            depth_score = np.clip(
                (p25_depth - MIN_DEPTH_MM) / (MAX_DEPTH_MM - MIN_DEPTH_MM),
                0.0,
                1.0,
            )

            # Repulsion score heavily weights the closest objects to push the cart away
            repulsion_score = np.clip((p5_depth - MIN_DEPTH_MM) / (MAX_DEPTH_MM - MIN_DEPTH_MM), 0, 1.0)

            # Base score from depth analysis
            base_score = (
                0.40 * repulsion_score
                + 0.30 * depth_score
                + 0.20 * (1.0 - close_turn_ratio)
                + 0.10 * valid_ratio
            )
            
            # Apply YOLO penalty subtractively
            final_score = max(0.0, base_score - penalty)
            
            # Directional hysteresis applied to score
            if self.last_stable_command == name:
                final_score += TURN_BIAS_BONUS

            metrics[name] = ZoneMetrics(
                name=name,
                valid_ratio=valid_ratio,
                p5_depth=p5_depth,
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

        # 1. Dead-End Check (True STOP)
        # Are we fundamentally trapped in all directions?
        if (center.p5_depth < STOP_DISTANCE_MM and left.p5_depth < STOP_DISTANCE_MM and right.p5_depth < STOP_DISTANCE_MM) or center.yolo_penalty >= 1.0:
            return "STOP"

        # 2. Corridor Navigation (Threading the Needle)
        # If left and right have obstacles, but center extends significantly past them
        # Note: Center must still be safe enough to not instantly crash ( > STOP_DISTANCE_MM)
        if center.p5_depth > STOP_DISTANCE_MM + 100:
            # If center is clearer than both side walls, commit to FORWARD to thread the needle
            if center.p5_depth > left.p5_depth + 300 and center.p5_depth > right.p5_depth + 300:
                if center.yolo_penalty < 0.5:
                    return "FORWARD"

        # 3. Obstacle Evasion (Handling Walls/Blocks ahead)
        # If center is starting to close in, we evaluate turning rather than blindly stopping
        if center.p5_depth < TURN_DISTANCE_MM or center.close_turn_ratio > 0.30:
            # We must pick the side with the most open space. 
            # We integrate the stable hysteresis score to prevent wiggling, but hard-override if one physical path is significantly wider
            
            # If one path has way more clearance, force a turn in that direction
            if left.p5_depth > right.p5_depth + 300:
                return "LEFT"
            if right.p5_depth > left.p5_depth + 300:
                return "RIGHT"
                
            # If physical spaces are roughly similar, trust the mathematically enhanced score (which includes hysteresis stickiness)
            return "LEFT" if left.score >= right.score else "RIGHT"

        # 4. Clear Path Default
        # Let's ensure turning isn't accidentally locked in if the center is totally clear
        if center.p5_depth > FORWARD_CLEAR_MM and center.yolo_penalty < 0.3:
             return "FORWARD"

        # 5. Fallback Soft-Centering
        # If we have slight variance ahead, lightly drift left/right to stay in the widest part of the path
        best_gap = abs(left.score - right.score)
        if best_gap > 0.20:
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
        floor_mask: np.ndarray = None
    ) -> np.ndarray:
        depth_vis = self._colorize_depth(depth_frame)
        h, w, _ = depth_vis.shape

        x1, y1, x2, y2 = roi_box
        
        # Erase the floor in the visualization so the user sees the AI ignoring it
        if floor_mask is not None:
            roi_vis = depth_vis[y1:y2, x1:x2]
            roi_vis[floor_mask] = [0, 0, 0]
            
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
                    f"p5: {int(m.p5_depth)} mm",
                    f"p25: {int(m.p25_depth)} mm",
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
