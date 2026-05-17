# utils/config.py
"""
Central configuration for the Smart Walker OAK-D navigation system.
All tunable parameters are defined here — no magic numbers elsewhere.
"""

import os
import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class WalkerConfig:
    """All tunable parameters for the smart walker navigation system."""

    # ── Walker Physical Dimensions ────────────────────────────
    # Required free width = walker body + safety margin on both sides (eligible corridor).
    WALKER_WIDTH_M: float = 0.64
    SIDE_MARGIN_M: float = 0.15

    @property
    def REQUIRED_CLEAR_WIDTH_M(self) -> float:
        return self.WALKER_WIDTH_M + 2 * self.SIDE_MARGIN_M

    # ── OAK-D Camera ─────────────────────────────────────────
    HFOV_DEG: float = 71.9  # OAK-D horizontal field of view
    DEPTH_WIDTH: int = 640  # Stereo output width
    DEPTH_HEIGHT: int = 400  # Stereo output height

    @property
    def HFOV_RAD(self) -> float:
        return math.radians(self.HFOV_DEG)

    # ── Depth Thresholds (millimeters) ────────────────────────
    MIN_DEPTH_MM: int = 300  # Ignore closer (noise)
    MAX_DEPTH_MM: int = 5000  # Ignore further (irrelevant)
    EMERGENCY_STOP_MM: int = 600  # Very close → emergency stop
    CLOSE_OBSTACLE_MM: int = 1200  # Nearby obstacle warning
    SAFE_CORRIDOR_MM: int = 1800  # Min depth to consider "safe"
    # Low percentile for clearance (catches thin poles: p20 stays “far”, tail is close)
    THIN_OBSTACLE_PERCENTILE: float = 2.0

    # ── ROI Crop Ratios ──────────────────────────────────────
    TOP_CROP_RATIO: float = 0.40  # Skip upper 40% (ceiling/sky)
    BOTTOM_CROP_RATIO: float = 0.95  # Skip bottom 5% (walker frame)
    SIDE_MARGIN_RATIO: float = 0.20  # Trim 8% each side

    # ── Corridor / Zone Configuration ────────────────────────
    NUM_ZONES: int = 7
    ZONE_NAMES: List[str] = field(
        default_factory=lambda: ["LEFT", "L2", "L1", "CENTER", "R1", "R2", "RIGHT"]
    )
    # Maps zone name -> Arduino GO command suffix
    ZONE_TO_CMD: dict = field(
        default_factory=lambda: {
            "LEFT": "GO:LEFT",
            "L2": "GO:L2",
            "L1": "GO:L1",
            "CENTER": "GO:CENTER",
            "R1": "GO:R1",
            "R2": "GO:R2",
            "RIGHT": "GO:RIGHT",
        }
    )

    # Set FLIP_LR=True when the camera is mounted so its image-left = walker's right.
    # Swaps all LEFT<->RIGHT zone commands and mirrors the LiDAR side distances.
    FLIP_LR: bool = False

    # ── Corridor Scoring Weights ─────────────────────────────
    WEIGHT_DEPTH_P20: float = 0.30  # 20th percentile depth score
    WEIGHT_CLEAR_WIDTH: float = 0.25  # Physical width score
    WEIGHT_CLOSE_OBS: float = 0.20  # Penalty for close obstacles
    WEIGHT_VALID_RATIO: float = 0.10  # Data quality bonus
    WEIGHT_CENTER_PREF: float = 0.15  # Center preference bias

    # Center preference: how much to favor center zone (0 = no bias)
    CENTER_BIAS: float = 0.12
    # Adjacent zones get partial center bias
    ADJACENT_BIAS: float = 0.06

    # ── Safety-Score-Based Steering ───────────────────────────
    # Soft additive bonus for CENTER during target comparison only
    CENTER_SAFETY_BIAS: float = 0.10
    # Fraction of center zones (L1/CENTER/R1) that must be in emergency
    # before triggering emergency stop (0.50 = majority)
    EMERGENCY_CENTER_COVERAGE: float = 0.50
    # Minimum p20 depth (mm) for a side path to override emergency stop
    MIN_SIDE_DEPTH_MM: float = 800.0
    # Safety score weights (pure safety, no center bias)
    SAFETY_W_DEPTH: float = 0.35  # p25 depth normalised
    SAFETY_W_CLOSE_OBS: float = 0.25  # 1 - close_obstacle_ratio
    SAFETY_W_VALID: float = 0.25  # valid_ratio
    SAFETY_W_FLOOR_INV: float = 0.15  # 1 - floor/invalid ratio
    # CENTER must be within this fraction of best zone to be chosen
    CENTER_ACCEPT_RATIO: float = 0.85

    # Minimum requirements for a corridor to be "passable" (navigable = not floor)
    MIN_VALID_RATIO: float = 0.14  # Min fraction of navigable pixels in zone
    MIN_P20_FOR_PASS: float = 700.0  # p20 must exceed emergency zone

    # ── Temporal Smoothing ───────────────────────────────────
    HISTORY_SIZE: int = 7  # Frames of decision history
    MIN_STABLE_COUNT: int = 4  # Votes needed to accept command
    STOP_CLEAR_COUNT: int = 5  # FORWARDs needed to exit STOP
    TURN_STICK_COUNT: int = 3  # Votes to break out of a turn

    # ── Command Rate Limiting ────────────────────────────────
    MIN_COMMAND_INTERVAL_S: float = 0.15  # Min time between any sends
    COMMAND_REFRESH_S: float = 2.0  # Re-send same command interval
    STOP_REPEAT_INTERVAL_S: float = 1.0  # Re-send STOP more often

    # ── STOP Recovery ────────────────────────────────────────────
    STOP_HOLD_SECONDS: float = 1.5
    CRITICAL_STOP_DISTANCE_MM: float = 600.0

    # ── Decision Hysteresis / Mode Switching ────────────────
    STOP_FRAMES_REQUIRED: int = 4
    FREE_FRAMES_REQUIRED: int = 6
    GO_FRAMES_REQUIRED: int = 4
    COMMAND_CHANGE_COOLDOWN_S: float = 0.7
    MIN_COMMAND_HOLD_MS: float = 800.0  # minimum ms before any non-critical transition
    UNSAFE_CONF_THRESHOLD: float = 0.35
    FREE_CONF_THRESHOLD: float = 0.55
    FREE_CENTER_CLOSE_OBS_MAX: float = 0.10
    FREE_CENTER_MIN_VALID_RATIO: float = 0.40
    FREE_CENTER_MIN_P20_MM: float = 1600.0

    # ── FREE Mode Stability ───────────────────────────────────────
    FREE_STABLE_FRAMES: int = 6
    FREE_CLEAR_DISTANCE_MM: float = 1200.0
    FREE_STICKY_SECONDS: float = 3.0  # how long FREE resists GO:CENTER re-entry
    SIDE_PREFER_MARGIN: float = 0.15  # side wins if safety beats CENTER by this much
    POST_RECOVERY_GRACE_S: float = 3.0  # grace after STOP->FREE recovery
    FREE_EXIT_BAD_FRAMES: int = 2  # consecutive "not free" frames required to exit FREE

    # Recenter after lateral GO:* when center path is open again
    RECENTER_MIN_P20_MM: float = 900.0
    RECENTER_SAFETY_GAP: float = 0.10  # prefer CENTER if within this of side safety
    # Require N consecutive low-confidence frames before STOP (reduces random stops)
    UNSAFE_CONF_STREAK_FRAMES: int = 3
    # When side beats center by this much, commit direction in fewer frames
    GO_FAST_MARGIN: float = 0.22
    GO_FAST_FRAMES_REQUIRED: int = 2

    # ── Lightweight Temporal Smoothing (DecisionEngine only) ─────────
    # EMA applied to p20_depth and close_obstacle_ratio per zone
    TEMP_EMA_ALPHA: float = 0.25
    TEMP_EMA_WARMUP_FRAMES: int = 4

    # ── Serial Communication ─────────────────────────────────
    ARDUINO_PORT: str = "MOCK"
    ARDUINO_BAUD: int = 9600

    # ── LiDAR Configuration ───────────────────────────────────
    LIDAR_PORT: str = "MOCK"
    # 460800 = RPLIDAR C1 (rplidarc1). Legacy A-series often uses 115200.
    LIDAR_BAUD: int = 460800
    LIDAR_LEGACY_BAUD: int = 115200
    # auto | c1 | legacy — auto tries rplidarc1 first, then legacy rplidar.
    LIDAR_BACKEND: str = "auto"
    LIDAR_SAFETY_MM: float = 600.0
    LIDAR_SIDE_ESCAPE_MM: float = 800.0
    LIDAR_FRONT_ARC_DEG: float = 30.0
    LIDAR_SIDE_ARC_START_DEG: float = 30.0
    LIDAR_SIDE_ARC_END_DEG: float = 90.0
    LIDAR_SCAN_TIMEOUT_S: float = 0.5
    # Height compensation: LiDAR (30cm) veto ignored when camera front >
    # this value — camera sees clear path, LiDAR seeing low obstacle.
    LIDAR_CAMERA_AGREE_MM: float = 1200.0
    # Raised from 800 → 2000: camera emergency is trusted up to 2 m even
    # when LiDAR reports clear.  Prevents walker driving into real obstacles.
    FUSION_DISAGREEMENT_THRESHOLD_MM: float = 2000.0

    # ── LiDAR Side-Distance Directional Bias ─────────────────
    # Applied only when OAK-D confidence < this threshold.
    LIDAR_SIDE_BIAS_CONF_THRESHOLD: float = 0.50
    # Minimum mm gap between left_min and right_min to apply a preference.
    # Prevents noise from flipping direction when both sides are almost equal.
    LIDAR_SIDE_BIAS_MM_MIN_GAP: float = 300.0
    # Additive safety-score bonus applied to candidates on the preferred side.
    # Kept small so it only breaks near-ties, not overrides strong OAK-D scores.
    LIDAR_SIDE_BIAS_BONUS: float = 0.08

    # ── Visualization / Optional Features ────────────────────
    DEBUG_DISPLAY: bool = True
    USE_TTS: bool = True
    ENABLE_SNAPS: bool = False

    # ── Floor / ground (traversable) vs obstacle ─────────────
    # Navigation uses raw depth (mm) only — never depth colormap RGB.
    FLOOR_GRADIENT_MIN: int = 1
    FLOOR_GRADIENT_MAX: int = 180
    FLOOR_MIN_DEPTH: int = 300
    FLOOR_MAX_DEPTH: int = 3200
    # Lower fraction of ROI where we aggressively merge floor-like regions (flat floor).
    GROUND_LOWER_BAND_RATIO: float = 0.42
    FLOOR_DILATE_KERNEL_V: int = 13
    FLOOR_DILATE_KERNEL_H: int = 9
    FLOOR_DILATE_ITERATIONS: int = 2
    # Compact / vertical obstacle detection (on navigable = valid & not floor)
    MIN_OBSTACLE_CLUSTER_PX: int = 72
    VERTICAL_CLOSE_RUN_FRAC: float = 0.20
    # “Spread” close obstacle: high emergency-pixel ratio on navigable surface
    SPREAD_EMERGENCY_RATIO: float = 0.28
    MIN_NAVIGABLE_RATIO: float = 0.10

    @classmethod
    def from_env(cls) -> "WalkerConfig":
        """Create config from environment variables with sensible defaults."""
        return cls(
            WALKER_WIDTH_M=float(os.getenv("WALKER_WIDTH_M", "0.64")),
            SIDE_MARGIN_M=float(os.getenv("SIDE_MARGIN_M", "0.15")),
            ARDUINO_PORT=os.getenv("ARDUINO_PORT", "MOCK"),
            ARDUINO_BAUD=int(os.getenv("ARDUINO_BAUD", "9600")),
            LIDAR_PORT=os.getenv("LIDAR_PORT", "MOCK"),
            LIDAR_BAUD=int(os.getenv("LIDAR_BAUD", "460800")),
            LIDAR_LEGACY_BAUD=int(os.getenv("LIDAR_LEGACY_BAUD", "115200")),
            LIDAR_BACKEND=os.getenv("LIDAR_BACKEND", "auto"),
            LIDAR_SAFETY_MM=float(os.getenv("LIDAR_SAFETY_MM", "600.0")),
            LIDAR_SIDE_ESCAPE_MM=float(os.getenv("LIDAR_SIDE_ESCAPE_MM", "800.0")),
            LIDAR_CAMERA_AGREE_MM=float(os.getenv("LIDAR_CAMERA_AGREE_MM", "1200.0")),
            FUSION_DISAGREEMENT_THRESHOLD_MM=float(
                os.getenv("FUSION_DISAGREEMENT_THRESHOLD_MM", "2000.0")
            ),
            LIDAR_SIDE_BIAS_CONF_THRESHOLD=float(
                os.getenv("LIDAR_SIDE_BIAS_CONF_THRESHOLD", "0.50")
            ),
            LIDAR_SIDE_BIAS_MM_MIN_GAP=float(
                os.getenv("LIDAR_SIDE_BIAS_MM_MIN_GAP", "300.0")
            ),
            LIDAR_SIDE_BIAS_BONUS=float(os.getenv("LIDAR_SIDE_BIAS_BONUS", "0.08")),
            DEBUG_DISPLAY=os.getenv("DEBUG_DISPLAY", "1") == "1",
            USE_TTS=os.getenv("USE_TTS", "1") == "1",
            ENABLE_SNAPS=os.getenv("ENABLE_SNAPS", "0") == "1",
            STOP_FRAMES_REQUIRED=int(os.getenv("STOP_FRAMES_REQUIRED", "4")),
            FREE_FRAMES_REQUIRED=int(os.getenv("FREE_FRAMES_REQUIRED", "6")),
            GO_FRAMES_REQUIRED=int(os.getenv("GO_FRAMES_REQUIRED", "4")),
            COMMAND_CHANGE_COOLDOWN_S=float(
                os.getenv("COMMAND_CHANGE_COOLDOWN_S", "0.7")
            ),
            UNSAFE_CONF_THRESHOLD=float(os.getenv("UNSAFE_CONF_THRESHOLD", "0.35")),
            FREE_CONF_THRESHOLD=float(os.getenv("FREE_CONF_THRESHOLD", "0.55")),
            FREE_CENTER_CLOSE_OBS_MAX=float(
                os.getenv("FREE_CENTER_CLOSE_OBS_MAX", "0.10")
            ),
            FREE_CENTER_MIN_VALID_RATIO=float(
                os.getenv("FREE_CENTER_MIN_VALID_RATIO", "0.40")
            ),
            FREE_CENTER_MIN_P20_MM=float(os.getenv("FREE_CENTER_MIN_P20_MM", "1600.0")),
            FREE_STABLE_FRAMES=int(os.getenv("FREE_STABLE_FRAMES", "6")),
            FREE_CLEAR_DISTANCE_MM=float(os.getenv("FREE_CLEAR_DISTANCE_MM", "1200.0")),
            FREE_STICKY_SECONDS=float(os.getenv("FREE_STICKY_SECONDS", "3.0")),
            SIDE_PREFER_MARGIN=float(os.getenv("SIDE_PREFER_MARGIN", "0.15")),
            POST_RECOVERY_GRACE_S=float(os.getenv("POST_RECOVERY_GRACE_S", "3.0")),
            CENTER_SAFETY_BIAS=float(os.getenv("CENTER_SAFETY_BIAS", "0.10")),
            CENTER_ACCEPT_RATIO=float(os.getenv("CENTER_ACCEPT_RATIO", "0.85")),
            STOP_HOLD_SECONDS=float(os.getenv("STOP_HOLD_SECONDS", "1.5")),
            CRITICAL_STOP_DISTANCE_MM=float(
                os.getenv("CRITICAL_STOP_DISTANCE_MM", "600.0")
            ),
            TEMP_EMA_ALPHA=float(os.getenv("TEMP_EMA_ALPHA", "0.25")),
            TEMP_EMA_WARMUP_FRAMES=int(os.getenv("TEMP_EMA_WARMUP_FRAMES", "4")),
            FREE_EXIT_BAD_FRAMES=int(os.getenv("FREE_EXIT_BAD_FRAMES", "2")),
            MIN_COMMAND_HOLD_MS=float(os.getenv("MIN_COMMAND_HOLD_MS", "800.0")),
            THIN_OBSTACLE_PERCENTILE=float(
                os.getenv("THIN_OBSTACLE_PERCENTILE", "2.0")
            ),
            RECENTER_MIN_P20_MM=float(os.getenv("RECENTER_MIN_P20_MM", "900.0")),
            RECENTER_SAFETY_GAP=float(os.getenv("RECENTER_SAFETY_GAP", "0.10")),
            UNSAFE_CONF_STREAK_FRAMES=int(os.getenv("UNSAFE_CONF_STREAK_FRAMES", "3")),
            GO_FAST_MARGIN=float(os.getenv("GO_FAST_MARGIN", "0.22")),
            GO_FAST_FRAMES_REQUIRED=int(os.getenv("GO_FAST_FRAMES_REQUIRED", "2")),
            FLIP_LR=os.getenv("FLIP_LR", "0") == "1",
        )

    def __post_init__(self) -> None:
        """Apply FLIP_LR: mirror all left<->right zone commands."""
        if self.FLIP_LR:
            flipped = {
                "LEFT":   "GO:RIGHT",
                "L2":     "GO:R2",
                "L1":     "GO:R1",
                "CENTER": "GO:CENTER",
                "R1":     "GO:L1",
                "R2":     "GO:L2",
                "RIGHT":  "GO:LEFT",
            }
            self.ZONE_TO_CMD = flipped
