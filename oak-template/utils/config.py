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
    WALKER_WIDTH_M: float = 0.64
    SIDE_MARGIN_M: float = 0.15

    @property
    def REQUIRED_CLEAR_WIDTH_M(self) -> float:
        return self.WALKER_WIDTH_M + 2 * self.SIDE_MARGIN_M

    # ── OAK-D Camera ─────────────────────────────────────────
    HFOV_DEG: float = 71.9          # OAK-D horizontal field of view
    DEPTH_WIDTH: int = 640          # Stereo output width
    DEPTH_HEIGHT: int = 400         # Stereo output height

    @property
    def HFOV_RAD(self) -> float:
        return math.radians(self.HFOV_DEG)

    # ── Depth Thresholds (millimeters) ────────────────────────
    MIN_DEPTH_MM: int = 300         # Ignore closer (noise)
    MAX_DEPTH_MM: int = 5000        # Ignore further (irrelevant)
    EMERGENCY_STOP_MM: int = 600    # Very close → emergency stop
    CLOSE_OBSTACLE_MM: int = 1200   # Nearby obstacle warning
    SAFE_CORRIDOR_MM: int = 1800    # Min depth to consider "safe"

    # ── ROI Crop Ratios ──────────────────────────────────────
    TOP_CROP_RATIO: float = 0.40    # Skip upper 40% (ceiling/sky)
    BOTTOM_CROP_RATIO: float = 0.95 # Skip bottom 5% (walker frame)
    SIDE_MARGIN_RATIO: float = 0.03 # Trim 3% each side

    # ── Corridor / Zone Configuration ────────────────────────
    NUM_ZONES: int = 7
    ZONE_NAMES: List[str] = field(default_factory=lambda: [
        "LEFT", "L2", "L1", "CENTER", "R1", "R2", "RIGHT"
    ])
    # Maps zone name → Arduino GO command suffix
    ZONE_TO_CMD: dict = field(default_factory=lambda: {
        "LEFT":   "GO:LEFT",
        "L2":     "GO:L2",
        "L1":     "GO:L1",
        "CENTER": "GO:CENTER",
        "R1":     "GO:R1",
        "R2":     "GO:R2",
        "RIGHT":  "GO:RIGHT",
    })

    # ── Corridor Scoring Weights ─────────────────────────────
    WEIGHT_DEPTH_P20: float = 0.30      # 20th percentile depth score
    WEIGHT_CLEAR_WIDTH: float = 0.25    # Physical width score
    WEIGHT_CLOSE_OBS: float = 0.20      # Penalty for close obstacles
    WEIGHT_VALID_RATIO: float = 0.10    # Data quality bonus
    WEIGHT_CENTER_PREF: float = 0.15    # Center preference bias

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
    SAFETY_W_DEPTH: float = 0.35        # p25 depth normalised
    SAFETY_W_CLOSE_OBS: float = 0.25    # 1 - close_obstacle_ratio
    SAFETY_W_VALID: float = 0.25        # valid_ratio
    SAFETY_W_FLOOR_INV: float = 0.15    # 1 - floor/invalid ratio
    # CENTER must be within this fraction of best zone to be chosen
    CENTER_ACCEPT_RATIO: float = 0.85

    # Minimum requirements for a corridor to be "passable"
    MIN_VALID_RATIO: float = 0.25       # At least 25% valid pixels
    MIN_P20_FOR_PASS: float = 700.0     # p20 must exceed emergency zone

    # ── Temporal Smoothing ───────────────────────────────────
    HISTORY_SIZE: int = 7               # Frames of decision history
    MIN_STABLE_COUNT: int = 4           # Votes needed to accept command
    STOP_CLEAR_COUNT: int = 5           # FORWARDs needed to exit STOP
    TURN_STICK_COUNT: int = 3           # Votes to break out of a turn

    # ── Command Rate Limiting ────────────────────────────────
    MIN_COMMAND_INTERVAL_S: float = 0.15    # Min time between any sends
    COMMAND_REFRESH_S: float = 2.0          # Re-send same command interval
    STOP_REPEAT_INTERVAL_S: float = 1.0     # Re-send STOP more often

    # ── STOP Recovery ────────────────────────────────────────────
    STOP_HOLD_SECONDS: float = 1.5
    CRITICAL_STOP_DISTANCE_MM: float = 600.0

    # ── Decision Hysteresis / Mode Switching ────────────────
    STOP_FRAMES_REQUIRED: int = 4
    FREE_FRAMES_REQUIRED: int = 6
    GO_FRAMES_REQUIRED: int = 4
    COMMAND_CHANGE_COOLDOWN_S: float = 0.7
    MIN_COMMAND_HOLD_MS: float = 800.0   # minimum ms before any non-critical transition
    UNSAFE_CONF_THRESHOLD: float = 0.35
    FREE_CONF_THRESHOLD: float = 0.55
    FREE_CENTER_CLOSE_OBS_MAX: float = 0.10
    FREE_CENTER_MIN_VALID_RATIO: float = 0.40
    FREE_CENTER_MIN_P20_MM: float = 1600.0

    # ── FREE Mode Stability ───────────────────────────────────────
    FREE_STABLE_FRAMES: int = 6
    FREE_CLEAR_DISTANCE_MM: float = 1200.0
    FREE_STICKY_SECONDS: float = 3.0   # how long FREE resists GO:CENTER re-entry

    # ── Serial Communication ─────────────────────────────────
    ARDUINO_PORT: str = "MOCK"
    ARDUINO_BAUD: int = 9600

    # ── Visualization / Optional Features ────────────────────
    DEBUG_DISPLAY: bool = True
    USE_TTS: bool = True
    ENABLE_SNAPS: bool = False

    # ── Floor Removal ────────────────────────────────────────
    FLOOR_GRADIENT_MIN: int = 1
    FLOOR_GRADIENT_MAX: int = 150
    FLOOR_MIN_DEPTH: int = 300
    FLOOR_MAX_DEPTH: int = 3000

    @classmethod
    def from_env(cls) -> "WalkerConfig":
        """Create config from environment variables with sensible defaults."""
        return cls(
            WALKER_WIDTH_M=float(os.getenv("WALKER_WIDTH_M", "0.64")),
            SIDE_MARGIN_M=float(os.getenv("SIDE_MARGIN_M", "0.15")),
            ARDUINO_PORT=os.getenv("ARDUINO_PORT", "MOCK"),
            ARDUINO_BAUD=int(os.getenv("ARDUINO_BAUD", "9600")),
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
            FREE_CENTER_MIN_P20_MM=float(
                os.getenv("FREE_CENTER_MIN_P20_MM", "1600.0")
            ),
            FREE_STABLE_FRAMES=int(
                os.getenv("FREE_STABLE_FRAMES", "6")
            ),
            FREE_CLEAR_DISTANCE_MM=float(
                os.getenv("FREE_CLEAR_DISTANCE_MM", "1200.0")
            ),
            MIN_COMMAND_HOLD_MS=float(
                os.getenv("MIN_COMMAND_HOLD_MS", "800.0")
            ),
        )
