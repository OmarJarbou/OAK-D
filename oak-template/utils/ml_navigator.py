# utils/ml_navigator.py
"""
ML-based navigation: depth CNN + LiDAR/depth safety + temporal smoothing.

Replaces CorridorAnalyzer + DecisionEngine when NAV_MODE=ml.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils.config import WalkerConfig
from utils.tiny_depth_navigator import (
    TinyDepthNavigator,
    IDX_TO_CLASS,
    CLASS_TO_IDX,
)

# Model labels → Arduino serial commands
LABEL_TO_ARDUINO = {
    "LEFT": "GO:L2",
    "CENTER": "GO:CENTER",
    "RIGHT": "GO:R2",
    "STOP": "STOP",
}


@dataclass
class MLDecisionResult:
    raw_command: str
    stable_command: str
    confidence: float
    model_label: str
    reason: str
    critical_stop: bool = False
    stable_count: int = 0
    allow_recenter: bool = False
    forward_min_depth_mm: float = 0.0


class SimpleSmoother:
    """Vote over recent raw Arduino commands."""

    def __init__(self, history_size: int, min_stable: int, stop_frames: int) -> None:
        self._history: deque[str] = deque(maxlen=history_size)
        self._min_stable = min_stable
        self._stop_frames = stop_frames
        self._last_stable = "STOP"

    def reset(self) -> None:
        self._history.clear()
        self._last_stable = "STOP"

    def smooth(self, raw_cmd: str) -> tuple[str, int]:
        self._history.append(raw_cmd)
        if not self._history:
            return self._last_stable, 0

        counts = Counter(self._history)
        top_cmd, top_count = counts.most_common(1)[0]

        stop_count = counts.get("STOP", 0)
        if stop_count >= self._stop_frames:
            self._last_stable = "STOP"
            return "STOP", stop_count

        if top_count >= self._min_stable:
            self._last_stable = top_cmd
            return top_cmd, top_count

        return self._last_stable, top_count


class MLNavigator:
    def __init__(self, cfg: WalkerConfig) -> None:
        self.cfg = cfg
        self._device = torch.device("cpu")
        self._model = TinyDepthNavigator(num_classes=4)
        self._load_weights(cfg.ML_MODEL_PATH)
        self._model.eval()
        self._smoother = SimpleSmoother(
            history_size=cfg.ML_HISTORY_SIZE,
            min_stable=cfg.ML_MIN_STABLE_COUNT,
            stop_frames=cfg.ML_STOP_FRAMES,
        )
        self._label_to_arduino = dict(LABEL_TO_ARDUINO)
        if cfg.FLIP_LR:
            self._label_to_arduino["LEFT"] = LABEL_TO_ARDUINO["RIGHT"]
            self._label_to_arduino["RIGHT"] = LABEL_TO_ARDUINO["LEFT"]

    def _load_weights(self, path: str) -> None:
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ML model not found: {model_path}. "
                "Train with train_navigator.py or set NAV_MODE=rules."
            )
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location="cpu")
        self._model.load_state_dict(state)
        print(f"[MLNav] Loaded model: {model_path}")

    def reset(self) -> None:
        self._smoother.reset()

    @staticmethod
    def preprocess_depth(
        depth_mm: np.ndarray,
        min_mm: int,
        max_mm: int,
    ) -> np.ndarray:
        clipped = np.clip(depth_mm.astype(np.float32), min_mm, max_mm)
        return clipped / float(max_mm)

    def _forward_min_depth_mm(self, depth_mm: np.ndarray) -> float:
        """Min depth in center-forward band (safety fallback)."""
        h, w = depth_mm.shape
        y0 = int(h * self.cfg.TOP_CROP_RATIO)
        y1 = int(h * self.cfg.BOTTOM_CROP_RATIO)
        x0 = int(w * 0.35)
        x1 = int(w * 0.65)
        roi = depth_mm[y0:y1, x0:x1]
        valid = roi[(roi >= self.cfg.MIN_DEPTH_MM) & (roi <= self.cfg.MAX_DEPTH_MM)]
        if valid.size == 0:
            return float(self.cfg.MAX_DEPTH_MM)
        return float(np.percentile(valid, 5))

    def _infer_label(self, depth_mm: np.ndarray) -> tuple[str, float]:
        norm = self.preprocess_depth(
            depth_mm, self.cfg.MIN_DEPTH_MM, self.cfg.MAX_DEPTH_MM
        )
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            idx = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, idx].item())
        return IDX_TO_CLASS[idx], conf

    def decide(
        self,
        depth_mm: np.ndarray,
        state: dict,
        lidar_front_mm: Optional[float] = None,
    ) -> MLDecisionResult:
        if not state.get("authorized", False):
            return MLDecisionResult(
                raw_command="NONE",
                stable_command="NONE",
                confidence=0.0,
                model_label="",
                reason="not_authorized",
            )

        if not state.get("ready", False):
            return MLDecisionResult(
                raw_command="NONE",
                stable_command="NONE",
                confidence=0.0,
                model_label="",
                reason="arduino_not_ready",
            )

        label, confidence = self._infer_label(depth_mm)
        reason_parts = [f"model={label} conf={confidence:.0%}"]

        # Depth forward safety (low obstacles / close wall)
        fwd_min = self._forward_min_depth_mm(depth_mm)
        if fwd_min < self.cfg.EMERGENCY_STOP_MM:
            label = "STOP"
            confidence = 0.99
            reason_parts.append(f"depth_fwd={fwd_min:.0f}mm")

        # LiDAR independent brake
        if lidar_front_mm is not None and lidar_front_mm < self.cfg.LIDAR_SAFETY_MM:
            label = "STOP"
            confidence = 0.99
            reason_parts.append(f"lidar_front={lidar_front_mm:.0f}mm")

        raw_arduino = self._label_to_arduino.get(label, "STOP")
        stable_arduino, stable_count = self._smoother.smooth(raw_arduino)

        critical = (
            stable_arduino == "STOP"
            and fwd_min > 0
            and fwd_min < self.cfg.CRITICAL_STOP_DISTANCE_MM
        )

        return MLDecisionResult(
            raw_command=raw_arduino,
            stable_command=stable_arduino,
            confidence=confidence,
            model_label=label,
            reason=" | ".join(reason_parts),
            critical_stop=critical,
            stable_count=stable_count,
            forward_min_depth_mm=fwd_min,
        )
