from abc import ABC, abstractmethod
from enum import Enum
from time import monotonic
from typing import Any, Dict, List, Optional
import logging

import depthai as dai
from box import Box
from pydantic import BaseModel, Field

from snapping.tracklet_analyzer import TrackletAnalyzer

logger = logging.getLogger(__name__)


class ConditionKey(str, Enum):
    """Unique identifiers for snapping conditions."""

    TIMED = "timed"
    NO_DETECTIONS = "noDetections"
    LOW_CONFIDENCE = "lowConfidence"
    LOST_MID = "lostMid"


class ConditionConfig(BaseModel):
    """Configuration for a single snapping condition (YAML/FE payload)."""

    enabled: bool = Field(default=True)
    cooldown: Optional[float] = Field(default=None, ge=0.0)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    margin: Optional[float] = Field(default=None, ge=0.0)


class Condition(ABC):
    """
    Abstract base class for all snap trigger conditions.

    Each condition has:
    - A unique KEY (class-level constant)
    - Human-readable name and optional tags
    - Configuration: enabled state, cooldown interval
    - Cooldown tracking to prevent rapid-fire triggers
    """

    KEY: ConditionKey  # Must be defined in subclasses

    def __init__(
        self,
        name: str,
        default_cooldown: float,
        tags: Optional[List[str]] = None,
    ):
        if not getattr(self, "KEY", None):
            raise ValueError(f"{self.__class__.__name__} must define a KEY constant")

        self.name = name
        self.tags = tags or []
        self.enabled: bool = False
        self.cooldown: float = max(0.0, float(default_cooldown))
        self._last_trigger_time: Optional[float] = None  # monotonic seconds

    @abstractmethod
    def should_trigger(self, detections: list, tracklets: dai.Tracklets) -> bool:
        """Return True if this condition should trigger a snap."""
        raise NotImplementedError

    @abstractmethod
    def make_extras(self) -> Dict[str, str]:
        """Return optional metadata to attach to the snap."""
        raise NotImplementedError

    def get_key(self) -> ConditionKey:
        return self.KEY

    def apply_config(self, conf: ConditionConfig) -> None:
        """Apply configuration from YAML / frontend."""
        self.enabled = conf.enabled
        if not self.enabled:
            self.reset_cooldown()
        if conf.cooldown is not None:
            self.cooldown = float(max(0.0, conf.cooldown))

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration for frontend."""
        return {"enabled": self.enabled, "cooldown": self.cooldown}

    def reset_cooldown(self) -> None:
        """Reset internal cooldown tracking."""
        self._last_trigger_time = None

    def _cooldown_passed(self) -> bool:
        """Return True if enough time has passed since last trigger."""
        if self._last_trigger_time is None:
            return True
        return (monotonic() - self._last_trigger_time) >= self.cooldown

    def mark_triggered(self) -> None:
        """Record that this condition has just fired."""
        self._last_trigger_time = monotonic()


class TimedCondition(Condition):
    """Triggers snaps at regular time intervals."""

    KEY = ConditionKey.TIMED

    def __init__(
        self, name: str, default_cooldown: float, tags: Optional[List[str]] = None
    ):
        super().__init__(name, default_cooldown, tags or [])

    def should_trigger(self, detections: list, tracklets: dai.Tracklets) -> bool:
        if self.enabled and self._cooldown_passed():
            self.mark_triggered()
            return True
        return False

    def make_extras(self) -> Dict[str, str]:
        return {"reason": "timed_snap"}


class NoDetectionsCondition(Condition):
    """Triggers when no objects are detected in the current frame."""

    KEY = ConditionKey.NO_DETECTIONS

    def __init__(
        self, name: str, default_cooldown: float, tags: Optional[List[str]] = None
    ):
        super().__init__(name, default_cooldown, tags or [])

    def should_trigger(self, detections: list, tracklets: dai.Tracklets) -> bool:
        if self.enabled and self._cooldown_passed():
            if not detections:
                self.mark_triggered()
                return True
        return False

    def make_extras(self) -> Dict[str, str]:
        return {"reason": "no_detections"}


class LowConfidenceCondition(Condition):
    """Triggers when any detection has confidence below a threshold."""

    KEY = ConditionKey.LOW_CONFIDENCE

    def __init__(
        self, name: str, default_cooldown: float, tags: Optional[List[str]] = None
    ):
        super().__init__(name, default_cooldown, tags or [])
        self.threshold: float = 0.3
        self.last_lowest: float = 0.0

    def should_trigger(self, detections: list, tracklets: dai.Tracklets) -> bool:
        if self.enabled and self._cooldown_passed():
            if self._check_detections(detections):
                self.mark_triggered()
                return True
        return False

    def _check_detections(self, detections: list) -> bool:
        if not detections:
            return False
        self.last_lowest = min((float(d.confidence) for d in detections), default=1.0)
        return self.last_lowest < self.threshold

    def apply_config(self, conf: ConditionConfig) -> None:
        super().apply_config(conf)
        if conf.threshold is not None:
            val = float(conf.threshold)
            self.threshold = max(0.0, min(1.0, val))

    def export_config(self) -> Dict[str, Any]:
        config = super().export_config()
        config["threshold"] = self.threshold
        return config

    def make_extras(self) -> Dict[str, str]:
        return {
            "reason": "low_confidence",
            "threshold": f"{round(self.threshold, 3)}",
            "min_conf": f"{round(self.last_lowest, 3)}",
        }


class LostMidCondition(Condition):
    """Triggers when an object is lost inside the frame center region."""

    KEY = ConditionKey.LOST_MID

    def __init__(
        self, name: str, default_cooldown: float, tags: Optional[List[str]] = None
    ):
        super().__init__(name, default_cooldown, tags or [])
        self.margin: float = 0.2
        self.prev_tracked: Dict[int, bool] = {}

    def should_trigger(self, detections: list, tracklets: dai.Tracklets) -> bool:
        if self.enabled and self._cooldown_passed():
            if self._check_tracklets(tracklets):
                self.mark_triggered()
                return True
        return False

    def _check_tracklets(self, tracklets: Optional[dai.Tracklets]) -> bool:
        if tracklets is None:
            return False

        triggered = False
        for t in getattr(tracklets, "tracklets", []):
            tr = TrackletAnalyzer(t)

            if tr.is_lost and tr.was_tracked(self.prev_tracked):
                rc = tr.center_area()
                if rc is not None:
                    cx, cy, _ = rc
                    if (
                        self.margin <= cx <= 1 - self.margin
                        and self.margin <= cy <= 1 - self.margin
                    ):
                        triggered = True

            tr.update_state(self.prev_tracked)

        return triggered

    def apply_config(self, conf: ConditionConfig) -> None:
        super().apply_config(conf)
        if conf.margin is not None:
            val = float(conf.margin)
            self.margin = max(0.0, min(0.49, val))

    def export_config(self) -> Dict[str, Any]:
        config = super().export_config()
        config["margin"] = self.margin
        return config

    def make_extras(self) -> Dict[str, str]:
        return {"reason": "lost_in_middle", "margin": f"{round(self.margin, 3)}"}


CONDITION_CLASSES = {
    ConditionKey.TIMED: TimedCondition,
    ConditionKey.NO_DETECTIONS: NoDetectionsCondition,
    ConditionKey.LOW_CONFIDENCE: LowConfidenceCondition,
    ConditionKey.LOST_MID: LostMidCondition,
}


def build_conditions(config: Box) -> Dict[ConditionKey, Condition]:
    """
    Build conditions from YAML configuration.

    @param config: Box config with 'conditions' list and 'cooldown' default.
    @return: Dictionary mapping ConditionKey to Condition instances.
    """
    conditions: Dict[ConditionKey, Condition] = {}

    default_cooldown = float(config.get("cooldown", 60.0))

    for entry in config.get("conditions", []):
        key_str = entry.get("key")
        if not key_str:
            continue

        try:
            key = ConditionKey(key_str)
            cls = CONDITION_CLASSES.get(key)
            if cls is None:
                logger.warning("Unknown condition key: %s", key_str)
                continue

            condition = cls(
                name=entry.get("name", key_str),
                default_cooldown=default_cooldown,
                tags=entry.get("tags", []),
            )

            initial_config = ConditionConfig(
                enabled=entry.get("enabled", False),
                cooldown=entry.get("cooldown"),
                threshold=entry.get("threshold"),
                margin=entry.get("margin"),
            )
            condition.apply_config(initial_config)

            conditions[key] = condition

        except ValueError as e:
            logger.error("Invalid condition key '%s': %s", key_str, e)

    return conditions
