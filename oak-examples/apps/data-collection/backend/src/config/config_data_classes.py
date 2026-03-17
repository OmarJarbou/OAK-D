from pathlib import Path
from dataclasses import dataclass

from box import Box
import depthai as dai


@dataclass
class ModelInfo:
    """Model metadata from DepthAI zoo."""

    name: str
    precision: str
    yaml_path: Path
    width: int
    height: int
    description: dai.NNModelDescription
    archive: dai.NNArchive


@dataclass
class VideoConfig:
    """Config for CameraSourceNode."""

    fps: int
    media_path: str
    width: int
    height: int


@dataclass
class NeuralNetworkConfig:
    """Config for NNDetectionNode."""

    model: ModelInfo
    backend_type: str
    runtime: str
    performance_profile: str
    num_inference_threads: int
    confidence_thr: float
    prompts: Box


@dataclass
class TrackingConfig:
    """Config for TrackingNode."""

    track_per_class: bool
    birth_threshold: int
    max_lifespan: int
    occlusion_ratio_threshold: float
    tracker_threshold: float
